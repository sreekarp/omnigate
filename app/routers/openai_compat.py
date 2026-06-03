"""OpenAI-compatible API surface.

``POST /v1/chat/completions`` is wire-compatible with the OpenAI Chat
Completions API, so the official ``openai`` SDK can point ``base_url`` at the
gateway and work unchanged (auth via ``Authorization: Bearer <gateway_key>`` or
``x-api-key``). ``GET /v1/models`` / ``GET /v1/models/{id}`` expose the catalog.

Reuses the full pipeline: auth -> rate limit -> budget (``enforce_budget``),
:func:`execute_chat` (cache/breaker/retry/fallback), and usage recording — so
cost tracking and budgets apply identically to compat traffic.
"""

import json
import time
import uuid
from collections.abc import AsyncIterator
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import JSONResponse, StreamingResponse

from app.db.session import SessionLocal, get_session
from app.logging_config import get_logger
from app.middleware import AuthContext, enforce_budget, get_auth_context
from app.observability import observe_request
from app.providers.base import ProviderError
from app.redis_client import get_redis
from app.schemas.chat import Usage
from app.schemas.openai_compat import (
    OAIChatCompletionRequest,
    build_chunk,
    build_completion_response,
    build_role_chunk,
    build_usage_chunk,
    to_internal_chat_request,
)
from app.services.catalog import model_card, model_cards
from app.services.circuit_breaker import circuit_key, get_circuit_breaker
from app.services.pricing import compute_cost
from app.services.routing import execute_chat, resolve_provider
from app.services.usage import record_usage

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["openai-compat"])


def _oai_error(message: str, status_code: int, err_type: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": err_type, "param": None, "code": None}},
    )


def _sse(payload: dict) -> str:
    return "data: " + json.dumps(payload, separators=(",", ":")) + "\n\n"


@router.post("/chat/completions")
async def chat_completions(
    payload: OAIChatCompletionRequest,
    request: Request,
    ctx: AuthContext = Depends(enforce_budget),
    session: AsyncSession = Depends(get_session),
    redis_client=Depends(get_redis),
):
    project = ctx.project
    try:
        internal = to_internal_chat_request(payload)
    except ValueError as exc:
        return _oai_error(str(exc), 400)

    request_id = "chatcmpl-" + uuid.uuid4().hex
    created = int(time.time())

    if internal.stream:
        return await _completions_stream(
            internal, ctx, session, redis_client, request_id, created, payload.wants_stream_usage()
        )

    started = time.perf_counter()
    try:
        outcome = await execute_chat(
            session, project=project, request=internal, redis_client=redis_client
        )
    except ProviderError as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        await record_usage(
            session, org_id=project.org_id, project_id=project.id, user_id=ctx.user_id,
            provider="unknown", model=internal.model, prompt_tokens=0, completion_tokens=0,
            cost=Decimal("0"), status="error", latency_ms=latency_ms, request_id=request_id,
        )
        observe_request(provider="unknown", model=internal.model, status="error", latency_ms=latency_ms)
        return _oai_error(exc.message, exc.status_code, "provider_error" if exc.status_code >= 500 else "invalid_request_error")

    latency_ms = int((time.perf_counter() - started) * 1000)
    result = outcome.response
    result.id = request_id
    await record_usage(
        session, org_id=project.org_id, project_id=project.id, user_id=ctx.user_id,
        provider=outcome.provider_used, model=result.model,
        prompt_tokens=result.usage.prompt_tokens, completion_tokens=result.usage.completion_tokens,
        cost=outcome.cost, status=outcome.status, latency_ms=latency_ms, request_id=request_id,
    )
    observe_request(
        provider=outcome.provider_used, model=result.model, status=outcome.status,
        latency_ms=latency_ms, prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens, cost_usd=float(outcome.cost),
    )
    return JSONResponse(content=build_completion_response(result, created))


async def _completions_stream(
    internal, ctx, session, redis_client, request_id, created, include_usage,
) -> StreamingResponse:
    project = ctx.project
    try:
        rp = await resolve_provider(session, project, internal.model)
    except ProviderError as exc:
        return _oai_error(exc.message, exc.status_code)

    breaker = get_circuit_breaker(redis_client=redis_client)
    breaker_key = circuit_key(rp.provider_name, project.id)
    if not await breaker.allow(breaker_key):
        return _oai_error(f"{rp.provider_name} circuit is open", 503, "provider_error")

    model_label = rp.model
    started = time.perf_counter()

    async def gen() -> AsyncIterator[str]:
        status_str = "ok"
        usage = Usage()
        finish = "stop"
        try:
            yield _sse(build_role_chunk(id=request_id, created=created, model=model_label))
            async for chunk in rp.provider.stream(internal, rp.api_key):
                if chunk.text:
                    yield _sse(build_chunk(chunk.text, id=request_id, created=created, model=model_label))
                if chunk.usage is not None:
                    usage = chunk.usage
                if chunk.finish_reason:
                    finish = chunk.finish_reason
            yield _sse(build_chunk("", id=request_id, created=created, model=model_label, finish_reason=finish))
            if include_usage:
                yield _sse(build_usage_chunk(
                    id=request_id, created=created, model=model_label,
                    prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                ))
        except ProviderError as exc:
            status_str = "error"
            logger.warning("compat stream error %s: %s", request_id, exc.message)
            yield _sse({"error": {"message": exc.message, "type": "provider_error"}})
        except Exception as exc:  # noqa: BLE001
            status_str = "error"
            logger.warning("compat stream unexpected %s: %s", request_id, exc)
            yield _sse({"error": {"message": str(exc), "type": "provider_error"}})
        finally:
            # Always terminate the SSE stream (even on error).
            yield "data: [DONE]\n\n"
            latency_ms = int((time.perf_counter() - started) * 1000)
            cost = (
                compute_cost(internal.model, usage.prompt_tokens, usage.completion_tokens)
                if status_str == "ok" else Decimal("0")
            )
            try:
                await (breaker.record_success(breaker_key) if status_str == "ok"
                       else breaker.record_failure(breaker_key))
            except Exception:  # noqa: BLE001
                pass
            observe_request(
                provider=rp.provider_name, model=internal.model, status=status_str,
                latency_ms=latency_ms, prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens, cost_usd=float(cost),
            )
            try:
                async with SessionLocal() as fresh:
                    await record_usage(
                        fresh, org_id=project.org_id, project_id=project.id, user_id=ctx.user_id,
                        provider=rp.provider_name, model=internal.model,
                        prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
                        cost=cost, status=status_str, latency_ms=latency_ms, request_id=request_id,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.error("compat stream usage record failed %s: %s", request_id, exc)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"x-request-id": request_id, "Cache-Control": "no-cache"},
    )


@router.get("/models")
async def list_models(ctx: AuthContext = Depends(get_auth_context)) -> dict:
    return {"object": "list", "data": model_cards()}


@router.get("/models/{model_id:path}")
async def get_model(model_id: str, ctx: AuthContext = Depends(get_auth_context)):
    card = model_card(model_id)
    if card is None:
        return _oai_error(f"model {model_id!r} not found", 404, "invalid_request_error")
    return card
