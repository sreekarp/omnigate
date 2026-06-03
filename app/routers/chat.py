"""POST /v1/chat — the main gateway endpoint.

Runs the full middleware chain (auth -> rate limit -> budget) via the
``enforce_budget`` dependency, then delegates to :func:`execute_chat` (cache,
circuit breaker, retry, fallback) for non-streaming requests, or streams
directly for streaming ones.

Streaming records REAL token usage now: providers emit a terminal usage chunk
(via ``stream_options.include_usage`` / Anthropic ``message_delta`` / Gemini
``usageMetadata``), which we accumulate and persist when the stream ends.

IMPORTANT (P0): a streaming generator's ``finally`` runs *after* the response
body is consumed — i.e. after the request's ``Depends(get_session)`` session is
closed. So usage is recorded from a FRESH session opened inside the generator.
"""

import time
import uuid
from collections.abc import AsyncIterator
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from app.config import get_settings
from app.db.session import SessionLocal, get_session
from app.logging_config import get_logger
from app.middleware import AuthContext, enforce_budget
from app.observability import observe_request
from app.providers.base import ProviderError
from app.providers.registry import provider_name_for_model
from app.redis_client import get_redis
from app.schemas.chat import ChatRequest, ChatResponse, Usage
from app.services.circuit_breaker import circuit_key, get_circuit_breaker
from app.services.pricing import compute_cost
from app.services.routing import ResolvedProvider, execute_chat, resolve_provider
from app.services.usage import record_usage

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["chat"])


def _safe_provider_name(model: str) -> str:
    try:
        return provider_name_for_model(model)
    except ProviderError:
        return "unknown"


@router.post("/chat")
async def chat(
    request: ChatRequest,
    response: Response,
    ctx: AuthContext = Depends(enforce_budget),
    session: AsyncSession = Depends(get_session),
    redis_client=Depends(get_redis),
):
    request_id = str(uuid.uuid4())
    project = ctx.project

    if request.stream:
        return await _start_stream(request, ctx, session, redis_client, request_id)

    started = time.perf_counter()
    try:
        outcome = await execute_chat(
            session, project=project, request=request, redis_client=redis_client
        )
    except ProviderError as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        provider_name = _safe_provider_name(request.model)
        await record_usage(
            session,
            org_id=project.org_id,
            project_id=project.id,
            user_id=ctx.user_id,
            provider=provider_name,
            model=request.model,
            prompt_tokens=0,
            completion_tokens=0,
            cost=Decimal("0"),
            status="error",
            latency_ms=latency_ms,
            request_id=request_id,
        )
        observe_request(
            provider=provider_name,
            model=request.model,
            status="error",
            latency_ms=latency_ms,
        )
        logger.warning("Provider error for request %s: %s", request_id, exc.message)
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    latency_ms = int((time.perf_counter() - started) * 1000)
    result = outcome.response
    result.latency_ms = latency_ms
    result.cached = outcome.cached

    await record_usage(
        session,
        org_id=project.org_id,
        project_id=project.id,
        user_id=ctx.user_id,
        provider=outcome.provider_used,
        model=result.model,
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        cost=outcome.cost,
        status=outcome.status,
        latency_ms=latency_ms,
        request_id=request_id,
    )
    observe_request(
        provider=outcome.provider_used,
        model=result.model,
        status=outcome.status,
        latency_ms=latency_ms,
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        cost_usd=float(outcome.cost),
    )

    response.headers["x-request-id"] = request_id
    response.headers["x-llmgw-provider"] = outcome.provider_used
    response.headers["x-llmgw-model"] = result.model
    response.headers["x-llmgw-cached"] = str(outcome.cached).lower()
    response.headers["x-llmgw-fallback"] = str(outcome.fallback_used).lower()
    return result


async def _start_stream(
    request: ChatRequest,
    ctx: AuthContext,
    session: AsyncSession,
    redis_client,
    request_id: str,
) -> StreamingResponse:
    """Pre-flight resolve (so missing-key/circuit errors surface as HTTP), then stream.

    Streaming uses the PRIMARY model only — no mid-stream fallback (you can't
    un-send bytes). Retry/cache do not apply to streams.
    """
    project = ctx.project
    settings = get_settings()
    try:
        rp = await resolve_provider(session, project, request.model)
    except ProviderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    breaker = get_circuit_breaker(settings, redis_client=redis_client)
    breaker_key = circuit_key(rp.provider_name, project.id)
    if not await breaker.allow(breaker_key):
        raise HTTPException(status_code=503, detail=f"{rp.provider_name} circuit is open")

    return _stream_response(request, ctx, rp, breaker, breaker_key, request_id)


def _stream_response(
    request: ChatRequest,
    ctx: AuthContext,
    rp: ResolvedProvider,
    breaker,
    breaker_key: str,
    request_id: str,
) -> StreamingResponse:
    project = ctx.project
    started = time.perf_counter()

    async def event_generator() -> AsyncIterator[str]:
        status_str = "ok"
        usage = Usage()
        try:
            async for chunk in rp.provider.stream(request, rp.api_key):
                if chunk.text:
                    yield chunk.text
                if chunk.usage is not None:
                    usage = chunk.usage
        except ProviderError as exc:
            status_str = "error"
            logger.warning("Streaming error for request %s: %s", request_id, exc.message)
            yield f"\n[error] {exc.message}"
        except Exception as exc:  # noqa: BLE001 - defensive
            status_str = "error"
            logger.warning("Unexpected streaming error %s: %s", request_id, exc)
            yield f"\n[error] {exc}"
        finally:
            latency_ms = int((time.perf_counter() - started) * 1000)
            cost = (
                compute_cost(request.model, usage.prompt_tokens, usage.completion_tokens)
                if status_str == "ok"
                else Decimal("0")
            )
            try:
                if status_str == "ok":
                    await breaker.record_success(breaker_key)
                else:
                    await breaker.record_failure(breaker_key)
            except Exception:  # noqa: BLE001 - breaker is best-effort
                pass
            observe_request(
                provider=rp.provider_name,
                model=request.model,
                status=status_str,
                latency_ms=latency_ms,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cost_usd=float(cost),
            )
            # Fresh session: the request-scoped one may already be closed here.
            try:
                async with SessionLocal() as fresh:
                    await record_usage(
                        fresh,
                        org_id=project.org_id,
                        project_id=project.id,
                        user_id=ctx.user_id,
                        provider=rp.provider_name,
                        model=request.model,
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        cost=cost,
                        status=status_str,
                        latency_ms=latency_ms,
                        request_id=request_id,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to record streaming usage for %s: %s", request_id, exc)

    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
        headers={
            "x-request-id": request_id,
            "x-llmgw-provider": rp.provider_name,
            "x-llmgw-model": request.model,
        },
    )
