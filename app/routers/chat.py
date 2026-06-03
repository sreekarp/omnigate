"""POST /v1/chat — the main gateway endpoint.

Runs the full middleware chain (auth -> rate limit -> budget) via the
``enforce_budget`` dependency, routes to the correct provider, and logs a
usage record. Supports streaming via StreamingResponse.

Streaming note: token usage from providers during streaming is not always
available, so for streamed requests we record token counts as 0 and cost 0,
with status "ok". Non-streaming requests record full usage and computed cost.
"""

import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from app.db.session import get_session
from app.logging_config import get_logger
from app.middleware import AuthContext, enforce_budget
from app.providers.base import ProviderError
from app.providers.registry import get_provider_for_model
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.credentials import get_provider_key
from app.services.pricing import compute_cost
from app.services.usage import record_usage

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat")
async def chat(
    request: ChatRequest,
    ctx: AuthContext = Depends(enforce_budget),
    session: AsyncSession = Depends(get_session),
):
    request_id = str(uuid.uuid4())
    project = ctx.project

    try:
        provider = get_provider_for_model(request.model)
    except ProviderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    # BYOK: use the developer's own stored provider key.
    api_key = await get_provider_key(
        session, project_id=project.id, provider=provider.name
    )
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No {provider.name} key configured for this project. "
                f"Add one via POST /v1/keys."
            ),
        )

    if request.stream:
        return await _stream(request, ctx, session, provider, api_key, request_id)

    started = time.perf_counter()
    try:
        result: ChatResponse = await provider.chat(request, api_key)
    except ProviderError as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        await record_usage(
            session,
            org_id=project.org_id,
            project_id=project.id,
            user_id=ctx.user_id,
            provider=provider.name,
            model=request.model,
            prompt_tokens=0,
            completion_tokens=0,
            cost=compute_cost(request.model, 0, 0),
            status="error",
            latency_ms=latency_ms,
            request_id=request_id,
        )
        logger.warning("Provider error for request %s: %s", request_id, exc.message)
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    latency_ms = int((time.perf_counter() - started) * 1000)
    cost = compute_cost(
        request.model,
        result.usage.prompt_tokens,
        result.usage.completion_tokens,
    )
    result.cost_usd = float(cost)

    await record_usage(
        session,
        org_id=project.org_id,
        project_id=project.id,
        user_id=ctx.user_id,
        provider=provider.name,
        model=result.model,
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        cost=cost,
        status="ok",
        latency_ms=latency_ms,
        request_id=request_id,
    )
    return result


async def _stream(
    request: ChatRequest,
    ctx: AuthContext,
    session: AsyncSession,
    provider,
    api_key: str,
    request_id: str,
) -> StreamingResponse:
    project = ctx.project
    started = time.perf_counter()

    async def event_generator() -> AsyncIterator[str]:
        status_str = "ok"
        try:
            async for piece in provider.stream(request, api_key):
                yield piece
        except ProviderError as exc:
            status_str = "error"
            logger.warning("Streaming error for request %s: %s", request_id, exc.message)
            yield f"\n[error] {exc.message}"
        finally:
            latency_ms = int((time.perf_counter() - started) * 1000)
            # Token usage is not reliably available mid-stream; record zeros.
            await record_usage(
                session,
                org_id=project.org_id,
                project_id=project.id,
                user_id=ctx.user_id,
                provider=provider.name,
                model=request.model,
                prompt_tokens=0,
                completion_tokens=0,
                cost=compute_cost(request.model, 0, 0),
                status=status_str,
                latency_ms=latency_ms,
                request_id=request_id,
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
        headers={"x-request-id": request_id},
    )
