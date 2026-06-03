"""Chat execution orchestrator.

Composes the resilience features around a provider call for the non-streaming
path, keeping the chat router thin:

    cache lookup -> [per candidate model: resolve -> circuit breaker -> retry]
                 -> cache store

Fallback models are tried in order; each may resolve to a *different* provider
with its own stored BYOK key. The circuit breaker only counts transient
failures (5xx/timeout); client errors (4xx) skip to the next candidate without
tripping it. Streaming is handled directly in the router (primary model only).
"""

import uuid
from dataclasses import dataclass
from decimal import Decimal

from app.config import get_settings
from app.logging_config import get_logger
from app.models.db import Project
from app.providers.azure_openai import AzureOpenAIProvider
from app.providers.base import AbstractProvider, ProviderError
from app.providers.registry import get_provider, provider_name_for_model
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.cache import cache_get, cache_key, cache_set, should_cache
from app.services.circuit_breaker import circuit_key, get_circuit_breaker
from app.services.credentials import get_provider_credential
from app.services.pricing import compute_cost
from app.utils.resilience import RetryPolicy, is_retryable, retry_async

logger = get_logger(__name__)

_MAX_CANDIDATES = 6  # primary + up to 5 fallbacks


@dataclass
class ResolvedProvider:
    provider: AbstractProvider
    provider_name: str
    api_key: str
    model: str


@dataclass
class ChatOutcome:
    response: ChatResponse
    model_used: str
    provider_used: str
    cached: bool
    fallback_used: bool
    status: str  # "ok" | "cache_hit"
    cost: Decimal


def candidate_models(request: ChatRequest) -> list[str]:
    """Ordered, de-duplicated list of [primary, *fallbacks] (capped)."""
    out: list[str] = []
    for model in [request.model, *request.fallback_models]:
        if model and model not in out:
            out.append(model)
    return out[:_MAX_CANDIDATES]


async def resolve_provider(
    session, project: Project, model: str
) -> ResolvedProvider:
    """Resolve a model to a ready-to-call provider + the project's BYOK key.

    Raises :class:`ProviderError` (400) for unknown models, missing keys, or
    incomplete Azure metadata.
    """
    provider_name = provider_name_for_model(model)
    cred = await get_provider_credential(
        session, project_id=project.id, provider=provider_name
    )
    if cred is None:
        raise ProviderError(
            f"No {provider_name} key configured for this project. "
            f"Add one via POST /v1/keys.",
            status_code=400,
        )
    api_key, meta = cred

    if provider_name == "azure":
        meta = meta or {}
        endpoint = meta.get("endpoint")
        if not endpoint:
            raise ProviderError(
                "Azure credential is missing 'endpoint' in its metadata.",
                status_code=400,
            )
        deployment: str | None = None
        if model.lower().startswith("azure/"):
            deployment = model.split("/", 1)[1] or None
        deployment = deployment or meta.get("deployment")
        if not deployment:
            raise ProviderError(
                "Azure deployment not specified. Use model 'azure/<deployment>' "
                "or set meta.deployment.",
                status_code=400,
            )
        kwargs: dict = {"endpoint": endpoint, "deployment": deployment}
        if meta.get("api_version"):
            kwargs["api_version"] = meta["api_version"]
        provider: AbstractProvider = AzureOpenAIProvider(**kwargs)
    else:
        provider = get_provider(provider_name)

    return ResolvedProvider(
        provider=provider,
        provider_name=provider_name,
        api_key=api_key,
        model=model,
    )


async def execute_chat(
    session,
    *,
    project: Project,
    request: ChatRequest,
    redis_client,
) -> ChatOutcome:
    """Run a non-streaming chat with caching, circuit-breaking, retry & fallback."""
    settings = get_settings()

    # --- Response cache (primary model only, deterministic requests) ---
    cache_enabled = should_cache(request, settings)
    primary_provider: str | None = None
    ckey: str | None = None
    if cache_enabled:
        try:
            primary_provider = provider_name_for_model(request.model)
            ckey = cache_key(project.id, primary_provider, request)
        except ProviderError:
            cache_enabled = False
        if cache_enabled and ckey is not None:
            hit = await cache_get(redis_client, ckey)
            if hit is not None:
                hit.cost_usd = 0.0  # cache hits are not billed
                return ChatOutcome(
                    response=hit,
                    model_used=request.model,
                    provider_used=primary_provider or hit.provider,
                    cached=True,
                    fallback_used=False,
                    status="cache_hit",
                    cost=Decimal("0"),
                )

    breaker = get_circuit_breaker(settings, redis_client=redis_client)
    policy = RetryPolicy.from_settings(settings)
    last_error: ProviderError | None = None

    for model in candidate_models(request):
        is_fallback = model != request.model
        try:
            rp = await resolve_provider(session, project, model)
        except ProviderError as exc:
            last_error = exc
            logger.warning("Skipping candidate %r: %s", model, exc.message)
            continue

        breaker_key = circuit_key(rp.provider_name, project.id)
        if not await breaker.allow(breaker_key):
            last_error = ProviderError(
                f"{rp.provider_name} circuit is open", status_code=503
            )
            logger.warning("Circuit open for %s; skipping candidate %r", rp.provider_name, model)
            continue

        req_for_model = (
            request
            if not is_fallback
            else request.model_copy(update={"model": model})
        )

        async def _call(
            provider: AbstractProvider = rp.provider,
            req: ChatRequest = req_for_model,
            key: str = rp.api_key,
        ) -> ChatResponse:
            return await provider.chat(req, key)

        try:
            result = await retry_async(_call, policy=policy)
        except ProviderError as exc:
            last_error = exc
            if is_retryable(exc):
                await breaker.record_failure(breaker_key)
            logger.warning("Candidate %r failed: %s", model, exc.message)
            continue
        except Exception as exc:  # noqa: BLE001 - defensive; never leak raw errors
            last_error = ProviderError(
                f"{rp.provider_name} call failed: {exc}", status_code=502
            )
            await breaker.record_failure(breaker_key)
            logger.warning("Candidate %r raised unexpectedly: %s", model, exc)
            continue

        await breaker.record_success(breaker_key)

        cost = compute_cost(
            result.model or model,
            result.usage.prompt_tokens,
            result.usage.completion_tokens,
        )
        result.cost_usd = float(cost)
        result.fallback_used = is_fallback

        if cache_enabled and ckey is not None and not is_fallback:
            await cache_set(
                redis_client, ckey, result, settings.response_cache_ttl_seconds
            )

        return ChatOutcome(
            response=result,
            model_used=model,
            provider_used=rp.provider_name,
            cached=False,
            fallback_used=is_fallback,
            status="ok",
            cost=cost,
        )

    if last_error is not None:
        raise last_error
    raise ProviderError("No provider candidate could handle the request", status_code=502)
