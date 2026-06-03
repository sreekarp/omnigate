"""Opt-in Redis response cache for deterministic, non-streaming chat calls.

Eligibility (all must hold) — see :func:`should_cache`:

* the request is **not** streaming (``request.stream is False``),
* the request is deterministic: ``temperature == 0`` *explicitly* (``None`` is
  excluded because provider defaults vary, e.g. OpenAI defaults to ``1.0``),
* caching is enabled: global ``response_cache_enabled`` **or** the per-request
  ``request.cache is True``, and the request did not opt out (``cache is False``).

The cache key is the gateway prefix followed by the SHA-256 of a compact,
key-sorted JSON of the cache-relevant fields, scoped per project + provider so
projects never serve each other's BYOK-billed completions.

All Redis access is **non-fatal**: any error is logged at ``WARNING`` and the
caller proceeds as a miss / no-op — the cache must never take down ``/v1/chat``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING

from app.schemas.chat import ChatRequest, ChatResponse

if TYPE_CHECKING:  # pragma: no cover - typing only
    import redis.asyncio as redis

    from app.config import Settings

logger = logging.getLogger(__name__)


def should_cache(request: ChatRequest, settings: "Settings") -> bool:
    """Return ``True`` if ``request`` is eligible for the response cache.

    Deterministic + non-streaming + (globally enabled or per-request opt-in)
    and not explicitly opted out.
    """
    if request.stream:
        return False
    if request.cache is False:
        return False
    if request.temperature != 0:
        # Excludes ``None`` (provider default, possibly random) and any t > 0.
        return False
    enabled = settings.response_cache_enabled or request.cache is True
    return bool(enabled)


def cache_key(project_id: object, provider: str, request: ChatRequest) -> str:
    """Build the cache key for a (project, provider, request) triple.

    The digest covers only fields that affect the deterministic output:
    provider, model (the *requested* model, not a fallback), messages, and the
    sampling/limit parameters ``max_tokens``, ``top_p``, ``stop``, ``seed``.
    Scoped by ``project_id`` for BYOK billing/privacy isolation.
    """
    from app.config import get_settings

    prefix = get_settings().response_cache_prefix
    payload = {
        "project_id": str(project_id),
        "provider": provider,
        "model": request.model,
        "messages": [m.model_dump() for m in request.messages],
        "max_tokens": request.max_tokens,
        "top_p": request.top_p,
        "stop": request.stop_sequences(),
        "seed": request.seed,
    }
    blob = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return f"{prefix}{digest}"


async def cache_get(redis_client: "redis.Redis", key: str) -> ChatResponse | None:
    """Return the cached :class:`ChatResponse` for ``key`` (with ``cached=True``).

    Returns ``None`` on a miss, on a malformed payload, or on any Redis error.
    """
    try:
        raw = await redis_client.get(key)
    except Exception:  # noqa: BLE001 - non-fatal: treat as a miss
        logger.warning("response cache get failed for %s", key, exc_info=True)
        return None
    if raw is None:
        return None
    try:
        # decode_responses=True yields str; validate either way.
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        response = ChatResponse.model_validate_json(raw)
    except Exception:  # noqa: BLE001 - corrupt entry: treat as a miss
        logger.warning("response cache decode failed for %s", key, exc_info=True)
        return None
    response.cached = True
    return response


async def cache_set(
    redis_client: "redis.Redis", key: str, response: ChatResponse, ttl: int
) -> None:
    """Store ``response`` under ``key`` with a TTL (seconds). Non-fatal on error."""
    if ttl <= 0:
        return
    try:
        await redis_client.set(key, response.model_dump_json(), ex=ttl)
    except Exception:  # noqa: BLE001 - non-fatal: silently skip caching
        logger.warning("response cache set failed for %s", key, exc_info=True)
