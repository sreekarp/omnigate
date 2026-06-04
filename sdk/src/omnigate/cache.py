"""In-memory TTL response cache for deterministic, non-streaming completions.

The in-process analog of the gateway's Redis response cache. Eligibility (see
:func:`should_cache`): non-streaming, deterministic (``temperature == 0``
*explicitly*), and enabled either globally or per-request (``cache=True``), and
not explicitly opted out (``cache=False``).

The key is the SHA-256 of a compact, key-sorted JSON of the cache-relevant
fields (provider, model, messages, and deterministic sampling/limit params).
There is no project scoping (single tenant in-process). All access is non-fatal:
any error is logged and treated as a miss / no-op.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Callable, Optional

from .models import ChatRequest, ChatResponse

logger = logging.getLogger("omnigate")


def should_cache(request: ChatRequest, *, enabled: bool) -> bool:
    """Return ``True`` if ``request`` is eligible for the response cache."""
    if request.stream:
        return False
    if request.cache is False:
        return False
    if request.temperature != 0:
        # Excludes ``None`` (provider default, possibly random) and any t > 0.
        return False
    return bool(enabled or request.cache is True)


def cache_key(provider: str, request: ChatRequest) -> str:
    """Build the cache key for a (provider, request) pair.

    The digest covers only fields that affect the deterministic output:
    provider, model, messages, and ``max_tokens``/``top_p``/``stop``/``seed``.
    """
    payload = {
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
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    """A tiny TTL cache mapping cache key -> serialised :class:`ChatResponse`."""

    def __init__(self, now: Callable[[], float] = time.monotonic) -> None:
        self._store: dict[str, tuple[float, str]] = {}
        self._now = now

    def get(self, key: str) -> Optional[ChatResponse]:
        """Return the cached response (``cached=True``) or ``None`` on miss/expiry."""
        try:
            item = self._store.get(key)
            if item is None:
                return None
            expiry, raw = item
            if expiry <= self._now():
                self._store.pop(key, None)
                return None
            resp = ChatResponse.model_validate_json(raw)
            resp.cached = True
            return resp
        except Exception:  # noqa: BLE001 - non-fatal: treat as a miss
            logger.warning("response cache get failed", exc_info=True)
            return None

    def set(self, key: str, response: ChatResponse, ttl: int) -> None:
        """Store ``response`` under ``key`` with a TTL (seconds). Non-fatal."""
        if ttl <= 0:
            return
        try:
            self._store[key] = (self._now() + ttl, response.model_dump_json())
        except Exception:  # noqa: BLE001 - non-fatal: silently skip caching
            logger.warning("response cache set failed", exc_info=True)

    def clear(self) -> None:
        """Drop all cached entries (test helper / manual flush)."""
        self._store.clear()


_cache = ResponseCache()


def get_cache() -> ResponseCache:
    """Return the process-wide response cache instance."""
    return _cache
