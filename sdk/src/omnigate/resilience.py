"""Async + sync retry helpers with exponential backoff and full jitter.

Retries only *transient* failures: an :class:`~omnigate.exceptions.APIError`
whose ``status_code`` is retryable, or transport/timeout errors from ``httpx``.
The caller passes a zero-argument thunk (so each attempt re-issues a fresh
provider call), a :class:`RetryPolicy`, and may inject ``sleep`` for tests.

Ported from the gateway's ``app/utils/resilience.py`` and made standalone:
``is_retryable`` keys off the SDK's own exception hierarchy, and a synchronous
twin (:func:`retry_sync`) is added for the in-process sync path.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

import httpx

from .exceptions import APIError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import EngineConfig

logger = logging.getLogger("omnigate")

T = TypeVar("T")

#: HTTP-ish status codes that are worth retrying (transient/server-side).
_RETRYABLE_STATUS: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})


def is_retryable(exc: BaseException) -> bool:
    """Return ``True`` if ``exc`` represents a transient, retryable failure.

    Retryable when it is an :class:`APIError` whose ``status_code`` is in
    :data:`_RETRYABLE_STATUS`, or an ``httpx`` timeout/transport error (which
    surface before any provider status is known).
    """
    if isinstance(exc, APIError):
        return exc.status_code in _RETRYABLE_STATUS
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    return False


@dataclass(slots=True)
class RetryPolicy:
    """Parameters controlling the backoff schedule."""

    max_attempts: int
    base_delay: float
    max_delay: float
    jitter: float

    @classmethod
    def from_config(cls, cfg: "EngineConfig") -> "RetryPolicy":
        """Build a policy from an :class:`~omnigate.config.EngineConfig`."""
        return cls(
            max_attempts=cfg.retry_max_attempts,
            base_delay=cfg.retry_base_delay,
            max_delay=cfg.retry_max_delay,
            jitter=cfg.retry_jitter,
        )

    def delay_for(self, attempt: int, *, retry_after: float | None = None) -> float:
        """Seconds to sleep before retrying after a failed ``attempt`` (0-based).

        Computes ``base_delay * 2 ** attempt`` capped at ``max_delay`` plus
        uniform jitter in ``[0, jitter]``. When ``retry_after`` is provided and
        exceeds the computed delay, it is used instead (provider hint wins).
        """
        backoff = self.base_delay * (2 ** attempt)
        capped = min(backoff, self.max_delay)
        delay = capped + random.uniform(0.0, self.jitter)
        if retry_after is not None and retry_after > delay:
            return retry_after
        return delay


#: Async zero-argument thunk producing a value of type ``T``.
AsyncThunk = Callable[[], Awaitable[T]]
#: Sync zero-argument thunk producing a value of type ``T``.
SyncThunk = Callable[[], T]
#: Optional per-retry callback ``(attempt, exc) -> None``.
OnRetry = Callable[[int, BaseException], None]


async def retry_async(
    func: AsyncThunk[T],
    *,
    policy: RetryPolicy,
    on_retry: OnRetry | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Call ``func`` (async thunk) with retry on transient errors."""
    attempts = max(1, policy.max_attempts)
    last_exc: BaseException | None = None

    for attempt in range(attempts):
        try:
            return await func()
        except BaseException as exc:  # noqa: BLE001 - re-raised below if needed
            last_exc = exc
            if attempt >= attempts - 1 or not is_retryable(exc):
                raise
            delay = policy.delay_for(
                attempt, retry_after=getattr(exc, "retry_after", None)
            )
            _notify(on_retry, attempt + 1, exc)
            logger.info(
                "retrying after transient error (attempt %d/%d, sleeping %.3fs): %s",
                attempt + 1, attempts, delay, exc,
            )
            await sleep(delay)

    assert last_exc is not None  # noqa: S101 - unreachable
    raise last_exc


def retry_sync(
    func: SyncThunk[T],
    *,
    policy: RetryPolicy,
    on_retry: OnRetry | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Synchronous twin of :func:`retry_async` (same backoff schedule)."""
    attempts = max(1, policy.max_attempts)
    last_exc: BaseException | None = None

    for attempt in range(attempts):
        try:
            return func()
        except BaseException as exc:  # noqa: BLE001 - re-raised below if needed
            last_exc = exc
            if attempt >= attempts - 1 or not is_retryable(exc):
                raise
            delay = policy.delay_for(
                attempt, retry_after=getattr(exc, "retry_after", None)
            )
            _notify(on_retry, attempt + 1, exc)
            logger.info(
                "retrying after transient error (attempt %d/%d, sleeping %.3fs): %s",
                attempt + 1, attempts, delay, exc,
            )
            sleep(delay)

    assert last_exc is not None  # noqa: S101 - unreachable
    raise last_exc


def _notify(on_retry: OnRetry | None, attempt: int, exc: BaseException) -> None:
    if on_retry is None:
        return
    try:
        on_retry(attempt, exc)
    except Exception:  # noqa: BLE001 - callback must never break retry
        logger.warning("retry on_retry callback raised", exc_info=True)
