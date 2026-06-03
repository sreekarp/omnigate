"""Hand-rolled async retry helper with exponential backoff and full jitter.

Retries only *transient* failures: :class:`~app.providers.base.ProviderError`
with a retryable HTTP-ish status, or transport/timeout errors from ``httpx``.
The caller passes a zero-argument async *thunk* (so each attempt can re-issue a
fresh provider call), a :class:`RetryPolicy`, and may inject ``sleep`` for tests.

Design notes
------------
* Backoff = ``base_delay * 2 ** attempt`` (0-based attempt index) capped at
  ``max_delay``, plus uniform jitter in ``[0, jitter]``.
* When a :class:`ProviderError` carries a ``retry_after`` hint that is *larger*
  than the computed delay, the hint wins (providers know better on 429/503).
* Non-retryable errors propagate immediately; the last error is re-raised after
  ``max_attempts`` are exhausted.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

import httpx

from app.providers.base import ProviderError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: HTTP-ish status codes that are worth retrying (transient/server-side).
_RETRYABLE_STATUS: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})

#: Type of an async, zero-argument thunk producing a value of type ``T``.
Thunk = Callable[[], Awaitable[T]]
#: Type of the optional per-retry callback ``(attempt, exc) -> None``.
OnRetry = Callable[[int, BaseException], None]
#: Type of the injectable async sleep ``(seconds) -> None``.
Sleep = Callable[[float], Awaitable[None]]


def is_retryable(exc: BaseException) -> bool:
    """Return ``True`` if ``exc`` represents a transient, retryable failure.

    Retryable when it is a :class:`ProviderError` whose ``status_code`` is in
    :data:`_RETRYABLE_STATUS`, or an ``httpx`` timeout/transport error (which
    surface before any provider status is known).
    """
    if isinstance(exc, ProviderError):
        return exc.status_code in _RETRYABLE_STATUS
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    return False


@dataclass(slots=True)
class RetryPolicy:
    """Parameters controlling :func:`retry_async` backoff."""

    max_attempts: int
    base_delay: float
    max_delay: float
    jitter: float

    @classmethod
    def from_settings(cls, settings: "Settings") -> "RetryPolicy":
        """Build a policy from the application :class:`Settings`."""
        return cls(
            max_attempts=settings.retry_max_attempts,
            base_delay=settings.retry_base_delay_seconds,
            max_delay=settings.retry_max_delay_seconds,
            jitter=settings.retry_jitter_seconds,
        )

    def delay_for(self, attempt: int, *, retry_after: float | None = None) -> float:
        """Return the sleep (seconds) before retrying after a failed ``attempt``.

        ``attempt`` is 0-based (0 = the first attempt just failed). Computes
        ``base_delay * 2 ** attempt`` capped at ``max_delay`` plus uniform
        jitter in ``[0, jitter]``. When ``retry_after`` is provided and exceeds
        the computed delay, it is used instead (provider hint wins).
        """
        backoff = self.base_delay * (2 ** attempt)
        capped = min(backoff, self.max_delay)
        delay = capped + random.uniform(0.0, self.jitter)
        if retry_after is not None and retry_after > delay:
            return retry_after
        return delay


async def retry_async(
    func: Thunk[T],
    *,
    policy: RetryPolicy,
    on_retry: OnRetry | None = None,
    sleep: Sleep = asyncio.sleep,
) -> T:
    """Call ``func`` (an async zero-arg thunk) with retry on transient errors.

    Retries on :func:`is_retryable` failures using exponential backoff + jitter
    from ``policy``, honouring :attr:`ProviderError.retry_after` when larger than
    the computed delay. Non-retryable errors raise immediately; after
    ``policy.max_attempts`` the last error is re-raised. ``on_retry`` (if given)
    is invoked as ``on_retry(attempt, exc)`` before each backoff sleep.
    """
    attempts = max(1, policy.max_attempts)
    last_exc: BaseException | None = None

    for attempt in range(attempts):
        try:
            return await func()
        except BaseException as exc:  # noqa: BLE001 - re-raised below if needed
            last_exc = exc
            is_last = attempt >= attempts - 1
            if is_last or not is_retryable(exc):
                raise
            retry_after = getattr(exc, "retry_after", None)
            delay = policy.delay_for(attempt, retry_after=retry_after)
            if on_retry is not None:
                try:
                    on_retry(attempt + 1, exc)
                except Exception:  # noqa: BLE001 - callback must never break retry
                    logger.warning("retry_async on_retry callback raised", exc_info=True)
            logger.info(
                "retrying after transient error (attempt %d/%d, sleeping %.3fs): %s",
                attempt + 1,
                attempts,
                delay,
                exc,
            )
            await sleep(delay)

    # Unreachable in practice: the loop either returns or raises. Present so the
    # type checker sees a definite terminal and to be safe if attempts == 0.
    assert last_exc is not None  # noqa: S101
    raise last_exc
