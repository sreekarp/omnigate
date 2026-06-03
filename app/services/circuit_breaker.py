"""Per-key circuit breaker for provider calls.

Keyed by ``f"{provider}:{project_id}"`` so a single project's bad BYOK key or a
single provider outage does not trip the breaker for unrelated projects.

Three states (classic breaker):

* **closed**     — calls flow; consecutive failures are counted.
* **open**       — calls are rejected until the cooldown elapses.
* **half_open**  — after cooldown a single trial is allowed: success closes the
  breaker, failure re-opens it (resetting the cooldown timer).

Two backends are provided behind a small :class:`CircuitBreakerBackend`
Protocol and selected by ``circuit_breaker_backend``:

* an in-memory backend (module-level dict, ideal for single-process / tests),
* a Redis-backed backend (``INCR``/``EXPIRE``/``GET``) for multi-worker setups.

All public methods are ``async`` and **non-fatal**: bookkeeping never raises —
errors are caught and logged so the breaker can never take down a request. The
caller may raise :class:`CircuitOpenError` itself when :meth:`allow` is ``False``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    import redis.asyncio as redis

    from app.config import Settings

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """The three breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised by callers when a breaker is OPEN (maps to HTTP 503 upstream)."""


def circuit_key(provider: str, project_id: object) -> str:
    """Build the canonical breaker key for a provider/project pair."""
    return f"{provider}:{project_id}"


@runtime_checkable
class CircuitBreakerBackend(Protocol):
    """Async breaker API shared by every backend."""

    async def allow(self, key: str) -> bool:
        """Return ``False`` when the breaker for ``key`` is OPEN."""
        ...

    async def record_success(self, key: str) -> None:
        """Record a successful call (closes/resets the breaker)."""
        ...

    async def record_failure(self, key: str) -> None:
        """Record a provider/infra failure (may OPEN the breaker)."""
        ...


@dataclass
class _Circuit:
    """Mutable per-key state for the in-memory backend."""

    consecutive_failures: int = 0
    opened_at: float | None = None

    def state(self, now: float, cooldown: float) -> CircuitState:
        if self.opened_at is None:
            return CircuitState.CLOSED
        if now - self.opened_at >= cooldown:
            return CircuitState.HALF_OPEN
        return CircuitState.OPEN


class InMemoryCircuitBreaker:
    """Process-local breaker. Trivially unit-testable via an injectable clock."""

    def __init__(
        self,
        *,
        fail_threshold: int,
        cooldown_seconds: float,
        now: "Callable[[], float]" = time.monotonic,
    ) -> None:
        self._threshold = fail_threshold
        self._cooldown = cooldown_seconds
        self._now = now
        self._circuits: dict[str, _Circuit] = {}

    def _get(self, key: str) -> _Circuit:
        circuit = self._circuits.get(key)
        if circuit is None:
            circuit = _Circuit()
            self._circuits[key] = circuit
        return circuit

    def state(self, key: str) -> CircuitState:
        """Return the current :class:`CircuitState` for ``key`` (test helper)."""
        return self._get(key).state(self._now(), self._cooldown)

    async def allow(self, key: str) -> bool:
        try:
            state = self._get(key).state(self._now(), self._cooldown)
            # HALF_OPEN lets a single trial through (returns True).
            return state is not CircuitState.OPEN
        except Exception:  # noqa: BLE001 - never fail closed-path on bookkeeping
            logger.warning("circuit breaker allow() failed for %s", key, exc_info=True)
            return True

    async def record_success(self, key: str) -> None:
        try:
            circuit = self._get(key)
            circuit.consecutive_failures = 0
            circuit.opened_at = None
        except Exception:  # noqa: BLE001
            logger.warning(
                "circuit breaker record_success() failed for %s", key, exc_info=True
            )

    async def record_failure(self, key: str) -> None:
        try:
            circuit = self._get(key)
            now = self._now()
            if circuit.state(now, self._cooldown) is CircuitState.HALF_OPEN:
                # Trial during half-open failed -> re-open and reset the timer.
                circuit.opened_at = now
                return
            circuit.consecutive_failures += 1
            if circuit.consecutive_failures >= self._threshold:
                circuit.opened_at = now
        except Exception:  # noqa: BLE001
            logger.warning(
                "circuit breaker record_failure() failed for %s", key, exc_info=True
            )

    def reset(self) -> None:
        """Clear all breaker state (test helper)."""
        self._circuits.clear()


class RedisCircuitBreaker:
    """Redis-backed breaker for multi-worker deployments.

    Cross-process HALF_OPEN cannot be modelled perfectly: while OPEN the key is
    rejected; once the open marker's TTL lapses, each worker may independently
    send one trial. That is acceptable — a handful of trial calls is fine.

    Keys (per breaker key ``k``):
        * ``cb:{k}:fails`` — INCR'd consecutive-failure counter.
        * ``cb:{k}:open``  — presence = OPEN; TTL = cooldown.
    """

    def __init__(
        self,
        client: "redis.Redis",
        *,
        fail_threshold: int,
        cooldown_seconds: float,
    ) -> None:
        self._redis = client
        self._threshold = fail_threshold
        self._cooldown = int(max(1, round(cooldown_seconds)))

    @staticmethod
    def _fails_key(key: str) -> str:
        return f"cb:{key}:fails"

    @staticmethod
    def _open_key(key: str) -> str:
        return f"cb:{key}:open"

    async def allow(self, key: str) -> bool:
        try:
            return not bool(await self._redis.exists(self._open_key(key)))
        except Exception:  # noqa: BLE001 - fail open if Redis is unavailable
            logger.warning(
                "circuit breaker allow() redis error for %s", key, exc_info=True
            )
            return True

    async def record_success(self, key: str) -> None:
        try:
            await self._redis.delete(self._fails_key(key), self._open_key(key))
        except Exception:  # noqa: BLE001
            logger.warning(
                "circuit breaker record_success() redis error for %s",
                key,
                exc_info=True,
            )

    async def record_failure(self, key: str) -> None:
        try:
            fails = int(await self._redis.incr(self._fails_key(key)))
            # Keep the counter from lingering forever; it should decay over a
            # window comparable to the cooldown.
            await self._redis.expire(self._fails_key(key), self._cooldown)
            if fails >= self._threshold:
                await self._redis.set(self._open_key(key), "1", ex=self._cooldown)
                await self._redis.delete(self._fails_key(key))
        except Exception:  # noqa: BLE001
            logger.warning(
                "circuit breaker record_failure() redis error for %s",
                key,
                exc_info=True,
            )


# --- Factory / process singleton --------------------------------------------

_breaker: CircuitBreakerBackend | None = None


def get_circuit_breaker(
    settings: "Settings | None" = None,
    *,
    redis_client: "redis.Redis | None" = None,
) -> CircuitBreakerBackend:
    """Return the process-wide breaker, building it lazily from settings.

    ``circuit_breaker_backend == "redis"`` selects the Redis backend (a client
    must be supplied or importable); anything else uses the in-memory backend.
    """
    global _breaker
    if _breaker is not None:
        return _breaker

    if settings is None:
        from app.config import get_settings

        settings = get_settings()

    backend = (settings.circuit_breaker_backend or "memory").strip().lower()
    if backend == "redis":
        if redis_client is None:
            from app.redis_client import redis_client as shared_client

            redis_client = shared_client
        _breaker = RedisCircuitBreaker(
            redis_client,
            fail_threshold=settings.circuit_breaker_fail_threshold,
            cooldown_seconds=settings.circuit_breaker_cooldown_seconds,
        )
    else:
        _breaker = InMemoryCircuitBreaker(
            fail_threshold=settings.circuit_breaker_fail_threshold,
            cooldown_seconds=settings.circuit_breaker_cooldown_seconds,
        )
    return _breaker


def reset_circuit_breaker() -> None:
    """Drop the cached breaker singleton (test helper)."""
    global _breaker
    _breaker = None
