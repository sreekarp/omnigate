"""Per-key in-memory circuit breaker for in-process provider calls.

Keyed by provider name so one provider's outage does not trip the breaker for
the others. Three classic states:

* **closed**    — calls flow; consecutive failures are counted.
* **open**      — calls are rejected until the cooldown elapses.
* **half_open** — after cooldown a single trial is admitted: success closes the
  breaker, failure re-opens it (resetting the cooldown timer).

Ported from the gateway's in-memory breaker (``app/services/circuit_breaker.py``)
with the Redis backend and async wrappers removed — in-process bookkeeping is
pure CPU, so the methods are plain ``def``. All methods are non-fatal: errors are
caught and logged so the breaker can never take down a request.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("omnigate")


class CircuitState(str, Enum):
    """The three breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


def circuit_key(provider: str) -> str:
    """Canonical breaker key for a provider (one breaker per provider)."""
    return provider


@dataclass
class _Circuit:
    """Mutable per-key state."""

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
        now: Callable[[], float] = time.monotonic,
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

    def allow(self, key: str) -> bool:
        """Return ``False`` when the breaker for ``key`` is OPEN."""
        try:
            circuit = self._get(key)
            state = circuit.state(self._now(), self._cooldown)
            if state is CircuitState.OPEN:
                return False
            if state is CircuitState.HALF_OPEN:
                # Admit a SINGLE trial: re-arm the cooldown so concurrent callers
                # see OPEN until this trial resolves.
                circuit.opened_at = self._now()
            return True
        except Exception:  # noqa: BLE001 - never fail the call path on bookkeeping
            logger.warning("circuit breaker allow() failed for %s", key, exc_info=True)
            return True

    def record_success(self, key: str) -> None:
        """Record a successful call (closes/resets the breaker)."""
        try:
            circuit = self._get(key)
            circuit.consecutive_failures = 0
            circuit.opened_at = None
        except Exception:  # noqa: BLE001
            logger.warning(
                "circuit breaker record_success() failed for %s", key, exc_info=True
            )

    def record_failure(self, key: str) -> None:
        """Record a provider/infra failure (may OPEN the breaker)."""
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
