"""Offline unit tests for the in-memory circuit breaker.

Uses an injectable clock to drive transitions deterministically: opens after
``fail_threshold`` consecutive failures, rejects while OPEN, half-opens after the
cooldown, closes on success, and re-opens (resetting the timer) if the half-open
trial fails. No Redis, no real time.
"""

import pytest

from app.services.circuit_breaker import (
    CircuitState,
    InMemoryCircuitBreaker,
    circuit_key,
)


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _breaker(clock: FakeClock, *, threshold: int = 3, cooldown: float = 30.0):
    return InMemoryCircuitBreaker(
        fail_threshold=threshold, cooldown_seconds=cooldown, now=clock
    )


def test_circuit_key():
    assert circuit_key("openai", "proj-1") == "openai:proj-1"


@pytest.mark.asyncio
async def test_starts_closed_and_allows():
    clock = FakeClock()
    cb = _breaker(clock)
    assert cb.state("k") == CircuitState.CLOSED
    assert await cb.allow("k") is True


@pytest.mark.asyncio
async def test_opens_after_threshold_failures():
    clock = FakeClock()
    cb = _breaker(clock, threshold=3)
    await cb.record_failure("k")
    await cb.record_failure("k")
    assert cb.state("k") == CircuitState.CLOSED  # 2 < 3
    assert await cb.allow("k") is True
    await cb.record_failure("k")  # 3rd -> open
    assert cb.state("k") == CircuitState.OPEN
    assert await cb.allow("k") is False


@pytest.mark.asyncio
async def test_half_opens_after_cooldown():
    clock = FakeClock()
    cb = _breaker(clock, threshold=2, cooldown=30.0)
    await cb.record_failure("k")
    await cb.record_failure("k")
    assert cb.state("k") == CircuitState.OPEN
    clock.advance(29.0)
    assert cb.state("k") == CircuitState.OPEN
    assert await cb.allow("k") is False
    clock.advance(1.0)  # total 30 >= cooldown
    assert cb.state("k") == CircuitState.HALF_OPEN
    assert await cb.allow("k") is True  # single trial allowed


@pytest.mark.asyncio
async def test_success_closes_breaker():
    clock = FakeClock()
    cb = _breaker(clock, threshold=2, cooldown=30.0)
    await cb.record_failure("k")
    await cb.record_failure("k")
    clock.advance(30.0)
    assert cb.state("k") == CircuitState.HALF_OPEN
    await cb.record_success("k")
    assert cb.state("k") == CircuitState.CLOSED
    assert await cb.allow("k") is True


@pytest.mark.asyncio
async def test_failure_during_half_open_reopens_and_resets_timer():
    clock = FakeClock()
    cb = _breaker(clock, threshold=2, cooldown=30.0)
    await cb.record_failure("k")
    await cb.record_failure("k")
    clock.advance(30.0)
    assert cb.state("k") == CircuitState.HALF_OPEN
    # Trial fails -> re-open with the timer reset to "now".
    await cb.record_failure("k")
    assert cb.state("k") == CircuitState.OPEN
    clock.advance(29.0)  # not yet past the fresh cooldown
    assert cb.state("k") == CircuitState.OPEN
    clock.advance(1.0)
    assert cb.state("k") == CircuitState.HALF_OPEN


@pytest.mark.asyncio
async def test_success_resets_failure_counter():
    clock = FakeClock()
    cb = _breaker(clock, threshold=3)
    await cb.record_failure("k")
    await cb.record_failure("k")
    await cb.record_success("k")  # resets count
    await cb.record_failure("k")
    await cb.record_failure("k")
    assert cb.state("k") == CircuitState.CLOSED  # only 2 since reset
    await cb.record_failure("k")
    assert cb.state("k") == CircuitState.OPEN


@pytest.mark.asyncio
async def test_keys_are_independent():
    clock = FakeClock()
    cb = _breaker(clock, threshold=1)
    await cb.record_failure("a")
    assert cb.state("a") == CircuitState.OPEN
    assert cb.state("b") == CircuitState.CLOSED
    assert await cb.allow("b") is True


@pytest.mark.asyncio
async def test_reset_clears_all_state():
    clock = FakeClock()
    cb = _breaker(clock, threshold=1)
    await cb.record_failure("a")
    assert cb.state("a") == CircuitState.OPEN
    cb.reset()
    assert cb.state("a") == CircuitState.CLOSED
