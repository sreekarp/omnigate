"""Offline unit tests for the async retry helper.

``retry_async`` retries retryable ``ProviderError`` (e.g. 503) up to
``max_attempts`` then re-raises; does NOT retry a 400; honours the injected
``sleep`` (delays recorded). Also exercises the ``is_retryable`` matrix. A small
``RetryPolicy`` keeps delays tiny and the injected sleep avoids real waits.
"""

import httpx
import pytest

from app.providers.base import ProviderError
from app.utils.resilience import RetryPolicy, is_retryable, retry_async


def _policy(max_attempts: int = 3) -> RetryPolicy:
    return RetryPolicy(
        max_attempts=max_attempts, base_delay=0.01, max_delay=0.05, jitter=0.0
    )


class _SleepRecorder:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


# --- is_retryable matrix ----------------------------------------------------


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_is_retryable_true_for_transient_statuses(status):
    assert is_retryable(ProviderError("x", status_code=status)) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 501])
def test_is_retryable_false_for_client_and_unlisted(status):
    assert is_retryable(ProviderError("x", status_code=status)) is False


def test_is_retryable_httpx_timeout_and_transport():
    assert is_retryable(httpx.TimeoutException("t")) is True
    assert is_retryable(httpx.ConnectError("c")) is True  # TransportError subclass


def test_is_retryable_other_exception_false():
    assert is_retryable(ValueError("nope")) is False


# --- retry_async behaviour --------------------------------------------------


@pytest.mark.asyncio
async def test_retries_then_succeeds():
    sleeper = _SleepRecorder()
    calls = {"n": 0}

    async def thunk():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ProviderError("temporarily down", status_code=503)
        return "ok"

    result = await retry_async(thunk, policy=_policy(3), sleep=sleeper)
    assert result == "ok"
    assert calls["n"] == 3
    # Two failures -> two sleeps before the third (successful) attempt.
    assert len(sleeper.delays) == 2


@pytest.mark.asyncio
async def test_raises_after_max_attempts():
    sleeper = _SleepRecorder()
    calls = {"n": 0}

    async def thunk():
        calls["n"] += 1
        raise ProviderError("still down", status_code=503)

    with pytest.raises(ProviderError) as exc:
        await retry_async(thunk, policy=_policy(3), sleep=sleeper)
    assert exc.value.status_code == 503
    assert calls["n"] == 3  # exactly max_attempts
    assert len(sleeper.delays) == 2  # slept between, not after the last


@pytest.mark.asyncio
async def test_does_not_retry_400():
    sleeper = _SleepRecorder()
    calls = {"n": 0}

    async def thunk():
        calls["n"] += 1
        raise ProviderError("bad request", status_code=400)

    with pytest.raises(ProviderError) as exc:
        await retry_async(thunk, policy=_policy(5), sleep=sleeper)
    assert exc.value.status_code == 400
    assert calls["n"] == 1  # no retries
    assert sleeper.delays == []


@pytest.mark.asyncio
async def test_injected_sleep_records_growing_delays():
    sleeper = _SleepRecorder()

    async def thunk():
        raise ProviderError("down", status_code=502)

    with pytest.raises(ProviderError):
        await retry_async(thunk, policy=_policy(4), sleep=sleeper)
    # base 0.01 * 2**0, *2**1, *2**2 -> 0.01, 0.02, 0.04 (capped at 0.05, no jitter)
    assert sleeper.delays == [0.01, 0.02, 0.04]


@pytest.mark.asyncio
async def test_retry_after_hint_overrides_when_larger():
    sleeper = _SleepRecorder()

    async def thunk():
        raise ProviderError("rate", status_code=429, retry_after=2.5)

    with pytest.raises(ProviderError):
        await retry_async(thunk, policy=_policy(2), sleep=sleeper)
    # computed delay (0.01) < retry_after (2.5) -> hint wins.
    assert sleeper.delays == [2.5]


@pytest.mark.asyncio
async def test_on_retry_callback_invoked():
    sleeper = _SleepRecorder()
    seen: list[int] = []

    async def thunk():
        raise ProviderError("down", status_code=503)

    with pytest.raises(ProviderError):
        await retry_async(
            thunk,
            policy=_policy(3),
            sleep=sleeper,
            on_retry=lambda attempt, exc: seen.append(attempt),
        )
    assert seen == [1, 2]


@pytest.mark.asyncio
async def test_succeeds_first_try_no_sleep():
    sleeper = _SleepRecorder()

    async def thunk():
        return 42

    assert await retry_async(thunk, policy=_policy(3), sleep=sleeper) == 42
    assert sleeper.delays == []
