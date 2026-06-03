"""Offline unit tests for the opt-in Redis response cache.

``should_cache`` eligibility (temperature == 0 only, ``None`` excluded, respects
the global enabled flag + per-request ``cache`` True/False, never streaming);
``cache_key`` stability + per-project differentiation; and ``cache_get`` /
``cache_set`` round-trip plus non-fatal behaviour on Redis errors, exercised
with a tiny dict-backed fake async Redis. No real Redis.
"""

import pytest

from app.config import Settings
from app.schemas.chat import ChatRequest, ChatResponse, Message, Usage
from app.services.cache import cache_get, cache_key, cache_set, should_cache


def _settings(*, enabled: bool) -> Settings:
    return Settings(
        database_url="postgresql+asyncpg://t:t@localhost/t",
        redis_url="redis://localhost:6379/0",
        secret_key="s",
        admin_api_key="a",
        response_cache_enabled=enabled,
    )


def _request(**kw) -> ChatRequest:
    base = {
        "model": "gpt-4o-mini",
        "messages": [Message(role="user", content="hi")],
        "temperature": 0,
    }
    base.update(kw)
    return ChatRequest(**base)


# --- Fake async redis -------------------------------------------------------


class FakeRedis:
    """Minimal dict-backed async Redis supporting get/set (ex ignored)."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, key: str):
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value


class BrokenRedis:
    """Async Redis whose every op raises (to test non-fatal behaviour)."""

    async def get(self, key: str):
        raise RuntimeError("redis down")

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        raise RuntimeError("redis down")


# --- should_cache -----------------------------------------------------------


def test_should_cache_temp_zero_globally_enabled():
    assert should_cache(_request(temperature=0), _settings(enabled=True)) is True


def test_should_cache_temp_none_excluded():
    assert should_cache(_request(temperature=None), _settings(enabled=True)) is False


def test_should_cache_temp_positive_excluded():
    assert should_cache(_request(temperature=0.7), _settings(enabled=True)) is False


def test_should_cache_streaming_never():
    assert (
        should_cache(_request(temperature=0, stream=True), _settings(enabled=True))
        is False
    )


def test_should_cache_per_request_optin_when_global_disabled():
    assert (
        should_cache(_request(temperature=0, cache=True), _settings(enabled=False))
        is True
    )


def test_should_cache_disabled_globally_and_no_optin():
    assert should_cache(_request(temperature=0), _settings(enabled=False)) is False


def test_should_cache_explicit_optout_overrides_global():
    assert (
        should_cache(_request(temperature=0, cache=False), _settings(enabled=True))
        is False
    )


# --- cache_key --------------------------------------------------------------


def test_cache_key_stable_for_identical_request():
    r1 = _request()
    r2 = _request()
    assert cache_key("proj-1", "openai", r1) == cache_key("proj-1", "openai", r2)


def test_cache_key_differs_per_project():
    r = _request()
    assert cache_key("proj-1", "openai", r) != cache_key("proj-2", "openai", r)


def test_cache_key_differs_per_provider():
    r = _request()
    assert cache_key("proj-1", "openai", r) != cache_key("proj-1", "azure", r)


def test_cache_key_differs_when_messages_change():
    r1 = _request(messages=[Message(role="user", content="a")])
    r2 = _request(messages=[Message(role="user", content="b")])
    assert cache_key("p", "openai", r1) != cache_key("p", "openai", r2)


def test_cache_key_has_prefix():
    key = cache_key("p", "openai", _request())
    assert key.startswith("respcache:")


# --- cache_get / cache_set round-trip --------------------------------------


def _response() -> ChatResponse:
    return ChatResponse(
        id="cmpl-1",
        provider="openai",
        model="gpt-4o-mini",
        content="cached!",
        usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        cost_usd=0.0001,
        finish_reason="stop",
    )


@pytest.mark.asyncio
async def test_round_trip_sets_cached_flag():
    redis = FakeRedis()
    key = "respcache:abc"
    await cache_set(redis, key, _response(), ttl=300)
    got = await cache_get(redis, key)
    assert got is not None
    assert got.content == "cached!"
    assert got.cached is True  # flag set on read
    assert got.usage.total_tokens == 3


@pytest.mark.asyncio
async def test_get_miss_returns_none():
    assert await cache_get(FakeRedis(), "respcache:missing") is None


@pytest.mark.asyncio
async def test_set_with_nonpositive_ttl_is_noop():
    redis = FakeRedis()
    await cache_set(redis, "respcache:x", _response(), ttl=0)
    assert redis.store == {}


@pytest.mark.asyncio
async def test_get_handles_corrupt_payload():
    redis = FakeRedis()
    redis.store["respcache:bad"] = "{not valid json"
    assert await cache_get(redis, "respcache:bad") is None


@pytest.mark.asyncio
async def test_get_decodes_bytes_payload():
    redis = FakeRedis()
    redis.store["respcache:b"] = _response().model_dump_json().encode("utf-8")
    got = await cache_get(redis, "respcache:b")
    assert got is not None
    assert got.content == "cached!"


@pytest.mark.asyncio
async def test_get_non_fatal_on_redis_error():
    assert await cache_get(BrokenRedis(), "respcache:x") is None


@pytest.mark.asyncio
async def test_set_non_fatal_on_redis_error():
    # Must not raise.
    await cache_set(BrokenRedis(), "respcache:x", _response(), ttl=300)
