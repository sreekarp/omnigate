"""Tests for the in-process omnigate engine (offline via httpx.MockTransport).

Run with: pytest sdk/tests/test_engine.py -q  (after `pip install -e sdk[dev]`).
"""

from __future__ import annotations

import json

import httpx
import pytest


# ---------------------------------------------------------------------------
# Task 1 — extended ChatRequest + APIError.retry_after
# ---------------------------------------------------------------------------

def test_chatrequest_has_engine_fields():
    from omnigate.models import ChatRequest, Message

    r = ChatRequest(
        model="gpt-4o-mini",
        messages=[Message(role="user", content="hi")],
        top_p=0.9, stop=["X"], presence_penalty=0.1, frequency_penalty=0.2,
        seed=7, fallback_models=["gpt-4o"], cache=True,
    )
    assert r.top_p == 0.9 and r.seed == 7
    assert r.stop_sequences() == ["X"]
    assert r.fallback_models == ["gpt-4o"] and r.cache is True


def test_apierror_has_retry_after():
    from omnigate.exceptions import APIError

    e = APIError("boom", status_code=503, retry_after=2.5)
    assert e.retry_after == 2.5


# ---------------------------------------------------------------------------
# Task 2 — pricing
# ---------------------------------------------------------------------------

def test_pricing_known_and_normalised():
    from omnigate.pricing import compute_cost, get_price

    assert get_price("gpt-4o-mini") is not None
    assert get_price("azure/gpt-4o") == get_price("gpt-4o")
    assert get_price("gpt-4o-2024-08-06") == get_price("gpt-4o")
    assert float(compute_cost("gpt-4o-mini", 1000, 1000)) > 0
    assert compute_cost("totally-unknown", 100, 100) == 0


# ---------------------------------------------------------------------------
# Task 3 — resilience
# ---------------------------------------------------------------------------

def test_retry_policy_delay_and_retryable():
    from omnigate.resilience import RetryPolicy, is_retryable
    from omnigate.exceptions import APIError, RateLimitError

    p = RetryPolicy(max_attempts=3, base_delay=1.0, max_delay=8.0, jitter=0.0)
    assert p.delay_for(0) == 1.0 and p.delay_for(1) == 2.0 and p.delay_for(10) == 8.0
    assert p.delay_for(0, retry_after=5.0) == 5.0
    assert is_retryable(APIError("x", status_code=503)) is True
    assert is_retryable(APIError("x", status_code=400)) is False
    assert is_retryable(RateLimitError("x", status_code=429)) is True


async def _aiosleep(_seconds):
    return None


async def test_retry_async_then_success():
    from omnigate.resilience import RetryPolicy, retry_async
    from omnigate.exceptions import ProviderError

    calls = {"n": 0}

    async def thunk():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ProviderError("boom", status_code=503)
        return "ok"

    p = RetryPolicy(max_attempts=3, base_delay=0.0, max_delay=0.0, jitter=0.0)
    out = await retry_async(thunk, policy=p, sleep=_aiosleep)
    assert out == "ok" and calls["n"] == 3


def test_retry_sync_then_success():
    from omnigate.resilience import RetryPolicy, retry_sync
    from omnigate.exceptions import ProviderError

    calls = {"n": 0}

    def thunk():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ProviderError("boom", status_code=500)
        return "ok"

    p = RetryPolicy(max_attempts=3, base_delay=0.0, max_delay=0.0, jitter=0.0)
    out = retry_sync(thunk, policy=p, sleep=lambda s: None)
    assert out == "ok" and calls["n"] == 2


def test_retry_sync_non_retryable_raises_immediately():
    from omnigate.resilience import RetryPolicy, retry_sync
    from omnigate.exceptions import APIError

    calls = {"n": 0}

    def thunk():
        calls["n"] += 1
        raise APIError("bad request", status_code=400)

    p = RetryPolicy(max_attempts=3, base_delay=0.0, max_delay=0.0, jitter=0.0)
    with pytest.raises(APIError):
        retry_sync(thunk, policy=p, sleep=lambda s: None)
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Task 4 — circuit breaker
# ---------------------------------------------------------------------------

def test_breaker_opens_and_recovers():
    from omnigate.circuit_breaker import InMemoryCircuitBreaker

    clock = {"t": 0.0}
    cb = InMemoryCircuitBreaker(
        fail_threshold=2, cooldown_seconds=10.0, now=lambda: clock["t"]
    )
    assert cb.allow("openai") is True
    cb.record_failure("openai")
    cb.record_failure("openai")
    assert cb.allow("openai") is False  # open
    clock["t"] = 11.0
    assert cb.allow("openai") is True  # half-open trial admitted
    cb.record_success("openai")
    assert cb.allow("openai") is True  # closed


# ---------------------------------------------------------------------------
# Task 5 — cache
# ---------------------------------------------------------------------------

def test_cache_eligibility_and_roundtrip():
    from omnigate.cache import ResponseCache, should_cache, cache_key
    from omnigate.models import ChatRequest, Message, ChatResponse, Usage

    req = ChatRequest(
        model="gpt-4o-mini", messages=[Message(role="user", content="hi")], temperature=0
    )
    assert should_cache(req, enabled=True) is True
    assert should_cache(req, enabled=False) is False
    assert should_cache(req.model_copy(update={"stream": True}), enabled=True) is False
    assert should_cache(req.model_copy(update={"temperature": 0.7}), enabled=True) is False

    c = ResponseCache(now=lambda: 0.0)
    k = cache_key("openai", req)
    assert c.get(k) is None
    resp = ChatResponse(
        id="1", provider="openai", model="gpt-4o-mini", content="x", usage=Usage()
    )
    c.set(k, resp, ttl=300)
    hit = c.get(k)
    assert hit is not None and hit.cached is True


# ---------------------------------------------------------------------------
# Task 6 — config
# ---------------------------------------------------------------------------

def test_engine_config_env(monkeypatch):
    from omnigate.config import EngineConfig

    monkeypatch.setenv("OMNIGATE_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("OMNIGATE_CACHE_ENABLED", "true")
    monkeypatch.setenv("OMNIGATE_MAX_SPEND_USD", "1.50")
    cfg = EngineConfig.from_env()
    assert cfg.timeout == 12.5 and cfg.cache_enabled is True and cfg.max_spend_usd == 1.5

    default = EngineConfig()
    assert default.timeout == 60.0
    assert default.cache_enabled is False
    assert default.max_spend_usd is None


# ---------------------------------------------------------------------------
# Task 7 — keys
# ---------------------------------------------------------------------------

def test_resolve_key_precedence(monkeypatch):
    from omnigate import keys
    from omnigate.exceptions import APIError

    keys.reset()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(APIError):
        keys.resolve_key("openai", None)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    assert keys.resolve_key("openai", None) == "sk-env"
    assert keys.resolve_key("openai", "sk-explicit") == "sk-explicit"

    keys.set_override("openai", "sk-override")
    assert keys.resolve_key("openai", None) == "sk-override"
    assert keys.resolve_key("openai", "sk-explicit") == "sk-explicit"  # explicit wins
    keys.reset()


def test_resolve_target_azure(monkeypatch):
    from omnigate import keys

    keys.reset()
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
    t = keys.resolve_target("azure", "azure/my-deploy", api_base=None, api_version=None)
    assert t is not None
    assert t.endpoint == "https://x.openai.azure.com" and t.deployment == "my-deploy"
    assert t.api_version  # defaulted
    assert keys.resolve_target("openai", "gpt-4o", api_base=None, api_version=None) is None


# ---------------------------------------------------------------------------
# Task 8 — callbacks
# ---------------------------------------------------------------------------

def test_callbacks_fire_and_swallow():
    from omnigate import callbacks
    from omnigate.callbacks import CallbackEvent

    callbacks.reset()
    seen = []

    def boom(_event):
        raise RuntimeError("boom")

    callbacks.register(on_success=lambda e: seen.append(("ok", e.model)), on_failure=boom)
    callbacks.fire_success(
        CallbackEvent(model="m", provider="openai", usage=None, cost_usd=0.1,
                      latency_ms=5, cached=False, fallback_used=False)
    )
    callbacks.fire_failure(
        CallbackEvent(model="m", provider="openai", usage=None, cost_usd=0.0,
                      latency_ms=1, cached=False, fallback_used=False,
                      exception=ValueError("x"))
    )
    assert seen == [("ok", "m")]  # failure callback raised but was swallowed
    callbacks.reset()


# ---------------------------------------------------------------------------
# Task 9 — OpenAI/Azure specs + registry
# ---------------------------------------------------------------------------

def test_openai_spec_payload_and_parse():
    from omnigate.providers.openai import OpenAISpec
    from omnigate.models import ChatRequest, Message

    s = OpenAISpec()
    req = ChatRequest(
        model="gpt-4o-mini", messages=[Message(role="user", content="hi")],
        max_tokens=10, temperature=0.5,
    )
    p = s.build_payload(req, stream=False)
    assert p["model"] == "gpt-4o-mini"
    assert p["messages"] == [{"role": "user", "content": "hi"}]
    assert p["max_tokens"] == 10

    data = {
        "id": "x", "model": "gpt-4o-mini",
        "choices": [{"message": {"content": "yo"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    resp = s.parse_response(data, "gpt-4o-mini")
    assert resp.content == "yo" and resp.usage.total_tokens == 3 and resp.provider == "openai"


def test_registry_routing():
    from omnigate.providers.registry import spec_for_model
    from omnigate.exceptions import APIError

    assert spec_for_model("gpt-4o-mini")[1] == "openai"
    assert spec_for_model("azure/my-deploy")[1] == "azure"
    assert spec_for_model("claude-3-5-haiku-latest")[1] == "anthropic"
    assert spec_for_model("gemini-1.5-flash")[1] == "gemini"
    with pytest.raises(APIError):
        spec_for_model("mystery-model")


# ---------------------------------------------------------------------------
# Task 10 — Anthropic/Gemini specs
# ---------------------------------------------------------------------------

def test_anthropic_spec_stream_accumulates_usage():
    from omnigate.providers.anthropic import AnthropicSpec

    s = AnthropicSpec()
    st = s.stream_begin()
    out = []
    out += s.stream_feed(st, {"type": "message_start", "message": {"usage": {"input_tokens": 5}}})
    out += s.stream_feed(st, {"type": "content_block_delta",
                              "delta": {"type": "text_delta", "text": "hi"}})
    out += s.stream_feed(st, {"type": "message_delta", "usage": {"output_tokens": 7},
                              "delta": {"stop_reason": "end_turn"}})
    term = s.stream_end(st)
    assert any(c.text == "hi" for c in out)
    assert term is not None
    assert term.usage.prompt_tokens == 5 and term.usage.completion_tokens == 7
    assert term.finish_reason == "end_turn"


def test_gemini_spec_parse():
    from omnigate.providers.gemini import GeminiSpec

    s = GeminiSpec()
    data = {
        "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 3, "totalTokenCount": 5},
    }
    resp = s.parse_response(data, "gemini-1.5-flash")
    assert resp.content == "ok" and resp.usage.total_tokens == 5
