"""Tests for the LLM Gateway SDK using httpx.MockTransport (no network).

Run with: pytest sdk/tests -q  (after `pip install -e sdk[dev]`).
"""

from __future__ import annotations

import json

import httpx
import pytest

from llm_gateway import (
    APIError,
    AsyncClient,
    AuthError,
    BudgetExceededError,
    ChatResponse,
    Client,
    ProviderError,
    RateLimitError,
    StreamChunk,
)
from llm_gateway import _retry as retry_mod
from llm_gateway._retry import RetryConfig, compute_delay, parse_retry_after, should_retry
from llm_gateway._transport import iter_text_chunks
from llm_gateway.exceptions import classify, normalise_detail
from llm_gateway.models import coerce_messages


BASE = "http://gw.test"

CHAT_OK = {
    "id": "cmpl-1",
    "provider": "openai",
    "model": "gpt-4o-mini",
    "content": "Bonjour!",
    "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
    "cost_usd": 0.0000042,
    "extra_unknown_field": "ignored",  # forward-compat
}


def make_client(handler, **kw) -> Client:
    return Client(
        api_key="llmg_test",
        base_url=BASE,
        transport=httpx.MockTransport(handler),
        **kw,
    )


def make_async_client(handler, **kw) -> AsyncClient:
    return AsyncClient(
        api_key="llmg_test",
        base_url=BASE,
        transport=httpx.MockTransport(handler),
        **kw,
    )


# --------------------------------------------------------------------------
# models / coercion
# --------------------------------------------------------------------------

def test_chatresponse_roundtrip_ignores_extra():
    resp = ChatResponse.model_validate(CHAT_OK)
    assert resp.content == "Bonjour!"
    assert resp.usage.total_tokens == 8
    assert resp.cost_usd == pytest.approx(0.0000042)


def test_coerce_messages_variants():
    assert coerce_messages("hi")[0].role == "user"
    assert coerce_messages("hi")[0].content == "hi"
    out = coerce_messages([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    assert [m.role for m in out] == ["system", "user"]
    single = coerce_messages({"role": "assistant", "content": "a"})
    assert single[0].role == "assistant"


def test_chatrequest_validation_rejects_bad_input():
    from llm_gateway.models import ChatRequest, Message
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        ChatRequest(model="m", messages=[])  # empty
    with pytest.raises(pydantic.ValidationError):
        ChatRequest(model="m", messages=[Message(role="user", content="x")], temperature=3.0)
    with pytest.raises(pydantic.ValidationError):
        ChatRequest(model="m", messages=[Message(role="user", content="x")], max_tokens=0)


# --------------------------------------------------------------------------
# chat happy path + header propagation
# --------------------------------------------------------------------------

def test_chat_happy_path_and_headers():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["x-api-key"] = request.headers.get("x-api-key")
        seen["x-user-id"] = request.headers.get("x-user-id")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=CHAT_OK)

    with make_client(handler, user_id="u-default") as c:
        resp = c.chat(model="gpt-4o-mini", messages="Say hi in French", user_id="u-override")

    assert isinstance(resp, ChatResponse)
    assert resp.content == "Bonjour!"
    assert seen["x-api-key"] == "llmg_test"
    assert seen["x-user-id"] == "u-override"  # per-call override wins
    assert seen["body"]["stream"] is False
    assert "max_tokens" not in seen["body"]  # None omitted
    assert seen["body"]["messages"] == [{"role": "user", "content": "Say hi in French"}]


def test_chat_includes_optional_params_when_set():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["max_tokens"] == 50
        assert body["temperature"] == 0.7
        return httpx.Response(200, json=CHAT_OK)

    with make_client(handler) as c:
        c.chat(model="m", messages="hi", max_tokens=50, temperature=0.7)


# --------------------------------------------------------------------------
# error mapping
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "status,detail,headers,exc_type",
    [
        (401, "Invalid API key", {}, AuthError),
        (402, "Project daily budget exceeded", {}, BudgetExceededError),
        (429, "Rate limit exceeded", {"Retry-After": "30"}, RateLimitError),
        (502, "upstream boom", {}, ProviderError),
        (401, "OpenAI error 401: bad key", {}, ProviderError),  # ambiguous 401 -> provider
        (400, "Some random bad request", {}, APIError),
    ],
)
def test_error_mapping(status, detail, headers, exc_type):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"detail": detail}, headers=headers)

    with make_client(handler, retries=0) as c:
        with pytest.raises(exc_type) as ei:
            c.chat(model="m", messages="hi")
    err = ei.value
    assert err.status_code == status
    if exc_type is RateLimitError:
        assert err.retry_after == 30.0


def test_validation_error_detail_list_is_stringified():
    detail = [{"loc": ["body", "messages"], "msg": "field required", "type": "missing"}]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": detail})

    with make_client(handler, retries=0) as c:
        with pytest.raises(APIError) as ei:
            c.chat(model="m", messages="hi")
    assert "field required" in ei.value.message
    assert ei.value.detail == detail


def test_classify_and_normalise_detail_units():
    assert isinstance(classify(401, "nope"), AuthError)
    assert isinstance(classify(401, "Anthropic error 401: x"), ProviderError)
    assert isinstance(classify(402, "budget"), BudgetExceededError)
    rl = classify(429, "slow", retry_after=12.0)
    assert isinstance(rl, RateLimitError) and rl.retry_after == 12.0
    assert "messages: field required" in normalise_detail(
        [{"loc": ["messages"], "msg": "field required"}]
    )


# --------------------------------------------------------------------------
# retry behaviour
# --------------------------------------------------------------------------

def test_retry_then_success(monkeypatch):
    calls = {"n": 0, "sleeps": []}
    monkeypatch.setattr("llm_gateway.client.time.sleep", lambda s: calls["sleeps"].append(s))

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, json={"detail": "slow"}, headers={"Retry-After": "0"})
        return httpx.Response(200, json=CHAT_OK)

    with make_client(handler, retries=2) as c:
        resp = c.chat(model="m", messages="hi")
    assert resp.content == "Bonjour!"
    assert calls["n"] == 3  # 1 initial + 2 retries
    assert len(calls["sleeps"]) == 2


def test_transport_error_becomes_connection_error(monkeypatch):
    monkeypatch.setattr("llm_gateway.client.time.sleep", lambda s: None)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    from llm_gateway import ConnectionError as GWConn

    with make_client(handler, retries=1) as c:
        with pytest.raises(GWConn):
            c.chat(model="m", messages="hi")


def test_retry_unit_helpers():
    cfg = RetryConfig(max_retries=2, jitter=0.0, backoff_base=1.0, backoff_max=8.0)
    assert should_retry(429, 0, cfg) is True
    assert should_retry(400, 0, cfg) is False
    assert should_retry(None, 0, cfg) is True  # transport error
    assert should_retry(429, 2, cfg) is False  # attempts exhausted
    assert compute_delay(0, cfg) == 1.0
    assert compute_delay(1, cfg) == 2.0
    assert compute_delay(10, cfg) == 8.0  # capped
    assert compute_delay(0, cfg, retry_after=3.0) == 3.0
    assert parse_retry_after("30") == 30.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("garbage") is None
    # HTTP-date form parses to a non-negative number of seconds.
    d = parse_retry_after("Wed, 21 Oct 2099 07:28:00 GMT")
    assert d is not None and d >= 0


# --------------------------------------------------------------------------
# streaming
# --------------------------------------------------------------------------

def _stream_handler(chunks, request_id="req-9"):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=httpx.ByteStream(b"".join(c.encode() for c in chunks)),
            headers={"content-type": "text/plain", "x-request-id": request_id},
        )

    return handler


def test_stream_text_reassembles():
    # iter_text on a single ByteStream yields one chunk; logic still must work.
    with make_client(_stream_handler(["Hello, ", "world", "!"])) as c:
        text = "".join(c.chat_stream(model="m", messages="hi"))
    assert text == "Hello, world!"


def test_stream_as_chunks_carries_request_id():
    with make_client(_stream_handler(["abc"])) as c:
        chunks = list(c.chat_stream(model="m", messages="hi", as_chunks=True))
    assert all(isinstance(ch, StreamChunk) for ch in chunks)
    assert "".join(ch.text for ch in chunks) == "abc"
    assert chunks[0].request_id == "req-9"


def test_stream_error_sentinel_raises_provider_error():
    body = "partial output\n[error] OpenAI error 500: boom"
    with make_client(_stream_handler([body])) as c:
        with pytest.raises(ProviderError) as ei:
            list(c.chat_stream(model="m", messages="hi"))
    assert "boom" in ei.value.detail
    assert ei.value.request_id == "req-9"


def test_iter_text_chunks_split_sentinel():
    # Sentinel "\n[error] " split across chunk boundaries must still trigger.
    pieces = ["good output\n[err", "or] kaboom happened"]
    collected = []
    with pytest.raises(ProviderError) as ei:
        for ch in iter_text_chunks(iter(pieces), request_id="r1"):
            collected.append(ch.text)
    assert "kaboom happened" in ei.value.detail
    assert "".join(collected) == "good output"


def test_iter_text_chunks_plain_passthrough():
    pieces = ["aa", "bb", "cc"]
    out = "".join(ch.text for ch in iter_text_chunks(iter(pieces), request_id="r"))
    assert out == "aabbcc"


# --------------------------------------------------------------------------
# account endpoints
# --------------------------------------------------------------------------

def test_signup_keyless_client():
    payload = {
        "org_id": "11111111-1111-1111-1111-111111111111",
        "project_id": "22222222-2222-2222-2222-222222222222",
        "email": "dev@acme.com",
        "api_key": "llmg_newkey",
        "message": "save it",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert "x-api-key" not in request.headers  # keyless
        return httpx.Response(201, json=payload)

    anon = Client(base_url=BASE, transport=httpx.MockTransport(handler))
    acct = anon.signup(email="dev@acme.com", org_name="Acme", project_name="prod")
    anon.close()
    assert acct.api_key == "llmg_newkey"
    assert acct.org_id == payload["org_id"]  # uuid as str


def test_set_provider_key_handles_204():
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["provider"] == "openai"
        return httpx.Response(204)

    with make_client(handler) as c:
        assert c.set_provider_key(provider="openai", api_key="sk-abcdef12") is None


def test_me_and_health():
    me_payload = {
        "project_id": "p", "org_id": "o", "project_name": "Default",
        "key_prefix": "llmg_ab", "rate_limit_per_min": 60,
        "configured_providers": ["openai"],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/me":
            return httpx.Response(200, json=me_payload)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "version": "0.1.0"})
        return httpx.Response(404, json={"detail": "nope"})

    with make_client(handler) as c:
        me = c.me()
        assert me.configured_providers == ["openai"]
        assert c.health()["status"] == "ok"


MODELS_OK = {
    "object": "list",
    "data": [
        {
            "id": "gpt-4o-mini",
            "object": "model",
            "created": 0,
            "owned_by": "openai",
            "provider": "openai",
            "pricing": {"input_per_1k_usd": 0.15, "output_per_1k_usd": 0.6},
        },
        {
            "id": "mystery-model",
            "object": "model",
            "created": 0,
            "owned_by": "unknown",
            "provider": "unknown",
            "pricing": None,  # unpriced models report pricing=None
        },
    ],
}

METRICS_OK = {
    "scope": "project",
    "scope_id": "22222222-2222-2222-2222-222222222222",
    "range_from": "2026-06-03T00:00:00+00:00",
    "range_to": "2026-06-04T00:00:00+00:00",
    "group_by": "status",
    "granularity": "hour",
    "totals": {
        "requests": 7,
        "prompt_tokens": 30,
        "completion_tokens": 50,
        "total_tokens": 80,
        "cost_usd": 1.25,
        "error_rate": 0.0,
        "cache_hit_rate": 0.5,
        "avg_latency_ms": 120.0,
        "p50_latency_ms": 100.0,
        "p95_latency_ms": 200.0,
        "p99_latency_ms": 250.0,
    },
    "breakdown": [
        {
            "key": "ok",
            "requests": 7,
            "total_tokens": 80,
            "cost_usd": 1.25,
            "avg_latency_ms": 120.0,
            "error_rate": 0.0,
        }
    ],
    "timeseries": [
        {
            "bucket": "2026-06-03T12:00:00+00:00",
            "requests": 7,
            "total_tokens": 80,
            "cost_usd": 1.25,
            "error_rate": 0.0,
        }
    ],
    "extra_unknown_field": "ignored",  # forward-compat
}


def test_models_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json=MODELS_OK)

    with make_client(handler) as c:
        models = c.models()
    assert [m.id for m in models] == ["gpt-4o-mini", "mystery-model"]
    first = models[0]
    assert first.owned_by == "openai" and first.provider == "openai"
    assert first.object == "model"
    assert first.pricing is not None
    assert first.pricing.input_per_1k_usd == 0.15
    assert first.pricing.output_per_1k_usd == 0.6
    assert models[1].pricing is None  # unpriced model


def test_metrics_uses_range_param_and_rich_shape():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/metrics"
        seen["range"] = request.url.params.get("range")
        seen["window"] = request.url.params.get("window")  # must be absent now
        return httpx.Response(200, json=METRICS_OK)

    with make_client(handler) as c:
        m = c.metrics(range="7d")

    assert seen["range"] == "7d"  # query param is 'range', not 'window'
    assert seen["window"] is None
    assert m.scope == "project"
    assert m.scope_id == "22222222-2222-2222-2222-222222222222"
    assert m.group_by == "status"
    assert m.granularity == "hour"
    assert m.totals.requests == 7
    assert m.totals.total_tokens == 80
    assert m.totals.cost_usd == pytest.approx(1.25)
    assert m.totals.cache_hit_rate == pytest.approx(0.5)
    assert m.totals.p95_latency_ms == 200.0
    assert m.breakdown[0].key == "ok" and m.breakdown[0].requests == 7
    assert m.timeseries[0].requests == 7
    assert m.timeseries[0].bucket.year == 2026


def test_metrics_defaults_to_24h():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("range") == "24h"
        return httpx.Response(200, json=METRICS_OK)

    with make_client(handler) as c:
        c.metrics()


def test_create_api_key_posts_to_keys_api():
    payload = {
        "id": "33333333-3333-3333-3333-333333333333",
        "name": "ci",
        "key_prefix": "llmg_abcd",
        "created_at": "2026-06-04T00:00:00+00:00",
        "last_used_at": None,
        "revoked_at": None,
        "api_key": "llmg_freshsecret",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/keys/api"  # NOT /v1/keys/create
        assert request.method == "POST"
        assert json.loads(request.content) == {"name": "ci"}
        return httpx.Response(201, json=payload)

    with make_client(handler) as c:
        key = c.create_api_key(name="ci")
    assert key.api_key == "llmg_freshsecret"
    assert key.name == "ci"
    assert key.key_prefix == "llmg_abcd"
    assert key.id == "33333333-3333-3333-3333-333333333333"
    assert key.revoked_at is None


# --------------------------------------------------------------------------
# async mirror
# --------------------------------------------------------------------------

async def test_async_chat_and_lifecycle():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("x-api-key") == "llmg_test"
        return httpx.Response(200, json=CHAT_OK)

    async with make_async_client(handler) as c:
        resp = await c.chat(model="m", messages="hi")
    assert resp.content == "Bonjour!"


async def test_async_streaming_and_sentinel():
    def ok_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=httpx.ByteStream(b"alpha beta"),
            headers={"content-type": "text/plain", "x-request-id": "areq"},
        )

    async with make_async_client(ok_handler) as c:
        pieces = [p async for p in c.chat_stream(model="m", messages="hi")]
        assert "".join(pieces) == "alpha beta"

    def err_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=httpx.ByteStream(b"out\n[error] Anthropic error 500: x"),
            headers={"content-type": "text/plain", "x-request-id": "areq"},
        )

    async with make_async_client(err_handler) as c:
        with pytest.raises(ProviderError):
            async for ch in c.chat_stream(model="m", messages="hi", as_chunks=True):
                pass


async def test_async_error_mapping():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"detail": "Organisation daily budget exceeded"})

    async with make_async_client(handler, retries=0) as c:
        with pytest.raises(BudgetExceededError):
            await c.chat(model="m", messages="hi")


async def test_async_metrics_and_create_api_key_shapes():
    seen: dict = {}
    key_payload = {
        "id": "44444444-4444-4444-4444-444444444444",
        "name": "svc",
        "key_prefix": "llmg_wxyz",
        "created_at": "2026-06-04T00:00:00+00:00",
        "last_used_at": None,
        "revoked_at": None,
        "api_key": "llmg_asyncsecret",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/metrics":
            seen["range"] = request.url.params.get("range")
            return httpx.Response(200, json=METRICS_OK)
        if request.url.path == "/v1/keys/api":
            assert request.method == "POST"
            return httpx.Response(201, json=key_payload)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=MODELS_OK)
        return httpx.Response(404, json={"detail": "nope"})

    async with make_async_client(handler) as c:
        m = await c.metrics(range="30d")
        assert seen["range"] == "30d"
        assert m.totals.total_tokens == 80
        models = await c.models()
        assert models[0].pricing.output_per_1k_usd == 0.6
        key = await c.create_api_key(name="svc")
        assert key.api_key == "llmg_asyncsecret" and key.key_prefix == "llmg_wxyz"
