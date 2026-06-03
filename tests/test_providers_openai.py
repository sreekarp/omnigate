"""Offline unit tests for the OpenAI adapter.

Exercises the pure module helpers (``build_chat_payload``,
``parse_chat_response``, ``parse_stream_chunk``) directly, and drives
``OpenAIProvider.chat``/``.stream`` over an ``httpx.MockTransport`` injected by
monkeypatching the module's ``httpx.AsyncClient``. No network, DB, or Redis.
"""

import json

import httpx
import pytest

import app.providers.openai as openai_mod
from app.providers.base import ProviderError
from app.providers.openai import (
    OpenAIProvider,
    build_chat_payload,
    parse_chat_response,
    parse_stream_chunk,
)
from app.schemas.chat import ChatRequest, Message


def _request(**kw) -> ChatRequest:
    base = {
        "model": "gpt-4o-mini",
        "messages": [Message(role="user", content="hi")],
    }
    base.update(kw)
    return ChatRequest(**base)


def _install_transport(monkeypatch, transport: httpx.MockTransport) -> None:
    """Patch the module's httpx.AsyncClient so it binds to ``transport``."""
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs.pop("transport", None)
        return real_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(openai_mod.httpx, "AsyncClient", factory)


# --- Pure helpers -----------------------------------------------------------


def test_build_payload_non_stream_omits_stream_options():
    payload = build_chat_payload(_request(), stream=False)
    assert payload["stream"] is False
    assert "stream_options" not in payload


def test_build_payload_stream_includes_usage():
    payload = build_chat_payload(_request(), stream=True)
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}


def test_build_payload_optional_params_only_when_set():
    payload = build_chat_payload(
        _request(temperature=0.5, max_tokens=10, top_p=0.9, seed=7), stream=False
    )
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 10
    assert payload["top_p"] == 0.9
    assert payload["seed"] == 7
    # Unset ones absent.
    assert "presence_penalty" not in payload
    assert "frequency_penalty" not in payload
    assert "stop" not in payload

    bare = build_chat_payload(_request(), stream=False)
    assert "temperature" not in bare
    assert "max_tokens" not in bare


def test_parse_chat_response_maps_usage_and_finish_reason():
    data = {
        "id": "cmpl-1",
        "model": "gpt-4o-mini-2024-07-18",
        "choices": [
            {"message": {"content": "hello"}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 5,
            "total_tokens": 8,
        },
    }
    resp = parse_chat_response(data, provider_name="openai", request_model="gpt-4o-mini")
    assert resp.id == "cmpl-1"
    assert resp.provider == "openai"
    assert resp.model == "gpt-4o-mini-2024-07-18"
    assert resp.content == "hello"
    assert resp.usage.prompt_tokens == 3
    assert resp.usage.completion_tokens == 5
    assert resp.usage.total_tokens == 8
    assert resp.finish_reason == "stop"


def test_parse_chat_response_handles_null_content_and_missing_usage():
    data = {"choices": [{"message": {"content": None}, "finish_reason": None}]}
    resp = parse_chat_response(data, provider_name="openai", request_model="gpt-4o")
    assert resp.content == ""
    assert resp.usage.total_tokens == 0
    assert resp.model == "gpt-4o"  # falls back to request model


def test_parse_stream_chunk_content_only():
    out = parse_stream_chunk({"choices": [{"delta": {"content": "hi"}}]})
    assert len(out) == 1
    assert out[0].text == "hi"
    assert out[0].usage is None


def test_parse_stream_chunk_finish_reason_emits_chunk():
    out = parse_stream_chunk(
        {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    )
    assert len(out) == 1
    assert out[0].finish_reason == "stop"
    assert out[0].text == ""


def test_parse_stream_chunk_empty_choices_yields_terminal_usage():
    out = parse_stream_chunk(
        {
            "choices": [],
            "model": "gpt-4o-mini",
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 4,
                "total_tokens": 6,
            },
        }
    )
    assert len(out) == 1
    chunk = out[0]
    assert chunk.text == ""
    assert chunk.usage is not None
    assert chunk.usage.prompt_tokens == 2
    assert chunk.usage.completion_tokens == 4
    assert chunk.usage.total_tokens == 6
    assert chunk.model == "gpt-4o-mini"


def test_parse_stream_chunk_no_content_no_finish_is_empty():
    assert parse_stream_chunk({"choices": [{"delta": {}}]}) == []


# --- chat() over MockTransport ---------------------------------------------


@pytest.mark.asyncio
async def test_chat_posts_and_parses(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "cmpl-x",
                "model": "gpt-4o-mini",
                "choices": [
                    {"message": {"content": "world"}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                },
            },
        )

    _install_transport(monkeypatch, httpx.MockTransport(handler))
    resp = await OpenAIProvider().chat(_request(), "sk-test")
    assert resp.content == "world"
    assert resp.usage.total_tokens == 3
    assert captured["headers"]["authorization"] == "Bearer sk-test"
    assert captured["body"]["stream"] is False


@pytest.mark.asyncio
async def test_chat_429_parses_retry_after(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "12"}, text="slow down")

    _install_transport(monkeypatch, httpx.MockTransport(handler))
    with pytest.raises(ProviderError) as exc:
        await OpenAIProvider().chat(_request(), "sk-test")
    assert exc.value.status_code == 429
    assert exc.value.retry_after == 12.0


@pytest.mark.asyncio
async def test_chat_500_has_no_retry_after(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, headers={"Retry-After": "5"}, text="boom")

    _install_transport(monkeypatch, httpx.MockTransport(handler))
    with pytest.raises(ProviderError) as exc:
        await OpenAIProvider().chat(_request(), "sk-test")
    assert exc.value.status_code == 500
    assert exc.value.retry_after is None  # only parsed on 429


@pytest.mark.asyncio
async def test_chat_no_key_raises_400(monkeypatch):
    # No transport needed: header build fails first.
    with pytest.raises(ProviderError) as exc:
        await OpenAIProvider().chat(_request(), "")
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_stream_yields_content_then_terminal_usage(monkeypatch):
    captured = {}

    sse = (
        'data: {"choices":[{"delta":{"content":"Hel"}}]}\n'
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n'
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n'
        'data: {"choices":[],"model":"gpt-4o-mini",'
        '"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}\n'
        "data: [DONE]\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=sse.encode("utf-8"))

    _install_transport(monkeypatch, httpx.MockTransport(handler))
    chunks = [c async for c in OpenAIProvider().stream(_request(), "sk-test")]

    # stream payload requested usage.
    assert captured["body"]["stream_options"] == {"include_usage": True}
    texts = [c.text for c in chunks if c.text]
    assert "".join(texts) == "Hello"
    usage_chunks = [c for c in chunks if c.usage is not None]
    assert len(usage_chunks) == 1
    assert usage_chunks[0].usage.total_tokens == 3
    assert any(c.finish_reason == "stop" for c in chunks)


@pytest.mark.asyncio
async def test_stream_error_status_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "3"}, content=b"nope")

    _install_transport(monkeypatch, httpx.MockTransport(handler))
    with pytest.raises(ProviderError) as exc:
        async for _ in OpenAIProvider().stream(_request(), "sk-test"):
            pass
    assert exc.value.status_code == 429
    assert exc.value.retry_after == 3.0
