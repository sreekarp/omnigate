"""Offline unit tests for the Anthropic Messages adapter streaming path.

Feeds a simulated SSE byte stream through an ``httpx.MockTransport`` to verify
the stream() accumulates ``input_tokens`` (message_start), cumulative
``output_tokens`` (message_delta, last-wins), ``stop_reason``, and emits exactly
ONE terminal usage chunk at ``message_stop``. An ``error`` event raises
``ProviderError``. No network.
"""

import json

import httpx
import pytest

import app.providers.anthropic as anthropic_mod
from app.providers.anthropic import AnthropicProvider
from app.providers.base import ProviderError
from app.schemas.chat import ChatRequest, Message


def _request(**kw) -> ChatRequest:
    base = {
        "model": "claude-3-5-sonnet-latest",
        "messages": [
            Message(role="system", content="be brief"),
            Message(role="user", content="hi"),
        ],
    }
    base.update(kw)
    return ChatRequest(**base)


def _install_transport(monkeypatch, transport: httpx.MockTransport) -> None:
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs.pop("transport", None)
        return real_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(anthropic_mod.httpx, "AsyncClient", factory)


def _sse(events: list[dict]) -> bytes:
    return ("".join(f"data: {json.dumps(e)}\n" for e in events)).encode("utf-8")


def test_payload_splits_system_and_defaults_max_tokens():
    payload = AnthropicProvider()._payload(_request(), stream=False)
    assert payload["system"] == "be brief"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["max_tokens"] == 1024  # default applied
    assert payload["model"] == "claude-3-5-sonnet-latest"


@pytest.mark.asyncio
async def test_stream_accumulates_usage_and_one_terminal_chunk(monkeypatch):
    captured = {}
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 11}}},
        {"type": "ping"},
        {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hel"},
        },
        {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "lo"},
        },
        # cumulative output tokens: 1 then 7 (last wins).
        {"type": "message_delta", "usage": {"output_tokens": 1}},
        {
            "type": "message_delta",
            "usage": {"output_tokens": 7},
            "delta": {"stop_reason": "end_turn"},
        },
        {"type": "message_stop"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=_sse(events))

    _install_transport(monkeypatch, httpx.MockTransport(handler))
    chunks = [c async for c in AnthropicProvider().stream(_request(), "ak-1")]

    # headers + version
    assert captured["headers"]["x-api-key"] == "ak-1"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["body"]["stream"] is True

    text = "".join(c.text for c in chunks if c.text)
    assert text == "Hello"

    usage_chunks = [c for c in chunks if c.usage is not None]
    assert len(usage_chunks) == 1
    terminal = usage_chunks[0]
    assert terminal.usage.prompt_tokens == 11
    assert terminal.usage.completion_tokens == 7  # cumulative last-wins
    assert terminal.usage.total_tokens == 18
    assert terminal.finish_reason == "end_turn"


@pytest.mark.asyncio
async def test_stream_emits_terminal_usage_without_message_stop(monkeypatch):
    # Regression: a truncated/proxied stream that ends WITHOUT a message_stop
    # event must still emit the accumulated usage (else the request bills $0).
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 5}}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}},
        {
            "type": "message_delta",
            "usage": {"output_tokens": 4},
            "delta": {"stop_reason": "max_tokens"},
        },
        # no message_stop
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse(events))

    _install_transport(monkeypatch, httpx.MockTransport(handler))
    chunks = [c async for c in AnthropicProvider().stream(_request(), "ak-1")]
    usage_chunks = [c for c in chunks if c.usage is not None]
    assert len(usage_chunks) == 1
    assert usage_chunks[0].usage.prompt_tokens == 5
    assert usage_chunks[0].usage.completion_tokens == 4
    assert usage_chunks[0].usage.total_tokens == 9
    assert usage_chunks[0].finish_reason == "max_tokens"


@pytest.mark.asyncio
async def test_stream_error_event_raises(monkeypatch):
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
        {"type": "error", "error": {"message": "overloaded"}},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse(events))

    _install_transport(monkeypatch, httpx.MockTransport(handler))
    with pytest.raises(ProviderError) as exc:
        async for _ in AnthropicProvider().stream(_request(), "ak-1"):
            pass
    assert exc.value.status_code == 502
    assert "overloaded" in str(exc.value)


@pytest.mark.asyncio
async def test_stream_http_error_status_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429, headers={"Retry-After": "9"}, content=b"rate limited"
        )

    _install_transport(monkeypatch, httpx.MockTransport(handler))
    with pytest.raises(ProviderError) as exc:
        async for _ in AnthropicProvider().stream(_request(), "ak-1"):
            pass
    assert exc.value.status_code == 429
    assert exc.value.retry_after == 9.0


@pytest.mark.asyncio
async def test_chat_parses_content_blocks_and_usage(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "msg-1",
                "model": "claude-3-5-sonnet-latest",
                "content": [
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "world"},
                    {"type": "thinking", "text": "ignored"},
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 4, "output_tokens": 6},
            },
        )

    _install_transport(monkeypatch, httpx.MockTransport(handler))
    resp = await AnthropicProvider().chat(_request(), "ak-1")
    assert resp.content == "Hello world"
    assert resp.usage.prompt_tokens == 4
    assert resp.usage.completion_tokens == 6
    assert resp.usage.total_tokens == 10
    assert resp.finish_reason == "end_turn"
