"""Offline unit tests for the Gemini (Generative Language API) adapter.

Drives ``GeminiProvider.chat``/``.stream`` over an ``httpx.MockTransport`` to
assert URL/path selection (generateContent vs streamGenerateContent?alt=sse),
the ``x-goog-api-key`` header, role mapping (assistant->model,
system->systemInstruction), ``generationConfig`` shaping, absolute
``usageMetadata`` parsing, and a single terminal usage chunk. No network.
"""

import json

import httpx
import pytest

import app.providers.gemini as gemini_mod
from app.providers.gemini import GeminiProvider
from app.schemas.chat import ChatRequest, Message


def _request(**kw) -> ChatRequest:
    base = {
        "model": "gemini-1.5-flash",
        "messages": [
            Message(role="system", content="sys-a"),
            Message(role="user", content="hi"),
            Message(role="assistant", content="prior"),
            Message(role="user", content="again"),
        ],
    }
    base.update(kw)
    return ChatRequest(**base)


def _install_transport(monkeypatch, transport: httpx.MockTransport) -> None:
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs.pop("transport", None)
        return real_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(gemini_mod.httpx, "AsyncClient", factory)


def _sse(chunks: list[dict]) -> bytes:
    return ("".join(f"data: {json.dumps(c)}\n" for c in chunks)).encode("utf-8")


# --- URL + payload (pure) ---------------------------------------------------


def test_url_non_stream():
    url = GeminiProvider()._url("gemini-1.5-flash", stream=False)
    assert url.endswith("/models/gemini-1.5-flash:generateContent")
    assert "alt=sse" not in url


def test_url_stream_has_alt_sse():
    url = GeminiProvider()._url("gemini-1.5-flash", stream=True)
    assert "/models/gemini-1.5-flash:streamGenerateContent" in url
    assert url.endswith("?alt=sse")


def test_url_strips_models_prefix():
    url = GeminiProvider()._url("models/gemini-2.0-flash", stream=False)
    assert url.endswith("/models/gemini-2.0-flash:generateContent")
    # not doubled
    assert "/models/models/" not in url


def test_payload_role_mapping_and_system_instruction():
    payload = GeminiProvider()._payload(
        _request(temperature=0.2, max_tokens=64, top_p=0.8, stop=["X"])
    )
    assert payload["systemInstruction"] == {"parts": [{"text": "sys-a"}]}
    roles = [c["role"] for c in payload["contents"]]
    assert roles == ["user", "model", "user"]
    assert payload["contents"][1]["parts"] == [{"text": "prior"}]

    gen = payload["generationConfig"]
    assert gen["temperature"] == 0.2
    assert gen["maxOutputTokens"] == 64
    assert gen["topP"] == 0.8
    assert gen["stopSequences"] == ["X"]


def test_payload_no_generation_config_when_unset():
    payload = GeminiProvider()._payload(
        ChatRequest(model="gemini-1.5-flash", messages=[Message(role="user", content="x")])
    )
    assert "generationConfig" not in payload
    assert "systemInstruction" not in payload


# --- chat() over MockTransport ---------------------------------------------


@pytest.mark.asyncio
async def test_chat_url_header_and_usage(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={
                "responseId": "g-1",
                "modelVersion": "gemini-1.5-flash-001",
                "candidates": [
                    {
                        "content": {"parts": [{"text": "hello"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 9,
                    "totalTokenCount": 14,
                },
            },
        )

    _install_transport(monkeypatch, httpx.MockTransport(handler))
    resp = await GeminiProvider().chat(_request(), "goog-key")

    assert captured["headers"]["x-goog-api-key"] == "goog-key"
    assert "authorization" not in captured["headers"]
    assert captured["url"].endswith("/models/gemini-1.5-flash:generateContent")
    assert resp.content == "hello"
    assert resp.usage.prompt_tokens == 5
    assert resp.usage.completion_tokens == 9
    assert resp.usage.total_tokens == 14
    assert resp.finish_reason == "STOP"


@pytest.mark.asyncio
async def test_stream_url_and_terminal_usage(monkeypatch):
    captured = {}
    chunks_in = [
        {"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]},
        {"candidates": [{"content": {"parts": [{"text": "lo"}]}}]},
        {
            "candidates": [
                {"content": {"parts": [{"text": ""}]}, "finishReason": "STOP"}
            ],
            "usageMetadata": {
                "promptTokenCount": 2,
                "candidatesTokenCount": 3,
                "totalTokenCount": 5,
            },
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, content=_sse(chunks_in))

    _install_transport(monkeypatch, httpx.MockTransport(handler))
    out = [c async for c in GeminiProvider().stream(_request(), "goog-key")]

    assert "streamGenerateContent" in captured["url"]
    assert "alt=sse" in captured["url"]

    text = "".join(c.text for c in out if c.text)
    assert text == "Hello"

    usage_chunks = [c for c in out if c.usage is not None]
    assert len(usage_chunks) == 1
    terminal = usage_chunks[0]
    assert terminal.usage.prompt_tokens == 2
    assert terminal.usage.completion_tokens == 3
    assert terminal.usage.total_tokens == 5
    assert terminal.finish_reason == "STOP"


@pytest.mark.asyncio
async def test_stream_terminal_chunk_is_last(monkeypatch):
    chunks_in = [
        {"candidates": [{"content": {"parts": [{"text": "a"}]}}]},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse(chunks_in))

    _install_transport(monkeypatch, httpx.MockTransport(handler))
    out = [c async for c in GeminiProvider().stream(_request(), "goog-key")]
    # Last chunk is always the terminal usage chunk (even without usageMetadata).
    assert out[-1].usage is not None
