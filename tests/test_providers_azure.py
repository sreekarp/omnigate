"""Offline unit tests for the Azure OpenAI adapter.

Verifies URL construction
(``endpoint.rstrip('/') + /openai/deployments/{deployment}/chat/completions
?api-version=...``), the ``api-key`` header (NOT ``Authorization: Bearer``),
and that parsing reuses the OpenAI helpers (response + stream). No network.
"""

import json

import httpx
import pytest

import app.providers.azure_openai as azure_mod
from app.providers.azure_openai import AzureOpenAIProvider
from app.providers.base import ProviderError
from app.schemas.chat import ChatRequest, Message


def _request(**kw) -> ChatRequest:
    base = {"model": "gpt-4o", "messages": [Message(role="user", content="hi")]}
    base.update(kw)
    return ChatRequest(**base)


def _install_transport(monkeypatch, transport: httpx.MockTransport) -> None:
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs.pop("transport", None)
        return real_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(azure_mod.httpx, "AsyncClient", factory)


def _provider(**kw) -> AzureOpenAIProvider:
    base = {
        "endpoint": "https://my-rsrc.openai.azure.com/",
        "deployment": "gpt4o-deploy",
        "api_version": "2024-10-21",
    }
    base.update(kw)
    return AzureOpenAIProvider(**base)


def test_url_builds_with_trailing_slash_trimmed():
    url = _provider()._url()
    assert url == (
        "https://my-rsrc.openai.azure.com/openai/deployments/gpt4o-deploy"
        "/chat/completions?api-version=2024-10-21"
    )


def test_url_honours_custom_api_version():
    url = _provider(api_version="2025-01-01-preview")._url()
    assert url.endswith("?api-version=2025-01-01-preview")


def test_headers_use_api_key_not_bearer():
    headers = _provider()._headers("azkey")
    assert headers["api-key"] == "azkey"
    assert "Authorization" not in headers


def test_headers_empty_key_raises_400():
    with pytest.raises(ProviderError) as exc:
        _provider()._headers("")
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_chat_uses_api_key_header_and_parses_like_openai(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "az-1",
                "model": "gpt-4o",
                "choices": [
                    {"message": {"content": "ok"}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    _install_transport(monkeypatch, httpx.MockTransport(handler))
    resp = await _provider().chat(_request(), "azkey")

    assert captured["headers"]["api-key"] == "azkey"
    assert "authorization" not in captured["headers"]
    assert "/openai/deployments/gpt4o-deploy/chat/completions" in captured["url"]
    assert "api-version=2024-10-21" in captured["url"]
    assert captured["body"]["stream"] is False
    assert resp.provider == "azure"
    assert resp.content == "ok"
    assert resp.usage.total_tokens == 2
    assert resp.finish_reason == "stop"


@pytest.mark.asyncio
async def test_chat_error_status_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="deployment not found")

    _install_transport(monkeypatch, httpx.MockTransport(handler))
    with pytest.raises(ProviderError) as exc:
        await _provider().chat(_request(), "azkey")
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_stream_parses_like_openai(monkeypatch):
    sse = (
        'data: {"choices":[{"delta":{"content":"hi"}}]}\n'
        'data: {"choices":[],"model":"gpt-4o",'
        '"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n'
        "data: [DONE]\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        # stream payload still includes include_usage (shared builder)
        assert json.loads(request.content)["stream_options"] == {
            "include_usage": True
        }
        return httpx.Response(200, content=sse.encode("utf-8"))

    _install_transport(monkeypatch, httpx.MockTransport(handler))
    out = [c async for c in _provider().stream(_request(), "azkey")]
    assert "".join(c.text for c in out if c.text) == "hi"
    usage_chunks = [c for c in out if c.usage is not None]
    assert len(usage_chunks) == 1
    assert usage_chunks[0].usage.total_tokens == 2
