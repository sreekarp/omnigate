"""OpenAI chat-completions adapter (async, httpx)."""

import json
from collections.abc import AsyncIterator

import httpx

from app.config import get_settings
from app.logging_config import get_logger
from app.providers.base import AbstractProvider, ProviderError
from app.schemas.chat import ChatRequest, ChatResponse, Usage

logger = get_logger(__name__)

_BASE_URL = "https://api.openai.com/v1/chat/completions"


class OpenAIProvider(AbstractProvider):
    name = "openai"

    def __init__(self) -> None:
        self._settings = get_settings()

    def _headers(self, api_key: str) -> dict[str, str]:
        if not api_key:
            raise ProviderError("No OpenAI API key provided", status_code=400)
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, request: ChatRequest, *, stream: bool) -> dict:
        payload: dict = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "stream": stream,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        return payload

    async def chat(self, request: ChatRequest, api_key: str) -> ChatResponse:
        timeout = self._settings.request_timeout_seconds
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                resp = await client.post(
                    _BASE_URL,
                    headers=self._headers(api_key),
                    json=self._payload(request, stream=False),
                )
            except httpx.HTTPError as exc:
                raise ProviderError(f"OpenAI request failed: {exc}") from exc

        if resp.status_code >= 400:
            raise ProviderError(
                f"OpenAI error {resp.status_code}: {resp.text}",
                status_code=resp.status_code,
            )

        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return ChatResponse(
            id=data.get("id", ""),
            provider=self.name,
            model=data.get("model", request.model),
            content=choice["message"]["content"] or "",
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
        )

    async def stream(self, request: ChatRequest, api_key: str) -> AsyncIterator[str]:
        timeout = self._settings.request_timeout_seconds
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                _BASE_URL,
                headers=self._headers(api_key),
                json=self._payload(request, stream=True),
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise ProviderError(
                        f"OpenAI error {resp.status_code}: {body.decode(errors='replace')}",
                        status_code=resp.status_code,
                    )
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed OpenAI stream chunk")
                        continue
                    delta = chunk["choices"][0].get("delta", {})
                    piece = delta.get("content")
                    if piece:
                        yield piece
