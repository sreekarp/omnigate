"""Anthropic Messages API adapter (async, httpx).

Translates the unified OpenAI-style request into Anthropic's format:
- system messages are concatenated into the top-level ``system`` field
- remaining messages map to ``messages`` with role user/assistant
- ``max_tokens`` is required by Anthropic, so we default it when absent
"""

import json
from collections.abc import AsyncIterator

import httpx

from app.config import get_settings
from app.logging_config import get_logger
from app.providers.base import AbstractProvider, ProviderError
from app.schemas.chat import ChatRequest, ChatResponse, Usage

logger = get_logger(__name__)

_BASE_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 1024


class AnthropicProvider(AbstractProvider):
    name = "anthropic"

    def __init__(self) -> None:
        self._settings = get_settings()

    def _headers(self, api_key: str) -> dict[str, str]:
        if not api_key:
            raise ProviderError("No Anthropic API key provided", status_code=400)
        return {
            "x-api-key": api_key,
            "anthropic-version": _API_VERSION,
            "Content-Type": "application/json",
        }

    def _payload(self, request: ChatRequest, *, stream: bool) -> dict:
        system_parts: list[str] = []
        messages: list[dict] = []
        for m in request.messages:
            if m.role == "system":
                system_parts.append(m.content)
            else:
                messages.append({"role": m.role, "content": m.content})

        payload: dict = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens or _DEFAULT_MAX_TOKENS,
            "stream": stream,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
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
                raise ProviderError(f"Anthropic request failed: {exc}") from exc

        if resp.status_code >= 400:
            raise ProviderError(
                f"Anthropic error {resp.status_code}: {resp.text}",
                status_code=resp.status_code,
            )

        data = resp.json()
        # content is a list of blocks; concatenate text blocks.
        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
        usage = data.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        return ChatResponse(
            id=data.get("id", ""),
            provider=self.name,
            model=data.get("model", request.model),
            content=text,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
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
                        f"Anthropic error {resp.status_code}: {body.decode(errors='replace')}",
                        status_code=resp.status_code,
                    )
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed Anthropic stream chunk")
                        continue
                    if event.get("type") == "content_block_delta":
                        piece = event.get("delta", {}).get("text")
                        if piece:
                            yield piece
