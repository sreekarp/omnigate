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
from app.schemas.chat import ChatRequest, ChatResponse, StreamChunk, Usage

logger = get_logger(__name__)

_BASE_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 1024


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a delta-seconds ``Retry-After`` header into a float (or ``None``)."""
    if not value:
        return None
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return None


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
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        stops = request.stop_sequences()
        if stops:
            payload["stop_sequences"] = stops
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
                retry_after=_parse_retry_after(resp.headers.get("Retry-After"))
                if resp.status_code == 429
                else None,
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
            finish_reason=data.get("stop_reason"),
        )

    async def stream(
        self, request: ChatRequest, api_key: str
    ) -> AsyncIterator[StreamChunk]:
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
                        f"Anthropic error {resp.status_code}: "
                        f"{body.decode(errors='replace')}",
                        status_code=resp.status_code,
                        retry_after=_parse_retry_after(
                            resp.headers.get("Retry-After")
                        )
                        if resp.status_code == 429
                        else None,
                    )
                # Usage is split across events: input_tokens from
                # message_start, output_tokens (cumulative, last-wins) from
                # each message_delta. We emit exactly ONE terminal usage chunk
                # after the loop (mirroring the other adapters) so usage is
                # never lost if the upstream stream ends without message_stop.
                input_tokens = 0
                output_tokens = 0
                finish_reason: str | None = None
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed Anthropic stream chunk")
                        continue
                    etype = event.get("type")
                    if etype == "message_start":
                        input_tokens = (
                            event.get("message", {})
                            .get("usage", {})
                            .get("input_tokens", 0)
                        )
                    elif etype == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            yield StreamChunk(text=delta["text"])
                    elif etype == "message_delta":
                        usage = event.get("usage") or {}
                        if "output_tokens" in usage:
                            # cumulative; last value wins
                            output_tokens = usage["output_tokens"]
                        stop = event.get("delta", {}).get("stop_reason")
                        if stop:
                            finish_reason = stop
                    elif etype == "error":
                        msg = event.get("error", {}).get("message", "stream error")
                        raise ProviderError(
                            f"Anthropic stream error: {msg}", status_code=502
                        )
                    # 'message_stop', 'ping' and other event types are ignored;
                    # the terminal usage chunk is yielded once below.

                yield StreamChunk(
                    usage=Usage(
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
                    ),
                    finish_reason=finish_reason,
                )
