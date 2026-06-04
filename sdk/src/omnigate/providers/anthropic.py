"""Anthropic Messages API provider spec (pure: builds payloads, parses bodies).

Translates the unified OpenAI-style request into Anthropic's format: system
messages concatenate into the top-level ``system`` field; the rest map to
``messages``; ``max_tokens`` is required by Anthropic so it is defaulted. Ported
from the gateway's ``app/providers/anthropic.py``, stripped of I/O.

Streaming usage is split across events: ``input_tokens`` from ``message_start``,
cumulative ``output_tokens`` (last-wins) from ``message_delta``. The accumulator
lives in :class:`StreamState`; exactly one terminal usage chunk is emitted from
:meth:`stream_end`.
"""

from __future__ import annotations

from typing import Optional

from ..exceptions import APIError, ProviderError
from ..models import ChatRequest, ChatResponse, StreamChunk, Usage
from .base import ProviderSpec, StreamState, Target

_BASE_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 1024


def parse_retry_after(value: Optional[str]) -> Optional[float]:
    """Parse a delta-seconds ``Retry-After`` header into a float (or ``None``)."""
    if not value:
        return None
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return None


class AnthropicSpec(ProviderSpec):
    name = "anthropic"

    def url(self, request: ChatRequest, target: Optional[Target]) -> str:
        return _BASE_URL

    def headers(self, api_key: str) -> dict[str, str]:
        if not api_key:
            raise APIError("No Anthropic API key provided", status_code=400)
        return {
            "x-api-key": api_key,
            "anthropic-version": _API_VERSION,
            "Content-Type": "application/json",
        }

    def build_payload(self, request: ChatRequest, *, stream: bool) -> dict:
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

    def parse_response(self, data: dict, request_model: str) -> ChatResponse:
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
            model=data.get("model", request_model),
            content=text,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            finish_reason=data.get("stop_reason"),
        )

    def stream_feed(self, state: StreamState, raw: dict) -> list[StreamChunk]:
        etype = raw.get("type")
        if etype == "message_start":
            state.data["in"] = (
                raw.get("message", {}).get("usage", {}).get("input_tokens", 0)
            )
            return []
        if etype == "content_block_delta":
            delta = raw.get("delta", {})
            if delta.get("type") == "text_delta" and delta.get("text"):
                return [StreamChunk(text=delta["text"])]
            return []
        if etype == "message_delta":
            usage = raw.get("usage") or {}
            if "output_tokens" in usage:
                state.data["out"] = usage["output_tokens"]  # cumulative; last wins
            stop = raw.get("delta", {}).get("stop_reason")
            if stop:
                state.data["finish"] = stop
            return []
        if etype == "error":
            msg = raw.get("error", {}).get("message", "stream error")
            raise ProviderError(f"Anthropic stream error: {msg}", status_code=502)
        # 'message_stop', 'ping', etc. are ignored; terminal chunk in stream_end.
        return []

    def stream_end(self, state: StreamState) -> Optional[StreamChunk]:
        input_tokens = state.data.get("in", 0)
        output_tokens = state.data.get("out", 0)
        return StreamChunk(
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            finish_reason=state.data.get("finish"),
        )
