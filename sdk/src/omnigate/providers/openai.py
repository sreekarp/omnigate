"""OpenAI chat-completions provider spec (pure: builds payloads, parses bodies).

The module-level helpers (:func:`build_chat_payload`, :func:`parse_chat_response`,
:func:`parse_stream_chunk`) are reused by the Azure spec, whose wire format is
byte-identical to OpenAI's chat-completions API. Ported from the gateway's
``app/providers/openai.py`` and stripped of all I/O — the engine performs the
HTTP call.
"""

from __future__ import annotations

from typing import Optional

from ..exceptions import APIError
from ..models import ChatRequest, ChatResponse, StreamChunk, Usage
from .base import ProviderSpec, StreamState, Target

_BASE_URL = "https://api.openai.com/v1/chat/completions"


def parse_retry_after(value: Optional[str]) -> Optional[float]:
    """Parse a delta-seconds ``Retry-After`` header into a float (or ``None``)."""
    if not value:
        return None
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return None


def build_chat_payload(request: ChatRequest, *, stream: bool) -> dict:
    """Build an OpenAI-style chat-completions request body.

    Optional sampling params are only included when set. When ``stream`` is true,
    ``stream_options.include_usage`` is added so the provider emits a final chunk
    carrying usage totals.
    """
    payload: dict = {
        "model": request.model,
        "messages": [m.model_dump() for m in request.messages],
        "stream": stream,
    }
    if request.max_tokens is not None:
        payload["max_tokens"] = request.max_tokens
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    stops = request.stop_sequences()
    if stops:
        payload["stop"] = stops
    if request.presence_penalty is not None:
        payload["presence_penalty"] = request.presence_penalty
    if request.frequency_penalty is not None:
        payload["frequency_penalty"] = request.frequency_penalty
    if request.seed is not None:
        payload["seed"] = request.seed
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def parse_chat_response(
    data: dict, *, provider_name: str, request_model: str
) -> ChatResponse:
    """Parse a non-streaming OpenAI-style response into a :class:`ChatResponse`."""
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    return ChatResponse(
        id=data.get("id", ""),
        provider=provider_name,
        model=data.get("model", request_model),
        content=choice["message"].get("content") or "",
        usage=Usage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        ),
        finish_reason=choice.get("finish_reason"),
    )


def parse_stream_chunk(chunk: dict) -> list[StreamChunk]:
    """Translate one decoded OpenAI-style SSE chunk into ``StreamChunk``\\ s.

    Returns a (possibly empty) list: a content/finish chunk when ``choices``
    carries a delta or finish_reason, plus a terminal usage chunk when ``usage``
    is present (the ``include_usage`` final chunk has ``choices == []``).
    """
    out: list[StreamChunk] = []
    choices = chunk.get("choices") or []
    if choices:
        delta = choices[0].get("delta") or {}
        piece = delta.get("content")
        finish = choices[0].get("finish_reason")
        if piece or finish:
            out.append(StreamChunk(text=piece or "", finish_reason=finish, model=chunk.get("model")))
    usage = chunk.get("usage")
    if usage:
        out.append(
            StreamChunk(
                usage=Usage(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                ),
                model=chunk.get("model"),
            )
        )
    return out


class OpenAISpec(ProviderSpec):
    name = "openai"

    def url(self, request: ChatRequest, target: Optional[Target]) -> str:
        return _BASE_URL

    def headers(self, api_key: str) -> dict[str, str]:
        if not api_key:
            raise APIError("No OpenAI API key provided", status_code=400)
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def build_payload(self, request: ChatRequest, *, stream: bool) -> dict:
        return build_chat_payload(request, stream=stream)

    def parse_response(self, data: dict, request_model: str) -> ChatResponse:
        return parse_chat_response(
            data, provider_name=self.name, request_model=request_model
        )

    def stream_feed(self, state: StreamState, raw: dict) -> list[StreamChunk]:
        return parse_stream_chunk(raw)
