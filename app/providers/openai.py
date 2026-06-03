"""OpenAI chat-completions adapter (async, httpx).

Also exposes small module-level helpers (:func:`build_chat_payload`,
:func:`parse_chat_response`, :func:`parse_stream_chunk`) that the
Azure OpenAI adapter reuses — Azure's wire format is byte-identical to
OpenAI's chat-completions API, so the parsing logic is shared rather than
duplicated. The :class:`OpenAIProvider` public surface is unchanged.
"""

import json
from collections.abc import AsyncIterator

import httpx

from app.config import get_settings
from app.logging_config import get_logger
from app.providers.base import AbstractProvider, ProviderError
from app.schemas.chat import ChatRequest, ChatResponse, StreamChunk, Usage

logger = get_logger(__name__)

_BASE_URL = "https://api.openai.com/v1/chat/completions"


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header (delta-seconds form) into a float.

    OpenAI/Azure send the delta-seconds variant (e.g. ``"20"``). The
    HTTP-date variant is uncommon here and is ignored (returns ``None``).
    """
    if not value:
        return None
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return None


def build_chat_payload(request: ChatRequest, *, stream: bool) -> dict:
    """Build an OpenAI-style chat-completions request body.

    Shared with the Azure adapter. Optional sampling parameters are only
    included when set on the request. When ``stream`` is true,
    ``stream_options.include_usage`` is added so the provider emits a final
    chunk carrying the usage totals.
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

    Returns a (possibly empty) list:
    - a content/finish chunk when ``choices`` carries a delta or finish_reason
    - a terminal usage chunk when ``usage`` is present and truthy (the
      ``include_usage`` final chunk has ``choices == []``).
    """
    out: list[StreamChunk] = []
    choices = chunk.get("choices") or []
    if choices:
        delta = choices[0].get("delta") or {}
        piece = delta.get("content")
        finish = choices[0].get("finish_reason")
        if piece or finish:
            out.append(StreamChunk(text=piece or "", finish_reason=finish))
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
        return build_chat_payload(request, stream=stream)

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
                retry_after=_parse_retry_after(resp.headers.get("Retry-After"))
                if resp.status_code == 429
                else None,
            )

        return parse_chat_response(
            resp.json(), provider_name=self.name, request_model=request.model
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
                        f"OpenAI error {resp.status_code}: "
                        f"{body.decode(errors='replace')}",
                        status_code=resp.status_code,
                        retry_after=_parse_retry_after(resp.headers.get("Retry-After"))
                        if resp.status_code == 429
                        else None,
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
                    for sc in parse_stream_chunk(chunk):
                        yield sc
