"""Google Gemini (Generative Language API, v1beta) adapter — async, httpx.

Wire notes:
- The model id lives in the URL path (``/models/{model}:generateContent``),
  not the body. A leading ``models/`` prefix is stripped defensively.
- Auth uses the ``x-goog-api-key`` header (keeps the BYOK secret out of URLs).
- Streaming uses ``?alt=sse`` so the response is line-delimited ``data:`` SSE.
- Roles map: system -> top-level ``systemInstruction`` (concatenated parts),
  user -> ``user``, assistant -> ``model``.
- ``usageMetadata`` token counts are absolute/last-wins, never re-summed.
"""

import json
from collections.abc import AsyncIterator
from urllib.parse import quote

import httpx

from app.config import get_settings
from app.logging_config import get_logger
from app.providers.base import AbstractProvider, ProviderError
from app.schemas.chat import ChatRequest, ChatResponse, StreamChunk, Usage

logger = get_logger(__name__)

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def _normalise_model(model: str) -> str:
    """Strip a leading ``models/`` so it isn't doubled in the URL path."""
    return model[len("models/") :] if model.startswith("models/") else model


class GeminiProvider(AbstractProvider):
    name = "gemini"

    def __init__(self) -> None:
        self._settings = get_settings()

    def _headers(self, api_key: str) -> dict[str, str]:
        if not api_key:
            raise ProviderError("No Gemini API key provided", status_code=400)
        return {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    def _url(self, model: str, *, stream: bool) -> str:
        action = "streamGenerateContent" if stream else "generateContent"
        model_id = quote(_normalise_model(model), safe="")
        url = f"{_BASE_URL}/models/{model_id}:{action}"
        return f"{url}?alt=sse" if stream else url

    def _payload(self, request: ChatRequest) -> dict:
        system_parts: list[str] = []
        contents: list[dict] = []
        for m in request.messages:
            if m.role == "system":
                system_parts.append(m.content)
                continue
            role = "model" if m.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m.content}]})

        payload: dict = {"contents": contents}
        if system_parts:
            payload["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_parts)}]
            }

        gen: dict = {}
        if request.max_tokens is not None:
            gen["maxOutputTokens"] = request.max_tokens
        if request.temperature is not None:
            gen["temperature"] = request.temperature
        if request.top_p is not None:
            gen["topP"] = request.top_p
        stops = request.stop_sequences()
        if stops:
            gen["stopSequences"] = stops
        if gen:
            payload["generationConfig"] = gen
        return payload

    @staticmethod
    def _extract_text(candidate: dict) -> str:
        parts = (candidate.get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts)

    @staticmethod
    def _usage_from_metadata(um: dict) -> Usage:
        prompt = um.get("promptTokenCount", 0)
        completion = um.get("candidatesTokenCount", 0)
        total = um.get("totalTokenCount", prompt + completion)
        return Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
        )

    async def chat(self, request: ChatRequest, api_key: str) -> ChatResponse:
        timeout = self._settings.request_timeout_seconds
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                resp = await client.post(
                    self._url(request.model, stream=False),
                    headers=self._headers(api_key),
                    json=self._payload(request),
                )
            except httpx.HTTPError as exc:
                raise ProviderError(f"Gemini request failed: {exc}") from exc

        if resp.status_code >= 400:
            raise ProviderError(
                f"Gemini error {resp.status_code}: {resp.text}",
                status_code=resp.status_code,
            )

        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get(
                "blockReason", "no candidates"
            )
            raise ProviderError(
                f"Gemini returned no candidates: {reason}", status_code=400
            )

        candidate = candidates[0]
        text = self._extract_text(candidate)
        usage = self._usage_from_metadata(data.get("usageMetadata") or {})
        return ChatResponse(
            id=data.get("responseId", ""),
            provider=self.name,
            model=data.get("modelVersion", request.model),
            content=text,
            usage=usage,
            finish_reason=candidate.get("finishReason"),
        )

    async def stream(
        self, request: ChatRequest, api_key: str
    ) -> AsyncIterator[StreamChunk]:
        timeout = self._settings.request_timeout_seconds
        # usageMetadata is absolute/last-wins; finishReason captured from
        # whichever chunk carries it. Emit one terminal usage chunk at the end.
        usage = Usage()
        finish_reason: str | None = None
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                self._url(request.model, stream=True),
                headers=self._headers(api_key),
                json=self._payload(request),
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise ProviderError(
                        f"Gemini error {resp.status_code}: "
                        f"{body.decode(errors='replace')}",
                        status_code=resp.status_code,
                    )
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if not data:
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed Gemini stream chunk")
                        continue
                    candidates = chunk.get("candidates") or []
                    if candidates:
                        piece = self._extract_text(candidates[0])
                        if piece:
                            yield StreamChunk(text=piece)
                        fr = candidates[0].get("finishReason")
                        if fr:
                            finish_reason = fr
                    um = chunk.get("usageMetadata")
                    if um:  # absolute / last-write-wins
                        usage = self._usage_from_metadata(um)
        yield StreamChunk(usage=usage, finish_reason=finish_reason)
