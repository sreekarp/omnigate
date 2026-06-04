"""Google Gemini (Generative Language API, v1beta) provider spec.

Wire notes: the model id lives in the URL path (``/models/{model}:generateContent``);
auth uses ``x-goog-api-key``; streaming uses ``?alt=sse``; roles map system ->
top-level ``systemInstruction``, user -> ``user``, assistant -> ``model``;
``usageMetadata`` counts are absolute/last-wins. Ported from the gateway's
``app/providers/gemini.py``, stripped of I/O.
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import quote

from ..exceptions import APIError, ProviderError
from ..models import ChatRequest, ChatResponse, StreamChunk, Usage
from .base import ProviderSpec, StreamState, Target

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def _normalise_model(model: str) -> str:
    """Strip a leading ``models/`` so it isn't doubled in the URL path."""
    return model[len("models/") :] if model.startswith("models/") else model


def _extract_text(candidate: dict) -> str:
    parts = (candidate.get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts)


def _usage_from_metadata(um: dict) -> Usage:
    prompt = um.get("promptTokenCount", 0)
    completion = um.get("candidatesTokenCount", 0)
    total = um.get("totalTokenCount", prompt + completion)
    return Usage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


class GeminiSpec(ProviderSpec):
    name = "gemini"

    def url(self, request: ChatRequest, target: Optional[Target]) -> str:
        model_id = quote(_normalise_model(request.model), safe="")
        return f"{_BASE_URL}/models/{model_id}:generateContent"

    def stream_url(self, request: ChatRequest) -> str:
        """Streaming URL variant (``streamGenerateContent`` + SSE)."""
        model_id = quote(_normalise_model(request.model), safe="")
        return f"{_BASE_URL}/models/{model_id}:streamGenerateContent?alt=sse"

    def headers(self, api_key: str) -> dict[str, str]:
        if not api_key:
            raise APIError("No Gemini API key provided", status_code=400)
        return {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    def build_payload(self, request: ChatRequest, *, stream: bool) -> dict:
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
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}

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

    def parse_response(self, data: dict, request_model: str) -> ChatResponse:
        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
            raise ProviderError(f"Gemini returned no candidates: {reason}", status_code=400)
        candidate = candidates[0]
        return ChatResponse(
            id=data.get("responseId", ""),
            provider=self.name,
            model=data.get("modelVersion", request_model),
            content=_extract_text(candidate),
            usage=_usage_from_metadata(data.get("usageMetadata") or {}),
            finish_reason=candidate.get("finishReason"),
        )

    def stream_feed(self, state: StreamState, raw: dict) -> list[StreamChunk]:
        out: list[StreamChunk] = []
        candidates = raw.get("candidates") or []
        if candidates:
            piece = _extract_text(candidates[0])
            if piece:
                out.append(StreamChunk(text=piece))
            fr = candidates[0].get("finishReason")
            if fr:
                state.data["finish"] = fr
        um = raw.get("usageMetadata")
        if um:  # absolute / last-write-wins
            state.data["usage"] = _usage_from_metadata(um)
        return out

    def stream_end(self, state: StreamState) -> Optional[StreamChunk]:
        usage = state.data.get("usage") or Usage()
        return StreamChunk(usage=usage, finish_reason=state.data.get("finish"))
