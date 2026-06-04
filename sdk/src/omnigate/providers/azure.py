"""Azure OpenAI provider spec. Wire format is byte-identical to OpenAI's.

Differences from OpenAI: the URL carries the deployment + an ``api-version``
query param, and auth uses the ``api-key`` header (not ``Authorization: Bearer``).
Payload building and response/stream parsing are reused from
:mod:`omnigate.providers.openai` so the two never diverge. The Azure target
(endpoint/deployment/version) is resolved per call in ``keys.resolve_target``.
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import quote

from ..exceptions import APIError
from ..models import ChatRequest, ChatResponse, StreamChunk
from .base import ProviderSpec, StreamState, Target
from .openai import build_chat_payload, parse_chat_response, parse_stream_chunk


class AzureSpec(ProviderSpec):
    name = "azure"

    def url(self, request: ChatRequest, target: Optional[Target]) -> str:
        if target is None:
            raise APIError("Azure call requires a resolved target", status_code=400)
        deployment = quote(target.deployment, safe="")
        return (
            f"{target.endpoint}/openai/deployments/{deployment}"
            f"/chat/completions?api-version={target.api_version}"
        )

    def headers(self, api_key: str) -> dict[str, str]:
        if not api_key:
            raise APIError("No Azure OpenAI API key provided", status_code=400)
        return {"api-key": api_key, "Content-Type": "application/json"}

    def build_payload(self, request: ChatRequest, *, stream: bool) -> dict:
        return build_chat_payload(request, stream=stream)

    def parse_response(self, data: dict, request_model: str) -> ChatResponse:
        return parse_chat_response(
            data, provider_name=self.name, request_model=request_model
        )

    def stream_feed(self, state: StreamState, raw: dict) -> list[StreamChunk]:
        return parse_stream_chunk(raw)
