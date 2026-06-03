"""Azure OpenAI chat-completions adapter (async, httpx). BYOK ``api-key`` auth.

The wire format is byte-identical to OpenAI's ``/v1/chat/completions`` (body,
response JSON, SSE stream). The only differences are:
- the URL carries the deployment and an ``api-version`` query param,
- auth uses the ``api-key`` header (NOT ``Authorization: Bearer``),
- the body ``model`` field is ignored by Azure (the URL deployment selects the
  model) — we still send it for harmless compatibility via the shared payload
  builder.

Payload building and response/chunk parsing are reused from
:mod:`app.providers.openai` so the two adapters never diverge. The adapter is
constructed per-request with its resolved target (endpoint/deployment/version)
and is therefore NOT process-cached.
"""

import json
from collections.abc import AsyncIterator
from urllib.parse import quote

import httpx

from app.config import get_settings
from app.logging_config import get_logger
from app.providers.base import AbstractProvider, ProviderError
from app.providers.openai import (
    build_chat_payload,
    parse_chat_response,
    parse_stream_chunk,
)
from app.schemas.chat import ChatRequest, ChatResponse, StreamChunk

logger = get_logger(__name__)

_DEFAULT_API_VERSION = "2024-10-21"  # latest GA


class AzureOpenAIProvider(AbstractProvider):
    name = "azure"

    def __init__(
        self,
        *,
        endpoint: str,
        deployment: str,
        api_version: str = _DEFAULT_API_VERSION,
    ) -> None:
        self._settings = get_settings()
        self._endpoint = endpoint.rstrip("/")
        self._deployment = deployment
        self._api_version = api_version

    def _url(self) -> str:
        deployment = quote(self._deployment, safe="")
        return (
            f"{self._endpoint}/openai/deployments/{deployment}"
            f"/chat/completions?api-version={self._api_version}"
        )

    def _headers(self, api_key: str) -> dict[str, str]:
        if not api_key:
            raise ProviderError(
                "No Azure OpenAI API key provided", status_code=400
            )
        return {"api-key": api_key, "Content-Type": "application/json"}

    def _payload(self, request: ChatRequest, *, stream: bool) -> dict:
        return build_chat_payload(request, stream=stream)

    async def chat(self, request: ChatRequest, api_key: str) -> ChatResponse:
        timeout = self._settings.request_timeout_seconds
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                resp = await client.post(
                    self._url(),
                    headers=self._headers(api_key),
                    json=self._payload(request, stream=False),
                )
            except httpx.HTTPError as exc:
                raise ProviderError(
                    f"Azure OpenAI request failed: {exc}"
                ) from exc

        if resp.status_code >= 400:
            raise ProviderError(
                f"Azure OpenAI error {resp.status_code}: {resp.text}",
                status_code=resp.status_code,
            )

        # Keep the canonical incoming model name for logging/pricing.
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
                self._url(),
                headers=self._headers(api_key),
                json=self._payload(request, stream=True),
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise ProviderError(
                        f"Azure OpenAI error {resp.status_code}: "
                        f"{body.decode(errors='replace')}",
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
                        logger.warning("Skipping malformed Azure stream chunk")
                        continue
                    for sc in parse_stream_chunk(chunk):
                        yield sc
