"""Asynchronous client for the LLM Gateway.

Mirrors :class:`llm_gateway.client.Client` one-to-one: identical constructor and
method *names*, but every method is ``async def`` and :meth:`chat_stream`
returns an ``AsyncIterator``. Use ``async with`` / ``await aclose()`` for
lifecycle management.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Optional, Union

import httpx

from . import _transport as T
from ._retry import RetryConfig, compute_delay, parse_retry_after, should_retry
from ._version import __version__
from .exceptions import ConnectionError as GatewayConnectionError, ProviderError
from .models import (
    ChatResponse,
    MeResponse,
    MetricsResponse,
    ModelInfo,
    SignupResponse,
    StreamChunk,
)

logger = logging.getLogger("llm_gateway")

_USER_AGENT = f"llm-gateway-sdk/{__version__}"


class AsyncClient:
    """Asynchronous LLM Gateway client.

    Example::

        async with AsyncClient(api_key="llmg_...") as c:
            resp = await c.chat(model="gpt-4o-mini", messages="Hello!")
            async for piece in c.chat_stream(model="gpt-4o-mini", messages="hi"):
                print(piece, end="")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: str = "http://localhost:8000",
        user_id: Optional[str] = None,
        timeout: float = 60.0,
        retries: int = 2,
        retry_config: Optional[RetryConfig] = None,
        headers: Optional[dict[str, str]] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.user_id = user_id
        self._retry = retry_config or RetryConfig(max_retries=retries)
        self._extra_headers = dict(headers or {})
        self._extra_headers.setdefault("user-agent", _USER_AGENT)
        self._client = httpx.AsyncClient(
            timeout=timeout,
            transport=transport,
        )

    # --- lifecycle --------------------------------------------------------

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # --- low-level request with retry ------------------------------------

    def _headers(self, user_id: Optional[str]) -> dict[str, str]:
        return T.auth_headers(
            self.api_key, user_id or self.user_id, self._extra_headers
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: Any = None,
        user_id: Optional[str] = None,
    ) -> httpx.Response:
        url = T.build_url(self.base_url, path)
        headers = self._headers(user_id)
        attempt = 0
        while True:
            try:
                response = await self._client.request(
                    method, url, json=json, params=params, headers=headers
                )
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                if should_retry(None, attempt, self._retry):
                    delay = compute_delay(attempt, self._retry, None)
                    logger.debug("transport error, retrying in %.2fs: %s", delay, exc)
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                raise GatewayConnectionError(
                    f"Could not reach gateway at {self.base_url}: {exc}"
                ) from exc

            if response.status_code >= 400 and should_retry(
                response.status_code, attempt, self._retry
            ):
                retry_after = parse_retry_after(response.headers.get("Retry-After"))
                delay = compute_delay(attempt, self._retry, retry_after)
                logger.debug(
                    "status %s, retrying in %.2fs", response.status_code, delay
                )
                await response.aclose()
                await asyncio.sleep(delay)
                attempt += 1
                continue
            return response

    # --- chat -------------------------------------------------------------

    async def chat(
        self,
        *,
        model: str,
        messages: Any,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        user_id: Optional[str] = None,
    ) -> ChatResponse:
        """Non-streaming chat completion (``POST /v1/chat``)."""
        body = T.prepare_chat_body(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
        resp = await self._request("POST", "/v1/chat", json=body, user_id=user_id)
        data = T.handle_json_response(resp)
        return ChatResponse.model_validate(data)

    def chat_stream(
        self,
        *,
        model: str,
        messages: Any,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        user_id: Optional[str] = None,
        as_chunks: bool = False,
    ) -> Union[AsyncIterator[str], AsyncIterator[StreamChunk]]:
        """Stream a chat completion as an async iterator.

        Yields ``str`` by default, or :class:`StreamChunk` when
        ``as_chunks=True``. Raises :class:`ProviderError` on the gateway's
        mid-stream error sentinel.
        """
        if as_chunks:
            return self._stream_chunks(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                user_id=user_id,
            )
        return self._stream_text(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            user_id=user_id,
        )

    async def _stream_chunks(
        self,
        *,
        model: str,
        messages: Any,
        max_tokens: Optional[int],
        temperature: Optional[float],
        user_id: Optional[str],
    ) -> AsyncIterator[StreamChunk]:
        body = T.prepare_chat_body(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        url = T.build_url(self.base_url, "/v1/chat")
        headers = self._headers(user_id)
        try:
            async with self._client.stream(
                "POST", url, json=body, headers=headers
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise T.error_for_response(response)
                request_id = T.request_id_of(response)
                async for chunk in _aiter_text_chunks(
                    response.aiter_text(), request_id
                ):
                    yield chunk
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            raise GatewayConnectionError(
                f"Could not reach gateway at {self.base_url}: {exc}"
            ) from exc

    async def _stream_text(
        self,
        *,
        model: str,
        messages: Any,
        max_tokens: Optional[int],
        temperature: Optional[float],
        user_id: Optional[str],
    ) -> AsyncIterator[str]:
        async for chunk in self._stream_chunks(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            user_id=user_id,
        ):
            if chunk.text:
                yield chunk.text

    async def completions(
        self,
        *,
        model: str,
        messages: Any,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """OpenAI-compatible completion (``POST /v1/chat/completions``)."""
        body = T.prepare_chat_body(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
        resp = await self._request(
            "POST", "/v1/chat/completions", json=body, user_id=user_id
        )
        return T.handle_json_response(resp)

    # --- catalog / metrics ------------------------------------------------

    async def models(self) -> list[ModelInfo]:
        """List available models (``GET /v1/models``; requires gateway task #6)."""
        resp = await self._request("GET", "/v1/models")
        data = T.handle_json_response(resp)
        items = data.get("data", data) if isinstance(data, dict) else data
        return [ModelInfo.model_validate(m) for m in (items or [])]

    async def metrics(self, *, window: str = "today") -> MetricsResponse:
        """Fetch usage metrics (``GET /v1/metrics``; requires gateway task #5)."""
        resp = await self._request("GET", "/v1/metrics", params={"window": window})
        data = T.handle_json_response(resp)
        return MetricsResponse.model_validate(data)

    # --- account / onboarding --------------------------------------------

    async def signup(
        self,
        *,
        email: str,
        org_name: Optional[str] = None,
        project_name: str = "Default",
    ) -> SignupResponse:
        """Self-serve signup (``POST /v1/signup``); works on a keyless client."""
        body: dict[str, Any] = {"email": email, "project_name": project_name}
        if org_name is not None:
            body["org_name"] = org_name
        resp = await self._request("POST", "/v1/signup", json=body)
        data = T.handle_json_response(resp)
        return SignupResponse.model_validate(data)

    async def set_provider_key(self, *, provider: str, api_key: str) -> None:
        """Store/replace your BYOK provider key (``POST /v1/keys`` -> 204)."""
        body = {"provider": provider, "api_key": api_key}
        resp = await self._request("POST", "/v1/keys", json=body)
        if resp.status_code >= 400:
            raise T.error_for_response(resp)

    async def create_api_key(self, *, name: str) -> dict[str, Any]:
        """Mint an additional gateway api key (``POST /v1/keys/create``)."""
        resp = await self._request("POST", "/v1/keys/create", json={"name": name})
        return T.handle_json_response(resp)

    async def me(self) -> MeResponse:
        """Account info for the current key (``GET /v1/me``)."""
        resp = await self._request("GET", "/v1/me")
        data = T.handle_json_response(resp)
        return MeResponse.model_validate(data)

    async def health(self) -> dict[str, Any]:
        """Gateway liveness (``GET /health``)."""
        resp = await self._request("GET", "/health")
        return T.handle_json_response(resp)


async def _aiter_text_chunks(
    raw_aiter: AsyncIterator[str], request_id: Optional[str]
) -> AsyncIterator[StreamChunk]:
    """Async twin of ``_transport.iter_text_chunks`` (same sentinel handling)."""
    pending = ""
    max_hold = len(T.ERROR_SENTINEL) - 1

    async for piece in raw_aiter:
        if not piece:
            continue
        buf = pending + piece
        idx = buf.find(T.ERROR_SENTINEL)
        if idx != -1:
            head = buf[:idx]
            if head:
                yield StreamChunk(text=head, request_id=request_id)
            message = buf[idx + len(T.ERROR_SENTINEL):]
            async for tail in raw_aiter:
                message += tail
            raise ProviderError(
                message.strip() or "streaming provider error",
                status_code=502,
                detail=message.strip(),
                request_id=request_id,
            )
        if len(buf) > max_hold:
            flush = buf[:-max_hold] if max_hold else buf
            pending = buf[-max_hold:] if max_hold else ""
            if flush:
                yield StreamChunk(text=flush, request_id=request_id)
        else:
            pending = buf

    if pending:
        yield StreamChunk(text=pending, request_id=request_id)
