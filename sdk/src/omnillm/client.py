"""Synchronous client for the OmniLLM.

Wraps an ``httpx.Client``, sends the ``x-api-key`` gateway key, retries 429/5xx
and transport errors with hand-rolled exponential backoff, and raises the SDK's
typed exceptions. Streaming uses the gateway's plain-text protocol (not SSE).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterator, Optional, Union

import httpx

from . import _transport as T
from ._retry import RetryConfig, compute_delay, parse_retry_after, should_retry
from ._version import __version__
from .exceptions import ConnectionError as GatewayConnectionError
from .models import (
    ApiKeyCreated,
    ChatResponse,
    MeResponse,
    MetricsResponse,
    ModelInfo,
    SignupResponse,
    StreamChunk,
)

logger = logging.getLogger("omnillm")

_USER_AGENT = f"omnillm/{__version__}"


class Client:
    """Synchronous OmniLLM client.

    Example::

        with Client(api_key="llmg_...", base_url="https://gw.example.com") as c:
            resp = c.chat(model="gpt-4o-mini", messages="Hello!")
            print(resp.content)

    ``api_key`` is optional so the public :meth:`signup` endpoint works on a
    keyless client.
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
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.user_id = user_id
        self._retry = retry_config or RetryConfig(max_retries=retries)
        self._extra_headers = dict(headers or {})
        self._extra_headers.setdefault("user-agent", _USER_AGENT)
        self._client = httpx.Client(
            timeout=timeout,
            transport=transport,
        )

    # --- lifecycle --------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- low-level request with retry ------------------------------------

    def _headers(self, user_id: Optional[str]) -> dict[str, str]:
        return T.auth_headers(
            self.api_key, user_id or self.user_id, self._extra_headers
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        user_id: Optional[str] = None,
    ) -> httpx.Response:
        """Issue a non-streaming request with retry/backoff; return the response."""
        url = T.build_url(self.base_url, path)
        headers = self._headers(user_id)
        attempt = 0
        last_exc: Optional[Exception] = None
        while True:
            try:
                response = self._client.request(
                    method, url, json=json, headers=headers
                )
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_exc = exc
                if should_retry(None, attempt, self._retry):
                    delay = compute_delay(attempt, self._retry, None)
                    logger.debug("transport error, retrying in %.2fs: %s", delay, exc)
                    time.sleep(delay)
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
                response.close()
                time.sleep(delay)
                attempt += 1
                continue
            return response

        # unreachable, but keeps type checkers happy
        raise GatewayConnectionError(str(last_exc))  # pragma: no cover

    # --- chat -------------------------------------------------------------

    def chat(
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
        resp = self._request("POST", "/v1/chat", json=body, user_id=user_id)
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
    ) -> Union[Iterator[str], Iterator[StreamChunk]]:
        """Stream a chat completion (``POST /v1/chat`` with ``stream=true``).

        Yields ``str`` text fragments by default, or :class:`StreamChunk`
        (carrying ``request_id``) when ``as_chunks=True``. Raises
        :class:`ProviderError` if the gateway emits its mid-stream error
        sentinel. Streaming requests are not retried once bytes are flowing.
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

    def _stream_chunks(
        self,
        *,
        model: str,
        messages: Any,
        max_tokens: Optional[int],
        temperature: Optional[float],
        user_id: Optional[str],
    ) -> Iterator[StreamChunk]:
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
            with self._client.stream(
                "POST", url, json=body, headers=headers
            ) as response:
                if response.status_code >= 400:
                    response.read()
                    raise T.error_for_response(response)
                request_id = T.request_id_of(response)
                yield from T.iter_text_chunks(
                    response.iter_text(), request_id
                )
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            raise GatewayConnectionError(
                f"Could not reach gateway at {self.base_url}: {exc}"
            ) from exc

    def _stream_text(
        self,
        *,
        model: str,
        messages: Any,
        max_tokens: Optional[int],
        temperature: Optional[float],
        user_id: Optional[str],
    ) -> Iterator[str]:
        for chunk in self._stream_chunks(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            user_id=user_id,
        ):
            if chunk.text:
                yield chunk.text

    def completions(
        self,
        *,
        model: str,
        messages: Any,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """OpenAI-compatible completion (``POST /v1/chat/completions``).

        Returns the raw OpenAI-shaped JSON dict so it can be consumed exactly
        like the OpenAI SDK's response. Requires gateway task #6 to be live.
        """
        body = T.prepare_chat_body(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
        resp = self._request(
            "POST", "/v1/chat/completions", json=body, user_id=user_id
        )
        return T.handle_json_response(resp)

    # --- catalog / metrics ------------------------------------------------

    def models(self) -> list[ModelInfo]:
        """List available models (``GET /v1/models``; requires gateway task #6)."""
        resp = self._request("GET", "/v1/models")
        data = T.handle_json_response(resp)
        items = data.get("data", data) if isinstance(data, dict) else data
        return [ModelInfo.model_validate(m) for m in (items or [])]

    def metrics(self, *, range: str = "24h") -> MetricsResponse:
        """Fetch project usage metrics (``GET /v1/metrics``).

        ``range`` is one of ``1h|24h|7d|30d`` (default ``24h``); the gateway also
        accepts explicit ``from``/``to`` ISO timestamps, but the SDK helper
        exposes the ``range`` shorthand. Returns the rich :class:`MetricsResponse`
        (``totals`` / ``breakdown`` / ``timeseries``).
        """
        url = T.build_url(self.base_url, "/v1/metrics")
        headers = self._headers(None)
        try:
            resp = self._client.get(url, params={"range": range}, headers=headers)
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            raise GatewayConnectionError(
                f"Could not reach gateway at {self.base_url}: {exc}"
            ) from exc
        data = T.handle_json_response(resp)
        return MetricsResponse.model_validate(data)

    # --- account / onboarding --------------------------------------------

    def signup(
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
        resp = self._request("POST", "/v1/signup", json=body)
        data = T.handle_json_response(resp)
        return SignupResponse.model_validate(data)

    def set_provider_key(self, *, provider: str, api_key: str) -> None:
        """Store/replace your BYOK provider key (``POST /v1/keys`` -> 204)."""
        body = {"provider": provider, "api_key": api_key}
        resp = self._request("POST", "/v1/keys", json=body)
        # 204 No Content: validate status, never parse a body.
        if resp.status_code >= 400:
            raise T.error_for_response(resp)

    def create_api_key(self, *, name: str) -> ApiKeyCreated:
        """Mint an additional gateway api key (``POST /v1/keys/api`` -> 201).

        Returns an :class:`ApiKeyCreated` carrying the one-time plaintext
        ``api_key`` (plus ``id``/``key_prefix``/timestamps); the plaintext is
        never recoverable afterwards, so persist it immediately.
        """
        resp = self._request("POST", "/v1/keys/api", json={"name": name})
        data = T.handle_json_response(resp)
        return ApiKeyCreated.model_validate(data)

    def me(self) -> MeResponse:
        """Account info for the current key (``GET /v1/me``)."""
        resp = self._request("GET", "/v1/me")
        data = T.handle_json_response(resp)
        return MeResponse.model_validate(data)

    def health(self) -> dict[str, Any]:
        """Gateway liveness (``GET /health``)."""
        resp = self._request("GET", "/health")
        return T.handle_json_response(resp)
