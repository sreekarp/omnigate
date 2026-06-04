"""Shared, transport-agnostic helpers used by both Client and AsyncClient.

These functions never perform I/O themselves; they build requests, interpret
responses, and parse the gateway's plain-text streaming protocol. The sync and
async clients supply the actual httpx transport and the retry sleep.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Iterator, Optional

import httpx

from .exceptions import APIError, classify
from .models import ChatRequest, Message, StreamChunk, coerce_messages
from ._retry import parse_retry_after

logger = logging.getLogger("omnillm")

#: The literal sentinel the gateway appends to a stream on mid-stream failure.
ERROR_SENTINEL = "\n[error] "


def build_url(base_url: str, path: str) -> str:
    """Join ``base_url`` and ``path`` without doubling slashes."""
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def auth_headers(
    api_key: Optional[str],
    user_id: Optional[str],
    extra: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """Build request headers: ``x-api-key`` / ``x-user-id`` plus any extras."""
    headers: dict[str, str] = {}
    if extra:
        headers.update(extra)
    if api_key:
        headers["x-api-key"] = api_key
    if user_id:
        headers["x-user-id"] = user_id
    return headers


def prepare_chat_body(
    *,
    model: str,
    messages: object,
    max_tokens: Optional[int],
    temperature: Optional[float],
    stream: bool,
) -> dict[str, Any]:
    """Validate inputs via ``ChatRequest`` and produce the JSON body.

    ``max_tokens``/``temperature`` are omitted when ``None`` to match the
    server's own defaults; ``stream`` is always present.
    """
    msgs: list[Message] = coerce_messages(messages)
    req = ChatRequest(
        model=model,
        messages=msgs,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
    )
    body = req.model_dump(exclude_none=True)
    body["stream"] = stream
    return body


def request_id_of(response: httpx.Response) -> Optional[str]:
    """Pull the ``x-request-id`` header off a response, if present."""
    return response.headers.get("x-request-id")


def _parse_detail(response: httpx.Response) -> Any:
    """Best-effort extraction of the FastAPI ``detail`` field from a response."""
    try:
        body = response.json()
    except (ValueError, httpx.DecodingError):
        text = response.text
        return text or None
    if isinstance(body, dict):
        return body.get("detail", body)
    return body


def error_for_response(response: httpx.Response) -> APIError:
    """Build the typed exception for a >=400 response (does not raise)."""
    detail = _parse_detail(response)
    retry_after = parse_retry_after(response.headers.get("Retry-After"))
    return classify(
        response.status_code,
        detail,
        retry_after=retry_after,
        request_id=request_id_of(response),
    )


def handle_json_response(response: httpx.Response) -> Any:
    """Raise a typed error on >=400, else return the parsed JSON body.

    Tolerates an empty body (e.g. a 204) by returning ``None``.
    """
    if response.status_code >= 400:
        raise error_for_response(response)
    if response.status_code == 204 or not response.content:
        return None
    return response.json()


def iter_text_chunks(
    raw_iter: Iterable[str], request_id: Optional[str]
) -> Iterator[StreamChunk]:
    """Yield ``StreamChunk`` for each text fragment of a plain-text stream.

    Watches for the ``\\n[error] `` sentinel the gateway emits on a mid-stream
    failure and raises :class:`ProviderError` with the rest-of-line message so
    streaming failures are typed rather than silently appended as text.

    The sentinel may straddle httpx chunk boundaries, so we keep a small tail
    of unflushed text and rescan across the join. Once the sentinel is seen,
    everything after it (across all remaining chunks) is the error message.
    """
    from .exceptions import ProviderError  # local import to avoid cycle at top

    pending = ""  # text held back in case it is a partial sentinel prefix
    max_hold = len(ERROR_SENTINEL) - 1

    for piece in raw_iter:
        if not piece:
            continue
        buf = pending + piece
        idx = buf.find(ERROR_SENTINEL)
        if idx != -1:
            head = buf[:idx]
            if head:
                yield StreamChunk(text=head, request_id=request_id)
            message = buf[idx + len(ERROR_SENTINEL):]
            # Drain the remainder of the message from following chunks.
            for tail in raw_iter:
                message += tail
            raise ProviderError(
                message.strip() or "streaming provider error",
                status_code=502,
                detail=message.strip(),
                request_id=request_id,
            )
        # No full sentinel yet. Flush everything except a possible partial
        # sentinel prefix at the very end of the buffer.
        if len(buf) > max_hold:
            flush = buf[:-max_hold] if max_hold else buf
            pending = buf[-max_hold:] if max_hold else ""
            if flush:
                yield StreamChunk(text=flush, request_id=request_id)
        else:
            pending = buf

    if pending:
        yield StreamChunk(text=pending, request_id=request_id)
