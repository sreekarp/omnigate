"""The ``ProviderSpec`` interface every in-process adapter implements.

A spec is **pure**: it builds request URLs/headers/payloads and parses
responses/stream events, but performs no network I/O. ``engine.py`` owns the two
executors (sync ``httpx.Client`` and async ``httpx.AsyncClient``) that call the
spec, which lets all four providers share one parser each across sync, async, and
streaming paths — and makes everything testable offline with ``MockTransport``.

Streaming contract: the engine frames the SSE/line protocol and, per decoded
JSON object, calls :meth:`ProviderSpec.stream_feed` (which may yield content
chunks and/or a terminal usage chunk). Providers that accumulate usage across
events (Anthropic, Gemini) carry state in :class:`StreamState` and emit their
single terminal usage chunk from :meth:`ProviderSpec.stream_end`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from ..keys import Target  # re-exported for engine/provider convenience
from ..models import ChatRequest, ChatResponse, StreamChunk

__all__ = ["ProviderSpec", "StreamState", "Target"]


class StreamState:
    """Mutable per-stream accumulator; ``data`` holds provider-defined fields."""

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}


class ProviderSpec(ABC):
    """Pure (I/O-free) description of how to talk to one provider."""

    #: Short identifier, e.g. "openai", "anthropic", "gemini", "azure".
    name: str

    @abstractmethod
    def url(self, request: ChatRequest, target: Optional[Target]) -> str:
        """The request URL (Azure uses ``target`` for endpoint/deployment/version)."""

    def stream_url(self, request: ChatRequest, target: Optional[Target]) -> str:
        """Streaming URL; defaults to :meth:`url` (Gemini overrides it)."""
        return self.url(request, target)

    @abstractmethod
    def headers(self, api_key: str) -> dict[str, str]:
        """Auth + content-type headers for the call."""

    @abstractmethod
    def build_payload(self, request: ChatRequest, *, stream: bool) -> dict:
        """Build the provider-native request body."""

    @abstractmethod
    def parse_response(self, data: dict, request_model: str) -> ChatResponse:
        """Parse a non-streaming response body into a :class:`ChatResponse`."""

    # --- streaming protocol ------------------------------------------------

    def stream_begin(self) -> StreamState:
        """Create a fresh per-stream accumulator."""
        return StreamState()

    @abstractmethod
    def stream_feed(self, state: StreamState, raw: dict) -> list[StreamChunk]:
        """Translate one decoded stream object into 0+ :class:`StreamChunk`\\ s."""

    def stream_end(self, state: StreamState) -> Optional[StreamChunk]:
        """Final terminal chunk (usage), or ``None`` if already emitted inline."""
        return None
