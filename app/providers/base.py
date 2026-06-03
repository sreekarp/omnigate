"""AbstractProvider interface that all provider adapters implement."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.schemas.chat import ChatRequest, ChatResponse, StreamChunk


class ProviderError(Exception):
    """Raised when a provider call fails. Carries an HTTP-ish status code.

    ``retry_after`` (seconds) is parsed from the provider's ``Retry-After``
    header when present, so the resilience layer can honour backoff hints.
    """

    def __init__(
        self,
        message: str,
        status_code: int = 502,
        *,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.retry_after = retry_after


class AbstractProvider(ABC):
    """Interface every provider adapter must implement.

    Adapters translate the unified :class:`ChatRequest` into provider-native
    payloads and back into a :class:`ChatResponse` (non-streaming) or a stream
    of :class:`StreamChunk` (streaming).
    """

    #: Short identifier, e.g. "openai", "anthropic", "gemini", "azure".
    name: str

    @abstractmethod
    async def chat(self, request: ChatRequest, api_key: str) -> ChatResponse:
        """Perform a non-streaming chat completion using the caller's key (BYOK)."""
        raise NotImplementedError

    @abstractmethod
    def stream(self, request: ChatRequest, api_key: str) -> AsyncIterator[StreamChunk]:
        """Yield :class:`StreamChunk`\\ s for a streaming chat completion (BYOK).

        Implementations are ``async def`` generators (which return an
        ``AsyncIterator`` when called), so this is intentionally *not* declared
        ``async`` itself — keep it a plain ``def`` in subclasses' generator form.

        Contract: yield content chunks (``text`` set) then exactly one terminal
        chunk carrying the final :class:`~app.schemas.chat.Usage` and
        ``finish_reason``.
        """
        raise NotImplementedError

    async def stream_text(
        self, request: ChatRequest, api_key: str
    ) -> AsyncIterator[str]:
        """Convenience wrapper yielding only non-empty text deltas."""
        async for chunk in self.stream(request, api_key):
            if chunk.text:
                yield chunk.text
