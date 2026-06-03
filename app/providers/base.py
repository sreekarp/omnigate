"""AbstractProvider interface that all provider adapters implement."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.schemas.chat import ChatRequest, ChatResponse


class ProviderError(Exception):
    """Raised when a provider call fails. Carries an HTTP-ish status code."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class AbstractProvider(ABC):
    """Interface every provider adapter must implement.

    Adapters are responsible for translating the unified :class:`ChatRequest`
    into provider-native payloads and back into a :class:`ChatResponse`.
    """

    #: Short identifier, e.g. "openai" or "anthropic".
    name: str

    @abstractmethod
    async def chat(self, request: ChatRequest, api_key: str) -> ChatResponse:
        """Perform a non-streaming chat completion using the caller's key (BYOK)."""
        raise NotImplementedError

    @abstractmethod
    def stream(self, request: ChatRequest, api_key: str) -> AsyncIterator[str]:
        """Yield response text chunks for a streaming chat completion (BYOK).

        Implementations should be ``async def`` generators (which return an
        ``AsyncIterator`` when called), so this is intentionally not declared
        ``async`` itself.
        """
        raise NotImplementedError
