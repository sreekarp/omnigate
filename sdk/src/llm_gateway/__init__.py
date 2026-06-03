"""llm-gateway-sdk — Python client for the LLM Gateway.

Sync and async, streaming-aware, fully typed. Talks the gateway's HTTP surface
(``/v1/chat``, ``/v1/signup``, ``/v1/keys``, ``/v1/me``, ``/health`` and the
forthcoming ``/v1/models`` + ``/v1/metrics``).

Quick start::

    from llm_gateway import Client

    with Client(api_key="llmg_...", base_url="https://gw.example.com") as c:
        print(c.chat(model="gpt-4o-mini", messages="Hello!").content)
"""

from __future__ import annotations

from ._retry import RetryConfig
from ._version import __version__
from .async_client import AsyncClient
from .client import Client
from .exceptions import (
    APIError,
    AuthError,
    BudgetExceededError,
    ConnectionError,
    GatewayError,
    ProviderError,
    RateLimitError,
    classify,
)
from .models import (
    ChatRequest,
    ChatResponse,
    Message,
    MeResponse,
    MetricsResponse,
    ModelInfo,
    SignupResponse,
    StreamChunk,
    Usage,
)

__all__ = [
    "__version__",
    # clients
    "Client",
    "AsyncClient",
    "RetryConfig",
    # models
    "Message",
    "ChatRequest",
    "ChatResponse",
    "Usage",
    "StreamChunk",
    "SignupResponse",
    "MeResponse",
    "ModelInfo",
    "MetricsResponse",
    # exceptions
    "GatewayError",
    "APIError",
    "AuthError",
    "RateLimitError",
    "BudgetExceededError",
    "ProviderError",
    "ConnectionError",
    "classify",
]
