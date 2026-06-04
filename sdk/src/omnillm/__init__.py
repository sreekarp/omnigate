"""omnillm — Python client for the OmniLLM.

Sync and async, streaming-aware, fully typed. Talks the gateway's HTTP surface:
``/v1/chat`` (text/plain streaming), ``/v1/chat/completions`` (OpenAI-compatible),
``/v1/models``, ``/v1/metrics``, ``/v1/keys`` (BYOK) + ``/v1/keys/api`` (gateway
key mgmt), ``/v1/signup``, ``/v1/me`` and ``/health``.

Quick start::

    from omnillm import Client

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
    ApiKeyCreated,
    ChatRequest,
    ChatResponse,
    Message,
    MeResponse,
    MetricsBreakdown,
    MetricsResponse,
    MetricsTimePoint,
    MetricsTotals,
    ModelInfo,
    ModelPricing,
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
    "ApiKeyCreated",
    "ModelInfo",
    "ModelPricing",
    "MetricsResponse",
    "MetricsTotals",
    "MetricsBreakdown",
    "MetricsTimePoint",
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
