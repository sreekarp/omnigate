"""omnigate — a litellm-style multi-provider LLM SDK.

Two ways to use it, both sync + async, streaming-aware, and fully typed:

1. **In-process (no hosting)** — call OpenAI/Anthropic/Gemini/Azure directly,
   with routing, retry, fallback, circuit breaking, cost tracking, an opt-in
   response cache, callbacks and a local spend cap. Keys come from the standard
   provider env vars (``OPENAI_API_KEY`` etc.) or an explicit ``api_key=``::

       import omnigate

       r = omnigate.completion(model="gpt-4o-mini", messages="Hello!")
       print(r.content, r.usage.total_tokens, r.cost_usd)

2. **Hosted gateway client** — point :class:`Client` / :class:`AsyncClient` at a
   running OmniGate server for centralised auth, budgets, rate limiting and
   metrics::

       from omnigate import Client

       with Client(api_key="llmg_...", base_url="https://gw.example.com") as c:
           print(c.chat(model="gpt-4o-mini", messages="Hello!").content)
"""

from __future__ import annotations

from ._retry import RetryConfig
from ._version import __version__
from .async_client import AsyncClient
from .callbacks import CallbackEvent
from .client import Client
from .config import EngineConfig
from .engine import acompletion, completion, configure, register_callback
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
    # in-process engine (litellm-style)
    "completion",
    "acompletion",
    "configure",
    "register_callback",
    "EngineConfig",
    "CallbackEvent",
    # hosted-gateway clients
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
