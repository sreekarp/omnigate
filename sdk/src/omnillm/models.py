"""Pydantic v2 models mirroring the OmniLLM wire schema.

These mirror ``app/schemas/chat.py`` and ``app/schemas/account.py`` on the
server side but import nothing from the server package, so the SDK is fully
standalone. Read models use ``extra="ignore"`` for forward compatibility with
new server fields; request models stay validated.

Note on UUIDs: the server stores ``org_id``/``project_id`` as UUIDs but
serialises them to JSON strings, so the SDK types them as ``str``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    """A single chat message."""

    role: Role
    content: str


class ChatRequest(BaseModel):
    """Outbound chat request body (validated on the client before sending)."""

    model: str
    messages: list[Message] = Field(min_length=1)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    stream: bool = False


class Usage(BaseModel):
    """Token usage. Zeros for streamed responses (server records no usage)."""

    model_config = ConfigDict(extra="ignore")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    """Non-streaming chat completion response."""

    model_config = ConfigDict(extra="ignore")

    id: str
    provider: str
    model: str
    content: str
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = 0.0
    # Additive server metadata (tolerated, may be absent on older servers).
    finish_reason: Optional[str] = None
    cached: bool = False
    fallback_used: bool = False
    latency_ms: int = 0


class StreamChunk(BaseModel):
    """A piece of streamed text plus provenance.

    Streaming is plain text (not SSE) and carries no usage/cost; only ``text``
    and the originating ``request_id`` (from the ``x-request-id`` header) are
    available.
    """

    text: str = ""
    request_id: Optional[str] = None


class SignupResponse(BaseModel):
    """Result of ``POST /v1/signup``. ``api_key`` is shown only once."""

    model_config = ConfigDict(extra="ignore")

    org_id: str
    project_id: str
    email: str
    api_key: str
    message: str = ""


class MeResponse(BaseModel):
    """Result of ``GET /v1/me``."""

    model_config = ConfigDict(extra="ignore")

    project_id: str
    org_id: str
    project_name: str
    key_prefix: str
    rate_limit_per_min: int
    configured_providers: list[str] = Field(default_factory=list)


class ApiKeyCreated(BaseModel):
    """Result of ``POST /v1/keys/api`` — a newly minted gateway API key.

    The plaintext ``api_key`` is returned only here and is never recoverable
    afterwards; the gateway stores only its hash.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    key_prefix: str
    created_at: datetime
    last_used_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    api_key: str


class ModelPricing(BaseModel):
    """Per-1k-token USD pricing for a model card."""

    model_config = ConfigDict(extra="ignore")

    input_per_1k_usd: float = 0.0
    output_per_1k_usd: float = 0.0


class ModelInfo(BaseModel):
    """An entry from ``GET /v1/models`` — an OpenAI-style model card.

    The server returns ``{"object": "list", "data": [card, ...]}`` where each
    card is ``{id, object, created, owned_by, provider, pricing}``. ``pricing``
    is ``None`` for models without a known price.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = ""
    provider: str = ""
    pricing: Optional[ModelPricing] = None


class MetricsTotals(BaseModel):
    """Window-wide aggregate totals from ``GET /v1/metrics``."""

    model_config = ConfigDict(extra="ignore")

    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    p99_latency_ms: Optional[float] = None


class MetricsBreakdown(BaseModel):
    """One grouped row (by provider/model/user/status)."""

    model_config = ConfigDict(extra="ignore")

    key: str
    requests: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0


class MetricsTimePoint(BaseModel):
    """One time-bucketed point in the metrics series."""

    model_config = ConfigDict(extra="ignore")

    bucket: datetime
    requests: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    error_rate: float = 0.0


class MetricsResponse(BaseModel):
    """Result of ``GET /v1/metrics`` — the rich project-scoped analytics payload.

    Carries window-wide ``totals`` (with latency percentiles + cache-hit rate),
    an optional grouped ``breakdown``, and a bucketed ``timeseries``.
    """

    model_config = ConfigDict(extra="ignore")

    scope: str
    scope_id: str
    range_from: datetime
    range_to: datetime
    group_by: Optional[str] = None
    granularity: str
    totals: MetricsTotals = Field(default_factory=MetricsTotals)
    breakdown: list[MetricsBreakdown] = Field(default_factory=list)
    timeseries: list[MetricsTimePoint] = Field(default_factory=list)


# --- coercion helpers -------------------------------------------------------

MessagesInput = "list[Message | dict] | str | Message | dict"


def coerce_messages(messages: object) -> list[Message]:
    """Coerce flexible ``messages`` input into a ``list[Message]``.

    Accepts:
      * a bare ``str`` -> ``[Message(role="user", content=str)]``
      * a single ``Message`` or ``dict``
      * a list of ``Message`` and/or ``dict`` entries
    """
    if isinstance(messages, str):
        return [Message(role="user", content=messages)]
    if isinstance(messages, Message):
        return [messages]
    if isinstance(messages, dict):
        return [Message(**messages)]
    if isinstance(messages, (list, tuple)):
        out: list[Message] = []
        for item in messages:
            if isinstance(item, Message):
                out.append(item)
            elif isinstance(item, dict):
                out.append(Message(**item))
            else:
                raise TypeError(
                    f"Unsupported message item type: {type(item).__name__!r}"
                )
        return out
    raise TypeError(f"Unsupported messages type: {type(messages).__name__!r}")
