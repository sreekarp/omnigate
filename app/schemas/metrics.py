"""Pydantic v2 response models for the metrics API.

These models describe aggregated usage analytics computed over
:class:`app.models.db.UsageRecord` rows. They are produced by
:mod:`app.services.metrics` and serialised by the metrics routers.

All cost fields are plain ``float`` (USD) to match the existing
``ChatResponse.cost_usd`` convention even though the underlying column is
``Numeric``; the service casts ``Decimal`` -> ``float`` when building these.
Latency percentiles may be ``None`` when the window contains no rows.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class Totals(BaseModel):
    """Window-wide aggregate totals."""

    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    p99_latency_ms: float | None = None


class Breakdown(BaseModel):
    """One grouped row (by provider/model/user/status)."""

    key: str
    requests: int
    total_tokens: int
    cost_usd: float
    avg_latency_ms: float
    error_rate: float


class TimePoint(BaseModel):
    """One time-bucketed point in the series (sparse: empty buckets absent)."""

    bucket: datetime
    requests: int
    total_tokens: int
    cost_usd: float
    error_rate: float


class MetricsResponse(BaseModel):
    """Full metrics payload for a project- or org-scoped query."""

    scope: str
    scope_id: str | UUID
    range_from: datetime
    range_to: datetime
    group_by: str | None = None
    granularity: str
    totals: Totals = Field(default_factory=Totals)
    breakdown: list[Breakdown] = Field(default_factory=list)
    timeseries: list[TimePoint] = Field(default_factory=list)
