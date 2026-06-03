"""Aggregation layer over :class:`UsageRecord` for the metrics API.

This module computes window totals (including latency percentiles), grouped
breakdowns, and a time-bucketed series, all via SQLAlchemy 2.0 async
``select()`` statements against Postgres.

Design notes / gotchas (see docs/design/04_metrics.md):

* A single async session/connection cannot run concurrent queries, so the
  three SELECTs are awaited **sequentially** here — never ``asyncio.gather``
  them on one session.
* ``cache_hit`` is encoded in ``UsageRecord.status`` (option A: zero
  migration). ``error_rate`` counts :data:`ERROR_STATUSES`; ``cache_hit_rate``
  counts :data:`CACHE_HIT_STATUS`.
* Latency percentiles use ``percentile_cont(q) FILTER (WHERE ...) WITHIN GROUP
  (ORDER BY latency_ms ASC)`` — the ``.filter().within_group()`` builder order
  is the form that compiles cleanly across SQLAlchemy 2.0.x.
* The half-open window ``[from, to)`` and a bare ``created_at`` comparison in
  the WHERE clause keep the composite ``(scope, created_at)`` index usable.
  ``date_trunc`` (3-arg, PG 12+, UTC zone) is only ever applied in SELECT /
  GROUP BY, never in WHERE.
* ``Decimal`` sums are cast to ``float`` in Python; divisions are guarded so a
  zero-request window yields ``0.0`` and percentiles stay ``None``.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import Select, and_, case, func, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.sql.elements import ColumnElement

from app.models.db import UsageRecord
from app.schemas.metrics import (
    Breakdown,
    MetricsResponse,
    TimePoint,
    Totals,
)

logger = logging.getLogger(__name__)

U = UsageRecord

# --- Canonical status predicates (option A: cache encoded in status) ---------
# Rows whose status counts as an error for rate computation.
ERROR_STATUSES: tuple[str, ...] = ("error", "rate_limited", "budget_exceeded")
# Status value marking a successful served-from-cache response.
CACHE_HIT_STATUS: str = "cache_hit"

# Valid ``range`` keyword -> window length.
RANGE_WINDOWS: dict[str, timedelta] = {
    "1h": timedelta(hours=1),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
}
DEFAULT_RANGE: str = "24h"

# Group-by aliases the caller may pass -> the column they map to.
GROUP_BY_COLUMNS: tuple[str, ...] = ("provider", "model", "user", "status")

VALID_GRANULARITIES: tuple[str, ...] = ("minute", "hour", "day")


@dataclass(frozen=True)
class MetricsScope:
    """Identifies the row-scope of a metrics query.

    ``field`` is ``"org_id"`` or ``"project_id"`` (the WHERE column name);
    ``value`` is the matching UUID.
    """

    field: str
    value: uuid.UUID

    @property
    def label(self) -> str:
        """Human-facing scope label: ``"org"`` or ``"project"``."""
        return "org" if self.field == "org_id" else "project"

    @property
    def column(self) -> InstrumentedAttribute:
        """The mapped ``UsageRecord`` column to filter on."""
        return getattr(U, self.field)


def parse_range(
    range: str | None,
    frm: str | None,
    to: str | None,
    *,
    now: datetime,
) -> tuple[datetime, datetime]:
    """Resolve the half-open window ``[from, to)``.

    ``now`` is supplied (UTC, tz-aware) for testability. Either ``range`` (one
    of ``1h``/``24h``/``7d``/``30d``) or an explicit ``frm``/``to`` pair may be
    given; ``range`` defaults to ``24h`` when nothing is supplied.

    Explicit ISO-8601 timestamps must be tz-aware; naive values raise
    ``ValueError``. ``to`` defaults to ``now`` when only ``frm`` is given.
    """
    now = _as_utc(now)

    if frm is not None or to is not None:
        if range is not None:
            raise ValueError("'range' is mutually exclusive with 'from'/'to'")
        if frm is None:
            raise ValueError("'from' is required when 'to' is supplied")
        start = _parse_iso_aware(frm, "from")
        end = _parse_iso_aware(to, "to") if to is not None else now
        if end <= start:
            raise ValueError("'to' must be after 'from'")
        return start, end

    key = range or DEFAULT_RANGE
    window = RANGE_WINDOWS.get(key)
    if window is None:
        valid = ", ".join(RANGE_WINDOWS)
        raise ValueError(f"invalid range {key!r}; expected one of {valid}")
    return now - window, now


def pick_granularity(window: timedelta) -> str:
    """Auto-select a bucket granularity for a window length.

    ``<= 1h`` -> ``minute``; ``<= 24h`` -> ``hour``; larger -> ``day``.
    """
    if window <= timedelta(hours=1):
        return "minute"
    if window <= timedelta(hours=24):
        return "hour"
    return "day"


# --- Reusable aggregate expressions -----------------------------------------
def _error_count() -> ColumnElement[int]:
    """``SUM(CASE WHEN status IN (errors) THEN 1 ELSE 0)``."""
    return func.sum(case((U.status.in_(ERROR_STATUSES), 1), else_=0))


def _cache_hit_count() -> ColumnElement[int]:
    """``SUM(CASE WHEN status = 'cache_hit' THEN 1 ELSE 0)``."""
    return func.sum(case((U.status == CACHE_HIT_STATUS, 1), else_=0))


def _percentile(quantile: float) -> ColumnElement[float | None]:
    """``percentile_cont(q) WITHIN GROUP (ORDER BY latency ASC) FILTER (WHERE not error)``.

    Postgres requires ``WITHIN GROUP`` to precede ``FILTER``; the builder must
    therefore be ``.within_group(...).filter(...)`` (the reverse order compiles
    to a string but is rejected by Postgres at execution time).
    """
    return (
        func.percentile_cont(quantile)
        .within_group(U.latency_ms.asc())
        .filter(~U.status.in_(ERROR_STATUSES))
    )


def _group_expr(group_by: str) -> ColumnElement:
    """Map a ``group_by`` alias to its grouping column expression."""
    if group_by == "user":
        return func.coalesce(U.user_id, "unknown")
    if group_by == "status":
        return U.status
    if group_by == "provider":
        return U.provider
    if group_by == "model":
        return U.model
    raise ValueError(
        f"invalid group_by {group_by!r}; expected one of "
        f"{', '.join(GROUP_BY_COLUMNS)}"
    )


def _bucket_expr(granularity: str) -> ColumnElement:
    """3-arg ``date_trunc(granularity, created_at, 'UTC')`` (PG 12+)."""
    return func.date_trunc(granularity, U.created_at, "UTC")


def _window_clause(
    scope: MetricsScope, frm: datetime, to: datetime
) -> ColumnElement[bool]:
    """Half-open ``[frm, to)`` WHERE clause for the given scope."""
    return and_(
        scope.column == scope.value,
        U.created_at >= frm,
        U.created_at < to,
    )


def build_totals_stmt(
    scope: MetricsScope, frm: datetime, to: datetime
) -> Select:
    """The single-row totals statement (counts, sums, percentiles)."""
    return select(
        func.count(U.id).label("requests"),
        func.coalesce(func.sum(U.prompt_tokens), 0).label("prompt_tokens"),
        func.coalesce(func.sum(U.completion_tokens), 0).label(
            "completion_tokens"
        ),
        func.coalesce(func.sum(U.total_tokens), 0).label("total_tokens"),
        func.coalesce(func.sum(U.cost), 0).label("cost"),
        _error_count().label("errors"),
        _cache_hit_count().label("cache_hits"),
        func.avg(U.latency_ms)
        .filter(~U.status.in_(ERROR_STATUSES))
        .label("avg_latency_ms"),
        _percentile(0.5).label("p50"),
        _percentile(0.95).label("p95"),
        _percentile(0.99).label("p99"),
    ).where(_window_clause(scope, frm, to))


def build_breakdown_stmt(
    scope: MetricsScope, frm: datetime, to: datetime, group_by: str
) -> Select:
    """The grouped-breakdown statement (capped at 100 rows by request count)."""
    key_expr = _group_expr(group_by)
    return (
        select(
            key_expr.label("key"),
            func.count(U.id).label("requests"),
            func.coalesce(func.sum(U.total_tokens), 0).label("total_tokens"),
            func.coalesce(func.sum(U.cost), 0).label("cost"),
            func.avg(U.latency_ms)
            .filter(~U.status.in_(ERROR_STATUSES))
            .label("avg_latency_ms"),
            _error_count().label("errors"),
        )
        .where(_window_clause(scope, frm, to))
        .group_by(key_expr)
        .order_by(func.count(U.id).desc())
        .limit(100)
    )


def build_timeseries_stmt(
    scope: MetricsScope, frm: datetime, to: datetime, granularity: str
) -> Select:
    """The time-bucketed statement (sparse: empty buckets absent)."""
    bucket = _bucket_expr(granularity).label("bucket")
    return (
        select(
            bucket,
            func.count(U.id).label("requests"),
            func.coalesce(func.sum(U.total_tokens), 0).label("total_tokens"),
            func.coalesce(func.sum(U.cost), 0).label("cost"),
            _error_count().label("errors"),
        )
        .where(_window_clause(scope, frm, to))
        .group_by(bucket)
        .order_by(bucket)
    )


def compile_percentile_sql() -> str:
    """Compile the totals statement against the Postgres dialect.

    Exposed so a SQL-compile test can assert the percentile / FILTER ordering
    without a live database.
    """
    scope = MetricsScope(field="project_id", value=uuid.UUID(int=0))
    now = datetime.now(timezone.utc)
    stmt = build_totals_stmt(scope, now - timedelta(hours=1), now)
    return str(
        stmt.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


def _ratio(numerator: int | None, denominator: int) -> float:
    """Guarded ratio: ``0.0`` when the denominator is zero."""
    if not denominator:
        return 0.0
    return float(numerator or 0) / float(denominator)


def _as_utc(value: datetime) -> datetime:
    """Return ``value`` as a tz-aware UTC datetime (raise if naive)."""
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(timezone.utc)


def _parse_iso_aware(raw: str, field: str) -> datetime:
    """Parse an ISO-8601 string, requiring it to be tz-aware (UTC-normalised)."""
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        raise ValueError(f"'{field}' must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _opt_float(value: object) -> float | None:
    """Coerce a possibly-``None`` Decimal/number to ``float`` (or ``None``)."""
    if value is None:
        return None
    return float(value)  # type: ignore[arg-type]


async def get_metrics(
    session: AsyncSession,
    scope: MetricsScope,
    *,
    frm: datetime,
    to: datetime,
    group_by: str | None,
    granularity: str,
) -> MetricsResponse:
    """Aggregate usage for ``scope`` over ``[frm, to)`` and build the response.

    Runs the three SELECTs sequentially on the one session (totals, then the
    optional breakdown, then the timeseries). Set ``group_by`` to one of
    ``provider``/``model``/``user``/``status`` to populate ``breakdown``;
    ``None`` leaves it empty. ``granularity`` is one of
    ``minute``/``hour``/``day``.
    """
    if granularity not in VALID_GRANULARITIES:
        raise ValueError(
            f"invalid granularity {granularity!r}; expected one of "
            f"{', '.join(VALID_GRANULARITIES)}"
        )

    # (1) Totals + latency percentiles.
    totals_row = (
        await session.execute(build_totals_stmt(scope, frm, to))
    ).one()
    requests = int(totals_row.requests or 0)
    totals = Totals(
        requests=requests,
        prompt_tokens=int(totals_row.prompt_tokens or 0),
        completion_tokens=int(totals_row.completion_tokens or 0),
        total_tokens=int(totals_row.total_tokens or 0),
        cost_usd=float(totals_row.cost or 0),
        error_rate=_ratio(totals_row.errors, requests),
        cache_hit_rate=_ratio(totals_row.cache_hits, requests),
        avg_latency_ms=float(totals_row.avg_latency_ms or 0),
        p50_latency_ms=_opt_float(totals_row.p50),
        p95_latency_ms=_opt_float(totals_row.p95),
        p99_latency_ms=_opt_float(totals_row.p99),
    )

    # (2) Breakdown (only when requested).
    breakdown: list[Breakdown] = []
    if group_by is not None:
        rows = (
            await session.execute(
                build_breakdown_stmt(scope, frm, to, group_by)
            )
        ).all()
        for row in rows:
            row_requests = int(row.requests or 0)
            breakdown.append(
                Breakdown(
                    key="unknown" if row.key is None else str(row.key),
                    requests=row_requests,
                    total_tokens=int(row.total_tokens or 0),
                    cost_usd=float(row.cost or 0),
                    avg_latency_ms=float(row.avg_latency_ms or 0),
                    error_rate=_ratio(row.errors, row_requests),
                )
            )

    # (3) Timeseries.
    ts_rows = (
        await session.execute(
            build_timeseries_stmt(scope, frm, to, granularity)
        )
    ).all()
    timeseries: list[TimePoint] = []
    for row in ts_rows:
        row_requests = int(row.requests or 0)
        timeseries.append(
            TimePoint(
                bucket=row.bucket,
                requests=row_requests,
                total_tokens=int(row.total_tokens or 0),
                cost_usd=float(row.cost or 0),
                error_rate=_ratio(row.errors, row_requests),
            )
        )

    return MetricsResponse(
        scope=scope.label,
        scope_id=scope.value,
        range_from=frm,
        range_to=to,
        group_by=group_by,
        granularity=granularity,
        totals=totals,
        breakdown=breakdown,
        timeseries=timeseries,
    )
