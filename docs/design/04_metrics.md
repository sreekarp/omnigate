This is a design-only subtask (the workflow tracks it under task #1, which stays in_progress for the broader design effort). I'll return the spec directly rather than mutating the shared task list, since my job is just to produce this one subsystem's spec. Here is the implementation spec.

---

# SUBSYSTEM SPEC: Org/Project Metrics JSON API + SQL

## 1. Endpoints

```
GET /v1/metrics                              # auth: x-api-key → caller's project (scoped to ctx.project.id)
GET /v1/admin/orgs/{org_id}/metrics          # auth: x-admin-key (require_admin), scoped to org_id
```

Mount a new router `app/routers/metrics.py` (`APIRouter(prefix="/v1", tags=["metrics"])`) for the project-scoped route, and add the admin variant to the **existing** `app/routers/admin.py` (it already has `prefix="/v1/admin"` + `require_admin`). Register the new router in `app/main.py` via `app.include_router(metrics.router)`.

Both routes share one service function; they differ only in the WHERE-scope column (`project_id` vs `org_id`), both of which have a covering composite index `(…, created_at)`.

### Query params (shared `MetricsQuery` dependency, parsed via `Annotated[..., Query()]`)

| param | type | default | notes |
|---|---|---|---|
| `range` | `str \| None` | `"24h"` | one of `1h,24h,7d,30d`. Mutually exclusive with `from_`/`to`. |
| `from_` (alias `from`) | `datetime \| None` | None | ISO 8601; **require tz-aware** (reject naive with 422). |
| `to` | `datetime \| None` | None | ISO 8601; defaults to `now(UTC)` if only `from` given. |
| `group_by` | `Literal["provider","model","user","day","status"] \| None` | None | drives the `breakdowns` array. |
| `granularity` | `Literal["minute","hour","day"] \| None` | auto | drives `timeseries` bucket. Auto: `1h→minute`, `24h→hour`, `7d/30d→day`. |

**Gotchas**
- `from` is a Python reserved word → declare `from_: datetime | None = Query(default=None, alias="from")`.
- Resolve `(start, end)` once in the dependency; `range` and `from/to` are mutually exclusive → 422 if both supplied. `range` values map to `end = now(UTC)`, `start = end - timedelta(...)`.
- All timestamps UTC, tz-aware. `UsageRecord.created_at` is `DateTime(timezone=True)`; compare against tz-aware `datetime` so asyncpg binds `timestamptz` correctly. `date_trunc` must run in UTC — pass the literal `'UTC'` form: `func.date_trunc("hour", UsageRecord.created_at)` operates in the session TZ; to be deterministic use `func.date_trunc("hour", func.timezone("UTC", UsageRecord.created_at))` **only if** the server TZ is not UTC. Given Postgres default + tz-aware storage, `date_trunc(text, timestamptz)` truncates in the connection `TimeZone` setting. **Set `start`/`end` to UTC and add `SET TIME ZONE 'UTC'`** at session level, OR truncate via `func.date_trunc("hour", UsageRecord.created_at, "UTC")` (3-arg form, PG 16+) which truncates a `timestamptz` in an explicit zone and returns `timestamptz`. Prefer the 3-arg form and document PG ≥ 12 requirement (3-arg `date_trunc(text, timestamptz, text)` is PG 12+).

---

## 2. `cache_hit_rate` — REQUIRED SCHEMA NOTE (read carefully)

The current `UsageRecord` has **no cache column**. `cache_hit_rate` cannot be computed today. Two options; spec recommends **(A)** for backward compatibility:

**(A) Encode in existing `status` (zero migration).** The cache subsystem (workflow task #4) records cache hits as `status="cache_hit"` (a *successful* served-from-cache response). Then:
- `cache_hit_rate = count(status='cache_hit') / count(all non-error rows)`.
- `error_rate` counts `status IN ('error','rate_limited','budget_exceeded')`.
- Until task #4 lands, `cache_hit_rate` is always `0.0` (no rows have that status) — correct and non-breaking.

**(B) Add nullable column** `cache_hit: Mapped[bool] = mapped_column(Boolean, server_default=text("false"), nullable=False)` via Alembic. Cleaner long-term but touches the hot-path INSERT and DB rows. **If chosen**, default `false` keeps existing rows valid; `cache_hit_rate = avg(cache_hit::int)`.

Spec proceeds with **(A)**. Define one canonical predicate set in the service module so every endpoint agrees:

```python
ERROR_STATUSES = ("error", "rate_limited", "budget_exceeded")
CACHE_HIT_STATUS = "cache_hit"
# "billable/attempted" rows for rates = all rows in window
```

---

## 3. Pydantic v2 response models (`app/schemas/metrics.py`)

```python
from datetime import datetime
from pydantic import BaseModel, Field

class LatencyPercentiles(BaseModel):
    p50: float | None = None      # ms; None when no rows
    p95: float | None = None
    p99: float | None = None

class MetricsTotals(BaseModel):
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    error_rate: float = 0.0       # 0.0–1.0
    cache_hit_rate: float = 0.0   # 0.0–1.0
    latency_ms: LatencyPercentiles = Field(default_factory=LatencyPercentiles)

class MetricsBreakdown(BaseModel):
    key: str                      # provider/model/user/day(ISO date)/status; user null -> "unknown"
    requests: int
    total_tokens: int
    cost_usd: float
    avg_latency_ms: float | None
    error_rate: float

class TimeseriesPoint(BaseModel):
    bucket: datetime              # tz-aware UTC, start of bucket
    requests: int
    total_tokens: int
    cost_usd: float
    error_rate: float

class MetricsWindow(BaseModel):
    start: datetime
    end: datetime
    granularity: str

class MetricsResponse(BaseModel):
    scope: str                    # "project" | "org"
    scope_id: str                 # UUID as str
    window: MetricsWindow
    totals: MetricsTotals
    group_by: str | None = None
    breakdowns: list[MetricsBreakdown] = Field(default_factory=list)
    timeseries: list[TimeseriesPoint] = Field(default_factory=list)
```

**Gotchas**
- `cost` is `Numeric(12,6)` → comes back as `Decimal`. Cast to `float` in the service when building the model (or annotate as `Decimal`; floats match the existing `ChatResponse.cost_usd: float` convention — use `float`).
- `percentile_cont` returns `double precision` → already a Python `float`. Empty window → `None` (no `NaN`). Coalesce sums but **not** percentiles.

---

## 4. Service layer (`app/services/metrics.py`)

### Signatures
```python
from dataclasses import dataclass
from datetime import datetime
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.metrics import MetricsResponse

@dataclass(frozen=True)
class MetricsScope:
    column: "InstrumentedAttribute[uuid.UUID]"   # UsageRecord.project_id or .org_id
    value: uuid.UUID
    label: str                                    # "project" | "org"

async def get_metrics(
    session: AsyncSession,
    *,
    scope: MetricsScope,
    start: datetime,
    end: datetime,
    group_by: str | None,
    granularity: str,
) -> MetricsResponse: ...
```
`get_metrics` runs three independent SELECTs (totals, breakdowns, timeseries) — they can be `await`ed sequentially on one session (asyncpg/SQLAlchemy async **does not allow concurrent ops on one session/connection**; do **not** `asyncio.gather` them on the same session). Then assembles `MetricsResponse`.

Routers build the `MetricsScope`:
- `/v1/metrics`: `MetricsScope(UsageRecord.project_id, ctx.project.id, "project")` via `Depends(get_auth_context)`.
- admin: `MetricsScope(UsageRecord.org_id, org_id, "org")`; first verify org exists → 404 if `session.get(Organisation, org_id)` is None.

---

## 5. Exact SQLAlchemy 2.0 async `select()` statements

Common imports:
```python
from sqlalchemy import select, func, case, and_, cast, Integer, Float
from app.models.db import UsageRecord
U = UsageRecord
```
Reusable filtered-count expressions (use `case`, since not all backends support FILTER and the within_group+filter combo is broken — see gotcha §6):
```python
window = and_(scope.column == scope.value,
              U.created_at >= start, U.created_at < end)   # half-open [start, end)

error_count = func.sum(case((U.status.in_(ERROR_STATUSES), 1), else_=0))
cache_count = func.sum(case((U.status == CACHE_HIT_STATUS, 1), else_=0))
req_count   = func.count(U.id)
```

### 5a. Totals + latency percentiles (one row)
```python
totals_stmt = (
    select(
        func.count(U.id).label("requests"),
        func.coalesce(func.sum(U.prompt_tokens), 0).label("prompt_tokens"),
        func.coalesce(func.sum(U.completion_tokens), 0).label("completion_tokens"),
        func.coalesce(func.sum(U.total_tokens), 0).label("total_tokens"),
        func.coalesce(func.sum(U.cost), 0).label("cost"),
        error_count.label("errors"),
        cache_count.label("cache_hits"),
        # percentiles over SUCCESSFUL latencies only -> filter BEFORE within_group:
        func.percentile_cont(0.5)
            .within_group(U.latency_ms.asc())
            .filter(~U.status.in_(ERROR_STATUSES)).label("p50"),
        func.percentile_cont(0.95)
            .within_group(U.latency_ms.asc())
            .filter(~U.status.in_(ERROR_STATUSES)).label("p95"),
        func.percentile_cont(0.99)
            .within_group(U.latency_ms.asc())
            .filter(~U.status.in_(ERROR_STATUSES)).label("p99"),
    )
    .where(window)
)
row = (await session.execute(totals_stmt)).one()
```
`error_rate = row.errors / row.requests if row.requests else 0.0`; `cache_hit_rate = row.cache_hits / row.requests if row.requests else 0.0`.

> NOTE on percentiles: if you want percentiles over **all** rows (incl. errors), drop the `.filter(...)`. Including the filter is recommended so a flood of fast error responses doesn't deflate p95.

### 5b. Breakdowns (grouped)
`group_by` → grouping expression map. For `user`, coalesce null → `'unknown'`; for `day`, truncate.
```python
GROUP_EXPR = {
    "provider": U.provider,
    "model":    U.model,
    "user":     func.coalesce(U.user_id, "unknown"),
    "status":   U.status,
    "day":      func.date_trunc("day", U.created_at, "UTC"),  # 3-arg, PG12+, returns timestamptz
}
key_expr = GROUP_EXPR[group_by]

breakdown_stmt = (
    select(
        key_expr.label("key"),
        func.count(U.id).label("requests"),
        func.coalesce(func.sum(U.total_tokens), 0).label("total_tokens"),
        func.coalesce(func.sum(U.cost), 0).label("cost"),
        func.avg(U.latency_ms).filter(~U.status.in_(ERROR_STATUSES)).label("avg_latency_ms"),
        error_count.label("errors"),
    )
    .where(window)
    .group_by(key_expr)
    .order_by(func.count(U.id).desc())
    .limit(100)   # cap cardinality (esp. group_by=user/model)
)
rows = (await session.execute(breakdown_stmt)).all()
# key serialization: if group_by == "day", key = row.key.date().isoformat(); else str(row.key)
```
Only run if `group_by is not None`; otherwise `breakdowns=[]`.

### 5c. Timeseries (bucketed)
```python
bucket = func.date_trunc(granularity, U.created_at, "UTC").label("bucket")  # 'minute'|'hour'|'day'
ts_stmt = (
    select(
        bucket,
        func.count(U.id).label("requests"),
        func.coalesce(func.sum(U.total_tokens), 0).label("total_tokens"),
        func.coalesce(func.sum(U.cost), 0).label("cost"),
        error_count.label("errors"),
    )
    .where(window)
    .group_by(bucket)
    .order_by(bucket)
)
ts_rows = (await session.execute(ts_stmt)).all()
```
Buckets with zero rows are **absent** (sparse series). If a dense series is needed, left-join against `generate_series(start, end, interval)` — document as optional v2; keep sparse for now (smaller, backward-compatible payloads).

---

## 6. Critical gotchas (verified)

1. **`within_group` + `filter` ordering (SQLAlchemy bug #11423, fixed/closed):** `func.percentile_cont(0.95).within_group(col).filter(pred)` historically raised `AttributeError: 'WithinGroup' object has no attribute 'filter'`. **Correct, stable order is `.within_group(...).filter(...)`** in current 2.0 (the fix added `.filter()` to `WithinGroup`), but to be safe across 2.0.x, the proven-working form is to apply `.filter()` on the **function element first, then `.within_group()`**:
   `func.percentile_cont(0.95).filter(pred).within_group(col.asc())`.
   **Spec mandates `.filter().within_group()`** (works on all 2.0.x; emits `percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) FILTER (WHERE …)`). Add a unit test asserting the compiled SQL string so a SQLAlchemy upgrade can't silently break it.
2. **No concurrent queries on one async session** — run the three SELECTs sequentially (do not `gather`). If you must parallelize, open separate sessions.
3. **`date_trunc` 3-arg form** (`date_trunc(field, timestamptz, zone)`) requires **PostgreSQL 12+** and returns `timestamptz`; keeps bucket boundaries deterministic in UTC regardless of server `TimeZone`. Document the version floor.
4. **Half-open interval `[start, end)`** to avoid double-counting the boundary across adjacent buckets/windows.
5. **Index usage:** `where(scope.column == value, created_at >= start, created_at < end)` matches `ix_usage_project_created (project_id, created_at)` / `ix_usage_org_created (org_id, created_at)` exactly — leading-column equality + range on the second. Don't wrap `created_at` in a function *in the WHERE* (only in SELECT/GROUP BY) or you lose the index.
6. **Decimal → float:** `func.sum(U.cost)` yields `Decimal`; cast in Python (`float(row.cost)`) when populating models. `percentile_cont`/`avg(latency_ms)` yield `float`/`Decimal` respectively — `avg` returns `Decimal`, coerce to `float`.
7. **Empty window:** `func.count` → 0, sums coalesced → 0, percentiles → `None`. Guard all divisions with `if requests else 0.0`.
8. **`group_by=user` cardinality / PII:** cap with `.limit(100)` ordered by requests desc; `user_id` is free-form header text — return as-is but never trust for SQL (it's parameterized, so safe).
9. **Backward compat:** purely additive — new routes, new schema/service modules, no change to `/v1/chat`, `record_usage`, or DB rows (under option A). Existing dashboard SQL untouched.

---

## 7. Test plan (pytest + httpx `ASGITransport`, in-memory/async; DB via test Postgres or SQLite-incompatible → use a Postgres test container or the project's existing async test session)

Percentile/`date_trunc`/`FILTER` are Postgres-specific → tests that execute SQL need **Postgres** (not SQLite). Split into compile-level (no DB) and integration (DB) tests.

**A. SQL-compile tests (no DB, fast):**
- `test_percentile_filter_compiles`: compile `totals_stmt` with `dialect=postgresql.dialect()`, assert string contains `percentile_cont(0.5) WITHIN GROUP (ORDER BY usage_records.latency_ms ASC) FILTER (WHERE ...)` — locks the #11423-safe ordering.
- `test_date_trunc_three_arg`: assert compiled timeseries SQL contains `date_trunc('hour', usage_records.created_at, 'UTC')`.
- `test_breakdown_user_coalesce`: assert `coalesce(usage_records.user_id, 'unknown')` present for `group_by=user`.

**B. Service integration tests (Postgres):** seed `UsageRecord` rows across providers/models/users/statuses/timestamps, then:
- `test_totals_counts_and_cost`: known rows → exact `requests`, token sums, `cost_usd`, `error_rate`, `cache_hit_rate`.
- `test_percentiles`: insert latencies `[10,20,30,40,100]` (status ok) + a couple error rows with huge latency → assert p50≈30 / error-rows excluded.
- `test_breakdown_by_provider` / `_by_model` / `_by_user` (null→"unknown") / `_by_status` / `_by_day`: keys + per-key requests/tokens/cost/error_rate.
- `test_timeseries_hourly`: rows in two hour buckets → two ordered `TimeseriesPoint`s; empty hour absent.
- `test_empty_window`: no rows → totals zeros, percentiles `None`, empty breakdowns/timeseries.
- `test_half_open_boundary`: a row exactly at `end` is excluded; at `start` included.

**C. Endpoint/auth tests (httpx `ASGITransport(app=app)`):**
- `test_project_metrics_requires_api_key`: 401 without `x-api-key`; 200 with valid; **scope isolation** — project A's key never sees project B's rows.
- `test_admin_org_metrics_requires_admin`: 401 w/o `x-admin-key`; 404 for unknown `org_id`; 200 aggregates across the org's projects.
- `test_range_and_from_to_mutually_exclusive`: supplying both → 422.
- `test_naive_datetime_rejected`: `from=2026-06-01T00:00:00` (no tz) → 422.
- `test_default_range_24h_hour_granularity`: no params → window 24h, `granularity="hour"`.

**Mocking:** No provider calls involved (pure DB read) → no `httpx.MockTransport` needed here; reuse the existing async session fixture.

---

## 8. Files to add / touch
- **add** `app/schemas/metrics.py` (models in §3)
- **add** `app/services/metrics.py` (`MetricsScope`, `get_metrics`, predicate constants, statement builders in §5)
- **add** `app/routers/metrics.py` (`GET /v1/metrics`, `Depends(get_auth_context)`)
- **edit** `app/routers/admin.py` (add `GET /orgs/{org_id}/metrics`, reuse `require_admin` + `get_metrics`)
- **edit** `app/main.py` (`app.include_router(metrics.router)`)
- **(only if option B)** Alembic migration adding `usage_records.cache_hit BOOLEAN NOT NULL DEFAULT false` + model field.

Sources:
- [SQLAlchemy 2.0 — SQL and Generic Functions (`percentile_cont`/`within_group`/`FunctionElement.filter`)](https://docs.sqlalchemy.org/en/20/core/functions.html)
- [SQLAlchemy issue #11423 — combining `within_group` and `filter`](https://github.com/sqlalchemy/sqlalchemy/issues/11423)