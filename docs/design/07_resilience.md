I have everything needed. Both wire details are confirmed: OpenAI's `stream_options:{include_usage:true}` yields a final chunk with full `usage` and empty `choices: []`; Anthropic's `message_start` carries `input_tokens` and `message_delta` carries cumulative `output_tokens`. These matter for the cache (caching streamed responses is out of scope, but usage capture informs why we don't cache them).

Now I'll write the implementation spec.

---

# Implementation Spec: Resilience + Response Cache + Multi-Key Auth + Monthly Budgets

Scope: `app/utils/resilience.py` (new), `app/services/cache.py` (new), `app/services/circuit_breaker.py` (new), `app/services/api_keys.py` (new), `app/routers/keys.py` (new), edits to `chat.py`, `auth.py`, `budget.py`, `registry.py`, `config.py`, `schemas/chat.py`, `models/db.py`, and migration `0003`. All async, type-hinted, Pydantic v2, `logging` only. No new runtime deps (stdlib `random`, `hashlib`, `asyncio`, `time`; existing `redis.asyncio`, `httpx`, SQLAlchemy).

---

## 1. Config knobs (`app/config.py`)

Add to `Settings` (env names are upper-case of the field):

```python
# --- Resilience ---
retry_max_attempts: int = 3            # total tries incl. first; 1 disables retry
retry_base_delay_seconds: float = 0.25 # backoff base
retry_max_delay_seconds: float = 8.0   # per-attempt cap
retry_jitter_seconds: float = 0.25     # +U(0, jitter) added each sleep

circuit_breaker_enabled: bool = True
circuit_breaker_fail_threshold: int = 5     # consecutive fails -> OPEN
circuit_breaker_cooldown_seconds: float = 30.0  # OPEN -> HALF_OPEN after this
circuit_breaker_backend: str = "memory"     # "memory" | "redis"

# --- Response cache ---
response_cache_enabled: bool = False   # opt-in globally; per-request flag also required
response_cache_ttl_seconds: int = 300
response_cache_prefix: str = "respcache:"
```

Gotcha: `get_settings()` is `@lru_cache`d and `redis_client`/`engine` capture `settings = get_settings()` at import time. Tests must set env before first import (existing tests already do this). Don't read settings at module top in new files — call `get_settings()` inside functions or accept values as args, so MockTransport tests can override.

---

## 2. Retry helper (`app/utils/resilience.py`)

Hand-rolled exponential backoff + full jitter. Retries only **transient** `ProviderError`s.

```python
import asyncio, random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar
from app.providers.base import ProviderError

T = TypeVar("T")

# Transient = worth retrying. 408 timeout, 409 conflict-ish no; 425,429,5xx yes.
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

def is_retryable(exc: ProviderError) -> bool:
    return exc.status_code in _RETRYABLE_STATUS

@dataclass(slots=True)
class RetryPolicy:
    max_attempts: int
    base_delay: float
    max_delay: float
    jitter: float

    @classmethod
    def from_settings(cls, s) -> "RetryPolicy":
        return cls(s.retry_max_attempts, s.retry_base_delay_seconds,
                   s.retry_max_delay_seconds, s.retry_jitter_seconds)

    def delay_for(self, attempt: int) -> float:  # attempt is 1-based
        backoff = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        return backoff + random.uniform(0.0, self.jitter)

async def retry_async(
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    retryable: Callable[[ProviderError], bool] = is_retryable,
    on_retry: Callable[[int, ProviderError], None] | None = None,
) -> T:
    last: ProviderError
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await fn()
        except ProviderError as exc:
            last = exc
            if attempt >= policy.max_attempts or not retryable(exc):
                raise
            if on_retry:
                on_retry(attempt, exc)
            await asyncio.sleep(policy.delay_for(attempt))
    raise last  # unreachable; satisfies type checker
```

Gotchas:
- **Honor `Retry-After`**: providers return it on 429/503. The current adapters discard headers (they raise `ProviderError(text, status_code)`). Extend `ProviderError` with an optional `retry_after: float | None = None` and have adapters parse `resp.headers.get("retry-after")` (seconds, integer; ignore HTTP-date form). In `delay_for`, prefer `max(retry_after, computed)` when set. This requires threading `retry_after` through — keep it optional and default `None` so existing raises still compile.
- **Idempotency**: chat completions are safe to retry (no side effects on provider; gateway only writes a usage row on the *final* outcome). Retrying is fine. Do **not** retry inside the streaming generator after bytes have been yielded to the client (you can't un-send). Retry streaming only on the *connect/first-status* failure (see §6).
- Retry is **per-provider-attempt**, nested *inside* the fallback loop (§5): each fallback target gets its own retry budget.

---

## 3. Circuit breaker (`app/services/circuit_breaker.py`)

Keyed per **`provider:project_id`** (a project's bad BYOK key must not trip the breaker for other projects). Three states: CLOSED, OPEN, HALF_OPEN.

```python
import time
from dataclasses import dataclass, field
from enum import Enum

class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitOpenError(Exception):
    """Raised when the breaker is OPEN; maps to HTTP 503."""

@dataclass
class _Circuit:
    fail_threshold: int
    cooldown: float
    consecutive_failures: int = 0
    opened_at: float | None = None

    def state(self, now: float) -> CircuitState:
        if self.opened_at is None:
            return CircuitState.CLOSED
        if now - self.opened_at >= self.cooldown:
            return CircuitState.HALF_OPEN
        return CircuitState.OPEN

class CircuitBreaker:
    """In-memory breaker. One instance per process (module singleton)."""
    def __init__(self, fail_threshold: int, cooldown: float) -> None:
        self._threshold = fail_threshold
        self._cooldown = cooldown
        self._circuits: dict[str, _Circuit] = {}

    def _get(self, key: str) -> _Circuit:
        c = self._circuits.get(key)
        if c is None:
            c = _Circuit(self._threshold, self._cooldown)
            self._circuits[key] = c
        return c

    def allow(self, key: str) -> bool:
        """Return False if OPEN. HALF_OPEN allows a single trial through."""
        return self._get(key).state(time.monotonic()) is not CircuitState.OPEN

    def record_success(self, key: str) -> None:
        c = self._get(key)
        c.consecutive_failures = 0
        c.opened_at = None  # close on any success (incl. half-open trial)

    def record_failure(self, key: str) -> None:
        c = self._get(key)
        now = time.monotonic()
        if c.state(now) is CircuitState.HALF_OPEN:
            c.opened_at = now            # trial failed -> re-open, reset timer
            return
        c.consecutive_failures += 1
        if c.consecutive_failures >= self._threshold:
            c.opened_at = now
```

Wire-up: module-level `_breaker` singleton built lazily from settings; `get_circuit_breaker()` accessor (so tests can reset). Only count **provider/infra** failures (`is_retryable(exc)` is True, or any 5xx/timeout) toward the breaker — do **not** trip on 400/401 (bad request, bad BYOK key) since those are caller errors, not provider outages. Use `time.monotonic()` not wall clock.

Optional Redis backend (`circuit_breaker_backend="redis"`): store `consecutive_failures` as an INCR'd key `cb:{provider}:{project}:fails` and `opened_at` as a key with `cooldown` TTL (`cb:{...}:open`). `allow()` = `not EXISTS open-key`. `record_failure` = INCR fails; if `>= threshold`, `SET open-key 1 EX cooldown`, `DEL fails`. `record_success` = `DEL fails`, `DEL open-key`. Note this can't perfectly model HALF_OPEN single-trial across processes — accept that multiple workers may each send one trial; that's acceptable. Keep `allow()`/`record_*` as `async` in the redis impl; define a tiny `Protocol` so both backends share the call sites. For the in-memory impl, wrap the sync methods in `async def` shims to keep call sites uniform.

---

## 4. Resilient routing core (`app/services/routing.py`, new)

Single function the router calls. Composes breaker + retry + fallback + cache around `provider.chat`. Keeps `chat.py` thin.

```python
@dataclass(slots=True)
class ChatOutcome:
    response: ChatResponse
    provider_name: str
    model_used: str
    cached: bool
    attempted: list[str]  # models tried, in order

async def execute_chat(
    session, *, request: ChatRequest, ctx, request_id: str,
) -> ChatOutcome: ...
```

Algorithm (non-streaming):
1. Build the ordered model list: `[request.model, *request.fallback_models]` (dedupe, preserve order).
2. **Cache lookup** (only if eligible — see §7): on hit, return `ChatOutcome(cached=True, attempted=[request.model])` immediately. Cache key uses the *originally requested* model, not the fallback that produced it (so a later identical request short-circuits before any provider call).
3. For each candidate model:
   - `provider = get_provider_for_model(model)`; resolve BYOK key (`get_provider_key`). Missing key on the primary → 400 (unchanged behavior); missing key on a *fallback* → skip to next candidate, log, don't 400.
   - `cb_key = f"{provider.name}:{ctx.project.id}"`. If `breaker.allow(cb_key)` is False → record this candidate as "skipped (circuit open)", continue to next.
   - Run `retry_async(lambda: provider.chat(request_for(model), key), policy, on_retry=...)`.
     - Success → `breaker.record_success`, optionally `cache_set`, return outcome.
     - `ProviderError` → if transient/5xx `breaker.record_failure`; record the error; continue to next candidate.
4. All candidates exhausted → raise the **last** `ProviderError` (or a synthesized 503 if all were circuit-open / skipped).

`request_for(model)` is a shallow `request.model_copy(update={"model": model})` so each attempt sends the right model string.

Gotcha: usage recording stays in `chat.py` (single source of truth), but now records `model=outcome.model_used` and `provider=outcome.provider_name`. For a **cached** hit, record a usage row with `status="ok"`, `cost=0`, `latency_ms≈0`, tokens from the cached payload (so analytics still attribute tokens but you don't double-bill) — document this choice; alternatively record `status="cache_hit"`. Recommend `status="ok"` + `cost=0` to keep dashboards/back-compat simple, and surface `cached` via response header/body only.

---

## 5. Fallback routing (schema + behavior)

`app/schemas/chat.py` — additive, back-compat (default empty list):

```python
class ChatRequest(BaseModel):
    ...
    fallback_models: list[str] = Field(default_factory=list, max_length=5)
    # opt-in cache override; None = follow global setting
    cache: bool | None = None
```

`ChatResponse` — additive fields with safe defaults (existing rows/clients unaffected):

```python
class ChatResponse(BaseModel):
    ...
    cached: bool = False
    fallback_used: bool = False   # True if model_used != originally requested model
```

Response headers set in `chat.py` (and exposed via `Access-Control-Expose-Headers` if CORS matters): `x-llmgw-model-used`, `x-llmgw-provider`, `x-llmgw-cached: true|false`, `x-llmgw-attempts: <n>`.

Gotchas:
- Validate fallback model strings resolve to a known provider lazily (inside the loop), not in the schema — an unknown fallback model should just be skipped/logged, never 500 the whole request.
- A fallback may target a **different provider** with a **different BYOK key**; that's the point. Each provider needs its own stored key.
- Cap `fallback_models` length (`max_length=5`) to bound worst-case latency = sum of per-model `retry_max_attempts` timeouts.

---

## 6. Streaming (`chat.py::_stream`)

Do **not** cache streams. Retry/fallback for streaming is limited: the adapter's `stream()` raises `ProviderError` on the initial non-2xx **before** yielding any bytes (see `openai.py`/`anthropic.py` lines that `aread()` the error body inside the `async with client.stream` block). You may safely retry/fallback **only** while no bytes have been yielded:

- Wrap stream startup: attempt provider N; pull the **first** chunk inside the resilience loop. If it raises before the first yield → breaker.record_failure + try next candidate. Once the first chunk is yielded, you are committed (no further fallback). Implement by having the generator buffer the first successful piece, then stream the rest.
- Practical simplification acceptable for v1: keep streaming on the **primary only** with retry on connect-failure, and document that `fallback_models` applies to non-streaming requests. Circuit breaker still applies (check `allow()` before opening the stream; `record_*` on connect outcome). Choose one and document it; the spec recommends the "first-chunk fallback" approach but the simplification is acceptable.

Gotcha: streaming usage *is* recoverable now (OpenAI `stream_options:{include_usage:true}` → final chunk with full `usage`, `choices:[]`; Anthropic `message_start.usage.input_tokens` + cumulative `message_delta.usage.output_tokens`). That's a separate workstream (#3); this subsystem leaves the existing zero-usage streaming behavior untouched to preserve back-compat, but the breaker/connect-retry wrapping is additive.

---

## 7. Redis response cache (`app/services/cache.py`)

Opt-in, non-streaming, deterministic only.

**Eligibility** (all must hold): global `response_cache_enabled` (or per-request `request.cache is True`) AND `request.cache is not False` AND `request.stream is False` AND `temperature in (None, 0, 0.0)`. Reason: non-deterministic temps would serve stale/wrong content.

**Key** = `f"{prefix}{sha256_hex}"` where the digest is over a **canonical JSON** of the cache-relevant fields:

```python
import hashlib, json
def cache_key(prefix: str, provider: str, request: ChatRequest) -> str:
    payload = {
        "provider": provider,
        "model": request.model,           # requested model, not fallback
        "messages": [m.model_dump() for m in request.messages],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return prefix + hashlib.sha256(blob.encode("utf-8")).hexdigest()
```

**Stored value** = JSON of the `ChatResponse` (use `response.model_dump_json()`), so usage/cost survive round-trips. `cache_get` returns `ChatResponse.model_validate_json(raw)` or `None`. `cache_set` uses `SET key value EX ttl`.

```python
async def cache_get(r, key) -> ChatResponse | None
async def cache_set(r, key, resp: ChatResponse, ttl: int) -> None
```

Gotchas:
- **Don't include the BYOK key, user_id, or stream flag in the key** — caching is per (provider+model+messages+params), shared within the gateway. *Decision:* scope is **gateway-global**, NOT per-project. If you want per-project isolation (recommended for BYOK billing/privacy: project A shouldn't get project B's cached completion that A never paid the provider for), prefix the key with `ctx.project.id`. **Recommend per-project** keying: `cache_key(prefix, f"{project_id}:{provider}", request)`. Document this; it's a one-line change and the safer default.
- Set `cached=True` and `cost_usd` from the stored payload on return; record a usage row with `cost=0` (you didn't hit the provider) — see §4 note. Don't re-bill the BYOK provider cost on a cache hit.
- `temperature` equality: treat `0` and `0.0` and `None` as deterministic; anything `> 0` is non-cacheable. `None` defaults vary by provider (OpenAI defaults to 1.0!) — so treating `None` as cacheable is a *gateway policy decision*. Safer: only cache when `temperature == 0` explicitly. **Recommend: cache only `temperature == 0`**, exclude `None`, to avoid serving a deterministic-looking cache for a provider-default-random request.
- Redis is `decode_responses=True` (see `redis_client.py`) so you get `str` back, not bytes — `model_validate_json(str)` is fine.
- Cache failures must be **non-fatal**: wrap `cache_get`/`cache_set` in try/except, log at WARNING, proceed to provider on any Redis error. Never let the cache take down `/v1/chat`.

---

## 8. Multi-key auth

### 8.1 Model (`app/models/db.py`) — new `ApiKey`

```python
class ApiKey(Base):
    __tablename__ = "api_keys"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    key_prefix: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    project: Mapped["Project"] = relationship(back_populates="api_keys")
    __table_args__ = (Index("ix_api_keys_project_id", "project_id"),)
```

Add to `Project`: `api_keys: Mapped[list["ApiKey"]] = relationship(back_populates="project", cascade="all, delete-orphan")`. Reuse the same `key_hash` algorithm as `Project.key_hash` (`hash_api_key`, SHA-256 hex, 64 chars) so a single hash can be looked up against both tables.

### 8.2 Auth resolution (`app/middleware/auth.py`)

Resolve the incoming `x-api-key` in this order, both via the **same hash**:

1. Active `ApiKey` row: `key_hash == h AND revoked_at IS NULL` → load its `Project` (with `selectinload(Project.organisation)`). Best-effort touch `last_used_at = now()` (fire it on the request's session; commit happens later in `record_usage`, or do a lightweight `UPDATE ... WHERE id=` — keep it cheap, and don't fail the request if it errors).
2. Legacy `Project.key_hash == h` (unchanged path) → back-compat for keys minted before 0003.
3. Neither → 401.

```python
h = hash_api_key(x_api_key)
api_key = (await session.execute(
    select(ApiKey).where(ApiKey.key_hash == h, ApiKey.revoked_at.is_(None))
    .options(selectinload(ApiKey.project).selectinload(Project.organisation))
)).scalar_one_or_none()
if api_key is not None:
    project = api_key.project
else:
    project = (await session.execute(
        select(Project).where(Project.key_hash == h)
        .options(selectinload(Project.organisation))
    )).scalar_one_or_none()
if project is None: raise 401
```

Gotchas:
- A **revoked** key must 401 (the `revoked_at IS NULL` filter handles step 1; ensure a revoked `ApiKey` doesn't accidentally fall through to a legacy `Project.key_hash` match — they're different hashes, so fine, but never mint an `ApiKey` whose plaintext equals the project's legacy key).
- `last_used_at` update is a write on a GET-ish path; make it non-blocking/non-fatal. Don't `await session.commit()` just for this in auth (the chat flow commits later); a stray commit mid-dependency-chain can interfere with the budget read. Prefer recording "touch" and letting `record_usage`'s commit flush it, or issue a separate short-lived session. Simplest safe option: `await session.execute(update(ApiKey).where(...).values(last_used_at=func.now()))` without commit, relying on the end-of-request commit; if no commit happens on error paths it's a tolerable miss.
- `AuthContext` should also carry `api_key_id: uuid.UUID | None` for audit (additive field, default `None`).

### 8.3 Key-management endpoints (`app/routers/keys.py`, authenticated via `get_auth_context`)

- `POST /v1/keys/api` → create a new key for the caller's project. Body `{name: str}`. Generates `generate_api_key()`, stores `ApiKey(name, key_hash, key_prefix)`, returns plaintext **once** + `id`, `key_prefix`, `created_at`. (201)
- `GET /v1/keys/api` → list caller's project keys (no secrets): `id, name, key_prefix, created_at, last_used_at, revoked_at, active`.
- `DELETE /v1/keys/api/{key_id}` → set `revoked_at = now()` (idempotent; 404 if not in caller's project; 204). Soft-revoke; never hard-delete (preserve audit/usage linkage).

Naming gotcha: the existing `POST /v1/keys` (BYOK *provider* keys, in `account.py`) must stay. Put the new endpoints under `/v1/keys/api` (or a `/v1/apikeys` prefix) to avoid collision. Spec uses `/v1/keys/api`. New schemas in `app/schemas/keys.py` (`ApiKeyCreate`, `ApiKeyCreated`, `ApiKeyOut`). Register router in `main.py`.

---

## 9. Monthly budgets + budget/rate-limit headers

### 9.1 Model/columns (`models/db.py`)

Add to **both** `Organisation` and `Project`: `monthly_budget: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False, server_default="0")`. `0` = unlimited, matching `daily_budget` semantics.

### 9.2 Spend queries (`app/services/usage.py`)

Add `_utc_month_start()` (`now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)`) and `project_spend_this_month`/`org_spend_this_month` mirroring the daily versions but filtering `created_at >= _utc_month_start()`. Reuse the `func.coalesce(func.sum(...), 0)` pattern. The `ix_usage_org_created` / `ix_usage_project_created` composite indexes already cover these range scans.

### 9.3 Budget middleware (`app/middleware/budget.py`)

After the existing daily checks, add monthly checks for project then org (402 with detail `"Project monthly budget exceeded"` / `"Organisation monthly budget exceeded"`). Keep ordering: project-daily → org-daily → project-monthly → org-monthly. Stash computed spends/limits on `AuthContext` (new optional fields) so the router can emit headers without re-querying.

### 9.4 Response headers

The budget dependency can't set response headers directly (it's a `Depends`), so either inject `response: Response` into the dependency or compute in the router from `AuthContext`. Emit on successful `/v1/chat`:

- `x-ratelimit-limit`, `x-ratelimit-remaining`, `x-ratelimit-reset` (seconds to window end). Compute in `rate_limit.py`: it already has `current` and `project.rate_limit_per_min`; add `response: Response` param and set `remaining = max(0, limit - current)`, `reset = _WINDOW_SECONDS - (now.second)`. On 429 keep existing `Retry-After`.
- `x-budget-daily-remaining`, `x-budget-monthly-remaining` (USD, project scope; `-1` or omit when unlimited).

Gotcha: FastAPI dependencies *can* mutate a `Response` if you declare `response: Response` as a param — use that, it's the clean way to set headers from `rate_limit`/`budget`. Don't try to set headers by raising.

---

## 10. DB migration outline (`alembic/versions/0003_resilience_multikey_monthly.py`)

```
revision = "0003_resilience_multikey_monthly"
down_revision = "0002_provider_credentials"
```

`upgrade()`:
1. `create_table("api_keys", ...)` — columns per §8.1: `id` (UUID PK), `project_id` (UUID, FK→projects.id ondelete CASCADE), `name` String(255), `key_hash` String(64), `key_prefix` String(16), `created_at` (timestamptz, server_default now()), `last_used_at` (timestamptz null), `revoked_at` (timestamptz null). `UniqueConstraint("key_hash", name="uq_api_keys_key_hash")`. `create_index("ix_api_keys_project_id", "api_keys", ["project_id"])`.
2. `op.add_column("organisations", sa.Column("monthly_budget", sa.Numeric(12,4), nullable=False, server_default="0"))`.
3. `op.add_column("projects", sa.Column("monthly_budget", sa.Numeric(12,4), nullable=False, server_default="0"))`.

`downgrade()`: drop the two columns, drop `ix_api_keys_project_id`, drop `api_keys`.

Gotchas:
- Use `server_default="0"` on the new NOT NULL columns so the migration succeeds on existing rows (mirrors how `daily_budget` is `nullable=False`). The model uses `default=Decimal("0")` (Python-side); add `server_default="0"` in the *model* too for consistency, or at least in the migration. (Note: existing `daily_budget` has no server_default in 0001 and is inserted Python-side — fine since those tables are written via ORM. For an `add_column` on a populated table you **must** provide `server_default`.)
- No data backfill needed; legacy `projects.key_hash` stays authoritative for existing keys.
- `api_keys.key_hash` UNIQUE is global (like `projects.key_hash`); collisions across the two tables are astronomically unlikely but auth checks `api_keys` first anyway.

---

## 11. Test plan (pytest + `httpx.MockTransport`, no live network)

Follow existing style: set env before import, override provider HTTP via `httpx.MockTransport`. Where the adapters build their own `AsyncClient`, inject the transport by patching (the adapters call `httpx.AsyncClient(timeout=...)`; add an optional `transport` seam or monkeypatch `httpx.AsyncClient` in tests — recommend adding a private `_client_factory` hook to each adapter for testability, defaulting to real client).

**Retry** (`test_resilience.py`):
- `delay_for` is monotonic-ish, capped at `max_delay`, and always `≥ backoff` and `< backoff + jitter`.
- `retry_async` returns on first success (transport called once).
- Transient 503 then 200 → succeeds on attempt 2; transport called twice; `asyncio.sleep` patched to no-op to keep tests fast.
- Non-retryable 400 → raised immediately, transport called once.
- Exhausts `max_attempts` on persistent 503 → raises last error, called exactly `max_attempts` times.
- `Retry-After: 2` honored (delay ≥ 2) when threaded through.

**Circuit breaker**:
- Opens after `fail_threshold` consecutive failures (`allow()` → False).
- Stays OPEN within cooldown; `allow()` → True after cooldown (HALF_OPEN) — patch `time.monotonic`.
- HALF_OPEN success closes; HALF_OPEN failure re-opens and resets timer.
- Keyed per `provider:project` — failures on project A don't open project B.
- 400/401 do **not** increment failures.

**Fallback** (`test_fallback.py`, MockTransport routing by URL host):
- Primary 503 (after retries) → fallback model used; `ChatOutcome.model_used`/`fallback_used`/header reflect it; usage row records fallback model.
- Fallback also fails → last error raised (502/503).
- Missing BYOK key for a fallback provider → that candidate skipped, next tried; primary missing key still 400.
- Circuit OPEN on primary → skips straight to fallback without calling primary transport.

**Cache** (`test_cache.py`, fakeredis or a dict-backed stub honoring `set ex`/`get`):
- Miss then set then hit: second identical request returns `cached=True`, transport **not** called second time, `x-llmgw-cached: true`.
- `temperature=0.7` → not cached (transport called both times).
- `temperature=None` → not cached (per §7 recommendation).
- `stream=True` → never cached.
- Different `messages`/`model`/`max_tokens` → different key → miss.
- Per-project isolation: project B with identical request gets a miss.
- Redis error in `cache_get`/`cache_set` → request still succeeds (non-fatal).
- Cached hit records usage row with `cost=0`.

**Multi-key auth** (`test_api_keys.py`):
- `POST /v1/keys/api` returns plaintext once; `GET` lists it without secret; key authenticates `/v1/chat`.
- Legacy `projects.key_hash` still authenticates (back-compat).
- Revoked key → 401; revoke is idempotent; revoking another project's key → 404.
- `last_used_at` populated after a successful authenticated call.
- Existing `POST /v1/keys` (provider BYOK) still works unchanged.

**Monthly budgets + headers** (`test_budget_monthly.py`):
- Project/org monthly budget exceeded → 402 with correct detail; daily-only still works.
- `0` monthly budget = unlimited.
- Month boundary: a record dated last month doesn't count toward this month (insert with backdated `created_at`).
- Success response carries `x-ratelimit-*` and `x-budget-*` headers with correct math; 429 carries `Retry-After`.

**Migration smoke**: `alembic upgrade head` then `downgrade -1` round-trips on a throwaway DB (or assert revision graph: `0003.down_revision == "0002_provider_credentials"`).

---

## Key gotchas (consolidated)

1. **`get_settings()`/`redis_client`/`engine` are import-time singletons** — don't capture settings at module top in new files; accept values or call inside functions for testability.
2. **Don't cache streaming**; cache only `temperature == 0`, non-stream; recommend **per-project** cache keys (BYOK billing/privacy).
3. **Cache & breaker failures must be non-fatal** — try/except + log, never 500 `/v1/chat`.
4. **Breaker keyed per `provider:project_id`** and only trips on provider/5xx/timeout, never on 400/401/bad-BYOK-key. Use `time.monotonic()`.
5. **Retry is nested inside fallback** (per-candidate retry budget); chat is idempotent so retry is safe; never retry a stream after first byte.
6. **Auth checks `api_keys` (revoked_at IS NULL) before legacy `projects.key_hash`**; same hash function; `last_used_at` touch must be non-fatal and must not trigger a stray mid-chain commit that perturbs the budget read.
7. **New endpoints under `/v1/keys/api`** to avoid colliding with existing BYOK `POST /v1/keys`.
8. **`add_column` NOT NULL needs `server_default="0"`** on populated tables (`monthly_budget`).
9. **Additive schema fields only** (`fallback_models`, `cache`, `cached`, `fallback_used` with defaults) → `/v1/chat` and stored `ChatResponse` consumers stay back-compatible; usage rows unchanged in shape (reuse existing `status`/`cost` columns; cache hit = `status="ok"`, `cost=0`).
10. **Headers from dependencies**: inject `response: Response` into `rate_limit`/`budget` deps to set `x-ratelimit-*`/`x-budget-*`; on cache/fallback set `x-llmgw-*` in the router.

New files: `app/utils/resilience.py`, `app/services/circuit_breaker.py`, `app/services/cache.py`, `app/services/routing.py`, `app/services/api_keys.py`, `app/routers/keys.py`, `app/schemas/keys.py`, `alembic/versions/0003_resilience_multikey_monthly.py`. Edited: `config.py`, `schemas/chat.py`, `models/db.py`, `middleware/auth.py`, `middleware/rate_limit.py`, `middleware/budget.py`, `routers/chat.py`, `main.py`, `services/usage.py`, `providers/base.py` (optional `retry_after` on `ProviderError`).

Sources: [OpenAI streaming usage / stream_options.include_usage](https://community.openai.com/t/usage-stats-now-available-when-using-streaming-with-the-chat-completions-api-or-completions-api/738156), [OpenAI chat streaming chunk object](https://platform.openai.com/docs/api-reference/chat-streaming/streaming), [Anthropic streaming messages (message_start / message_delta usage)](https://platform.claude.com/docs/en/build-with-claude/streaming)