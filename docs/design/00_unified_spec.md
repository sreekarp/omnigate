I have all the specs I need. This is a synthesis task — no codebase exploration required since the specs already cite exact file paths. I'll produce the unified spec directly.

# UNIFIED IMPLEMENTATION SPEC — LLM Gateway Enhancements

This is the single source of truth. Build in the order given in §3. All code is async, type-hinted, Pydantic v2, `logging` only (no `print` in `app/`).

---

## 1. Consolidated DB schema changes — single Alembic migration `0003`

`alembic/versions/0003_resilience_multikey_monthly_meta.py`

```python
revision = "0003_resilience_multikey_monthly_meta"
down_revision = "0002_provider_credentials"
```

**De-conflict note:** four specs touched the DB. The Azure spec's `provider_credentials.meta` column, the resilience spec's `api_keys` table + monthly-budget columns, and the metrics spec's *optional* `cache_hit` column all collapse into this one migration. The metrics spec recommended **option (A)** — encode cache hits in the existing `status` string (`status="cache_hit"`), **no column**. The resilience spec independently decided cache hits record `status="ok"`, `cost=0`. **RESOLUTION: cache hits record `status="cache_hit"`** (satisfies both — metrics gets its rate from the status string, resilience still sets `cost=0`). No `cache_hit` boolean column. Zero migration cost for cache-hit-rate.

`upgrade()` does exactly four things:

1. **Add `provider_credentials.meta`** (Azure non-secret config):
   ```python
   op.add_column("provider_credentials", sa.Column("meta", sa.JSON(), nullable=True))
   ```
   Portable `sa.JSON` (not JSONB) so SQLite test fixtures work. Existing rows → `NULL`, openai/anthropic never read it.

2. **Create `api_keys` table** (multi-key auth):
   | col | type | notes |
   |---|---|---|
   | `id` | UUID PK | `default uuid4` |
   | `project_id` | UUID FK→projects.id | `ondelete CASCADE` |
   | `name` | String(255) NOT NULL | |
   | `key_hash` | String(64) NOT NULL | `UniqueConstraint uq_api_keys_key_hash` |
   | `key_prefix` | String(16) NOT NULL | |
   | `created_at` | timestamptz NOT NULL | `server_default now()` |
   | `last_used_at` | timestamptz NULL | |
   | `revoked_at` | timestamptz NULL | |

   `create_index("ix_api_keys_project_id", "api_keys", ["project_id"])`.

3. **Add `organisations.monthly_budget`** and **4. `projects.monthly_budget`**:
   ```python
   sa.Column("monthly_budget", sa.Numeric(12, 4), nullable=False, server_default="0")
   ```
   `server_default="0"` is **mandatory** — these are NOT NULL adds on populated tables. `0` = unlimited (mirrors `daily_budget`).

`downgrade()`: drop both `monthly_budget` columns, drop `ix_api_keys_project_id`, drop `api_keys`, drop `provider_credentials.meta`.

**ORM model changes (`app/models/db.py`)** that pair with the migration:
- `ProviderCredential`: `meta: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=None)`.
- New `ApiKey` model (cols above) + `Project.api_keys` relationship (`cascade="all, delete-orphan"`). Reuse `hash_api_key` (SHA-256, 64 hex) so one hash matches both `api_keys` and legacy `projects.key_hash`.
- `Organisation.monthly_budget` + `Project.monthly_budget`: `Mapped[Decimal] = mapped_column(Numeric(12,4), nullable=False, server_default="0", default=Decimal("0"))`.

No `UsageRecord` shape change. Existing `ix_usage_project_created (project_id, created_at)` and `ix_usage_org_created (org_id, created_at)` already cover metrics + monthly-spend range scans.

---

## 2. Final enriched provider interface

Two specs proposed competing mechanisms to surface streaming usage: a side-channel `StreamUsage` out-param (Gemini spec) vs. a typed `StreamChunk` yield (streaming-usage spec). **RESOLUTION: adopt the typed `StreamChunk` design** — it is strictly more expressive, has no shared-mutable-state/ordering hazards, and is trivially testable. The `StreamUsage` out-param is dropped.

**`app/schemas/chat.py`** (additive only — `Usage`/`ChatResponse` field names unchanged for DB/wire stability):
```python
class StreamChunk(BaseModel):
    text: str = ""                  # delta forwarded to wire; may be ""
    usage: Usage | None = None      # populated on the FINAL usage-bearing chunk only
    finish_reason: str | None = None
    model: str | None = None        # provider-confirmed model id, when known
```
Also additive (resilience spec): `ChatResponse.cached: bool = False`, `ChatResponse.fallback_used: bool = False`; `ChatRequest.fallback_models: list[str] = Field(default_factory=list, max_length=5)`, `ChatRequest.cache: bool | None = None`.

**`app/providers/base.py`**:
```python
class AbstractProvider(ABC):
    name: str

    @abstractmethod
    async def chat(self, request: ChatRequest, api_key: str) -> ChatResponse: ...

    @abstractmethod
    def stream(self, request: ChatRequest, api_key: str) -> AsyncIterator[StreamChunk]:
        """Async generator. Terminal usage chunk (text may be '') carries final Usage."""
        ...

    async def stream_text(self, request: ChatRequest, api_key: str) -> AsyncIterator[str]:
        async for chunk in self.stream(request, api_key):
            if chunk.text:
                yield chunk.text
```
**Critical invariant (do not regress):** base `stream` stays a plain `def` abstractmethod; concrete impls are `async def` generators. Only `stream_text` is `async def`. Making base `stream` async turns every adapter into a coroutine-returning-generator.

`ProviderError` gains optional `retry_after: float | None = None` (parsed from `resp.headers.get("retry-after")`, integer seconds; ignore HTTP-date form). Default `None` keeps all existing `raise` sites compiling.

**Per-adapter contract** — every adapter yields `StreamChunk(text=...)` for content and exactly one terminal `StreamChunk(usage=Usage(...), finish_reason=...)` before the generator ends. Usage values are **absolute/last-wins**, never re-summed:
- **OpenAI/Azure**: add `stream_options={"include_usage": True}` only when streaming. Guard `choices = chunk.get("choices") or []` (final usage chunk has `choices:[]`; Azure's first content-filter chunk also empty). Emit usage when `chunk.get("usage")` is truthy.
- **Anthropic**: `message_start.message.usage.input_tokens` = prompt (ignore its priming `output_tokens`); `message_delta.usage.output_tokens` = cumulative completion (last wins); stash `stop_reason`; emit terminal chunk at `message_stop`. Handle `error` events → `ProviderError(status_code=502)`; ignore `ping`.
- **Gemini**: `?alt=sse`; `usageMetadata.{promptTokenCount,candidatesTokenCount,totalTokenCount}` are absolute — overwrite on every chunk, last wins. Trust server `totalTokenCount` for `total_tokens` (it includes thoughts/tool tokens).

**Azure construction (Option A — no signature change):** the Azure adapter is **NOT** `lru_cache`d. `registry.get_provider_for_model` only recognizes the `azure/` prefix; the **router** reads `meta`, resolves the deployment, and constructs `AzureOpenAIProvider(endpoint=..., deployment=..., api_version=...)` per request. `chat(request, api_key)`/`stream(request, api_key)` signatures stay intact.

---

## 3. File-by-file change/addition list

### GROUP A — Foundation / spine (build first, in this order; everything else depends on it)

| Path | Purpose |
|---|---|
| `app/models/db.py` | EDIT. `ProviderCredential.meta` (JSON); new `ApiKey` model + `Project.api_keys` rel; `monthly_budget` on `Organisation`+`Project`. |
| `alembic/versions/0003_resilience_multikey_monthly_meta.py` | NEW. The single migration from §1. `down_revision="0002_provider_credentials"`. |
| `app/schemas/chat.py` | EDIT. Add `StreamChunk`; add `fallback_models`,`cache` to `ChatRequest`; `cached`,`fallback_used` to `ChatResponse`. `Usage`/existing fields untouched. |
| `app/providers/base.py` | EDIT. `stream()` → `AsyncIterator[StreamChunk]`; add `stream_text()` default; `ProviderError.retry_after`. |
| `app/config.py` | EDIT. All new env knobs (§5). Do NOT read settings at module top in new files. |
| `app/services/credentials.py` | EDIT. `SUPPORTED_PROVIDERS += ("gemini","azure")`; `set_provider_key(..., meta=None)`; new `get_provider_credential(...)->(key,meta)`; keep `get_provider_key`. |
| `app/services/pricing.py` | EDIT. Add `gemini-*` and `azure/*` price rows; add public `get_price(model)->tuple[Decimal,Decimal]|None` and `known_models()->list[str]` (keep `_PRICING` private). |
| `app/providers/registry.py` | EDIT. Add `gemini` prefix → `@lru_cache _gemini()`; add `azure/`/`azure-` prefix recognition → `@lru_cache _azure()` factory **type only** (router builds the per-request instance). |

These 8 are mutually coupled (the migration must match the models; registry/pricing/credentials are read by every router; `StreamChunk`/`base.py` define the contract all adapters implement). Get them coherent before parallelizing.

### GROUP B — Isolated new modules (safe to build in parallel once Group A lands)

| Path | Purpose |
|---|---|
| `app/providers/openai.py` | EDIT. `stream()` yields `StreamChunk`; `stream_options` on stream; empty-`choices` guard; emit usage chunk. |
| `app/providers/anthropic.py` | EDIT. `stream()` yields `StreamChunk`; accumulate `message_start`/`message_delta`; terminal usage at `message_stop`; handle `error`. |
| `app/providers/gemini.py` | NEW. Full adapter per Gemini spec, but `stream()` yields `StreamChunk` (not `StreamUsage` out-param). |
| `app/providers/azure_openai.py` | NEW. Per-request-constructed adapter (endpoint/deployment/api_version in `__init__`); `api-key` header; `stream()` yields `StreamChunk`. |
| `app/utils/resilience.py` | NEW. `RetryPolicy`, `retry_async`, `is_retryable`, `_RETRYABLE_STATUS={408,429,500,502,503,504}`. |
| `app/services/circuit_breaker.py` | NEW. Per-`provider:project_id` breaker, `time.monotonic()`, memory+optional redis backend behind a Protocol. |
| `app/services/cache.py` | NEW. `cache_key` (per-project), `cache_get`/`cache_set`; non-fatal on Redis error; only `temperature==0`, non-stream. |
| `app/services/api_keys.py` | NEW. create/list/revoke logic over `ApiKey`. |
| `app/services/metrics.py` | NEW. `MetricsScope`, `get_metrics`, predicate constants, 3 sequential SELECTs (§6 gotchas). |
| `app/schemas/metrics.py` | NEW. Response models (`MetricsResponse` etc.). |
| `app/schemas/keys.py` | NEW. `ApiKeyCreate`,`ApiKeyCreated`,`ApiKeyOut`. |
| `app/schemas/openai_compat.py` | NEW. OpenAI-wire DTOs + `to_internal_chat_request` translation. |
| `app/cli.py` | NEW. argparse CLI (`serve`/`db`/`org`/`project`/`key`/`usage`/`config check`/`version`). |
| `sdk/**` (separate package) | NEW. Independent `llm-gateway-sdk` per its spec; imports nothing from `app/`. Tests use MockTransport; `models()`/`metrics()` target the new endpoints below. |
| `pyproject.toml` (repo root) | NEW. Server packaging + `llm-gateway = "app.cli:main"` console script. |

### GROUP C — Wiring (build last; integrates Groups A+B)

| Path | Purpose |
|---|---|
| `app/services/usage.py` | EDIT. Add `_utc_month_start()`, `project_spend_this_month`, `org_spend_this_month`. |
| `app/services/routing.py` | NEW. `execute_chat()` — composes cache→breaker→retry→fallback around `provider.chat`; returns `ChatOutcome`. Keeps `chat.py` thin. |
| `app/middleware/auth.py` | EDIT. Accept `Authorization: Bearer` (prefer `x-api-key`); resolve `api_keys` (revoked_at IS NULL) **before** legacy `projects.key_hash`; non-fatal `last_used_at` touch (no stray commit); `AuthContext.api_key_id`. |
| `app/middleware/rate_limit.py` | EDIT. Inject `response: Response`; set `x-ratelimit-limit/remaining/reset`; keep `Retry-After` on 429. |
| `app/middleware/budget.py` | EDIT. Add monthly checks (project-daily→org-daily→project-monthly→org-monthly); stash spends on `AuthContext`; set `x-budget-*` headers. |
| `app/routers/chat.py` | EDIT. Call `execute_chat`; Azure branch (fetch key+meta, resolve deployment, build adapter); rewrite `_stream` to consume `StreamChunk`, forward `.text`, accumulate usage, record **real** tokens+cost in `finally` (media_type `text/plain`, `x-request-id` unchanged); set `x-llmgw-*` headers. |
| `app/routers/openai_compat.py` | NEW. `POST /v1/chat/completions` (non-stream + SSE `text/event-stream`), `GET /v1/models`, `GET /v1/models/{id}`. Reuses `enforce_budget`, `execute_chat`/provider pipeline, `record_usage`. |
| `app/routers/metrics.py` | NEW. `GET /v1/metrics` (project-scoped via `get_auth_context`). |
| `app/routers/admin.py` | EDIT. Add `GET /orgs/{org_id}/metrics` (reuse `require_admin` + `get_metrics`; 404 if org missing). |
| `app/routers/keys.py` | NEW. `POST/GET/DELETE /v1/keys/api*` (multi-key mgmt). NOT colliding with existing BYOK `POST /v1/keys`. |
| `app/routers/account.py` | EDIT. `POST /v1/keys` (BYOK): validate Azure `meta.endpoint`+`meta.deployment` (https://), persist `meta`. |
| `app/main.py` | EDIT. `include_router` for `openai_compat`, `metrics`, `keys`. |

---

## 4. Final new/changed HTTP endpoints

| Method | Path | Auth | Notes |
|---|---|---|---|
| POST | `/v1/chat` | gateway key | EXISTING. Now: streaming records real tokens; `fallback_models`/`cache` honored; `x-llmgw-*` headers. Wire-compatible. |
| POST | `/v1/chat/completions` | gateway key (`x-api-key` **or** `Authorization: Bearer`) | NEW. OpenAI-compatible, non-stream + SSE (`text/event-stream`, `data: [DONE]`). |
| GET | `/v1/models` | gateway key (auth only) | NEW. `{object:"list", data:[model_card]}`; only routable+priced models. |
| GET | `/v1/models/{id}` | gateway key | NEW. Single card; 404 envelope if unknown. |
| GET | `/v1/metrics` | gateway key (project scope) | NEW. totals/breakdowns/timeseries. |
| GET | `/v1/admin/orgs/{org_id}/metrics` | `x-admin-key` | NEW. Org-scoped; 404 if org missing. |
| POST | `/v1/keys/api` | gateway key | NEW. Create named key, plaintext returned once (201). |
| GET | `/v1/keys/api` | gateway key | NEW. List keys (no secrets). |
| DELETE | `/v1/keys/api/{key_id}` | gateway key | NEW. Soft-revoke (`revoked_at`), idempotent, 204; 404 if not caller's project. |
| POST | `/v1/keys` | gateway key | EXISTING (BYOK). Now accepts `provider:"gemini"\|"azure"` + Azure `meta`. |

`Authorization: Bearer` support is added in `get_auth_context`, so it transparently flows to `/v1/chat` too (`x-api-key` still preferred). Don't confuse the gateway key (project) with BYOK provider keys (server-resolved from `ProviderCredential`).

---

## 5. Config knobs (new env vars in `app/config.py` `Settings`)

```python
# Resilience — retry
retry_max_attempts: int = 3
retry_base_delay_seconds: float = 0.25
retry_max_delay_seconds: float = 8.0
retry_jitter_seconds: float = 0.25
# Resilience — circuit breaker
circuit_breaker_enabled: bool = True
circuit_breaker_fail_threshold: int = 5
circuit_breaker_cooldown_seconds: float = 30.0
circuit_breaker_backend: str = "memory"        # "memory" | "redis"
# Response cache
response_cache_enabled: bool = False           # global opt-in; per-request flag also required
response_cache_ttl_seconds: int = 300
response_cache_prefix: str = "respcache:"
```
No new env vars for Gemini/Azure/metrics — **BYOK keys + Azure `meta` live in `provider_credentials`**, not env. Existing `OPENAI_API_KEY`/`ANTHROPIC_API_KEY`/`DATABASE_URL`/`REDIS_URL`/`SECRET_KEY`/`request_timeout_seconds` unchanged. `get_settings()` is `@lru_cache`d — tests must set env before first import; new modules must call `get_settings()` inside functions, never at module top.

---

## 6. Prioritized risk list + top correctness gotchas

**P0 — will break at runtime if wrong:**
1. **Streaming DB-session lifetime.** `chat.py::_stream`'s `finally` runs *after* the StreamingResponse body finishes — i.e. after the handler returns. The `Depends(get_session)` session may already be closed. **Verify `get_session` stays open for the generator's life; if not, acquire a fresh session from the sessionmaker inside `event_generator()`.** This is the single most likely runtime bug. Test with the disconnect case.
2. **No concurrent ops on one async session.** Metrics runs 3 SELECTs — `await` them **sequentially**, never `asyncio.gather` on the same session/connection (asyncpg raises). Same caution anywhere in the chat pipeline.
3. **`async`-ness of base `stream`.** Keep base `stream` a plain `def` abstractmethod with `async def` generator impls. `stream_text` is the only `async def`. Regressing this breaks every adapter.
4. **`add_column` NOT NULL needs `server_default="0"`** for `monthly_budget` on populated tables, else migration fails.

**P1 — correctness/billing:**
5. **Postgres percentile SQL ordering.** Mandate `func.percentile_cont(0.95).filter(pred).within_group(col.asc())` (filter-then-within_group) — works across all SQLAlchemy 2.0.x (issue #11423). Add a SQL-compile unit test asserting the emitted string `percentile_cont(0.5) WITHIN GROUP (ORDER BY ... ASC) FILTER (WHERE ...)` so an upgrade can't silently break it. Filter percentiles to non-error rows so fast error floods don't deflate p95.
6. **`date_trunc` 3-arg form** `date_trunc('hour', created_at, 'UTC')` is PG12+, returns deterministic `timestamptz`. Document the version floor. **Never wrap `created_at` in a function in the WHERE clause** (only in SELECT/GROUP BY) — you'd lose the composite index.
7. **Half-open interval `[start, end)`** for all windows/buckets to avoid double-counting boundaries.
8. **Streaming usage absolute, not delta.** Anthropic `output_tokens` is cumulative (last-wins); Gemini/OpenAI usage is absolute on the final chunk. Never re-sum; trust the server's `total`. Interrupted stream may never deliver the usage chunk — record whatever accumulated; consider `status="incomplete"` if `total_tokens==0` after `CancelledError`.
9. **Decimal→float.** `func.sum(cost)`/`avg(latency)` return `Decimal`; cast to `float` when building Pydantic models. `percentile_cont` is already `float`. Empty window → `None` (no NaN); coalesce sums but **not** percentiles; guard all divisions with `if requests else 0.0`.

**P2 — backward compatibility & isolation:**
10. **Auth order & revocation.** Check `api_keys WHERE revoked_at IS NULL` before legacy `projects.key_hash`. `last_used_at` touch must be non-fatal and must **not** issue a stray mid-dependency-chain commit (it can perturb the budget read). Prefer letting `record_usage`'s end-of-request commit flush it.
11. **Cache & breaker must be non-fatal** — try/except + WARNING log, always fall through to the provider. Breaker keyed per `provider:project_id`, only trips on 5xx/timeout/transient (never 400/401/bad-BYOK). Use `time.monotonic()`.
12. **Cache keying per-project** (`{project_id}:{provider}` in the key) for BYOK billing/privacy. **Cache only `temperature == 0`** (exclude `None` — OpenAI's `None` default is 1.0). Never cache streams. Cache hit → `status="cache_hit"`, `cost=0`, tokens from cached payload.
13. **Additive-only schema fields** (`fallback_models`,`cache`,`cached`,`fallback_used`,`meta`,`StreamChunk`) keep `/v1/chat`, stored `ChatResponse`, and existing `UsageRecord` rows compatible.
14. **OpenAI-compat edge cases:** `choices:[]` on final/filter chunks (guard `or []`); `content` can be `None` (`.get` + truthiness); `role:"tool"`→400 envelope, `developer`→`system`; list/multimodal content→400; SDK requires `media_type="text/event-stream"` and `data: [DONE]` always (even on error, in `finally`); compact JSON `separators=(",",":")`.
15. **Azure specifics:** header `api-key` (no Bearer); `api-version` query mandatory; body `model` ignored (deployment in URL wins); `rstrip("/")` the stored endpoint; build adapter per-request (not `lru_cache`d).
16. **Fallback bounds:** validate fallback models lazily (skip+log unknowns, never 500); `max_length=5`; retry nested *inside* the fallback loop (per-candidate budget); never retry a stream after the first byte (first-chunk fallback at most, or document primary-only streaming).

---

## 7. Minimal-dependency decision

**No new runtime dependencies anywhere.** Confirmed hand-rolled vs. existing-package:

| Concern | Decision |
|---|---|
| Retry/backoff + jitter | **Hand-roll** (`asyncio.sleep`, stdlib `random`). No `tenacity`. |
| Circuit breaker | **Hand-roll** (`time.monotonic` + dict; optional Redis via existing `redis.asyncio`). No `pybreaker`. |
| Response cache | **Existing** `redis.asyncio` (`SET ... EX`) + stdlib `hashlib`/`json`. No new cache lib. |
| Gemini / Azure / OpenAI-compat HTTP | **Existing** `httpx.AsyncClient`. No provider SDKs. |
| Metrics SQL (percentiles, date_trunc) | **Existing** SQLAlchemy 2.0 + Postgres native funcs. No analytics lib. |
| OpenAI-compat SSE framing | **Hand-roll** with `json.dumps(separators=(",",":"))`. No SSE lib. |
| CLI | **Hand-roll** with stdlib `argparse`. No Click/Typer. |
| SDK runtime | **`httpx` + `pydantic` only** (separate package); backoff/SSE-sentinel parsing hand-rolled. No `tenacity`/`anyio`. |
| Packaging | `hatchling` build backend (build-time only, both `pyproject.toml`s). |
| Tests | Existing `pytest`/`pytest-asyncio` + `httpx.MockTransport`/`ASGITransport`. Postgres-specific metrics tests need a real Postgres (percentile/date_trunc/FILTER aren't in SQLite) — split compile-level (no DB) from integration (DB) tests. SDK may use `fakeredis` as a **dev-only** extra if a dict stub is insufficient. |

Build-backend additions (`hatchling`) and any test-only extras are not runtime deps. The gateway's `pyproject.toml` should mirror the pinned `requirements.txt` versions (single source of truth; keep `requirements.txt` for Docker layer caching).