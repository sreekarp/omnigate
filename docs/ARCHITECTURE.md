# Architecture

This document describes how the OmniLLM is put together: the request
pipeline, the provider abstraction, the resilience layer, metrics/observability,
the data model, and the client SDK. It complements `README.md` (usage) and the
detailed design notes under `docs/design/`.

## Overview

```
client ──HTTP──▶ FastAPI app ──▶ provider (OpenAI / Anthropic / Gemini / Azure)
                    │                       │
                    │                  (BYOK key, encrypted at rest)
                    ▼
            Postgres (durable: orgs, projects, keys, credentials, usage)
            Redis    (ephemeral: rate-limit counters, response cache,
                      optional circuit-breaker state)
```

- **FastAPI** async app (`app/main.py`).
- **Postgres** via `asyncpg` + SQLAlchemy 2.0 async ORM (`app/models/db.py`) for
  all durable data.
- **Redis** via `redis.asyncio` for rate-limit counters, the opt-in response
  cache, and the optional Redis circuit-breaker backend.
- **Alembic** owns the schema (migrations under `alembic/versions/`); the app
  never creates tables itself.
- Configuration is centralised in `app/config.py` (pydantic-settings).
- A request-id is bound in an ASGI middleware (`app/main.py`) and injected into
  every log record (`app/logging_config.py`); it is echoed as `x-request-id`.

## Request pipeline

The pipeline is built from **chained FastAPI dependencies** (not Starlette
middleware), so ordering is guaranteed by the dependency graph:

```
get_auth_context  →  enforce_rate_limit  →  enforce_budget  →  route handler
```

1. **Auth** (`app/middleware/auth.py`) — extracts the gateway key from
   `x-api-key` or `Authorization: Bearer`, resolves it against the `api_keys`
   table (unrevoked, multi-key per project) and falls back to the legacy
   `projects.key_hash`. Produces an `AuthContext(project, user_id, api_key_id)`.
2. **Rate limit** (`app/middleware/rate_limit.py`) — Redis `INCR` on a
   per-project, per-minute window key; `429` with `Retry-After` when exceeded.
3. **Budget** (`app/middleware/budget.py`) — sums today's and this-month's spend
   (UTC) for both the project and the org against their `daily_budget` and
   `monthly_budget` (`0` = unlimited); `402` on exceed. Echoes `x-budget-*`
   headers. Reads are sequential on the one async session (never concurrent).
4. **Route** — the handler resolves the model to a provider + BYOK key and runs
   the resilience pipeline, then records a `UsageRecord`.

### Endpoints

- `POST /v1/chat` (`app/routers/chat.py`) — native endpoint; JSON `ChatResponse`
  or a `text/plain` chunked stream when `stream` is true.
- `POST /v1/chat/completions` (`app/routers/openai_compat.py`) — OpenAI
  wire-compatible; JSON, or SSE terminated by `data: [DONE]`. The official
  `openai` SDK can point `base_url` at the gateway.
- `GET /v1/models`, `GET /v1/models/{id}` — model catalog cards.
- `GET /v1/metrics` — project-scoped usage analytics.
- `GET /v1/admin/orgs/{id}/metrics` — org-scoped analytics (`x-admin-key`).
- `POST/GET/DELETE /v1/keys/api` — gateway key management.
- `POST /v1/signup`, `POST /v1/keys` (BYOK), `GET /v1/me` — self-serve account.
- `GET /metrics` — Prometheus text exposition.
- `GET /health`, `/health/live`, `/health/ready`, `/version`.

## Provider abstraction

Every provider adapter implements `AbstractProvider`
(`app/providers/base.py`):

```python
async def chat(self, request: ChatRequest, api_key: str) -> ChatResponse: ...
def stream(self, request: ChatRequest, api_key: str) -> AsyncIterator[StreamChunk]: ...
```

- Adapters translate the unified `ChatRequest` to the provider-native payload
  and back. The caller's BYOK key is passed per call.
- `stream(...)` is an async generator: it yields content `StreamChunk`s (with
  `text` set) followed by **exactly one** terminal chunk carrying the final
  `Usage` and `finish_reason`. Token counts are absolute / last-wins, never
  re-summed by consumers.
- Failures raise `ProviderError(message, status_code, retry_after=…)`;
  `retry_after` is parsed from the upstream `Retry-After` header so the retry
  layer can honour backoff hints.

`StreamChunk` (`app/schemas/chat.py`):

```python
class StreamChunk(BaseModel):
    text: str = ""
    usage: Usage | None = None
    finish_reason: str | None = None
    model: str | None = None
```

### Adapters

- **OpenAI** (`openai.py`) — also exposes module helpers
  `build_chat_payload` / `parse_chat_response` / `parse_stream_chunk`.
- **Anthropic** (`anthropic.py`) — Messages API; `message_delta` carries usage.
- **Gemini** (`gemini.py`) — `GeminiProvider()`; v1beta Generative Language API.
  Model id lives in the URL path; auth via `x-goog-api-key`; streaming uses
  `?alt=sse`; `usageMetadata` is absolute/last-wins.
- **Azure OpenAI** (`azure_openai.py`) — `AzureOpenAIProvider(*, endpoint,
  deployment, api_version)`. Reuses the OpenAI payload/parse helpers; auth via
  the `api-key` header; constructed **per request** from stored credential
  `meta` (so it is not process-cached).

### Routing

`app/providers/registry.py::provider_name_for_model` maps a model id to a
provider by prefix (`gpt-/o1/o3/o4/chatgpt` → openai, `claude-` → anthropic,
`gemini-`/`models/gemini-` → gemini, `azure/<deployment>` or `azure-` → azure);
unknown models raise `ProviderError(400)`. Non-Azure adapters are lazily
instantiated and `lru_cache`d; Azure is built by the router/resolver from the
project's stored endpoint/deployment/api-version.

## Resilience layer

The non-streaming path runs through
`app/services/routing.py::execute_chat`:

```
cache lookup
  → for each candidate model in [primary, *fallback_models] (capped at 6):
        resolve_provider (model → provider + BYOK key/meta)
        circuit-breaker allow?  (skip candidate if open)
        retry_async( provider.chat )      # transient failures only
  → on success: compute cost, set fallback_used, cache store (primary only)
```

- **Retry** (`app/utils/resilience.py`) — `RetryPolicy.from_settings` +
  `retry_async`. Backoff = `base_delay * 2**attempt` capped at `max_delay`, plus
  uniform jitter in `[0, jitter]`. Retryable: `ProviderError` with status in
  `{408,429,500,502,503,504}`, or httpx timeout/transport errors. A larger
  `retry_after` hint wins. Non-retryable errors propagate immediately.
- **Circuit breaker** (`app/services/circuit_breaker.py`) — classic
  closed/open/half-open breaker keyed by `f"{provider}:{project_id}"`, so one
  project's bad key or one provider's outage doesn't trip unrelated traffic.
  Two backends behind a `CircuitBreakerBackend` Protocol: in-memory (default,
  single-process/tests) and Redis (`INCR`/`EXPIRE`/`GET`, multi-worker). All
  bookkeeping is non-fatal and fails open. Only transient failures trip it; 4xx
  client errors skip to the next candidate.
- **Fallback** — `ChatRequest.fallback_models` (≤5) are tried in order; each may
  resolve to a different provider. Streaming uses the primary model only (you
  can't un-send bytes), so retry/fallback/cache do not apply to streams.
- **Response cache** (`app/services/cache.py`) — opt-in Redis cache, eligible
  only for non-streaming, deterministic (`temperature == 0`) requests, enabled
  globally (`response_cache_enabled`) or per-request (`cache: true`). The key is
  `prefix + sha256` of project_id + provider + model + messages + sampling/limit
  params, isolating projects from each other's BYOK-billed completions. Hits are
  billed `$0` and marked `cached`. Cache access is always non-fatal.

## Metrics & observability

### Usage analytics (`app/services/metrics.py`, `app/routers/metrics.py`)

Aggregates `UsageRecord` rows into a `MetricsResponse`
(`app/schemas/metrics.py`):

- `totals` — requests, prompt/completion/total tokens, `cost_usd`,
  `error_rate`, `cache_hit_rate`, `avg_latency_ms`, and p50/p95/p99 latency
  (Postgres `percentile_cont … WITHIN GROUP`).
- `breakdown` — grouped rows by `provider|model|user|status`.
- `timeseries` — sparse buckets at `minute|hour|day` granularity.

Windows come from `range=1h|24h|7d|30d` or explicit `from`/`to` ISO timestamps.
`error_rate` counts `error|rate_limited|budget_exceeded`; `cache_hit_rate`
counts the `cache_hit` status (cache hits are encoded in `UsageRecord.status`,
needing no extra column). The three SELECTs run sequentially on one async
session. A `MetricsScope(field, value)` selects project- vs org-scope; the admin
org endpoint reuses the same code path.

### Prometheus (`app/observability/prometheus.py`, `GET /metrics`)

A tiny hand-rolled, thread-safe registry (no `prometheus-client` dependency).
`observe_request(...)` is called from each chat handler; `render_prometheus()`
emits text exposition for:

- `llmgw_requests_total{provider,model,status}` (counter)
- `llmgw_tokens_total{provider,model,kind}` (counter)
- `llmgw_cost_usd_total{provider,model}` (counter)
- `llmgw_request_latency_ms{provider,model}` (histogram)

### Logging

`app/logging_config.py` configures a single stdout handler in `text` (default)
or `json` (`LOG_FORMAT=json`) form, idempotently, and injects the per-request
`request_id` into every record. No `print()` in `app/` (the CLI excepted).

## Data model

SQLAlchemy 2.0 async ORM (`app/models/db.py`); money uses `Numeric` for exact
decimal arithmetic. Migrations: `0001_initial`, `0002_provider_credentials`,
`0003_multikey_monthly_meta`.

- **Organisation** — `id`, `name`, `daily_budget`, `monthly_budget`,
  `created_at`. Has many projects.
- **Project** — `id`, `org_id`, `name`, legacy `key_hash`/`key_prefix`,
  `daily_budget`, `monthly_budget`, `rate_limit_per_min`, `created_at`. Has many
  api_keys, provider_credentials, usage_records.
- **ApiKey** — `id`, `project_id`, `name`, `key_hash` (SHA-256, unique),
  `key_prefix`, `created_at`, `last_used_at`, `revoked_at`. Multiple named,
  soft-revocable gateway keys per project; auth checks these first.
- **ProviderCredential** — `id`, `project_id`, `provider`
  (`openai|anthropic|gemini|azure`), `encrypted_key` (Fernet, never plaintext),
  `meta` (JSON, nullable — e.g. Azure `{endpoint, deployment, api_version}`),
  timestamps. Unique on `(project_id, provider)`.
- **UsageRecord** — one row per completed/failed request: `org_id`,
  `project_id`, `user_id` (from `x-user-id`), `provider`, `model`, token counts,
  `cost` (`Numeric`), `status` (`ok|error|rate_limited|budget_exceeded|cache_hit`),
  `latency_ms`, `request_id`, `created_at`. Indexed on `(org_id, created_at)`
  and `(project_id, created_at)` for the metrics queries.

BYOK keys are encrypted at rest with Fernet (`app/crypto.py`), the key derived
from `SECRET_KEY`. Gateway keys are stored only as a SHA-256 hash plus a
non-secret display prefix; plaintext is returned exactly once at creation.

## Client SDK

`sdk/` is a standalone, fully-typed package (`omnillm`) — sync `Client` and
async `AsyncClient` with identical constructors and method names. It imports
nothing from the server and mirrors the wire schema with its own Pydantic
models. Features: flexible `messages` input, streaming (`chat_stream`, raw text
or `StreamChunk`s), `set_provider_key`, `me`, `models`, `metrics`,
`create_api_key`, a thin OpenAI-shaped `completions` helper, typed errors
(`GatewayError` and subclasses), and built-in retry with backoff + jitter
(honouring `Retry-After`). See `sdk/README.md`.
