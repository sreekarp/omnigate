# Changelog

All notable changes to the OmniLLM are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-06-04

A large feature release: two new providers, a full resilience layer, a metrics
API, Prometheus exposition, an OpenAI-compatible surface, multi-key auth,
monthly budgets, a client SDK, packaging, and a management CLI. All additions
are backward-compatible — existing `/v1/chat` clients and stored data continue
to work unchanged.

### Added

- **Providers**
  - Google **Gemini** adapter (`app/providers/gemini.py`) — v1beta Generative
    Language API, `x-goog-api-key` auth, SSE streaming, absolute `usageMetadata`.
  - **Azure OpenAI** adapter (`app/providers/azure_openai.py`) — OpenAI-compatible
    wire format, `api-key` auth, per-request construction from stored
    endpoint/deployment/api-version, reusing the OpenAI payload/parse helpers.
  - Model routing extended for `gemini-*` and `azure/<deployment>`
    (`app/providers/registry.py`); pricing for Gemini models added.
- **Streaming usage** — OpenAI/Anthropic/Gemini/Azure adapters now emit a
  terminal usage chunk, so streamed requests record real token counts and cost
  (previously recorded as zero). Unified `StreamChunk` schema.
- **Resilience layer**
  - Async **retry** with exponential backoff + full jitter and `Retry-After`
    awareness (`app/utils/resilience.py`).
  - Per-`provider:project` **circuit breaker** with in-memory and Redis backends
    (`app/services/circuit_breaker.py`).
  - **Fallback models** — `ChatRequest.fallback_models` (≤5) tried in order,
    possibly across providers; `fallback_used` surfaced on the response.
  - Opt-in Redis **response cache** for deterministic, non-streaming requests
    (`app/services/cache.py`); cache hits billed `$0` and marked `cached`.
  - Orchestrated by `app/services/routing.py::execute_chat`.
- **Metrics & observability**
  - `GET /v1/metrics` — project-scoped analytics: totals (incl. p50/p95/p99
    latency, error/cache-hit rates), grouped breakdown, and a bucketed
    timeseries (`app/services/metrics.py`, `app/routers/metrics.py`).
  - `GET /v1/admin/orgs/{id}/metrics` — org-scoped analytics (`x-admin-key`).
  - `GET /metrics` — hand-rolled Prometheus text exposition (no new dependency).
  - Per-request `x-request-id` propagation and request-id-aware logging, plus a
    structured JSON log format via `LOG_FORMAT=json`.
- **OpenAI-compatible API**
  - `POST /v1/chat/completions` — JSON + SSE (`data: [DONE]`), usable with the
    official `openai` SDK by pointing `base_url` at the gateway.
  - `GET /v1/models` and `GET /v1/models/{id}` — catalog cards with pricing.
- **Auth & budgets**
  - Multiple named, soft-revocable gateway keys per project
    (`POST/GET/DELETE /v1/keys/api`; `api_keys` table; `last_used_at` tracking).
  - `Authorization: Bearer <key>` accepted in addition to `x-api-key`.
  - **Monthly budgets** on organisations and projects, alongside daily budgets;
    `x-budget-*` response headers expanded.
- **BYOK** now covers `openai|anthropic|gemini|azure`, with an optional non-secret
  `meta` blob (`provider_credentials.meta`) for Azure
  `{endpoint, deployment, api_version}`.
- **Health/ops** — `GET /health/live`, `GET /health/ready` (Postgres + Redis
  checks), and `GET /version`.
- **Client SDK** (`sdk/`) — standalone, fully-typed `Client` + `AsyncClient`,
  streaming-aware, typed errors, built-in retry; depends only on httpx +
  pydantic. `pip install ./sdk`.
- **CLI** — `omnillm-gateway` console script (`app/cli.py`): `serve`, `db
  upgrade/downgrade`, `config-check`, `version`, `org`, `project`, `usage`.
- **Packaging** — `pyproject.toml` (hatchling) for the server with the
  `omnillm-gateway` entry point; separate SDK distribution.
- New resilience/cache/logging configuration env vars (`RETRY_*`,
  `CIRCUIT_BREAKER_*`, `RESPONSE_CACHE_*`, `LOG_FORMAT`) — all optional with
  sensible defaults; documented in `.env.example` and passed through in
  `docker-compose.yml`.

### Changed

- `UsageRecord.status` now also encodes `cache_hit` (powers `cache_hit_rate`
  with zero schema cost).
- `Redis` is no longer used solely for rate limiting; it also backs the response
  cache and the optional circuit-breaker backend.
- README, CLAUDE.md, and the new `docs/ARCHITECTURE.md` rewritten to reflect the
  multi-provider, resilient, observable architecture.

### Migrations

- `0003_multikey_monthly_meta` — adds `provider_credentials.meta`, the `api_keys`
  table, and `monthly_budget` on `organisations` and `projects`. Run
  `alembic upgrade head` (or `omnillm-gateway db upgrade`).

## [0.1.0]

- Initial release: FastAPI gateway with OpenAI and Anthropic providers,
  gateway-key auth, per-project Redis rate limiting, daily budgets,
  per-org/project/user cost tracking, `POST /v1/chat` (with plain-text
  streaming), self-serve signup + single-key BYOK vault (Fernet-encrypted),
  admin provisioning endpoints, and a Jinja2 dashboard.
