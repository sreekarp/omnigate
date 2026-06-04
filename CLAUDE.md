# OmniGate

A production-grade OmniGate that sits between applications and LLM
providers (OpenAI, Anthropic, Google Gemini, Azure OpenAI). Adds auth, rate
limiting, daily + monthly budget controls, per-org/project/user cost tracking,
resilience (retry, circuit breaker, fallback, response cache), a metrics API,
Prometheus exposition, an OpenAI-compatible API, a typed client SDK, and a live
dashboard.

## Architecture
- FastAPI async app
- Postgres (via asyncpg + SQLAlchemy 2.0 async) for persistent data
- Redis (via redis.asyncio) for rate-limit counters, the response cache, and
  the optional Redis circuit-breaker backend
- Alembic for schema migrations
- Docker Compose for local dev
- Packaged with pyproject.toml; ships an `omnigate-gateway` console script

## Hierarchy
Organisation → Projects → Users
- Org: top-level billing unit, has daily + monthly budget
- Project: has its own gateway API key(s), daily + monthly budget, rate limit
- User: identified by x-user-id header per request (cost attribution)

## Providers
- All adapters implement AbstractProvider (app/providers/base.py):
  `async chat(request, api_key) -> ChatResponse` and an async-generator
  `stream(request, api_key)` yielding StreamChunk(text, usage, finish_reason,
  model) — content chunks then exactly one terminal usage chunk.
- openai.py — OpenAI; module helpers build_chat_payload / parse_chat_response /
  parse_stream_chunk are reused by the Azure adapter.
- anthropic.py — Anthropic Messages API.
- gemini.py — GeminiProvider(); Google Generative Language API (v1beta).
- azure_openai.py — AzureOpenAIProvider(*, endpoint, deployment, api_version);
  OpenAI-compatible wire format, constructed per-request (NOT process-cached)
  from the project's stored credential meta.
- registry.py — model-prefix → provider routing + lazy cached instances.

## Request middleware chain (in order)
Chained FastAPI dependencies (NOT Starlette middleware):
1. Auth — validate gateway key (x-api-key OR Authorization: Bearer) → resolve
   project + org. Resolves api_keys table first, then legacy projects.key_hash.
2. Rate limit — Redis INCR per project per minute window
3. Budget check — Postgres SUM of today's + this-month's spend vs daily/monthly
   budgets (0 = unlimited); 402 on exceed; x-budget-* headers
4. Route — resolve model→provider + BYOK key, apply resilience, call provider,
   log usage

## Key files
- app/models/db.py — SQLAlchemy 2.0 async models: Organisation, Project, ApiKey,
  ProviderCredential (with `meta`), UsageRecord (do not use sync sessions)
- app/config.py — pydantic-settings; resilience/cache/log_format tunables
- app/providers/base.py — AbstractProvider + ProviderError(status_code, retry_after)
- app/providers/{openai,anthropic,gemini,azure_openai}.py — adapters
- app/providers/registry.py — model→provider routing
- app/services/routing.py — execute_chat orchestrator (cache→breaker→retry→fallback)
- app/services/cache.py — opt-in Redis response cache (deterministic only)
- app/services/circuit_breaker.py — per provider:project breaker (memory/redis)
- app/services/metrics.py — UsageRecord aggregation for the metrics API
- app/services/catalog.py — model cards for /v1/models
- app/services/api_keys.py — multi-key gateway-key CRUD + auth resolution
- app/services/credentials.py — BYOK encrypted provider-key storage (+ meta)
- app/services/pricing.py — per-model USD/1k token pricing + compute_cost
- app/utils/resilience.py — RetryPolicy + retry_async (backoff + jitter)
- app/observability/prometheus.py — observe_request + render_prometheus
- app/middleware/ — each middleware is a FastAPI dependency, not Starlette
- app/routers/chat.py — POST /v1/chat (text/plain streaming via StreamingResponse)
- app/routers/openai_compat.py — POST /v1/chat/completions (JSON + SSE [DONE]),
  GET /v1/models, GET /v1/models/{id}
- app/routers/metrics.py — GET /v1/metrics (project-scoped analytics)
- app/routers/keys.py — POST/GET/DELETE /v1/keys/api (gateway key mgmt)
- app/routers/account.py — POST /v1/signup, POST /v1/keys (BYOK), GET /v1/me
- app/routers/admin.py — /v1/admin/* provisioning + org metrics (x-admin-key)
- app/main.py — app wiring, request-id middleware, /health[/live|/ready],
  /version, /metrics (Prometheus)
- app/cli.py — `omnigate-gateway` CLI (serve, db, config-check, org, project, usage)
- sdk/ — standalone typed client SDK (Client, AsyncClient); imports nothing from app

## Env vars
Required: DATABASE_URL, REDIS_URL, SECRET_KEY, ADMIN_API_KEY.
Optional: OPENAI_API_KEY, ANTHROPIC_API_KEY (BYOK is preferred),
DEFAULT_RATE_LIMIT_PER_MIN, REQUEST_TIMEOUT_SECONDS, LOG_LEVEL, LOG_FORMAT,
RETRY_*, CIRCUIT_BREAKER_*, RESPONSE_CACHE_*. See .env.example.

## Style rules
- Async everywhere (asyncpg, httpx.AsyncClient, redis.asyncio)
- Type hints on all functions
- Pydantic v2 for schemas
- No print() in app/ — use the Python logging module (app/logging_config.py).
  The CLI (app/cli.py) is the only place print() is acceptable (user-facing).
