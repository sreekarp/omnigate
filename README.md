# LLM Gateway

A production-grade gateway between your applications and LLM providers. It adds
authentication, rate limiting, daily **and** monthly budget controls,
per-org/project/user cost tracking, resilience (retry, circuit breaking,
multi-model fallback, response caching), a metrics API, Prometheus exposition,
an OpenAI-compatible API surface, a typed client SDK, and a live dashboard.

Providers supported: **OpenAI**, **Anthropic**, **Google Gemini**, and
**Azure OpenAI** — each callable bring-your-own-key (BYOK), per project.

## Architecture

- **FastAPI** async application
- **Postgres** (asyncpg + SQLAlchemy 2.0 async) for persistent data
- **Redis** (`redis.asyncio`) for rate-limit counters, the response cache, and
  the optional Redis circuit-breaker backend
- **Alembic** for schema migrations
- **Docker Compose** for local development
- Packaged with `pyproject.toml`; ships a console script `llm-gateway`

### Hierarchy

```
Organisation  → top-level billing unit, daily + monthly budget
  └── Project → its own gateway API key(s), daily + monthly budget, rate limit
        └── User → identified by the x-user-id header per request (cost attribution)
```

### Request pipeline (in order)

Implemented as **chained FastAPI dependencies** (not Starlette middleware):

1. **Auth** — validate the gateway key (`x-api-key` **or** `Authorization: Bearer`)
   → resolve Project + Organisation
2. **Rate limit** — Redis `INCR` per project per minute window
3. **Budget check** — Postgres `SUM` of today's / this-month's spend vs the
   daily and monthly budgets (a budget of `0` means unlimited)
4. **Route** — resolve the model to a provider + the project's BYOK key, apply
   resilience (cache → circuit breaker → retry → fallback), call the provider,
   then log a usage record

`/v1/chat` and `/v1/chat/completions` depend on `enforce_budget`, which depends
on `enforce_rate_limit`, which depends on `get_auth_context` — guaranteeing the
order above.

## Quick start (Docker)

```bash
cp .env.example .env          # fill in SECRET_KEY / ADMIN_API_KEY; provider keys are BYOK
docker compose up --build     # runs migrations, then the API on :8000
```

In a second terminal, seed a demo org + project + gateway key:

```bash
docker compose exec app python -m scripts.seed
# copy the printed llmg_... key
```

## Quick start (local, without Docker)

Requires Postgres and Redis (or just `docker compose up db redis`).

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows
pip install -r requirements.txt
cp .env.example .env

alembic upgrade head            # create the schema
python -m scripts.seed          # create a demo org/project/key
uvicorn app.main:app --reload   # http://localhost:8000

# …or use the CLI (installed by `pip install -e .`):
llm-gateway db upgrade
llm-gateway serve --reload
```

## Providers & model routing

The provider is chosen by model-name prefix (`app/providers/registry.py`):

| Model prefix / form                          | Provider  | `owned_by` |
|----------------------------------------------|-----------|------------|
| `gpt-*`, `o1*`, `o3*`, `o4*`, `chatgpt*`     | OpenAI    | `openai`   |
| `claude-*`                                   | Anthropic | `anthropic`|
| `gemini-*`, `models/gemini-*`                | Gemini    | `google`   |
| `azure/<deployment>`, `azure-*`              | Azure OpenAI | `azure` |

Examples: `gpt-4o-mini`, `claude-3-5-sonnet-latest`, `gemini-1.5-flash`,
`azure/my-gpt4o-deployment`. For Azure the `<deployment>` after `azure/`
selects the deployment, overriding `meta.deployment`.

Add a provider by implementing `AbstractProvider` (`app/providers/base.py`) and
registering it in the registry. Every adapter implements
`async chat(request, api_key) -> ChatResponse` and an async-generator
`stream(request, api_key)` yielding `StreamChunk`s (content chunks followed by
exactly one terminal usage chunk).

## Bring-your-own-key (BYOK)

The gateway is multi-tenant: each developer signs up, gets their own gateway API
key (their unspoofable identity), and stores their own provider keys, which are
**encrypted at rest** (Fernet, derived from `SECRET_KEY`). Their provider key —
not yours — is charged.

```bash
# 1. Sign up (public) — returns your gateway api_key once
curl -s -X POST localhost:8000/v1/signup -H "content-type: application/json" \
  -d '{"email":"dev@example.com","project_name":"My App"}'

# 2. Store an OpenAI key (provider: openai | anthropic | gemini | azure)
curl -s -X POST localhost:8000/v1/keys \
  -H "x-api-key: llmg_..." -H "content-type: application/json" \
  -d '{"provider":"openai","api_key":"sk-..."}'

# 3. Store an Azure OpenAI key — requires meta.endpoint + meta.deployment
curl -s -X POST localhost:8000/v1/keys \
  -H "x-api-key: llmg_..." -H "content-type: application/json" \
  -d '{"provider":"azure","api_key":"<azure-key>",
       "meta":{"endpoint":"https://my-res.openai.azure.com",
               "deployment":"my-gpt4o","api_version":"2024-10-21"}}'

# 4. See your account + which providers are configured
curl -s localhost:8000/v1/me -H "x-api-key: llmg_..."
```

`/v1/chat` returns `400` until a key for the relevant provider is stored. The
`/v1/admin/*` endpoints remain for operator-side provisioning (`x-admin-key`).

## Making chat requests

### Native endpoint — `POST /v1/chat`

Returns the gateway's `ChatResponse` JSON (non-streaming) or a `text/plain`
chunked stream (when `"stream": true`). Echoes `x-request-id` and
`x-llmgw-provider` / `x-llmgw-model` / `x-llmgw-cached` / `x-llmgw-fallback`
response headers.

```bash
curl -s -X POST localhost:8000/v1/chat \
  -H "x-api-key: llmg_..." -H "x-user-id: alice@acme.com" \
  -H "content-type: application/json" \
  -d '{
        "model": "gpt-4o-mini",
        "messages": [{"role":"user","content":"Hello!"}],
        "fallback_models": ["claude-3-5-haiku-latest", "gemini-1.5-flash"],
        "temperature": 0,
        "cache": true
      }'
```

`ChatResponse`:

```json
{
  "id": "…", "provider": "openai", "model": "gpt-4o-mini",
  "content": "Bonjour !",
  "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
  "cost_usd": 0.0000033, "finish_reason": "stop",
  "cached": false, "fallback_used": false, "latency_ms": 412
}
```

Streaming (plain-text chunks; primary model only, no fallback/retry/cache):

```bash
curl -N -X POST localhost:8000/v1/chat \
  -H "x-api-key: llmg_..." -H "content-type: application/json" \
  -d '{"model":"claude-3-5-sonnet-latest","messages":[{"role":"user","content":"Hi"}],"stream":true}'
```

### OpenAI-compatible endpoint — `POST /v1/chat/completions`

Wire-compatible with OpenAI Chat Completions (JSON, plus SSE streaming
terminated by `data: [DONE]`). Point the official `openai` SDK at the gateway:

```python
from openai import OpenAI

oai = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="llmg_...",                       # your gateway key (sent as bearer)
    default_headers={"x-user-id": "alice@acme.com"},
)
print(oai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hi"}],
).choices[0].message.content)
```

Pass `{"stream": true, "stream_options": {"include_usage": true}}` to get a
terminal usage chunk in the SSE stream. The same pipeline (auth → rate limit →
budget → resilience → usage recording) applies, so cost tracking and budgets
work identically for compat traffic.

### Model catalog

```bash
curl -s localhost:8000/v1/models -H "x-api-key: llmg_..."
# {"object":"list","data":[{ "id":"gpt-4o-mini","object":"model","created":0,
#   "owned_by":"openai","provider":"openai",
#   "pricing":{"input_per_1k_usd":0.00015,"output_per_1k_usd":0.0006} }, …]}

curl -s localhost:8000/v1/models/gpt-4o-mini -H "x-api-key: llmg_..."
```

## Resilience

Non-streaming `/v1/chat` and `/v1/chat/completions` flow through
`app/services/routing.py::execute_chat`:

```
cache lookup → for each candidate model: resolve → circuit breaker → retry → cache store
```

- **Retry** (`app/utils/resilience.py`) — exponential backoff + full jitter on
  transient failures (HTTP 408/429/500/502/503/504, httpx timeouts/transport
  errors); honours a provider `Retry-After` hint when larger than the computed
  delay.
- **Circuit breaker** (`app/services/circuit_breaker.py`) — per
  `provider:project` breaker (closed → open → half-open). In-memory or
  Redis-backed. Only transient failures trip it; 4xx client errors skip to the
  next candidate without tripping.
- **Fallback** — `fallback_models` (up to 5) are tried in order; each may
  resolve to a different provider with its own BYOK key. `fallback_used` is set
  on the response. Streaming uses the primary model only.
- **Response cache** (`app/services/cache.py`) — opt-in Redis cache for
  deterministic (`temperature == 0`), non-streaming requests, scoped per
  project + provider. Enabled globally via `RESPONSE_CACHE_ENABLED` or
  per-request via `"cache": true`; cache hits are billed `$0` and marked
  `"cached": true`. All cache access is non-fatal.

Tunables (all optional, see `.env.example`):

| Env var | Default | Purpose |
|---|---|---|
| `RETRY_MAX_ATTEMPTS` | `3` | total attempts per provider call |
| `RETRY_BASE_DELAY_SECONDS` | `0.25` | base backoff |
| `RETRY_MAX_DELAY_SECONDS` | `8.0` | backoff cap |
| `RETRY_JITTER_SECONDS` | `0.25` | uniform jitter added to backoff |
| `CIRCUIT_BREAKER_ENABLED` | `true` | enable the breaker |
| `CIRCUIT_BREAKER_FAIL_THRESHOLD` | `5` | consecutive failures to open |
| `CIRCUIT_BREAKER_COOLDOWN_SECONDS` | `30` | open-state cooldown |
| `CIRCUIT_BREAKER_BACKEND` | `memory` | `memory` or `redis` |
| `RESPONSE_CACHE_ENABLED` | `false` | enable the response cache globally |
| `RESPONSE_CACHE_TTL_SECONDS` | `300` | cache entry TTL |
| `RESPONSE_CACHE_PREFIX` | `respcache:` | Redis key prefix |

## Metrics & observability

### Usage analytics — `GET /v1/metrics`

Project-scoped (authenticated by the gateway key). Query params:
`range=1h|24h|7d|30d` **or** `from`/`to` (ISO-8601); optional
`group_by=provider|model|user|status` and `granularity=minute|hour|day`.

```bash
curl -s "localhost:8000/v1/metrics?range=7d&group_by=model" -H "x-api-key: llmg_..."
```

```json
{
  "scope": "project", "scope_id": "…",
  "range_from": "2026-05-28T00:00:00Z", "range_to": "2026-06-04T00:00:00Z",
  "group_by": "model", "granularity": "hour",
  "totals": {
    "requests": 1280, "prompt_tokens": 410233, "completion_tokens": 188190,
    "total_tokens": 598423, "cost_usd": 4.182931,
    "error_rate": 0.012, "cache_hit_rate": 0.18, "avg_latency_ms": 642.5,
    "p50_latency_ms": 520.0, "p95_latency_ms": 1480.0, "p99_latency_ms": 2310.0
  },
  "breakdown": [
    {"key": "gpt-4o-mini", "requests": 900, "total_tokens": 410000,
     "cost_usd": 2.91, "avg_latency_ms": 600.0, "error_rate": 0.01}
  ],
  "timeseries": [
    {"bucket": "2026-06-04T09:00:00Z", "requests": 73, "total_tokens": 31000,
     "cost_usd": 0.21, "error_rate": 0.0}
  ]
}
```

### Org-wide analytics — `GET /v1/admin/orgs/{org_id}/metrics`

Same payload, scoped to an organisation. Requires `x-admin-key`.

### Prometheus — `GET /metrics`

Hand-rolled text exposition (no extra dependency) of in-process counters and a
latency histogram, labelled by `provider`/`model`/`status`:
`llmgw_requests_total`, `llmgw_tokens_total`, `llmgw_cost_usd_total`,
`llmgw_request_latency_ms`.

## Multi-key gateway management

Each project can mint multiple named gateway keys (rotate without downtime).
These are distinct from BYOK provider keys.

```bash
curl -s -X POST localhost:8000/v1/keys/api \
  -H "x-api-key: llmg_..." -H "content-type: application/json" \
  -d '{"name":"ci-pipeline"}'           # returns api_key plaintext ONCE
curl -s localhost:8000/v1/keys/api -H "x-api-key: llmg_..."     # list (no secrets)
curl -s -X DELETE localhost:8000/v1/keys/api/<key_id> -H "x-api-key: llmg_..."  # soft-revoke
```

## Budgets

Both organisations and projects carry a `daily_budget` and a `monthly_budget`
(USD, UTC; `0` = unlimited). Exceeding any configured limit returns
`402 Payment Required`. Current spend vs limits is echoed in `x-budget-*`
response headers on every chat call.

## Health, readiness & version

| Endpoint | Purpose |
|---|---|
| `GET /health` | basic ok + version |
| `GET /health/live` | liveness (process up) |
| `GET /health/ready` | readiness — checks Postgres + Redis (`503` if down) |
| `GET /version` | `{"name":"llm-gateway","version":"…"}` |

## CLI

Installed as `llm-gateway` (stdlib argparse, no extra deps):

```bash
llm-gateway serve [--host H] [--port P] [--reload]
llm-gateway db upgrade|downgrade [--rev REV]
llm-gateway config-check
llm-gateway version
llm-gateway org create --name N [--daily-budget D] [--monthly-budget M]
llm-gateway org list
llm-gateway project create --org-id ID --name N [--daily-budget D] [--monthly-budget M] [--rate-limit N]
llm-gateway project list [--org-id ID]
llm-gateway usage (--org-id ID | --project-id ID) [--range 24h]
```

## Client SDK

A standalone, fully-typed Python client (`sdk/`) — sync **and** async,
streaming-aware, with typed errors. It imports nothing from the server.

```bash
pip install ./sdk
```

```python
from llm_gateway import Client

with Client(api_key="llmg_...", base_url="http://localhost:8000", user_id="u-42") as c:
    c.set_provider_key(provider="openai", api_key="sk-...")     # POST /v1/keys
    resp = c.chat(model="gpt-4o-mini", messages="Say hi in French")
    print(resp.content, resp.usage.total_tokens, resp.cost_usd)

    for piece in c.chat_stream(model="gpt-4o-mini", messages="Stream a haiku"):
        print(piece, end="", flush=True)
```

```python
import asyncio
from llm_gateway import AsyncClient

async def main() -> None:
    async with AsyncClient(api_key="llmg_...", base_url="http://localhost:8000") as c:
        async for chunk in c.chat_stream(
            model="claude-3-5-sonnet-latest",
            messages=[{"role": "user", "content": "hi"}],
            as_chunks=True,
        ):
            print(chunk.text, end="")

asyncio.run(main())
```

Errors derive from `GatewayError` (`AuthError` 401, `RateLimitError` 429 with
`.retry_after`, `BudgetExceededError` 402, `ProviderError`, `APIError`,
`ConnectionError`); 429/5xx/timeouts are retried with backoff + jitter. See
`sdk/README.md` for the full reference.

## Dashboard

Open <http://localhost:8000/> for the org overview, then click into a project
for per-request detail. Interactive API docs are at `/docs`.

## Pricing

Per-model token prices live in `app/services/pricing.py` (USD per 1K tokens),
covering OpenAI, Anthropic, and Gemini models. Resolution is forgiving: an
`azure/` prefix is stripped and dated/version suffixes (e.g. `-2024-08-06`,
`-002`) are trimmed so versioned ids price correctly. Unknown models record a
cost of `0` and log a warning.

## Tests

The offline unit tests (pricing, routing, security, resilience, …) need no
DB/Redis and use `httpx.MockTransport` for provider HTTP (no network):

```bash
pytest tests/ -q
```

The SDK has its own suite:

```bash
pytest sdk/tests/ -q
```

## Environment variables

See `.env.example`. Required: `DATABASE_URL`, `REDIS_URL`, `SECRET_KEY`,
`ADMIN_API_KEY`. Provider keys are BYOK (stored per project); `OPENAI_API_KEY` /
`ANTHROPIC_API_KEY` env vars are optional fallbacks only. The retry,
circuit-breaker, response-cache, and `LOG_FORMAT` vars are all optional with the
defaults shown above.

## Packaging

`pyproject.toml` (hatchling) packages the `app` package, exposes the
`llm-gateway` console script, and declares runtime + `dev` (pytest,
pytest-asyncio) dependencies. The SDK is a separate distribution under `sdk/`.
