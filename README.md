# LLM Gateway

A production-grade gateway between your applications and LLM providers
(OpenAI, Anthropic). It adds authentication, rate limiting, budget controls,
per-org/project/user cost tracking, and a live dashboard.

## Architecture

- **FastAPI** async application
- **Postgres** (asyncpg + SQLAlchemy 2.0 async) for persistent data
- **Redis** (redis.asyncio) for rate-limit counters only
- **Alembic** for schema migrations
- **Docker Compose** for local development

### Hierarchy

```
Organisation  → top-level billing unit, global daily budget
  └── Project → own API key, daily budget, per-minute rate limit
        └── User → identified by the x-user-id header per request
```

### Request pipeline (in order)

Implemented as **chained FastAPI dependencies** (not Starlette middleware):

1. **Auth** — validate `x-api-key` → resolve Project + Organisation
2. **Rate limit** — Redis `INCR` per project per minute window
3. **Budget check** — Postgres `SUM` of today's spend vs daily budgets
4. **Route** — call the provider, then log a usage record

The chat endpoint depends on `enforce_budget`, which depends on
`enforce_rate_limit`, which depends on `get_auth_context` — guaranteeing the
order above.

## Project layout

```
app/
  config.py            # pydantic-settings (env vars)
  logging_config.py    # logging setup (no print())
  security.py          # API key generation / hashing
  redis_client.py      # async Redis (rate limits only)
  db/session.py        # async engine + session dependency
  models/db.py         # SQLAlchemy async models
  schemas/             # Pydantic v2 request/response models
  providers/           # AbstractProvider + OpenAI/Anthropic adapters + registry
  services/            # pricing + usage accounting
  middleware/          # auth → rate_limit → budget dependencies
  routers/             # chat (streaming), admin, dashboard
  templates/ static/   # Jinja2 dashboard
alembic/               # migrations
scripts/seed.py        # demo org/project/key
tests/                 # offline unit tests
```

## Quick start (Docker)

```bash
cp .env.example .env          # then fill in OPENAI_API_KEY / ANTHROPIC_API_KEY etc.
docker compose up --build     # runs migrations, then the API on :8000
```

In a second terminal, seed a demo org + project + API key:

```bash
docker compose exec app python -m scripts.seed
# copy the printed llmg_... key
```

## Quick start (local, without Docker)

Requires a Postgres and Redis running locally (or just use the compose `db`
and `redis` services: `docker compose up db redis`).

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows
pip install -r requirements.txt
cp .env.example .env

alembic upgrade head            # create the schema
python -m scripts.seed          # create demo org/project/key
uvicorn app.main:app --reload   # http://localhost:8000
```

## Self-serve (BYOK) developer flow

The gateway is multi-tenant and **bring-your-own-key**: each developer signs up,
gets their own gateway API key (their unspoofable identity), and stores their
own OpenAI/Anthropic key, which is **encrypted at rest** (Fernet, derived from
`SECRET_KEY`). Their provider key — not yours — is charged.

```bash
# 1. Sign up (public) — returns YOUR gateway api_key once
curl -s -X POST localhost:8000/v1/signup -H "content-type: application/json" \
  -d '{"email":"dev@example.com","project_name":"My App"}'

# 2. Store your own provider key (encrypted server-side)
curl -s -X POST localhost:8000/v1/keys \
  -H "x-api-key: llmg_..." -H "content-type: application/json" \
  -d '{"provider":"openai","api_key":"sk-..."}'

# 3. See your account + which providers you've configured
curl -s localhost:8000/v1/me -H "x-api-key: llmg_..."

# 4. Call the gateway — it uses YOUR stored key
curl -s -X POST localhost:8000/v1/chat \
  -H "x-api-key: llmg_..." -H "content-type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Hello!"}]}'
```

`/v1/chat` returns `400` until a key for the relevant provider is stored. The
`/v1/admin/*` endpoints still exist for operator-side provisioning.

## Using the gateway

### Provision via the admin API (alternative to the seed script)

```bash
# Create an organisation
curl -s -X POST localhost:8000/v1/admin/orgs \
  -H "x-admin-key: $ADMIN_API_KEY" -H "content-type: application/json" \
  -d '{"name":"Acme","daily_budget":"100.00"}'

# Create a project (returns the plaintext api_key ONCE)
curl -s -X POST localhost:8000/v1/admin/projects \
  -H "x-admin-key: $ADMIN_API_KEY" -H "content-type: application/json" \
  -d '{"org_id":"<ORG_ID>","name":"Backend","daily_budget":"10.00","rate_limit_per_min":60}'
```

### Make a chat request

```bash
curl -s -X POST localhost:8000/v1/chat \
  -H "x-api-key: llmg_..." \
  -H "x-user-id: alice@acme.com" \
  -H "content-type: application/json" \
  -d '{
        "model": "gpt-4o-mini",
        "messages": [{"role":"user","content":"Hello!"}]
      }'
```

Streaming (plain-text chunks):

```bash
curl -N -X POST localhost:8000/v1/chat \
  -H "x-api-key: llmg_..." -H "content-type: application/json" \
  -d '{"model":"claude-3-5-sonnet-latest","messages":[{"role":"user","content":"Hi"}],"stream":true}'
```

### Dashboard

Open <http://localhost:8000/> for the org overview, and click into a project
for per-request detail. Interactive API docs are at `/docs`.

## Model routing

The provider is chosen by model-name prefix (`app/providers/registry.py`):

| Prefix             | Provider  |
|--------------------|-----------|
| `gpt`, `o1`, `o3`  | OpenAI    |
| `claude`           | Anthropic |

Add new providers by implementing `AbstractProvider` and registering them.

## Pricing

Per-model token prices live in `app/services/pricing.py` (USD per 1K tokens).
Unknown models record a cost of 0 and log a warning — update the table to keep
budgets accurate. Note: streaming requests record token counts as 0 because
providers don't reliably return usage mid-stream.

## Tests

The offline unit tests (pricing, routing, security) need no DB/Redis:

```bash
pytest tests/test_pricing.py tests/test_routing.py tests/test_security.py -q
```

## Environment variables

See `.env.example`. Required: `DATABASE_URL`, `REDIS_URL`, `SECRET_KEY`,
`ADMIN_API_KEY`. Provider keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) are
needed only for the providers you actually call.
