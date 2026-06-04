<div align="center">

# 🛰️ OmniGate

### One OpenAI-compatible API for every LLM provider.

Call **OpenAI, Anthropic, Google Gemini, and Azure OpenAI** through a single self-hosted gateway —
with authentication, per-org/project budgets, rate limiting, automatic fallbacks, response caching,
cost tracking, and a live metrics API.

![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![providers](https://img.shields.io/badge/providers-OpenAI%20·%20Anthropic%20·%20Gemini%20·%20Azure-8A2BE2)
![api](https://img.shields.io/badge/API-OpenAI--compatible-success)

</div>

---

OmniGate sits between your applications and LLM providers. Point your existing **OpenAI SDK** at it and
keep your code unchanged — OmniGate handles routing, keys, spend limits, retries, and observability.

```
            ┌───────────────────────────── OmniGate ─────────────────────────────┐
 your app   │  auth → rate limit → budget → cache → breaker → retry → fallback   │   OpenAI
  (OpenAI ──┼──►  /v1/chat/completions  ·  /v1/chat  ·  /v1/metrics  ·  /metrics  ┼──► Anthropic
   SDK)     │                  cost tracking · BYOK vault · dashboard             │   Gemini · Azure
            └────────────────────────────────────────────────────────────────────┘
```

## ✨ Features

- 🌐 **4 providers, 1 API** — OpenAI, Anthropic, Google Gemini, Azure OpenAI behind one endpoint. Routing is by model name (`gpt-4o`, `claude-3-5-sonnet-latest`, `gemini-1.5-flash`, `azure/<deployment>`).
- 🔌 **Drop-in OpenAI compatibility** — `POST /v1/chat/completions` (JSON **and** SSE streaming). Set your OpenAI SDK's `base_url` and it just works.
- 🔑 **BYOK vault** — each project stores its own provider keys, **encrypted at rest** (Fernet). Their key is charged, not yours.
- 💸 **Budgets & cost tracking** — daily *and* monthly budgets per org and project; every request's cost is computed and recorded.
- 🛟 **Resilience** — automatic retries with backoff, a per-provider **circuit breaker**, and **fallback models** (`fallback_models: [...]`) when a provider fails.
- ⚡ **Response cache** — optional Redis cache for deterministic (`temperature=0`) requests.
- 📊 **Metrics API** — `GET /v1/metrics` returns spend, tokens, latency **p50/p95/p99**, error & cache-hit rates, and breakdowns by provider/model/user/day. Plus a Prometheus `/metrics` endpoint and a live dashboard.
- 🪪 **Multi-key auth** — multiple named API keys per project (create / list / revoke), `x-api-key` or `Authorization: Bearer`.
- 🚦 **Rate limiting** — per-project per-minute window with `x-ratelimit-*` headers.
- 🧰 **Batteries included** — a `pip`-installable client SDK (sync + async), a management CLI, Docker Compose, and Alembic migrations.

## 🚀 Quick start (run the gateway)

```bash
git clone https://github.com/sreekarp/omnigate.git
cd omnigate
cp .env.example .env            # set SECRET_KEY + ADMIN_API_KEY (generate: python -c "import secrets;print(secrets.token_urlsafe(48))")
docker compose up --build       # runs migrations, then serves on :8000
```

Create an account and store your provider key (self-serve, BYOK):

```bash
# 1) Sign up — returns your gateway api_key once
curl -s -X POST localhost:8000/v1/signup -H 'content-type: application/json' \
  -d '{"email":"dev@example.com","project_name":"My App"}'

# 2) Store your own OpenAI key (encrypted server-side)
curl -s -X POST localhost:8000/v1/keys \
  -H 'x-api-key: llmg_...' -H 'content-type: application/json' \
  -d '{"provider":"openai","api_key":"sk-..."}'
```

## 🔁 Use it with the OpenAI SDK (zero code changes)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="llmg_...",          # your OmniGate gateway key
)

resp = client.chat.completions.create(
    model="gpt-4o-mini",          # or claude-3-5-sonnet-latest, gemini-1.5-flash, azure/<deployment>
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

The same key now routes to **any** provider you've configured — just change the `model`.

## 🐍 Use the OmniGate Python SDK

A typed client (sync + async, streaming, retries) for talking to your gateway:

```bash
pip install omnigate           # the client SDK
```

```python
from omnigate import Client

with Client(api_key="llmg_...", base_url="http://localhost:8000") as gw:
    # simple chat
    print(gw.chat(model="gpt-4o-mini", messages="Hello!").content)

    # streaming
    for piece in gw.chat_stream(model="claude-3-5-sonnet-latest", messages="Write a haiku"):
        print(piece, end="", flush=True)

    # catalog + usage metrics
    print(gw.models())
    print(gw.metrics(range="24h"))
```

```python
import asyncio
from omnigate import AsyncClient

async def main():
    async with AsyncClient(api_key="llmg_...", base_url="http://localhost:8000") as gw:
        r = await gw.chat(model="gemini-1.5-flash", messages="Hi")
        print(r.content, r.usage, r.cost_usd)

asyncio.run(main())
```

> **Two packages, one project:** `pip install omnigate` is the **client SDK** (call a running gateway);
> `pip install omnigate-gateway` installs the **server** + the `omnigate-gateway` CLI.

## 🧭 Supported models & routing

| Model prefix | Provider | BYOK key (`POST /v1/keys`) |
|---|---|---|
| `gpt*`, `o1*`, `o3*`, `o4*`, `chatgpt*` | OpenAI | `{"provider":"openai","api_key":"sk-..."}` |
| `claude*` | Anthropic | `{"provider":"anthropic","api_key":"sk-ant-..."}` |
| `gemini*` | Google Gemini | `{"provider":"gemini","api_key":"AIza..."}` |
| `azure/<deployment>` | Azure OpenAI | `{"provider":"azure","api_key":"...","meta":{"endpoint":"https://<res>.openai.azure.com","deployment":"<name>","api_version":"2024-10-21"}}` |

Per-model prices live in `app/services/pricing.py` (USD per 1K tokens) — update them to match current provider pricing.

## 🛟 Resilience & cost controls

```jsonc
// POST /v1/chat — extra knobs (all optional; an OpenAI-compatible request otherwise)
{
  "model": "gpt-4o",
  "messages": [{"role": "user", "content": "..."}],
  "fallback_models": ["claude-3-5-sonnet-latest", "gemini-1.5-flash"],  // tried in order on failure
  "cache": true                                                          // cache deterministic responses
}
```

- **Budgets** — set `daily_budget` / `monthly_budget` on orgs & projects; requests over budget return `402`.
- **Rate limits** — per-project `rate_limit_per_min`; over-limit returns `429` with `Retry-After`.
- **Circuit breaker** — trips per provider after repeated transient failures, then half-opens.

## 📊 Metrics

```bash
curl -s "localhost:8000/v1/metrics?range=24h&group_by=model" -H 'x-api-key: llmg_...'
```

```json
{
  "scope": "project",
  "totals": {
    "requests": 1280, "total_tokens": 935210, "cost_usd": 4.7193,
    "error_rate": 0.012, "cache_hit_rate": 0.21,
    "avg_latency_ms": 812, "p50_latency_ms": 640, "p95_latency_ms": 1980, "p99_latency_ms": 3120
  },
  "breakdown": [{"key": "gpt-4o-mini", "requests": 900, "cost_usd": 1.21, "error_rate": 0.004}],
  "timeseries": [{"bucket": "2026-06-04T00:00:00Z", "requests": 53, "cost_usd": 0.19}]
}
```

- Org-wide (admin): `GET /v1/admin/orgs/{org_id}/metrics` with `x-admin-key`.
- Prometheus: `GET /metrics` exposes `llmgw_requests_total`, `llmgw_tokens_total`, `llmgw_cost_usd_total`, `llmgw_request_latency_ms`.
- Dashboard: open <http://localhost:8000/> (overview) and <http://localhost:8000/dashboard/metrics> (charts).

## 🛠️ Management CLI

```bash
pip install omnigate-gateway          # installs the `omnigate-gateway` command + server

omnigate-gateway serve                # run the API (uvicorn)
omnigate-gateway db upgrade           # apply migrations
omnigate-gateway org create --name Acme --daily-budget 100 --monthly-budget 2000
omnigate-gateway project create --org-id <ID> --name Backend --rate-limit 120   # prints the api key once
omnigate-gateway usage --org-id <ID> --range 7d
omnigate-gateway config-check
```

## ⚙️ Configuration

Required env vars: `DATABASE_URL`, `REDIS_URL`, `SECRET_KEY`, `ADMIN_API_KEY`.
Optional: `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` (BYOK is preferred), `LOG_FORMAT` (`text`|`json`),
`RETRY_*`, `CIRCUIT_BREAKER_*`, `RESPONSE_CACHE_*`. See [`.env.example`](.env.example).

## 🏗️ Architecture

FastAPI (async) · PostgreSQL (asyncpg + SQLAlchemy 2.0) · Redis (rate limits, cache, breaker) ·
httpx · Pydantic v2 · Alembic. The request pipeline is a chain of FastAPI dependencies:
**auth → rate limit → budget → route**. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

```
app/
  providers/     OpenAI · Anthropic · Gemini · Azure adapters + registry
  services/      routing (cache→breaker→retry→fallback) · metrics · pricing · cache · credentials · api_keys · catalog
  middleware/    auth · rate_limit · budget (FastAPI dependencies)
  routers/       chat · openai_compat · metrics · keys · account · admin · dashboard
  observability/ Prometheus metrics
sdk/             omnigate — the installable Python client (sync + async)
```

## 🧪 Development

```bash
python -m venv .venv && source .venv/Scripts/activate    # Windows
pip install -e ".[dev]"
docker compose up -d db redis        # Postgres + Redis
alembic upgrade head
pytest -q                            # server tests
PYTHONPATH=sdk/src pytest sdk/tests -q   # SDK tests
uvicorn app.main:app --reload        # http://localhost:8000  (/docs for OpenAPI)
```

## 🤝 Contributing

Contributions are welcome - see [CONTRIBUTING.md](CONTRIBUTING.md) for local setup, conventions, and how to add a provider. Please run `pytest -q` and `pytest -q sdk/tests` before opening a PR.

## 📄 License

[MIT](LICENSE) (c) Sreekar Paruchuru
