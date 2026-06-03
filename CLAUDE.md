# LLM Gateway

A production-grade LLM gateway that sits between applications and LLM 
providers (OpenAI, Anthropic). Adds auth, rate limiting, budget controls, 
per-org/project/user cost tracking, and a live dashboard.

## Architecture
- FastAPI async app
- Postgres (via asyncpg + SQLAlchemy async) for persistent data
- Redis (via aioredis) for rate limiting counters only
- Docker Compose for local dev

## Hierarchy
Organisation → Projects → Users
- Org: top-level billing unit, has global budget
- Project: has its own API key, budget, rate limits
- User: identified by x-user-id header per request

## Request middleware chain (in order)
1. Auth — validate x-api-key → resolve project + org
2. Rate limit — Redis INCR per project per minute window
3. Budget check — Postgres SUM of today's spend vs daily_budget
4. Route — call provider, log result

## Key files
- app/models/db.py — SQLAlchemy models (do not use sync sessions)
- app/providers/base.py — AbstractProvider interface all adapters implement
- app/middleware/ — each middleware is a FastAPI dependency, not a Starlette middleware
- app/routers/chat.py — POST /v1/chat, supports streaming via StreamingResponse

## Env vars needed
DATABASE_URL, REDIS_URL, OPENAI_API_KEY, ANTHROPIC_API_KEY, SECRET_KEY

## Style rules
- Async everywhere (asyncpg, httpx.AsyncClient, aioredis)
- Type hints on all functions
- Pydantic v2 for schemas
- No print() — use Python logging module