"""FastAPI application entrypoint.

Wires routers, static files, logging, request-id propagation, Prometheus
metrics, health probes, and lifecycle hooks. Database schema is managed by
Alembic (run ``alembic upgrade head``), not created here.
"""

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from starlette.responses import JSONResponse, PlainTextResponse

from app import __version__
from app.config import get_settings
from app.db.session import engine
from app.logging_config import configure_logging, get_logger, request_id_ctx
from app.observability import render_prometheus
from app.redis_client import close_redis, redis_client
from app.routers import account, admin, chat, dashboard, keys, metrics, openai_compat

settings = get_settings()
configure_logging(settings.log_level, settings.log_format)
logger = get_logger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("OmniGate v%s starting up", __version__)
    yield
    await close_redis()
    logger.info("OmniGate shutting down")


app = FastAPI(
    title="OmniGate",
    version=__version__,
    summary="One OpenAI-compatible API for OpenAI, Anthropic, Gemini & Azure.",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Bind a request id to the logging context and echo it as a header."""
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex
    token = request_id_ctx.set(rid)
    try:
        response = await call_next(request)
        response.headers.setdefault("x-request-id", rid)
        return response
    finally:
        request_id_ctx.reset(token)


app.include_router(account.router)
app.include_router(chat.router)
app.include_router(openai_compat.router)
app.include_router(metrics.router)
app.include_router(keys.router)
app.include_router(admin.router)
app.include_router(dashboard.router)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/health/live", tags=["meta"])
async def live() -> dict[str, str]:
    """Liveness: the process is up and serving."""
    return {"status": "alive"}


@app.get("/health/ready", tags=["meta"])
async def ready() -> JSONResponse:
    """Readiness: dependencies (Postgres, Redis) are reachable."""
    checks: dict[str, str] = {}
    ok = True
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["database"] = f"error: {exc}"
        ok = False
    try:
        await redis_client.ping()
        checks["redis"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["redis"] = f"error: {exc}"
        ok = False
    return JSONResponse(
        {"status": "ready" if ok else "not_ready", "checks": checks},
        status_code=200 if ok else 503,
    )


@app.get("/version", tags=["meta"])
async def version() -> dict[str, str]:
    return {"name": "omnigate", "version": __version__}


@app.get("/metrics", tags=["meta"], include_in_schema=True)
async def prometheus_metrics() -> PlainTextResponse:
    """Prometheus text exposition of in-process gateway metrics."""
    return PlainTextResponse(
        render_prometheus(), media_type="text/plain; version=0.0.4; charset=utf-8"
    )
