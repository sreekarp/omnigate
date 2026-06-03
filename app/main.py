"""FastAPI application entrypoint.

Wires routers, static files, logging, and lifecycle hooks. Database schema is
managed by Alembic (run `alembic upgrade head`), not created here.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.config import get_settings
from app.logging_config import configure_logging, get_logger
from app.redis_client import close_redis
from app.routers import account, admin, chat, dashboard

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("LLM Gateway v%s starting up", __version__)
    yield
    await close_redis()
    logger.info("LLM Gateway shutting down")


app = FastAPI(title="LLM Gateway", version=__version__, lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

app.include_router(account.router)
app.include_router(chat.router)
app.include_router(admin.router)
app.include_router(dashboard.router)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
