"""Async Redis client, used ONLY for rate-limit counters (see CLAUDE.md).

`redis.asyncio` is the modern home of what used to be the separate `aioredis`
package; it ships inside `redis>=4.2`.
"""

from collections.abc import AsyncIterator

import redis.asyncio as redis

from app.config import get_settings

settings = get_settings()

# A single connection pool shared across the process.
redis_client: redis.Redis = redis.from_url(
    settings.redis_url,
    encoding="utf-8",
    decode_responses=True,
)


async def get_redis() -> AsyncIterator[redis.Redis]:
    """FastAPI dependency yielding the shared Redis client."""
    yield redis_client


async def close_redis() -> None:
    """Close the Redis connection pool on shutdown."""
    await redis_client.aclose()
