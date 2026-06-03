"""Stage 2: Rate limiting.

Fixed-window counter in Redis: one key per project per UTC minute, INCR'd on
each request and expired after the window. Depends on auth so it only runs for
authenticated requests.
"""

from datetime import datetime, timezone

import redis.asyncio as redis
from fastapi import Depends, HTTPException, status

from app.logging_config import get_logger
from app.middleware.auth import AuthContext, get_auth_context
from app.redis_client import get_redis

logger = get_logger(__name__)

_WINDOW_SECONDS = 60


async def enforce_rate_limit(
    ctx: AuthContext = Depends(get_auth_context),
    r: redis.Redis = Depends(get_redis),
) -> AuthContext:
    project = ctx.project
    minute = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    key = f"ratelimit:{project.id}:{minute}"

    # INCR then set TTL on first hit of the window.
    current = await r.incr(key)
    if current == 1:
        await r.expire(key, _WINDOW_SECONDS)

    if current > project.rate_limit_per_min:
        logger.info(
            "Rate limit exceeded for project %s (%s/%s)",
            project.id,
            current,
            project.rate_limit_per_min,
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(_WINDOW_SECONDS)},
        )

    return ctx
