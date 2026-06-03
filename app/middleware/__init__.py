"""Request-pipeline dependencies (FastAPI Depends), not Starlette middleware.

Chained so the resolution order matches CLAUDE.md:
    auth -> rate limit -> budget

Import ``enforce_budget`` into the chat router; it transitively runs the
earlier stages.
"""

from starlette.responses import Response

from app.middleware.auth import AuthContext, get_auth_context
from app.middleware.budget import enforce_budget
from app.middleware.rate_limit import enforce_rate_limit

__all__ = [
    "AuthContext",
    "get_auth_context",
    "enforce_rate_limit",
    "enforce_budget",
    "propagate_gateway_headers",
]

# Prefixes of headers set by the rate-limit / budget dependencies on the
# FastAPI-injected Response. When an endpoint returns its OWN Response (JSON or
# Streaming), FastAPI discards the injected one, so these must be copied over.
_PASSTHROUGH_PREFIXES = ("x-ratelimit", "x-budget")


def propagate_gateway_headers(src: Response, dst: Response) -> None:
    """Copy rate-limit/budget headers from the injected ``src`` onto ``dst``."""
    for name, value in src.headers.items():
        if name.lower().startswith(_PASSTHROUGH_PREFIXES):
            dst.headers.setdefault(name, value)
