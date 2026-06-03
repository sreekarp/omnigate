"""Request-pipeline dependencies (FastAPI Depends), not Starlette middleware.

Chained so the resolution order matches CLAUDE.md:
    auth -> rate limit -> budget

Import ``enforce_budget`` into the chat router; it transitively runs the
earlier stages.
"""

from app.middleware.auth import AuthContext, get_auth_context
from app.middleware.budget import enforce_budget
from app.middleware.rate_limit import enforce_rate_limit

__all__ = [
    "AuthContext",
    "get_auth_context",
    "enforce_rate_limit",
    "enforce_budget",
]
