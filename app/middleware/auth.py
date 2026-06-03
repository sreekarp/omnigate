"""Stage 1: Authentication.

Validates the gateway key (``x-api-key`` header, or an ``Authorization: Bearer``
token for OpenAI-SDK compatibility), resolves it to a Project (and its
Organisation), and exposes both via :class:`AuthContext`.

Resolution order:
1. The ``api_keys`` table (multiple named keys per project), unrevoked only.
2. Legacy ``projects.key_hash`` (backward compatibility).
"""

import uuid
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.session import get_session
from app.logging_config import get_logger
from app.models.db import Project
from app.security import hash_api_key
from app.services.api_keys import resolve_api_key, touch_last_used

logger = get_logger(__name__)


@dataclass
class AuthContext:
    """Resolved request identity, passed down the dependency chain."""

    project: Project
    user_id: str | None
    api_key_id: uuid.UUID | None = None


def _extract_key(x_api_key: str | None, authorization: str | None) -> str | None:
    if x_api_key:
        return x_api_key
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[len("bearer ") :].strip() or None
    return None


async def get_auth_context(
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
    authorization: str | None = Header(default=None, alias="authorization"),
    x_user_id: str | None = Header(default=None, alias="x-user-id"),
    session: AsyncSession = Depends(get_session),
) -> AuthContext:
    presented = _extract_key(x_api_key, authorization)
    if not presented:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing x-api-key header (or Authorization: Bearer token)",
        )

    # 1. Multi-key table (preferred).
    resolved = await resolve_api_key(session, presented)
    if resolved is not None:
        project, api_key_id = resolved
        try:
            await touch_last_used(session, api_key_id)
        except Exception:  # noqa: BLE001 - last_used is best-effort
            logger.debug("Failed to touch last_used for api key %s", api_key_id)
        return AuthContext(project=project, user_id=x_user_id, api_key_id=api_key_id)

    # 2. Legacy single-key path.
    key_hash = hash_api_key(presented)
    stmt = (
        select(Project)
        .where(Project.key_hash == key_hash)
        .options(selectinload(Project.organisation))
    )
    project = (await session.execute(stmt)).scalar_one_or_none()
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return AuthContext(project=project, user_id=x_user_id, api_key_id=None)
