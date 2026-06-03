"""Stage 1: Authentication.

Validates the ``x-api-key`` header, resolves it to a Project (and its
Organisation), and exposes both via :class:`AuthContext`.
"""

from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.session import get_session
from app.models.db import Project
from app.security import hash_api_key


@dataclass
class AuthContext:
    """Resolved request identity, passed down the dependency chain."""

    project: Project
    user_id: str | None


async def get_auth_context(
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
    x_user_id: str | None = Header(default=None, alias="x-user-id"),
    session: AsyncSession = Depends(get_session),
) -> AuthContext:
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing x-api-key header",
        )

    key_hash = hash_api_key(x_api_key)
    stmt = (
        select(Project)
        .where(Project.key_hash == key_hash)
        .options(selectinload(Project.organisation))
    )
    result = await session.execute(stmt)
    project = result.scalar_one_or_none()

    if project is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return AuthContext(project=project, user_id=x_user_id)
