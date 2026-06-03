"""Business logic over the :class:`~app.models.db.ApiKey` model.

Provides CRUD + auth resolution for named, per-project gateway API keys
(multiple keys per project, soft-revocable). Plaintext keys are returned
exactly once on creation; only a SHA-256 hash and a non-secret display prefix
are persisted.

Auth resolution (:func:`resolve_api_key`) hashes the presented key and matches
an *unrevoked* ``api_keys`` row, eager-loading the owning project and its
organisation. Callers fall back to the legacy ``projects.key_hash`` lookup when
this returns ``None``.
"""

import uuid

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.sql import func

from app.models.db import ApiKey, Project
from app.security import generate_api_key, hash_api_key, key_display_prefix


async def create_api_key(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    name: str,
) -> tuple[ApiKey, str]:
    """Create a new gateway API key for a project.

    Generates a fresh plaintext key, stores its hash + non-secret prefix, and
    returns ``(row, plaintext)``. The plaintext is shown only here and is never
    recoverable afterwards. Commits the new row so the caller can rely on
    ``row.id``/``row.created_at`` being populated.
    """
    plaintext = generate_api_key()
    row = ApiKey(
        project_id=project_id,
        name=name,
        key_hash=hash_api_key(plaintext),
        key_prefix=key_display_prefix(plaintext),
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row, plaintext


async def list_api_keys(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
) -> list[ApiKey]:
    """Return all API keys for a project, oldest first.

    Includes revoked keys (the caller decides what to surface); secrets are not
    stored so nothing sensitive is returned beyond the non-secret prefix.
    """
    stmt = (
        select(ApiKey)
        .where(ApiKey.project_id == project_id)
        .order_by(ApiKey.created_at)
    )
    return list((await session.execute(stmt)).scalars().all())


async def revoke_api_key(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    key_id: uuid.UUID,
) -> bool:
    """Soft-revoke a key (set ``revoked_at = now()``); never hard-delete.

    Scoped to the project so one project cannot revoke another's keys.
    Idempotent: revoking an already-revoked key is a no-op that still returns
    ``True`` (the key exists and is in the revoked state). Returns ``False``
    only when no such key exists for this project.
    """
    stmt = select(ApiKey).where(
        ApiKey.id == key_id,
        ApiKey.project_id == project_id,
    )
    row = (await session.execute(stmt)).scalar_one_or_none()
    if row is None:
        return False
    if row.revoked_at is None:
        row.revoked_at = func.now()
        await session.commit()
    return True


async def resolve_api_key(
    session: AsyncSession,
    presented_key: str,
) -> tuple[Project, uuid.UUID | None] | None:
    """Resolve a presented plaintext key to its project + api-key id.

    Hashes the key and matches an *unrevoked* ``api_keys`` row, eager-loading
    ``project`` and ``project.organisation`` (so downstream auth/budget code can
    touch them without lazy-load I/O on the async session). Returns
    ``(project, api_key_id)`` on success, or ``None`` when no active key matches
    — in which case the caller falls back to the legacy ``projects.key_hash``
    lookup.
    """
    digest = hash_api_key(presented_key)
    stmt = (
        select(ApiKey)
        .where(ApiKey.key_hash == digest, ApiKey.revoked_at.is_(None))
        .options(selectinload(ApiKey.project).selectinload(Project.organisation))
    )
    row = (await session.execute(stmt)).scalar_one_or_none()
    if row is None:
        return None
    return row.project, row.id


async def touch_last_used(
    session: AsyncSession,
    api_key_id: uuid.UUID,
) -> None:
    """Best-effort update of ``last_used_at`` for an API key.

    Issues an ``UPDATE ... SET last_used_at = now()`` but deliberately does NOT
    commit: this runs inside the request's auth dependency, and the later
    request commit (e.g. in ``record_usage``) flushes it. This keeps the touch
    cheap and avoids a stray mid-dependency-chain commit that could perturb the
    budget read. If the request errors before any commit, the touch is simply
    lost — an acceptable miss. Non-fatal by design; callers may ignore failures.
    """
    stmt = (
        update(ApiKey)
        .where(ApiKey.id == api_key_id)
        .values(last_used_at=func.now())
    )
    await session.execute(stmt)
