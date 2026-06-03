"""BYOK provider-credential storage.

Stores/retrieves each project's provider API keys, encrypted at rest. Plaintext
keys exist only transiently in memory while encrypting or making a request. A
non-secret ``meta`` JSON blob holds provider config that is safe to store in the
clear (e.g. Azure endpoint/deployment/api-version).
"""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.crypto import decrypt_secret, encrypt_secret
from app.models.db import ProviderCredential

# Providers we accept BYOK keys for.
SUPPORTED_PROVIDERS = ("openai", "anthropic", "gemini", "azure")


async def set_provider_key(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    provider: str,
    api_key: str,
    meta: dict | None = None,
) -> ProviderCredential:
    """Insert or update (upsert) a project's encrypted key for a provider."""
    stmt = select(ProviderCredential).where(
        ProviderCredential.project_id == project_id,
        ProviderCredential.provider == provider,
    )
    existing = (await session.execute(stmt)).scalar_one_or_none()

    if existing is None:
        cred = ProviderCredential(
            project_id=project_id,
            provider=provider,
            encrypted_key=encrypt_secret(api_key),
            meta=meta,
        )
        session.add(cred)
    else:
        existing.encrypted_key = encrypt_secret(api_key)
        existing.meta = meta
        cred = existing

    await session.commit()
    await session.refresh(cred)
    return cred


async def get_provider_key(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    provider: str,
) -> str | None:
    """Return the decrypted provider key for a project, or None if not set."""
    stmt = select(ProviderCredential.encrypted_key).where(
        ProviderCredential.project_id == project_id,
        ProviderCredential.provider == provider,
    )
    token = (await session.execute(stmt)).scalar_one_or_none()
    if token is None:
        return None
    return decrypt_secret(token)


async def get_provider_credential(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    provider: str,
) -> tuple[str, dict | None] | None:
    """Return ``(decrypted_key, meta)`` for a project+provider, or None."""
    stmt = select(
        ProviderCredential.encrypted_key, ProviderCredential.meta
    ).where(
        ProviderCredential.project_id == project_id,
        ProviderCredential.provider == provider,
    )
    row = (await session.execute(stmt)).one_or_none()
    if row is None:
        return None
    encrypted_key, meta = row
    return decrypt_secret(encrypted_key), meta


async def list_configured_providers(
    session: AsyncSession, *, project_id: uuid.UUID
) -> list[str]:
    """Return the provider names a project has stored a key for (no secrets)."""
    stmt = select(ProviderCredential.provider).where(
        ProviderCredential.project_id == project_id
    )
    return list((await session.execute(stmt)).scalars().all())
