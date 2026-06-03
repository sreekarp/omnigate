"""Pydantic v2 schemas for gateway API-key management.

These back the ``/v1/keys/api`` endpoints. The plaintext key appears exactly
once, in :class:`ApiKeyCreated`; list/read responses (:class:`ApiKeyOut`) carry
only the non-secret prefix and metadata.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ApiKeyCreate(BaseModel):
    """Request body for creating a new gateway API key."""

    name: str = Field(min_length=1, max_length=255)


class ApiKeyOut(BaseModel):
    """A gateway API key as exposed to clients (no secret material)."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    key_prefix: str
    created_at: datetime
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None


class ApiKeyCreated(ApiKeyOut):
    """Creation response: the full key metadata plus the plaintext key.

    The ``api_key`` plaintext is returned only here and is never recoverable
    afterwards — the gateway stores only its hash.
    """

    api_key: str
