"""Schemas for self-serve signup and provider-key management."""

import uuid
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field

ProviderName = Literal["openai", "anthropic", "gemini", "azure"]


class SignupRequest(BaseModel):
    # Plain string (not EmailStr) to avoid the extra email-validator dependency.
    email: str = Field(..., min_length=3, max_length=255)
    org_name: str | None = Field(default=None, max_length=255)
    project_name: str = Field(default="Default", max_length=255)


class SignupResponse(BaseModel):
    org_id: uuid.UUID
    project_id: uuid.UUID
    email: str
    api_key: str  # the gateway key; shown once, store it now
    message: str = (
        "Save your api_key now — it cannot be retrieved later. "
        "Add a provider key via POST /v1/keys before calling /v1/chat."
    )


class SetProviderKeyRequest(BaseModel):
    provider: ProviderName
    api_key: str = Field(..., min_length=8, description="Your own provider key")
    meta: dict | None = Field(
        default=None,
        description=(
            "Non-secret provider config. For Azure: "
            '{"endpoint": "https://<res>.openai.azure.com", '
            '"deployment": "<name>", "api_version": "2024-10-21"}.'
        ),
    )


class MeResponse(BaseModel):
    project_id: uuid.UUID
    org_id: uuid.UUID
    project_name: str
    key_prefix: str
    rate_limit_per_min: int
    daily_budget: Decimal
    monthly_budget: Decimal
    configured_providers: list[str]
