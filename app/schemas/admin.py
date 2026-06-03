"""Schemas for the admin provisioning API."""

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field


# --- Organisations ---
class OrganisationCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    daily_budget: Decimal = Field(default=Decimal("0"), ge=0)


class OrganisationOut(BaseModel):
    id: uuid.UUID
    name: str
    daily_budget: Decimal
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Projects ---
class ProjectCreate(BaseModel):
    org_id: uuid.UUID
    name: str = Field(..., min_length=1, max_length=255)
    daily_budget: Decimal = Field(default=Decimal("0"), ge=0)
    rate_limit_per_min: int = Field(default=60, ge=1)


class ProjectOut(BaseModel):
    id: uuid.UUID
    org_id: uuid.UUID
    name: str
    key_prefix: str
    daily_budget: Decimal
    rate_limit_per_min: int
    created_at: datetime

    model_config = {"from_attributes": True}


class ProjectCreated(ProjectOut):
    """Returned once on creation; includes the plaintext key (never stored)."""

    api_key: str
