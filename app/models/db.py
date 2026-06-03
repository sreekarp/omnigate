"""SQLAlchemy 2.0 async ORM models.

Hierarchy: Organisation -> Project -> (usage attributed to users by header).
Costs and budgets use Numeric for exact decimal arithmetic.

NOTE: Only async sessions may be used with these models (see CLAUDE.md).
"""

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


class Base(DeclarativeBase):
    """Declarative base for all models."""


class Organisation(Base):
    __tablename__ = "organisations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # Daily budget in USD. 0 means "no limit".
    daily_budget: Mapped[Decimal] = mapped_column(
        Numeric(12, 4), nullable=False, default=Decimal("0")
    )
    # Monthly budget in USD (calendar month, UTC). 0 means "no limit".
    monthly_budget: Mapped[Decimal] = mapped_column(
        Numeric(12, 4), nullable=False, server_default="0", default=Decimal("0")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    projects: Mapped[list["Project"]] = relationship(
        back_populates="organisation",
        cascade="all, delete-orphan",
    )


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    org_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organisations.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # SHA-256 hex digest of the API key (never store plaintext).
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    # Non-secret leading chars of the key, for dashboard display.
    key_prefix: Mapped[str] = mapped_column(String(16), nullable=False)

    daily_budget: Mapped[Decimal] = mapped_column(
        Numeric(12, 4), nullable=False, default=Decimal("0")
    )
    # Monthly budget in USD (calendar month, UTC). 0 means "no limit".
    monthly_budget: Mapped[Decimal] = mapped_column(
        Numeric(12, 4), nullable=False, server_default="0", default=Decimal("0")
    )
    rate_limit_per_min: Mapped[int] = mapped_column(
        Integer, nullable=False, default=60
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    organisation: Mapped["Organisation"] = relationship(back_populates="projects")
    usage_records: Mapped[list["UsageRecord"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )
    provider_credentials: Mapped[list["ProviderCredential"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )
    api_keys: Mapped[list["ApiKey"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )

    __table_args__ = (Index("ix_projects_org_id", "org_id"),)


class ApiKey(Base):
    """A named gateway API key for a project (multiple per project).

    Auth resolves a presented key against ``api_keys`` (unrevoked) first, then
    falls back to the legacy ``projects.key_hash`` for backward compatibility.
    Only the SHA-256 hash and a short non-secret prefix are stored.
    """

    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    key_prefix: Mapped[str] = mapped_column(String(16), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    project: Mapped["Project"] = relationship(back_populates="api_keys")

    __table_args__ = (Index("ix_api_keys_project_id", "project_id"),)


class ProviderCredential(Base):
    """A developer's own provider API key (BYOK), encrypted at rest.

    One row per (project, provider). The plaintext key is never stored; only
    the Fernet-encrypted token in ``encrypted_key``.
    """

    __tablename__ = "provider_credentials"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    # "openai" | "anthropic" | "gemini" | "azure"
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    encrypted_key: Mapped[str] = mapped_column(Text, nullable=False)
    # Non-secret provider config (e.g. Azure endpoint/deployment/api_version).
    meta: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=None)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    project: Mapped["Project"] = relationship(back_populates="provider_credentials")

    __table_args__ = (
        UniqueConstraint("project_id", "provider", name="uq_provider_cred"),
    )


class UsageRecord(Base):
    """One row per completed (or failed) provider request."""

    __tablename__ = "usage_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    org_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organisations.id", ondelete="CASCADE"), nullable=False
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    # From the x-user-id request header (free-form, optional).
    user_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)

    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Computed cost in USD for this request.
    cost: Mapped[Decimal] = mapped_column(
        Numeric(12, 6), nullable=False, default=Decimal("0")
    )

    # "ok" | "error" | "rate_limited" | "budget_exceeded"
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ok")
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    project: Mapped["Project"] = relationship(back_populates="usage_records")

    __table_args__ = (
        Index("ix_usage_org_created", "org_id", "created_at"),
        Index("ix_usage_project_created", "project_id", "created_at"),
    )
