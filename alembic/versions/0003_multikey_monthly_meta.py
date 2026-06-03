"""multi-key auth, monthly budgets, provider-credential meta

Adds:
- provider_credentials.meta (JSON, nullable) for non-secret provider config (Azure).
- api_keys table for multiple named gateway keys per project.
- organisations.monthly_budget and projects.monthly_budget (NOT NULL, default 0).

Revision ID: 0003_resilience_multikey_monthly_meta
Revises: 0002_provider_credentials
Create Date: 2026-06-04

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0003_multikey_monthly_meta"
down_revision: Union[str, None] = "0002_provider_credentials"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Non-secret provider config (e.g. Azure endpoint/deployment/api_version).
    op.add_column(
        "provider_credentials",
        sa.Column("meta", sa.JSON(), nullable=True),
    )

    # 2. Multiple named gateway API keys per project.
    op.create_table(
        "api_keys",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("key_hash", sa.String(length=64), nullable=False),
        sa.Column("key_prefix", sa.String(length=16), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key_hash", name="uq_api_keys_key_hash"),
    )
    op.create_index("ix_api_keys_project_id", "api_keys", ["project_id"])

    # 3 & 4. Monthly budgets. server_default is mandatory for NOT NULL adds on
    # populated tables. 0 means "no limit", mirroring daily_budget.
    op.add_column(
        "organisations",
        sa.Column(
            "monthly_budget",
            sa.Numeric(precision=12, scale=4),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "projects",
        sa.Column(
            "monthly_budget",
            sa.Numeric(precision=12, scale=4),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    op.drop_column("projects", "monthly_budget")
    op.drop_column("organisations", "monthly_budget")
    op.drop_index("ix_api_keys_project_id", table_name="api_keys")
    op.drop_table("api_keys")
    op.drop_column("provider_credentials", "meta")
