"""Usage accounting: recording requests and summing today's spend.

"Today" is defined in UTC (calendar day boundary at 00:00 UTC).
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db import UsageRecord


def _utc_day_start() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


async def project_spend_today(session: AsyncSession, project_id: uuid.UUID) -> Decimal:
    """Sum of today's (UTC) cost for a project."""
    stmt = select(func.coalesce(func.sum(UsageRecord.cost), 0)).where(
        UsageRecord.project_id == project_id,
        UsageRecord.created_at >= _utc_day_start(),
    )
    result = await session.execute(stmt)
    return Decimal(result.scalar_one())


async def org_spend_today(session: AsyncSession, org_id: uuid.UUID) -> Decimal:
    """Sum of today's (UTC) cost across all projects in an org."""
    stmt = select(func.coalesce(func.sum(UsageRecord.cost), 0)).where(
        UsageRecord.org_id == org_id,
        UsageRecord.created_at >= _utc_day_start(),
    )
    result = await session.execute(stmt)
    return Decimal(result.scalar_one())


async def record_usage(
    session: AsyncSession,
    *,
    org_id: uuid.UUID,
    project_id: uuid.UUID,
    user_id: str | None,
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost: Decimal,
    status: str,
    latency_ms: int,
    request_id: str,
) -> UsageRecord:
    """Insert a usage record and commit it."""
    record = UsageRecord(
        org_id=org_id,
        project_id=project_id,
        user_id=user_id,
        provider=provider,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cost=cost,
        status=status,
        latency_ms=latency_ms,
        request_id=request_id,
    )
    session.add(record)
    await session.commit()
    return record
