"""Stage 3: Budget check.

Compares today's spend (Postgres SUM) against the project and org daily
budgets. A budget of 0 means "unlimited". Depends on rate limit so the full
chain runs in order: auth -> rate limit -> budget.
"""

from decimal import Decimal

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.logging_config import get_logger
from app.middleware.auth import AuthContext
from app.middleware.rate_limit import enforce_rate_limit
from app.services.usage import org_spend_today, project_spend_today

logger = get_logger(__name__)

_ZERO = Decimal("0")


async def enforce_budget(
    ctx: AuthContext = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_session),
) -> AuthContext:
    project = ctx.project

    if project.daily_budget > _ZERO:
        spent = await project_spend_today(session, project.id)
        if spent >= project.daily_budget:
            logger.info(
                "Project budget exceeded for %s (%s/%s)",
                project.id,
                spent,
                project.daily_budget,
            )
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail="Project daily budget exceeded",
            )

    org = project.organisation
    if org.daily_budget > _ZERO:
        spent = await org_spend_today(session, org.id)
        if spent >= org.daily_budget:
            logger.info(
                "Org budget exceeded for %s (%s/%s)",
                org.id,
                spent,
                org.daily_budget,
            )
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail="Organisation daily budget exceeded",
            )

    return ctx
