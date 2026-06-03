"""Stage 3: Budget check.

Compares spend against the project and org daily *and* monthly budgets (UTC).
A budget of 0 means "unlimited". Depends on rate limit so the full chain runs
in order: auth -> rate limit -> budget. Sets ``x-budget-*`` response headers.

Checks run sequentially (never concurrently on one async session).
"""

from decimal import Decimal

from fastapi import Depends, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.logging_config import get_logger
from app.middleware.auth import AuthContext
from app.middleware.rate_limit import enforce_rate_limit
from app.services.usage import (
    org_spend_this_month,
    org_spend_today,
    project_spend_this_month,
    project_spend_today,
)

logger = get_logger(__name__)

_ZERO = Decimal("0")


def _exceeded(spend: Decimal, budget: Decimal) -> bool:
    return budget > _ZERO and spend >= budget


async def enforce_budget(
    response: Response,
    ctx: AuthContext = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_session),
) -> AuthContext:
    project = ctx.project
    org = project.organisation

    # Daily (always computed — drives headers).
    project_day = await project_spend_today(session, project.id)
    org_day = await org_spend_today(session, org.id)
    response.headers["x-budget-project-daily-spend"] = f"{project_day:.6f}"
    response.headers["x-budget-project-daily-limit"] = f"{project.daily_budget:.4f}"
    response.headers["x-budget-org-daily-spend"] = f"{org_day:.6f}"
    response.headers["x-budget-org-daily-limit"] = f"{org.daily_budget:.4f}"

    if _exceeded(project_day, project.daily_budget):
        logger.info("Project daily budget exceeded for %s (%s/%s)", project.id, project_day, project.daily_budget)
        raise HTTPException(status.HTTP_402_PAYMENT_REQUIRED, "Project daily budget exceeded")
    if _exceeded(org_day, org.daily_budget):
        logger.info("Org daily budget exceeded for %s (%s/%s)", org.id, org_day, org.daily_budget)
        raise HTTPException(status.HTTP_402_PAYMENT_REQUIRED, "Organisation daily budget exceeded")

    # Monthly (only when a monthly budget is configured).
    if project.monthly_budget > _ZERO:
        project_month = await project_spend_this_month(session, project.id)
        response.headers["x-budget-project-monthly-spend"] = f"{project_month:.6f}"
        response.headers["x-budget-project-monthly-limit"] = f"{project.monthly_budget:.4f}"
        if project_month >= project.monthly_budget:
            logger.info("Project monthly budget exceeded for %s", project.id)
            raise HTTPException(status.HTTP_402_PAYMENT_REQUIRED, "Project monthly budget exceeded")

    if org.monthly_budget > _ZERO:
        org_month = await org_spend_this_month(session, org.id)
        response.headers["x-budget-org-monthly-spend"] = f"{org_month:.6f}"
        response.headers["x-budget-org-monthly-limit"] = f"{org.monthly_budget:.4f}"
        if org_month >= org.monthly_budget:
            logger.info("Org monthly budget exceeded for %s", org.id)
            raise HTTPException(status.HTTP_402_PAYMENT_REQUIRED, "Organisation monthly budget exceeded")

    return ctx
