"""Server-rendered (Jinja2) dashboard showing usage and spend.

Read-only. Routes:
    GET /            -> overview of all orgs with today's spend
    GET /projects/{project_id} -> per-project detail + recent requests
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.templating import Jinja2Templates
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.models.db import Organisation, Project, UsageRecord

router = APIRouter(tags=["dashboard"])

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def _utc_day_start() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


@router.get("/")
async def overview(request: Request, session: AsyncSession = Depends(get_session)):
    day_start = _utc_day_start()

    orgs = list(
        (await session.execute(select(Organisation).order_by(Organisation.name)))
        .scalars()
        .all()
    )

    # Today's spend + request count per org.
    spend_stmt = (
        select(
            UsageRecord.org_id,
            func.coalesce(func.sum(UsageRecord.cost), 0).label("spend"),
            func.count(UsageRecord.id).label("requests"),
        )
        .where(UsageRecord.created_at >= day_start)
        .group_by(UsageRecord.org_id)
    )
    spend_rows = (await session.execute(spend_stmt)).all()
    spend_by_org = {row.org_id: (Decimal(row.spend), row.requests) for row in spend_rows}

    org_views = []
    for org in orgs:
        spend, requests = spend_by_org.get(org.id, (Decimal("0"), 0))
        org_views.append(
            {
                "org": org,
                "spend": spend,
                "requests": requests,
                "budget": org.daily_budget,
            }
        )

    return templates.TemplateResponse(
        request,
        "overview.html",
        {"orgs": org_views, "generated_at": datetime.now(timezone.utc)},
    )


@router.get("/projects/{project_id}")
async def project_detail(
    project_id: uuid.UUID,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    project = await session.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    org = await session.get(Organisation, project.org_id)
    day_start = _utc_day_start()

    spend_stmt = select(
        func.coalesce(func.sum(UsageRecord.cost), 0),
        func.count(UsageRecord.id),
    ).where(
        UsageRecord.project_id == project_id,
        UsageRecord.created_at >= day_start,
    )
    spend, request_count = (await session.execute(spend_stmt)).one()

    recent_stmt = (
        select(UsageRecord)
        .where(UsageRecord.project_id == project_id)
        .order_by(desc(UsageRecord.created_at))
        .limit(50)
    )
    recent = list((await session.execute(recent_stmt)).scalars().all())

    return templates.TemplateResponse(
        request,
        "project.html",
        {
            "project": project,
            "org": org,
            "spend": Decimal(spend),
            "request_count": request_count,
            "recent": recent,
            "generated_at": datetime.now(timezone.utc),
        },
    )
