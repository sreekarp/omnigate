"""Admin provisioning API: create orgs, projects (API keys), and inspect usage.

All endpoints require the ``x-admin-key`` header to match ADMIN_API_KEY.
The plaintext project API key is returned exactly once, at creation.
"""

import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.session import get_session
from app.models.db import Organisation, Project
from app.routers.metrics import _metrics_for_scope as metrics_for_scope
from app.schemas.admin import (
    OrganisationCreate,
    OrganisationOut,
    ProjectCreate,
    ProjectCreated,
    ProjectOut,
)
from app.schemas.metrics import MetricsResponse
from app.services.metrics import MetricsScope
from app.security import (
    constant_time_compare,
    generate_api_key,
    hash_api_key,
    key_display_prefix,
)

router = APIRouter(prefix="/v1/admin", tags=["admin"])


async def require_admin(
    x_admin_key: str | None = Header(default=None, alias="x-admin-key"),
) -> None:
    settings = get_settings()
    if not x_admin_key or not constant_time_compare(x_admin_key, settings.admin_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing x-admin-key",
        )


@router.post(
    "/orgs",
    response_model=OrganisationOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_admin)],
)
async def create_org(
    payload: OrganisationCreate,
    session: AsyncSession = Depends(get_session),
) -> Organisation:
    org = Organisation(
        name=payload.name,
        daily_budget=payload.daily_budget,
        monthly_budget=payload.monthly_budget,
    )
    session.add(org)
    await session.commit()
    await session.refresh(org)
    return org


@router.get(
    "/orgs",
    response_model=list[OrganisationOut],
    dependencies=[Depends(require_admin)],
)
async def list_orgs(session: AsyncSession = Depends(get_session)) -> list[Organisation]:
    result = await session.execute(select(Organisation).order_by(Organisation.created_at))
    return list(result.scalars().all())


@router.post(
    "/projects",
    response_model=ProjectCreated,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_admin)],
)
async def create_project(
    payload: ProjectCreate,
    session: AsyncSession = Depends(get_session),
) -> ProjectCreated:
    org = await session.get(Organisation, payload.org_id)
    if org is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organisation not found",
        )

    api_key = generate_api_key()
    project = Project(
        org_id=payload.org_id,
        name=payload.name,
        key_hash=hash_api_key(api_key),
        key_prefix=key_display_prefix(api_key),
        daily_budget=payload.daily_budget,
        monthly_budget=payload.monthly_budget,
        rate_limit_per_min=payload.rate_limit_per_min,
    )
    session.add(project)
    await session.commit()
    await session.refresh(project)

    # Return the plaintext key once; it is not recoverable later.
    return ProjectCreated(
        id=project.id,
        org_id=project.org_id,
        name=project.name,
        key_prefix=project.key_prefix,
        daily_budget=project.daily_budget,
        monthly_budget=project.monthly_budget,
        rate_limit_per_min=project.rate_limit_per_min,
        created_at=project.created_at,
        api_key=api_key,
    )


@router.get(
    "/projects",
    response_model=list[ProjectOut],
    dependencies=[Depends(require_admin)],
)
async def list_projects(
    org_id: uuid.UUID | None = None,
    session: AsyncSession = Depends(get_session),
) -> list[Project]:
    stmt = select(Project).order_by(Project.created_at)
    if org_id is not None:
        stmt = stmt.where(Project.org_id == org_id)
    result = await session.execute(stmt)
    return list(result.scalars().all())


@router.get(
    "/orgs/{org_id}/metrics",
    response_model=MetricsResponse,
    dependencies=[Depends(require_admin)],
)
async def org_metrics(
    org_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    range_: str | None = Query(default=None, alias="range"),
    frm: str | None = Query(default=None, alias="from"),
    to: str | None = Query(default=None),
    group_by: str | None = Query(default=None),
    granularity: str | None = Query(default=None),
) -> MetricsResponse:
    org = await session.get(Organisation, org_id)
    if org is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Organisation not found"
        )
    scope = MetricsScope(field="org_id", value=org_id)
    return await metrics_for_scope(
        session, scope, range_=range_, frm=frm, to=to,
        group_by=group_by, granularity=granularity,
    )
