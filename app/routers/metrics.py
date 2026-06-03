"""GET /v1/metrics — project-scoped usage analytics for the calling project.

Returns totals (with latency percentiles), an optional grouped breakdown, and a
bucketed timeseries over a time window. Authenticated by the gateway key (no
rate-limit/budget needed for a read-only metrics call).
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.middleware import AuthContext, get_auth_context
from app.schemas.metrics import MetricsResponse
from app.services.metrics import (
    GROUP_BY_COLUMNS,
    VALID_GRANULARITIES,
    MetricsScope,
    get_metrics,
    parse_range,
    pick_granularity,
)

router = APIRouter(prefix="/v1", tags=["metrics"])


async def _metrics_for_scope(
    session: AsyncSession,
    scope: MetricsScope,
    *,
    range_: str | None,
    frm: str | None,
    to: str | None,
    group_by: str | None,
    granularity: str | None,
) -> MetricsResponse:
    now = datetime.now(timezone.utc)
    try:
        window_from, window_to = parse_range(range_, frm, to, now=now)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if group_by is not None and group_by not in GROUP_BY_COLUMNS:
        raise HTTPException(
            status_code=422,
            detail=f"invalid group_by; expected one of {', '.join(GROUP_BY_COLUMNS)}",
        )

    gran = granularity or pick_granularity(window_to - window_from)
    if gran not in VALID_GRANULARITIES:
        raise HTTPException(
            status_code=422,
            detail=f"invalid granularity; expected one of {', '.join(VALID_GRANULARITIES)}",
        )

    try:
        return await get_metrics(
            session, scope, frm=window_from, to=window_to,
            group_by=group_by, granularity=gran,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/metrics", response_model=MetricsResponse)
async def project_metrics(
    ctx: AuthContext = Depends(get_auth_context),
    session: AsyncSession = Depends(get_session),
    range_: str | None = Query(default=None, alias="range"),
    frm: str | None = Query(default=None, alias="from"),
    to: str | None = Query(default=None),
    group_by: str | None = Query(default=None),
    granularity: str | None = Query(default=None),
) -> MetricsResponse:
    scope = MetricsScope(field="project_id", value=ctx.project.id)
    return await _metrics_for_scope(
        session, scope, range_=range_, frm=frm, to=to,
        group_by=group_by, granularity=granularity,
    )
