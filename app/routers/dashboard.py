"""Server-rendered (Jinja2) dashboard showing usage and spend.

Read-only. Routes:
    GET /                         -> overview of all orgs with today's spend
    GET /projects/{project_id}    -> per-project detail + recent requests
    GET /dashboard/metrics        -> gateway-wide analytics for a time range
                                     (totals, per-provider / per-model
                                     breakdowns, and an inline-SVG timeseries)

All queries run sequentially on the one async session (a single
connection cannot multiplex concurrent statements — see
``app/services/metrics.py``). The metrics page reuses the range/granularity
helpers and the canonical aggregate expressions from that service, but issues
its own *gateway-wide* (un-scoped) statements since :func:`get_metrics` is
scoped to a single org or project.
"""

import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.templating import Jinja2Templates
from sqlalchemy import ColumnElement, Select, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.models.db import Organisation, Project, UsageRecord
from app.services.metrics import (
    RANGE_WINDOWS,
    _bucket_expr,
    _error_count,
    parse_range,
    pick_granularity,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["dashboard"])

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

U = UsageRecord

# Ranges the metrics page offers, in display order.
_RANGE_CHOICES: tuple[str, ...] = ("1h", "24h", "7d", "30d")
_DEFAULT_RANGE = "24h"


def _utc_day_start() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _window_clause(frm: datetime, to: datetime) -> ColumnElement[bool]:
    """Gateway-wide half-open ``[frm, to)`` WHERE clause (no org/project scope)."""
    return (U.created_at >= frm) & (U.created_at < to)


def _gateway_totals_stmt(frm: datetime, to: datetime) -> Select:
    """Single-row gateway-wide totals (counts, token/cost sums, latency, errors)."""
    return select(
        func.count(U.id).label("requests"),
        func.coalesce(func.sum(U.prompt_tokens), 0).label("prompt_tokens"),
        func.coalesce(func.sum(U.completion_tokens), 0).label("completion_tokens"),
        func.coalesce(func.sum(U.total_tokens), 0).label("total_tokens"),
        func.coalesce(func.sum(U.cost), 0).label("cost"),
        _error_count().label("errors"),
        func.avg(U.latency_ms).label("avg_latency_ms"),
    ).where(_window_clause(frm, to))


def _gateway_breakdown_stmt(frm: datetime, to: datetime, column: ColumnElement) -> Select:
    """Grouped gateway-wide breakdown by ``column`` (provider or model)."""
    return (
        select(
            column.label("key"),
            func.count(U.id).label("requests"),
            func.coalesce(func.sum(U.total_tokens), 0).label("total_tokens"),
            func.coalesce(func.sum(U.cost), 0).label("cost"),
            func.avg(U.latency_ms).label("avg_latency_ms"),
            _error_count().label("errors"),
        )
        .where(_window_clause(frm, to))
        .group_by(column)
        .order_by(func.coalesce(func.sum(U.cost), 0).desc(), func.count(U.id).desc())
        .limit(50)
    )


def _gateway_timeseries_stmt(frm: datetime, to: datetime, granularity: str) -> Select:
    """Time-bucketed gateway-wide series (sparse: empty buckets are absent)."""
    bucket = _bucket_expr(granularity).label("bucket")
    return (
        select(
            bucket,
            func.count(U.id).label("requests"),
            func.coalesce(func.sum(U.total_tokens), 0).label("total_tokens"),
            func.coalesce(func.sum(U.cost), 0).label("cost"),
            _error_count().label("errors"),
        )
        .where(_window_clause(frm, to))
        .group_by(bucket)
        .order_by(bucket)
    )


def _ratio(numerator: int | None, denominator: int) -> float:
    """Guarded ratio: ``0.0`` when the denominator is zero."""
    if not denominator:
        return 0.0
    return float(numerator or 0) / float(denominator)


def _breakdown_view(rows: list) -> list[dict]:
    """Shape grouped rows into template dicts (with cost-bar fractions)."""
    items = [
        {
            "key": "unknown" if row.key is None else str(row.key),
            "requests": int(row.requests or 0),
            "total_tokens": int(row.total_tokens or 0),
            "cost": float(row.cost or 0),
            "avg_latency_ms": float(row.avg_latency_ms or 0),
            "error_rate": _ratio(row.errors, int(row.requests or 0)),
        }
        for row in rows
    ]
    max_cost = max((item["cost"] for item in items), default=0.0)
    for item in items:
        item["cost_frac"] = (item["cost"] / max_cost) if max_cost > 0 else 0.0
    return items


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


@router.get("/dashboard/metrics")
async def metrics_page(
    request: Request,
    range: str = Query(_DEFAULT_RANGE),
    session: AsyncSession = Depends(get_session),
):
    """Gateway-wide analytics: totals, per-provider / per-model breakdowns, and
    a time-bucketed series, all rendered server-side with inline-SVG charts.

    The ``range`` query param is one of ``1h``/``24h``/``7d``/``30d``; an
    unrecognised value falls back to the default rather than erroring (this is a
    human-facing page, not the JSON API).
    """
    now = datetime.now(timezone.utc)
    if range not in RANGE_WINDOWS:
        range = _DEFAULT_RANGE
    frm, to = parse_range(range, None, None, now=now)
    granularity = pick_granularity(to - frm)

    # (1) Totals.
    totals_row = (await session.execute(_gateway_totals_stmt(frm, to))).one()
    requests = int(totals_row.requests or 0)
    totals = {
        "requests": requests,
        "prompt_tokens": int(totals_row.prompt_tokens or 0),
        "completion_tokens": int(totals_row.completion_tokens or 0),
        "total_tokens": int(totals_row.total_tokens or 0),
        "cost": float(totals_row.cost or 0),
        "error_rate": _ratio(totals_row.errors, requests),
        "avg_latency_ms": float(totals_row.avg_latency_ms or 0),
    }

    # (2) Per-provider breakdown.
    provider_rows = (
        await session.execute(_gateway_breakdown_stmt(frm, to, U.provider))
    ).all()
    by_provider = _breakdown_view(provider_rows)

    # (3) Per-model breakdown.
    model_rows = (
        await session.execute(_gateway_breakdown_stmt(frm, to, U.model))
    ).all()
    by_model = _breakdown_view(model_rows)

    # (4) Timeseries -> chart-ready bars.
    ts_rows = (
        await session.execute(_gateway_timeseries_stmt(frm, to, granularity))
    ).all()
    series = [
        {
            "bucket": row.bucket,
            "requests": int(row.requests or 0),
            "total_tokens": int(row.total_tokens or 0),
            "cost": float(row.cost or 0),
            "error_rate": _ratio(row.errors, int(row.requests or 0)),
        }
        for row in ts_rows
    ]
    chart = _build_chart(series, granularity)

    return templates.TemplateResponse(
        request,
        "metrics.html",
        {
            "totals": totals,
            "by_provider": by_provider,
            "by_model": by_model,
            "series": series,
            "chart": chart,
            "range": range,
            "range_choices": _RANGE_CHOICES,
            "granularity": granularity,
            "range_from": frm,
            "range_to": to,
            "generated_at": now,
        },
    )


# --- Inline-SVG chart geometry (computed server-side; no client JS) ----------
_CHART_W = 920
_CHART_H = 180
_CHART_PAD = 8


def _bucket_label(bucket: datetime, granularity: str) -> str:
    """Short axis label for a timeseries bucket."""
    if granularity == "day":
        return bucket.strftime("%m-%d")
    return bucket.strftime("%H:%M")


def _build_chart(series: list[dict], granularity: str) -> dict:
    """Pre-compute bar geometry (cost + requests) for the inline-SVG chart.

    Returns a dict consumed directly by ``metrics.html``: width/height, a list
    of bar rectangles (x, y, w, h for both the cost and request scales), and a
    handful of axis labels. Empty series yield ``bars=[]`` so the template can
    show a placeholder.
    """
    n = len(series)
    if n == 0:
        return {
            "width": _CHART_W,
            "height": _CHART_H,
            "bars": [],
            "max_cost": 0.0,
            "max_requests": 0,
            "labels": [],
        }

    max_cost = max((p["cost"] for p in series), default=0.0)
    max_requests = max((p["requests"] for p in series), default=0)
    inner_w = _CHART_W - 2 * _CHART_PAD
    inner_h = _CHART_H - 2 * _CHART_PAD
    slot = inner_w / n
    bar_w = max(1.0, slot * 0.72)
    gap = (slot - bar_w) / 2

    bars: list[dict] = []
    # Label at most ~8 evenly spaced ticks to avoid crowding.
    label_step = max(1, n // 8)
    labels: list[dict] = []
    for i, point in enumerate(series):
        x = _CHART_PAD + i * slot + gap
        cost_h = (point["cost"] / max_cost * inner_h) if max_cost > 0 else 0.0
        req_h = (point["requests"] / max_requests * inner_h) if max_requests > 0 else 0.0
        bars.append(
            {
                "x": round(x, 2),
                "w": round(bar_w, 2),
                "cost_y": round(_CHART_PAD + (inner_h - cost_h), 2),
                "cost_h": round(cost_h, 2),
                "req_y": round(_CHART_PAD + (inner_h - req_h), 2),
                "req_h": round(req_h, 2),
                "cost": point["cost"],
                "requests": point["requests"],
                "has_error": point["error_rate"] > 0,
                "label": _bucket_label(point["bucket"], granularity),
            }
        )
        if i % label_step == 0:
            labels.append(
                {
                    "x": round(x + bar_w / 2, 2),
                    "text": _bucket_label(point["bucket"], granularity),
                }
            )

    return {
        "width": _CHART_W,
        "height": _CHART_H,
        "bars": bars,
        "max_cost": max_cost,
        "max_requests": max_requests,
        "labels": labels,
    }


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
