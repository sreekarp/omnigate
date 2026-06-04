"""``omnillm-gateway`` management CLI (stdlib argparse, no extra deps).

Subcommands::

    omnillm-gateway serve [--host H] [--port P] [--reload]
    omnillm-gateway db upgrade|downgrade [--rev REV]
    omnillm-gateway config-check
    omnillm-gateway version
    omnillm-gateway org create --name N [--daily-budget D] [--monthly-budget M]
    omnillm-gateway org list
    omnillm-gateway project create --org-id ID --name N [--daily-budget D] \\
                               [--monthly-budget M] [--rate-limit N]
    omnillm-gateway project list [--org-id ID]
    omnillm-gateway usage (--org-id ID | --project-id ID) [--range 24h]

DB-touching commands talk to the database directly (like the seed script).
"""

import argparse
import asyncio
import sys
import uuid
from decimal import Decimal


def _print(*args: object) -> None:
    print(*args)  # CLI output is user-facing; print is appropriate here.


# --- serve / db ---------------------------------------------------------------
def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "app.main:app", host=args.host, port=args.port, reload=args.reload
    )
    return 0


def _cmd_db(args: argparse.Namespace) -> int:
    from alembic import command
    from alembic.config import Config

    cfg = Config(args.config)
    if args.db_action == "upgrade":
        command.upgrade(cfg, args.rev)
    else:
        command.downgrade(cfg, args.rev)
    return 0


def _cmd_version(_: argparse.Namespace) -> int:
    from app import __version__

    _print(f"omnillm-gateway {__version__}")
    return 0


def _cmd_config_check(_: argparse.Namespace) -> int:
    from app.config import get_settings

    s = get_settings()
    _print("Configuration:")
    _print(f"  database_url      : {'set' if s.database_url else 'MISSING'}")
    _print(f"  redis_url         : {'set' if s.redis_url else 'MISSING'}")
    _print(f"  secret_key        : {'set' if s.secret_key else 'MISSING'}")
    _print(f"  admin_api_key     : {'set' if s.admin_api_key else 'MISSING'}")
    _print(f"  log_level/format  : {s.log_level}/{s.log_format}")
    _print(f"  retry_max_attempts: {s.retry_max_attempts}")
    _print(f"  circuit_breaker   : {'on' if s.circuit_breaker_enabled else 'off'} ({s.circuit_breaker_backend})")
    _print(f"  response_cache    : {'on' if s.response_cache_enabled else 'off'} (ttl {s.response_cache_ttl_seconds}s)")
    return 0


# --- org / project (async DB) -------------------------------------------------
async def _org_create(name: str, daily: Decimal, monthly: Decimal) -> None:
    from app.db.session import SessionLocal
    from app.models.db import Organisation

    async with SessionLocal() as session:
        org = Organisation(name=name, daily_budget=daily, monthly_budget=monthly)
        session.add(org)
        await session.commit()
        await session.refresh(org)
        _print(f"Created organisation {org.name!r} id={org.id}")


async def _org_list() -> None:
    from sqlalchemy import select

    from app.db.session import SessionLocal
    from app.models.db import Organisation

    async with SessionLocal() as session:
        rows = (await session.execute(select(Organisation).order_by(Organisation.created_at))).scalars().all()
        if not rows:
            _print("(no organisations)")
        for o in rows:
            _print(f"{o.id}  {o.name}  daily=${o.daily_budget}  monthly=${o.monthly_budget}")


async def _project_create(
    org_id: uuid.UUID, name: str, daily: Decimal, monthly: Decimal, rate_limit: int
) -> None:
    from app.db.session import SessionLocal
    from app.models.db import Organisation, Project
    from app.security import generate_api_key, hash_api_key, key_display_prefix

    async with SessionLocal() as session:
        org = await session.get(Organisation, org_id)
        if org is None:
            _print(f"ERROR: organisation {org_id} not found", )
            raise SystemExit(2)
        api_key = generate_api_key()
        project = Project(
            org_id=org_id, name=name, key_hash=hash_api_key(api_key),
            key_prefix=key_display_prefix(api_key), daily_budget=daily,
            monthly_budget=monthly, rate_limit_per_min=rate_limit,
        )
        session.add(project)
        await session.commit()
        await session.refresh(project)
        _print(f"Created project {project.name!r} id={project.id}")
        _print("API KEY (store it now, shown once):")
        _print(f"    {api_key}")


async def _project_list(org_id: uuid.UUID | None) -> None:
    from sqlalchemy import select

    from app.db.session import SessionLocal
    from app.models.db import Project

    async with SessionLocal() as session:
        stmt = select(Project).order_by(Project.created_at)
        if org_id is not None:
            stmt = stmt.where(Project.org_id == org_id)
        rows = (await session.execute(stmt)).scalars().all()
        if not rows:
            _print("(no projects)")
        for p in rows:
            _print(f"{p.id}  {p.name}  org={p.org_id}  key={p.key_prefix}...  rl={p.rate_limit_per_min}/min")


async def _usage(scope_field: str, scope_value: uuid.UUID, range_: str) -> None:
    from datetime import datetime, timezone

    from app.db.session import SessionLocal
    from app.services.metrics import MetricsScope, get_metrics, parse_range, pick_granularity

    now = datetime.now(timezone.utc)
    frm, to = parse_range(range_, None, None, now=now)
    scope = MetricsScope(field=scope_field, value=scope_value)
    async with SessionLocal() as session:
        m = await get_metrics(
            session, scope, frm=frm, to=to, group_by="model",
            granularity=pick_granularity(to - frm),
        )
    t = m.totals
    _print(f"Usage for {m.scope} {m.scope_id} over {range_}:")
    _print(f"  requests={t.requests}  tokens={t.total_tokens}  cost=${t.cost_usd:.6f}")
    _print(f"  error_rate={t.error_rate:.2%}  cache_hit_rate={t.cache_hit_rate:.2%}")
    _print(f"  latency avg={t.avg_latency_ms:.0f}ms p95={t.p95_latency_ms}")
    for b in m.breakdown:
        _print(f"    {b.key:<28} req={b.requests} tokens={b.total_tokens} cost=${b.cost_usd:.6f}")


def _cmd_org(args: argparse.Namespace) -> int:
    if args.org_action == "create":
        asyncio.run(_org_create(args.name, Decimal(str(args.daily_budget)), Decimal(str(args.monthly_budget))))
    else:
        asyncio.run(_org_list())
    return 0


def _cmd_project(args: argparse.Namespace) -> int:
    if args.project_action == "create":
        asyncio.run(_project_create(
            uuid.UUID(args.org_id), args.name,
            Decimal(str(args.daily_budget)), Decimal(str(args.monthly_budget)), args.rate_limit,
        ))
    else:
        asyncio.run(_project_list(uuid.UUID(args.org_id) if args.org_id else None))
    return 0


def _cmd_usage(args: argparse.Namespace) -> int:
    if args.org_id:
        asyncio.run(_usage("org_id", uuid.UUID(args.org_id), args.range))
    elif args.project_id:
        asyncio.run(_usage("project_id", uuid.UUID(args.project_id), args.range))
    else:
        _print("ERROR: provide --org-id or --project-id")
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="omnillm-gateway", description="OmniLLM management CLI")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="Run the API server (uvicorn)")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=_cmd_serve)

    d = sub.add_parser("db", help="Run Alembic migrations")
    d.add_argument("db_action", choices=["upgrade", "downgrade"])
    d.add_argument("--rev", default="head")
    d.add_argument("--config", default="alembic.ini")
    d.set_defaults(func=_cmd_db)

    sub.add_parser("version", help="Print version").set_defaults(func=_cmd_version)
    sub.add_parser("config-check", help="Validate configuration").set_defaults(func=_cmd_config_check)

    org = sub.add_parser("org", help="Manage organisations")
    org_sub = org.add_subparsers(dest="org_action", required=True)
    oc = org_sub.add_parser("create")
    oc.add_argument("--name", required=True)
    oc.add_argument("--daily-budget", default="0")
    oc.add_argument("--monthly-budget", default="0")
    org_sub.add_parser("list")
    org.set_defaults(func=_cmd_org)

    proj = sub.add_parser("project", help="Manage projects")
    proj_sub = proj.add_subparsers(dest="project_action", required=True)
    pc = proj_sub.add_parser("create")
    pc.add_argument("--org-id", required=True)
    pc.add_argument("--name", required=True)
    pc.add_argument("--daily-budget", default="0")
    pc.add_argument("--monthly-budget", default="0")
    pc.add_argument("--rate-limit", type=int, default=60)
    pl = proj_sub.add_parser("list")
    pl.add_argument("--org-id", default=None)
    proj.set_defaults(func=_cmd_project)

    u = sub.add_parser("usage", help="Show usage metrics")
    u.add_argument("--org-id", default=None)
    u.add_argument("--project-id", default=None)
    u.add_argument("--range", default="24h")
    u.set_defaults(func=_cmd_usage)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
