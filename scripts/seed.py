"""Seed the database with a demo organisation, project, and API key.

Run after applying migrations:

    python -m scripts.seed

Idempotent-ish: if a project named "Demo Project" already exists it is left
alone (a new key would otherwise be unrecoverable). Prints the plaintext API
key exactly once on creation.
"""

import asyncio
from decimal import Decimal

from sqlalchemy import select

from app.db.session import SessionLocal
from app.logging_config import configure_logging, get_logger
from app.models.db import Organisation, Project
from app.security import generate_api_key, hash_api_key, key_display_prefix

configure_logging("INFO")
logger = get_logger("seed")

DEMO_ORG = "Demo Org"
DEMO_PROJECT = "Demo Project"


async def main() -> None:
    async with SessionLocal() as session:
        existing = await session.execute(
            select(Project).where(Project.name == DEMO_PROJECT)
        )
        if existing.scalar_one_or_none() is not None:
            logger.info(
                "Project %r already exists; skipping seed. "
                "Delete it first if you need a fresh key.",
                DEMO_PROJECT,
            )
            return

        org = Organisation(name=DEMO_ORG, daily_budget=Decimal("50.00"))
        session.add(org)
        await session.flush()  # populate org.id

        api_key = generate_api_key()
        project = Project(
            org_id=org.id,
            name=DEMO_PROJECT,
            key_hash=hash_api_key(api_key),
            key_prefix=key_display_prefix(api_key),
            daily_budget=Decimal("10.00"),
            rate_limit_per_min=60,
        )
        session.add(project)
        await session.commit()

        logger.info("Created organisation %r (id=%s)", org.name, org.id)
        logger.info("Created project %r (id=%s)", project.name, project.id)
        logger.info("=" * 60)
        logger.info("API KEY (store it now, it will not be shown again):")
        logger.info("    %s", api_key)
        logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
