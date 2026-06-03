"""Self-serve developer onboarding and BYOK key management.

- POST /v1/signup  (public)        -> create account, return gateway api key once
- POST /v1/keys    (authenticated) -> store/replace your provider key (encrypted)
- GET  /v1/me      (authenticated) -> your account + which providers are configured

The gateway api key returned by signup is the developer's identity; it is what
authenticates the /v1/keys and /v1/chat calls.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.logging_config import get_logger
from app.middleware import AuthContext, get_auth_context
from app.models.db import Organisation, Project
from app.schemas.account import (
    MeResponse,
    SetProviderKeyRequest,
    SignupRequest,
    SignupResponse,
)
from app.security import generate_api_key, hash_api_key, key_display_prefix
from app.services.credentials import list_configured_providers, set_provider_key

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["account"])


@router.post("/signup", response_model=SignupResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    payload: SignupRequest,
    session: AsyncSession = Depends(get_session),
) -> SignupResponse:
    """Public self-serve registration: one org + one project + a gateway key."""
    org = Organisation(name=payload.org_name or payload.email)
    session.add(org)
    await session.flush()  # populate org.id

    api_key = generate_api_key()
    project = Project(
        org_id=org.id,
        name=payload.project_name,
        key_hash=hash_api_key(api_key),
        key_prefix=key_display_prefix(api_key),
    )
    session.add(project)
    await session.commit()
    await session.refresh(project)

    logger.info("New signup: org=%s project=%s", org.id, project.id)
    return SignupResponse(
        org_id=org.id,
        project_id=project.id,
        email=payload.email,
        api_key=api_key,
    )


def _validate_azure_meta(meta: dict | None) -> dict:
    """Validate the Azure credential metadata (endpoint + deployment)."""
    meta = meta or {}
    endpoint = meta.get("endpoint")
    deployment = meta.get("deployment")
    if not endpoint or not str(endpoint).startswith("https://"):
        raise HTTPException(
            status_code=400,
            detail="Azure provider requires meta.endpoint (an https:// URL).",
        )
    if not deployment:
        raise HTTPException(
            status_code=400,
            detail="Azure provider requires meta.deployment (the deployment name).",
        )
    return meta


@router.post("/keys", status_code=status.HTTP_204_NO_CONTENT)
async def set_key(
    payload: SetProviderKeyRequest,
    ctx: AuthContext = Depends(get_auth_context),
    session: AsyncSession = Depends(get_session),
) -> None:
    """Store (or replace) the caller's own provider key, encrypted at rest.

    For Azure, ``meta`` must carry ``endpoint`` and ``deployment``.
    """
    meta = payload.meta
    if payload.provider == "azure":
        meta = _validate_azure_meta(meta)

    await set_provider_key(
        session,
        project_id=ctx.project.id,
        provider=payload.provider,
        api_key=payload.api_key,
        meta=meta,
    )
    logger.info(
        "Stored %s key for project %s", payload.provider, ctx.project.id
    )


@router.get("/me", response_model=MeResponse)
async def me(
    ctx: AuthContext = Depends(get_auth_context),
    session: AsyncSession = Depends(get_session),
) -> MeResponse:
    project = ctx.project
    providers = await list_configured_providers(session, project_id=project.id)
    return MeResponse(
        project_id=project.id,
        org_id=project.org_id,
        project_name=project.name,
        key_prefix=project.key_prefix,
        rate_limit_per_min=project.rate_limit_per_min,
        daily_budget=project.daily_budget,
        monthly_budget=project.monthly_budget,
        configured_providers=providers,
    )
