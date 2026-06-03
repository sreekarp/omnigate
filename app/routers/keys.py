"""Gateway API-key management (multiple named keys per project).

- POST   /v1/keys/api          create a named key (plaintext returned once)
- GET    /v1/keys/api          list this project's keys (no secrets)
- DELETE /v1/keys/api/{key_id} soft-revoke a key (idempotent)

These manage the *gateway* keys that authenticate callers — distinct from the
BYOK provider keys managed by ``POST /v1/keys`` in the account router.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.logging_config import get_logger
from app.middleware import AuthContext, get_auth_context
from app.schemas.keys import ApiKeyCreate, ApiKeyCreated, ApiKeyOut
from app.services.api_keys import create_api_key, list_api_keys, revoke_api_key

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/keys", tags=["api-keys"])


@router.post("/api", response_model=ApiKeyCreated, status_code=status.HTTP_201_CREATED)
async def create_key(
    payload: ApiKeyCreate,
    ctx: AuthContext = Depends(get_auth_context),
    session: AsyncSession = Depends(get_session),
) -> ApiKeyCreated:
    row, plaintext = await create_api_key(
        session, project_id=ctx.project.id, name=payload.name
    )
    logger.info("Created API key %s for project %s", row.id, ctx.project.id)
    return ApiKeyCreated(
        id=row.id,
        name=row.name,
        key_prefix=row.key_prefix,
        created_at=row.created_at,
        last_used_at=row.last_used_at,
        revoked_at=row.revoked_at,
        api_key=plaintext,
    )


@router.get("/api", response_model=list[ApiKeyOut])
async def list_keys(
    ctx: AuthContext = Depends(get_auth_context),
    session: AsyncSession = Depends(get_session),
) -> list[ApiKeyOut]:
    rows = await list_api_keys(session, project_id=ctx.project.id)
    return [ApiKeyOut.model_validate(row) for row in rows]


@router.delete("/api/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_key(
    key_id: uuid.UUID,
    ctx: AuthContext = Depends(get_auth_context),
    session: AsyncSession = Depends(get_session),
) -> None:
    found = await revoke_api_key(session, project_id=ctx.project.id, key_id=key_id)
    if not found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="API key not found"
        )
