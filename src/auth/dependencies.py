"""FastAPI dependencies for authentication.

Identity is populated on ``request.state.user`` by w2v_common's ``AuthMiddleware``
(re-exported from ``auth.middleware``); these helpers read it back.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Coroutine
from typing import Any, cast

from fastapi import Depends, HTTPException, Request, status

from services.metering import MissingTenantError, require_tenant

from .models import User

ENABLE_AUTH = os.getenv("ENABLE_AUTH", "true").lower() == "true"

# ── PRODUCT-OWNER RULING: a tenant_id may NEVER hold a sentinel ──────────────
# Do NOT re-add ``UNATTRIBUTED_TENANT_ID``, ``DEFAULT_TENANT_ID``, a
# ``DEFAULT_TENANT_ID`` env read, or an ``_ANONYMOUS_USER`` carrying a
# manufactured tenant. Every request carries a REAL registered tenant forwarded
# from the requestor; an absent tenant is an ERROR (401), never a substituted
# default. The nil UUID this module used to mint was the most dangerous
# spelling: a VALID uuid that passes every cast, which billing then buckets as
# unattributed and NEVER CHARGES. Real usage, silently unbilled.


async def get_current_user(request: Request) -> User:
    """Return the authenticated user, or raise 401.

    ``ENABLE_AUTH=false`` disables license-token INTROSPECTION, not tenancy: the
    caller must still have forwarded a real tenant. It used to return a
    hard-coded anonymous user stamped with the nil-UUID sentinel, so every
    bypass-path transcription was emitted unattributed and never billed.
    """
    if not ENABLE_AUTH:
        try:
            tenant_id = require_tenant(getattr(request.state, "tenant_id", None))
        except MissingTenantError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(exc),
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc
        # ``sub`` is a free-form str and ``tenant_id`` is now a ``uuid.UUID``
        # (w2v_common #127). Passing the UUID straight into ``sub`` would fail
        # pydantic validation and take the whole ENABLE_AUTH=false lane down.
        return User(
            sub=str(tenant_id),
            preferred_username="anonymous",
            tenant_id=tenant_id,
            roles=["admin"],
        )

    user = getattr(request.state, "user", None)
    if isinstance(user, User):
        return user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_role(role: str) -> Callable[..., Coroutine[Any, Any, User]]:
    """Dependency factory enforcing a role."""

    async def checker(user: User = Depends(get_current_user)) -> User:
        if role not in user.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' required",
            )
        return user

    return cast("Callable[..., Coroutine[Any, Any, User]]", checker)
