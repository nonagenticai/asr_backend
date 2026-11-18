"""Unit tests for the auth FastAPI dependencies.

PRODUCT-OWNER RULING: a ``tenant_id`` may NEVER hold a sentinel. These tests
used to assert the OPPOSITE — that the no-auth path minted an ``_ANONYMOUS_USER``
stamped with the nil-UUID sentinel. They are rewritten, not deleted: the nil
UUID is a VALID uuid that passes every cast, and billing then buckets it as
unattributed and NEVER CHARGES it. Real usage, silently unbilled.
"""

from __future__ import annotations

import types
import uuid
from typing import cast

import pytest
from fastapi import HTTPException, Request

import auth.dependencies as deps
from auth.models import User


def test_no_sentinel_names_are_reachable_from_the_package() -> None:
    # The substitution must be GONE, not merely unused: a re-added constant is
    # how this bug comes back.
    assert not hasattr(deps, "UNATTRIBUTED_TENANT_ID")
    assert not hasattr(deps, "DEFAULT_TENANT_ID")
    assert not hasattr(deps, "_ANONYMOUS_USER")


def _request(**state: object) -> Request:
    """Minimal stand-in for the parts of ``Request`` the dependency reads."""
    return cast("Request", types.SimpleNamespace(state=types.SimpleNamespace(**state)))


@pytest.mark.asyncio
async def test_no_tenant_under_disabled_auth_is_401_not_a_synthetic_tenant() -> None:
    """ENABLE_AUTH=false disables INTROSPECTION, not tenancy."""
    assert deps.ENABLE_AUTH is False  # set by conftest
    with pytest.raises(HTTPException) as exc:
        await deps.get_current_user(_request())
    assert exc.value.status_code == 401
    assert "no sentinel, no default, no substitution" in str(exc.value.detail)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sentinel",
    [
        "00000000-0000-0000-0000-000000000000",
        "00000000000000000000000000000000",
        "  00000000-0000-0000-0000-000000000000  ",
        "",
        "   ",
        None,
    ],
)
async def test_nil_uuid_from_upstream_is_rejected_by_value(sentinel: object) -> None:
    """A None-only guard would leave the hole OPEN.

    ``w2v_common.auth.middleware`` writes ``DEFAULT_TENANT_ID`` — defaulting to
    the nil UUID — onto ``request.state.tenant_id`` on the legacy
    trusted-network path, so the sentinel arrives from OUTSIDE this repo.
    """
    with pytest.raises(HTTPException) as exc:
        await deps.get_current_user(_request(tenant_id=sentinel))
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_real_forwarded_tenant_is_used_verbatim() -> None:
    real = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
    user = await deps.get_current_user(_request(tenant_id=real))
    # ``tenant_id`` is a ``uuid.UUID`` from w2v-common 0.67.10 (#127); ``sub`` is
    # a free-form str and stays the string spelling. "Verbatim" still means "the
    # forwarded value, never a substituted default" — that is what this pins.
    assert user.tenant_id == uuid.UUID(real)
    assert user.sub == real


@pytest.mark.asyncio
async def test_enabled_auth_returns_the_middleware_user() -> None:
    user = User(sub="s", tenant_id="3f2504e0-4f89-11d3-9a0c-0305e82c3301")
    monkey = deps.ENABLE_AUTH
    deps.ENABLE_AUTH = True
    try:
        assert await deps.get_current_user(_request(user=user)) is user  # type: ignore[arg-type]
        with pytest.raises(HTTPException) as exc:
            await deps.get_current_user(_request())
        assert exc.value.status_code == 401
    finally:
        deps.ENABLE_AUTH = monkey
