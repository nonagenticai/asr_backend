"""Unit tests for the asr-backend metering facade.

The emit/stream/field/Redis logic now lives in ``w2v_common.metering`` (the
fleet-canonical helper) and is covered by that package's own suite. These tests
pin only the service-specific surface this module is responsible for: the
routing identity, the "a tenant is REQUIRED" contract, and that the facade
re-exports the shared helpers.
"""

from __future__ import annotations

import uuid

import pytest
import w2v_common.metering as shared

from services import metering


def test_cost_source_and_service_key_are_asr_routing_slugs() -> None:
    assert metering.COST_SOURCE == "w2v-asr"
    assert metering.SERVICE_KEY == "w2v-asr"  # pragma: allowlist secret -- routing slug


def test_no_sentinel_names_are_reachable_from_the_facade() -> None:
    # PRODUCT-OWNER RULING: a tenant_id may NEVER hold a sentinel. This test
    # asserted the OPPOSITE (that DEFAULT_TENANT_ID == the nil UUID); it is
    # rewritten, not deleted. The substitution must be GONE, not merely unused —
    # a re-added constant is how this bug comes back.
    assert not hasattr(metering, "DEFAULT_TENANT_ID")
    assert not hasattr(metering, "UNATTRIBUTED_TENANT_ID")
    assert "DEFAULT_TENANT_ID" not in metering.__all__
    assert "UNATTRIBUTED_TENANT_ID" not in metering.__all__


@pytest.mark.parametrize(
    "absent",
    ["", "   ", None, 0, b"3f2504e0-4f89-11d3-9a0c-0305e82c3301"],
)
def test_require_tenant_rejects_an_absent_tenant(absent: object) -> None:
    with pytest.raises(metering.MissingTenantError):
        metering.require_tenant(absent)


@pytest.mark.parametrize(
    "nil",
    [
        "00000000-0000-0000-0000-000000000000",
        "00000000000000000000000000000000",
        "  00000000-0000-0000-0000-000000000000  ",
        "{00000000-0000-0000-0000-000000000000}",
        "urn:uuid:00000000-0000-0000-0000-000000000000",
    ],
)
def test_require_tenant_rejects_the_nil_uuid_by_value(nil: str) -> None:
    """The load-bearing half: a None-only guard leaves the upstream hole OPEN.

    ``w2v_common.auth.middleware`` still writes the nil UUID onto
    ``request.state.tenant_id`` on the legacy trusted-network path, so the
    sentinel arrives from OUTSIDE this repo and must be rejected BY VALUE.
    """
    with pytest.raises(metering.MissingTenantError):
        metering.require_tenant(nil)


def test_missing_tenant_error_is_not_swallowed_by_the_uuid_parse() -> None:
    # MissingTenantError subclasses ValueError, so raising it inside the
    # uuid.UUID() try/except would be swallowed by that very handler.
    assert issubclass(metering.MissingTenantError, ValueError)


# The dash-less spelling is DERIVED, not written out: a bare 32-char hex literal
# trips detect-secrets as a "Hex High Entropy String", and widening the secrets
# baseline to admit a test fixture is the wrong trade. `.hex` is the same value.
_REAL = uuid.UUID("3f2504e0-4f89-11d3-9a0c-0305e82c3301")


@pytest.mark.parametrize(
    "real",
    [
        str(_REAL),
        f"  {_REAL}  ",
        str(_REAL).upper(),
        _REAL.hex,
        f"{{{_REAL}}}",
    ],
)
def test_require_tenant_returns_a_real_tenant_as_a_uuid(real: str) -> None:
    """Every spelling collapses to the one canonical ``uuid.UUID``.

    The return type became ``uuid.UUID`` with w2v-common 0.67.10 — the cost bus
    parses the tenant itself, so handing it a pre-parsed value is what keeps the
    parse in ONE place instead of two that can disagree.
    """
    assert metering.require_tenant(real) == _REAL


@pytest.mark.parametrize("slug", ["acme-tenant", "n8n-dev", "anonymous", "unknown"])
def test_require_tenant_refuses_a_non_uuid_slug(slug: str) -> None:
    """A slug must be refused AT THE DOOR, not at the emit.

    This is the load-bearing test for the 0.66.1 -> 0.67.10 lift. A slug used to
    pass this guard and travel all the way to ``schedule_cost_event``, which
    calls ``parse_tenant_id`` synchronously in the caller's frame (deliberately
    outside its never-raises swallow). In this service the emit happens AFTER the
    audio has been transcribed, so a slug produced a 500 on completed, unbillable
    work. Refusing here turns that into a 401 that costs nothing.

    Reverting ``require_tenant`` to the old "non-uuid slugs are allowed" branch
    fails exactly this test.
    """
    with pytest.raises(metering.MissingTenantError):
        metering.require_tenant(slug)


def test_facade_reexports_shared_helpers() -> None:
    # The facade must delegate to w2v_common.metering, not re-implement it.
    assert metering.emit_cost_event is shared.emit_cost_event
    assert metering.schedule_cost_event is shared.schedule_cost_event
    assert metering.init_metering_redis is shared.init_metering_redis
    assert metering.close_metering_redis is shared.close_metering_redis


@pytest.mark.asyncio
async def test_init_metering_redis_empty_url_returns_none() -> None:
    assert await metering.init_metering_redis("") is None


@pytest.mark.asyncio
async def test_close_metering_redis_none_is_noop() -> None:
    await metering.close_metering_redis(None)  # must not raise
