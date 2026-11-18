"""Cost metering — thin facade over the fleet-canonical ``w2v_common.metering``.

All emit / stream-resolution / field-shaping / Redis-connection logic lives in
``w2v_common.metering`` (the shared helper). This module keeps only the
service-specific routing identity (``COST_SOURCE`` / ``SERVICE_KEY``) and the
"a tenant is REQUIRED" contract, and re-exports the helper surface so the rest
of asr-backend has a single import point.

mlx-whisper runs locally on Apple Silicon, so the emitted ``cost_usd`` is only
an advisory hint — the metering service recomputes the authoritative cost from
the centralised ``provider_pricing`` table (ASR is priced per minute of audio
against the ``mlx-whisper`` provider row). The billable unit is audio duration:
callers pass ``input_tokens`` = whole seconds of audio, ``output_tokens`` = 0.
"""

from __future__ import annotations

import uuid as _uuid

from w2v_common.metering import (
    close_metering_redis,
    emit_cost_event,
    init_metering_redis,
    schedule_cost_event,
)
from w2v_common.tenant import TenantError, parse_tenant_id

COST_SOURCE = "w2v-asr"
SERVICE_KEY = "w2v-asr"  # pragma: allowlist secret -- routing slug, not a credential

# ── PRODUCT-OWNER RULING: a tenant_id may NEVER hold a sentinel ──────────────
# Do NOT re-add ``DEFAULT_TENANT_ID``, ``UNATTRIBUTED_TENANT_ID``, a
# ``DEFAULT_TENANT_ID`` env read, or ``config.auth.default_tenant_id`` here.
# Every message carries a REAL registered tenant forwarded from the requestor;
# an absent tenant is an ERROR (HTTP 401 at the route layer), never a
# substituted default.
#
# The nil UUID was the most dangerous spelling of the substitution: it is a
# VALID uuid, so it passes every uuid cast and dirty-row check, and billing then
# buckets it as unattributed and NEVER CHARGES it (billing-service
# ``_parsers.py`` + ``cost_flusher.py``). Real usage, silently unbilled.

_NIL_TENANT = _uuid.UUID(int=0)


class MissingTenantError(ValueError):
    """No real tenant was forwarded on a call that MUST be tenant-scoped.

    FAIL LOUD, never substitute. Raised by :func:`require_tenant` and mapped to
    HTTP 401 by the route helpers in :mod:`routers.asr_routes` and by
    :func:`auth.dependencies.get_current_user`.
    """

    def __init__(self, field: str = "tenant_id") -> None:
        super().__init__(
            f"{field} is required and must be a real registered tenant forwarded by the "
            "caller — no sentinel, no default, no substitution. Nothing was recorded."
        )


def require_tenant(tenant_id: object, field: str = "tenant_id") -> _uuid.UUID:
    """Return the forwarded tenant as a real ``uuid.UUID``, or raise.

    Rejects everything that is not a usable tenant:

    * ``None`` / non-``str`` / blank — nothing was forwarded at all;
    * a **non-UUID slug** (``n8n-dev``, ``anonymous``) — see below;
    * the **nil UUID in any spelling** — this is the load-bearing half. A
      ``None``-only guard would leave the hole OPEN: the nil UUID is a VALID
      uuid, so it survives every cast, and billing then buckets it as
      unattributed and NEVER CHARGES it. Spellings collapse for free now that
      the value is parsed: braced, urn-prefixed, upper-case and dash-less forms
      all become the same ``UUID(int=0)``.

    Why slugs are refused HERE, from 0.67.10 (they used to pass through). The
    cost bus parses the tenant itself: ``schedule_cost_event`` calls
    ``parse_tenant_id`` SYNCHRONOUSLY in the caller's frame, deliberately
    outside its own never-raises swallow. So a slug no longer reaches the stream
    — it raises ``InvalidTenantIdError`` at the emit call, which in this service
    happens AFTER the audio has already been transcribed: a 500 on completed,
    unbillable work. Refusing at the door turns that into a 401 that costs
    nothing, which is what ``asr_routes`` already says it wants ("Resolve the
    tenant BEFORE any work"). This is not a new restriction in practice:
    ``AuthMiddleware`` writes ``str(tenant_id)`` of a parsed ``uuid.UUID`` at
    every one of its three lanes, so a slug cannot arrive from upstream anyway.
    """
    try:
        parsed = parse_tenant_id(tenant_id, source=field)
    except TenantError as exc:
        # Collapsed to one local error on purpose: the route layer maps this to
        # 401 and the caller cannot act on the missing/invalid distinction.
        raise MissingTenantError(field) from exc
    if parsed == _NIL_TENANT:
        raise MissingTenantError(field)
    return parsed


__all__ = [
    "COST_SOURCE",
    "SERVICE_KEY",
    "MissingTenantError",
    "close_metering_redis",
    "emit_cost_event",
    "init_metering_redis",
    "require_tenant",
    "schedule_cost_event",
]
