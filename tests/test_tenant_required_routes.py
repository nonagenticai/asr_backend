"""Route-level pin: a tenant-less transcription FAILS and BILLS NOTHING.

PRODUCT-OWNER RULING: a ``tenant_id`` may NEVER hold a sentinel. Both halves of
this contract are load-bearing and are asserted together:

1. no tenant  -> HTTP 401 (the request does not succeed), and
2. no tenant  -> ``redis.xadd`` is NEVER awaited (nothing reaches the cost bus).

Half (2) is the one that actually cost money. The old code substituted the nil
UUID, so the request SUCCEEDED and emitted a cost event that billing-service
buckets as unattributed and never charges — real transcription minutes, silently
unbilled. A test asserting only the 401 would still pass if someone re-added a
substitution behind a soft-fail.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from routers.asr_routes import router
from services.admission import build_gate
from services.whisper_service import TranscriptionResult


class _StubBackend:
    is_loaded = True
    name = "stub"
    model_size = "stub-model"
    device = "cpu"
    compute_type = "fp16"

    def transcribe(self, *_a: object, **_kw: object) -> TranscriptionResult:
        return TranscriptionResult(
            backend="stub", model="stub-model", language="en", duration=60.0, text="hi", segments=[]
        )


def _app(tenant_id: object) -> tuple[FastAPI, AsyncMock]:
    """App whose middleware forwards ``tenant_id`` exactly as AuthMiddleware would."""
    app = FastAPI()

    @app.middleware("http")
    async def _forward_tenant(request: Request, call_next: Any) -> Any:
        if tenant_id is not None:
            request.state.tenant_id = tenant_id
        return await call_next(request)

    app.include_router(router)
    redis = AsyncMock()
    app.state.whisper_service = _StubBackend()
    app.state.metering_redis = redis
    app.state.gpu_gate = build_gate(environ={})
    return app, redis


_SENTINELS: list[object] = [
    None,  # nothing forwarded at all
    "",  # blank
    "   ",  # whitespace
    "00000000-0000-0000-0000-000000000000",  # the nil UUID, from w2v_common upstream
    "00000000000000000000000000000000",  # same sentinel, undashed
]


@pytest.mark.parametrize("tenant_id", _SENTINELS)
def test_tenantless_transcribe_is_401_and_emits_no_cost_event(tenant_id: object) -> None:
    app, redis = _app(tenant_id)
    with TestClient(app) as client:
        r = client.post("/api/transcribe", files={"file": ("a.wav", b"RIFFfake", "audio/wav")})

    # Assert the money half FIRST so it cannot be masked by the status assertion.
    redis.xadd.assert_not_awaited()
    assert r.status_code == 401, r.text
    assert "no sentinel, no default, no substitution" in r.text


def test_real_tenant_transcribes_and_emits_a_cost_event_with_that_tenant() -> None:
    """Positive control: the 401 above is the tenant check, not a broken route."""
    real = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
    app, redis = _app(real)
    with TestClient(app) as client:
        r = client.post("/api/transcribe", files={"file": ("a.wav", b"RIFFfake", "audio/wav")})

    assert r.status_code == 200, r.text
    redis.xadd.assert_awaited()
    fields = redis.xadd.await_args.args[1]
    assert fields["tenant_id"] == real
