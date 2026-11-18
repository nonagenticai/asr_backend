"""The entitlement gate on the ASR router actually refuses an unlicensed tenant.

Mirror of tts_backend's ``tests/test_license_gate.py`` (w2v_common#222). Until
this change ASR authenticated the licence token for tenant IDENTITY and stopped
there: an expired or revoked tenant was transcribed and billed. ``LICENSE_GATE``
is mounted router-wide on ``src/routers/asr_routes.py`` to close that.

Every test drives a REAL HTTP request through the REAL router: FastAPI includes
routers lazily, so static introspection of ``route.dependant`` would report
these routes as ungated (a false negative). The paired 200s (entitled, flag-off)
stop the file passing vacuously.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
import w2v_common.auth.middleware as w2v_auth
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient
from w2v_common.auth import clear_license_status_cache
from w2v_common.auth.licenses_read_client import LicensesClientError

from auth.middleware import AuthMiddleware
from routers.asr_routes import router as asr_router
from services.admission import build_gate
from services.whisper_service import TranscriptionResult

TENANT = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
SUBJECT = "user-under-test"

#: Cheap gated route: reads only the stub backend, no decode, no cost emit.
GATED_PATH = "/api/models"
#: The BILLED route. Gating only /models would leave the revenue leak open.
GATED_TRANSCRIBE_PATH = "/api/transcribe"
PROBE_PATH = "/health/live"
_AUDIO = {"file": ("a.wav", b"RIFFfake", "audio/wav")}


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Record:
    def __init__(self, **fields: Any) -> None:
        self._fields = fields

    def model_dump(self) -> dict[str, Any]:
        return dict(self._fields)


class _Detail(_Record):
    conflict = False


class _FakeLicenses:
    """Stands in for ``LicensesReader``; counts calls so a test can prove the
    gate was (or was not) consulted."""

    def __init__(
        self,
        *,
        active: _Detail | None = None,
        inventory: list[_Record] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._active = active
        self._inventory = inventory or []
        self._error = error
        self.calls = 0

    async def get_active_license(self, *, tenant_id: str, app_key: str) -> _Detail | None:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._active

    async def list_user_licenses(self, subject_id: str, *, tenant_id: str) -> list[_Record]:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return list(self._inventory)


def _revoked() -> list[_Record]:
    return [_Record(app={"key": "w2v"}, tenant_id=TENANT, status="revoked")]


def _expired() -> list[_Record]:
    return [_Record(app={"key": "w2v"}, tenant_id=TENANT, status="expired", expires_at="2020-01-01T00:00:00+00:00")]


def _entitled() -> _Detail:
    return _Detail(
        app={"key": "w2v"},
        tenant_id=TENANT,
        status="active",
        expires_at="2099-01-01T00:00:00+00:00",
        plan={"code": "pro"},
    )


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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_cached_verdicts() -> Any:
    clear_license_status_cache()
    yield
    clear_license_status_cache()


@pytest.fixture(autouse=True)
def _gate_disarmed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default the flag OFF, exactly as it ships. Each test arms it explicitly."""
    monkeypatch.delenv("W2V_LICENSE_GATE_ENABLED", raising=False)


def _wire(app: FastAPI) -> AsyncMock:
    app.include_router(asr_router)
    redis = AsyncMock()
    app.state.whisper_service = _StubBackend()
    app.state.metering_redis = redis
    app.state.gpu_gate = build_gate(environ={})
    return redis


@pytest.fixture
def gated_app() -> tuple[FastAPI, AsyncMock]:
    """The REAL router with identity stamped the way AuthMiddleware stamps it, so
    a refusal can only be the licence verdict, not a missing tenant."""
    application = FastAPI()

    @application.middleware("http")
    async def _stamp_identity(request: Request, call_next: Any) -> Any:
        request.state.tenant_id = TENANT
        request.state.subject_id = SUBJECT
        return await call_next(request)

    return application, _wire(application)


async def _get(app: FastAPI, path: str) -> Any:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.get(path)


async def _post_audio(app: FastAPI) -> Any:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.post(GATED_TRANSCRIBE_PATH, files=_AUDIO)


def _licenses(monkeypatch: pytest.MonkeyPatch, fake: _FakeLicenses) -> None:
    monkeypatch.setattr("w2v_common.auth.entitlement.get_licenses_read_client", lambda *a, **k: fake)


# ---------------------------------------------------------------------------
# Unauthenticated: 401 from the auth layer, gate armed or not
# ---------------------------------------------------------------------------


async def test_unauthenticated_transcribe_is_401_with_gate_armed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Production shape: real AuthMiddleware in front of the gated router, no
    token. 401 — and the licences service is never consulted, nothing billed."""
    w2v_auth._license_cache = None
    w2v_auth._license_neg_cache = None
    w2v_auth._introspect_locks.clear()
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    fake = _FakeLicenses(active=_entitled())
    _licenses(monkeypatch, fake)
    application = FastAPI()
    application.add_middleware(AuthMiddleware)
    redis = _wire(application)

    response = await _post_audio(application)

    redis.xadd.assert_not_awaited()
    assert response.status_code == 401, response.text
    assert fake.calls == 0


# ---------------------------------------------------------------------------
# Armed: an unentitled tenant is refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("inventory", "code"),
    [([], "license_none"), (_revoked(), "license_revoked"), (_expired(), "license_expired")],
    ids=["none", "revoked", "expired"],
)
async def test_armed_gate_402s_an_unentitled_tenant(
    gated_app: tuple[FastAPI, AsyncMock], monkeypatch: pytest.MonkeyPatch, inventory: list[_Record], code: str
) -> None:
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=inventory))

    response = await _get(gated_app[0], GATED_PATH)

    assert response.status_code == 402, response.text
    assert response.json()["detail"]["error"] == code


async def test_armed_gate_402s_the_billed_transcribe_route(
    gated_app: tuple[FastAPI, AsyncMock], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refused BEFORE the body runs: no decode, no cost event."""
    app, redis = gated_app
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _post_audio(app)

    redis.xadd.assert_not_awaited()
    assert response.status_code == 402, response.text
    assert response.json()["detail"]["error"] == "license_revoked"


async def test_armed_gate_503s_when_entitlement_cannot_be_established(
    gated_app: tuple[FastAPI, AsyncMock], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(error=LicensesClientError("licenses unreachable")))

    response = await _get(gated_app[0], GATED_PATH)

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["error"] == "license_unavailable"
    assert response.headers["Retry-After"]


# ---------------------------------------------------------------------------
# Armed: an entitled tenant is served (the non-vacuity pair)
# ---------------------------------------------------------------------------


async def test_armed_gate_serves_an_entitled_tenant_on_transcribe(
    gated_app: tuple[FastAPI, AsyncMock], monkeypatch: pytest.MonkeyPatch
) -> None:
    app, redis = gated_app
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    fake = _FakeLicenses(active=_entitled())
    _licenses(monkeypatch, fake)

    response = await _post_audio(app)

    assert response.status_code == 200, response.text
    assert response.json()["text"] == "hi"
    assert fake.calls >= 1  # the gate really ran
    redis.xadd.assert_awaited()


# ---------------------------------------------------------------------------
# Dormant on merge
# ---------------------------------------------------------------------------


async def test_gate_is_dormant_while_the_flag_is_unset(
    gated_app: tuple[FastAPI, AsyncMock], monkeypatch: pytest.MonkeyPatch
) -> None:
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _get(gated_app[0], GATED_PATH)

    assert response.status_code == 200, response.text


async def test_gate_is_dormant_when_the_flag_is_explicitly_false(
    gated_app: tuple[FastAPI, AsyncMock], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "false")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _get(gated_app[0], GATED_PATH)

    assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# Probes are never gated
# ---------------------------------------------------------------------------


async def test_the_real_apps_probe_is_served_for_a_revoked_tenant(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    fake = _FakeLicenses(active=None, inventory=_revoked())
    _licenses(monkeypatch, fake)

    for path in (PROBE_PATH, "/health"):
        response = await client.get(path)
        assert response.status_code == 200, (path, response.text)
    assert fake.calls == 0


async def test_the_services_own_probe_paths_are_exempt_by_path() -> None:
    from w2v_common.auth import is_license_exempt_path

    for path in ("/health", "/health/live", "/health/ready", "/healthz", "/readyz"):
        assert is_license_exempt_path(path), path
    assert not is_license_exempt_path(GATED_PATH)
    assert not is_license_exempt_path(GATED_TRANSCRIBE_PATH)
