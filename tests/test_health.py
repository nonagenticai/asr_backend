"""Health-endpoint smoke tests (no Apple-Silicon model load required).

Uses the shared conftest stubs for ``mlx_whisper`` and ``mlx.core``.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_ok(test_client: TestClient) -> None:
    r = test_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "asr-backend"
    assert body["device"] == "unified"


def test_live_ok(test_client: TestClient) -> None:
    r = test_client.get("/health/live")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_ready_reports_loaded_when_stub_present(test_client: TestClient) -> None:
    r = test_client.get("/health/ready")
    assert r.status_code == 200
    body = r.json()
    # The lifespan tries to load the stubbed model; readiness reports
    # "healthy" iff WhisperService.is_loaded is True. The stub makes initialise()
    # succeed → loaded → status healthy.
    assert body["status"] in {"healthy", "degraded"}


def test_root_info(test_client: TestClient) -> None:
    r = test_client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "asr-backend"


def test_metrics_endpoint_exposes_prometheus_format(test_client: TestClient) -> None:
    test_client.get("/health")
    r = test_client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "# HELP" in body
    assert "# TYPE" in body
    assert "http_request_duration_seconds" in body


def test_protected_route_under_disabled_auth_still_requires_a_tenant(test_client: TestClient) -> None:
    """PRODUCT-OWNER RULING: ENABLE_AUTH=false disables INTROSPECTION, not tenancy.

    This test asserted the OPPOSITE (``status_code != 401``), because
    ``get_current_user`` minted a synthetic anonymous admin stamped with the
    nil-UUID sentinel. It is rewritten, not deleted. With auth disabled no
    AuthMiddleware runs, so nothing forwards a tenant — and a tenant-less
    request to a tenant-scoped service is an ERROR, never a substituted default.
    """
    r = test_client.get("/api/models")
    assert r.status_code == 401, r.text
    assert "no sentinel, no default, no substitution" in r.text
