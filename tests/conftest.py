"""Shared fixtures + module-level test environment.

Sets the env vars the FastAPI app reads at import time, stubs the
Apple-Silicon-only `mlx_whisper` + `mlx.core` modules so the rest of the
codebase imports cleanly on Linux CI, and exposes an async HTTP client
fixture that talks to the real ASGI app (no network).
"""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

# --- Env setup (must happen BEFORE `main` is imported) ---
os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("CORS_ORIGINS", "http://localhost:3000")
os.environ.setdefault("LICENSE_INTROSPECT_URL", "http://license-introspect.test/license/introspect")
os.environ.setdefault("ASR_MODEL", "stub-model")


# --- Stub Apple-Silicon-only deps so tests import on any platform ---
def _install_stubs() -> None:
    fake_whisper = cast(Any, type(sys)("mlx_whisper"))

    def _transcribe(*_a: object, **_kw: object) -> dict[str, Any]:
        return {"text": "", "segments": [], "language": "en", "duration": 0.0}

    fake_whisper.transcribe = _transcribe

    fake_mlx = cast(Any, type(sys)("mlx"))
    fake_mlx_core = cast(Any, type(sys)("mlx.core"))

    class _StreamCtx:
        def __enter__(self) -> _StreamCtx:
            return self

        def __exit__(self, *_a: object) -> bool:
            return False

    fake_mlx_core.gpu = object()
    fake_mlx_core.stream = lambda _g: _StreamCtx()
    fake_mlx.core = fake_mlx_core

    sys.modules.setdefault("mlx_whisper", fake_whisper)
    sys.modules.setdefault("mlx", fake_mlx)
    sys.modules.setdefault("mlx.core", fake_mlx_core)


_install_stubs()


# --- Async client fixture ---
@pytest.fixture
async def client() -> AsyncIterator[Any]:
    import httpx

    # Import inside the fixture so env + stubs are set before `main` is loaded.
    from main import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        # Trigger lifespan via ASGITransport's startup
        yield ac


@pytest.fixture
def test_client() -> Any:
    """Synchronous TestClient — drives FastAPI lifespan."""
    from fastapi.testclient import TestClient

    from main import app

    return TestClient(app)
