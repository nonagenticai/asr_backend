"""GPU admission control on /api/transcribe (nonagenticai/asr_backend#34).

Drives the REAL ``main.app`` (all middleware + installed GpuBusy/ClientGone
handlers) over httpx's ASGI transport, with the mlx-whisper backend replaced by a
stub whose ``transcribe`` blocks on an event, so the lane can be saturated
deterministically.

Contract pinned here (lane ``transcribe``, code defaults 1 / 8 / 60 s):
- exactly ``max_active`` decodes run at once; ``max_waiting`` more queue;
- everything beyond is an IMMEDIATE 503 + ``Retry-After`` + gpu_busy body;
- /health stays < 200 ms while the lane is saturated (decode is off-loop);
- a client that is gone before admission is skipped — the decode never runs;
- the temp audio file exists only while its decode runs: a queued, rejected or
  skipped request never writes one, and an admitted one is always unlinked.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, MutableMapping
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from services.admission import TRANSCRIBE_DEFAULTS, TRANSCRIBE_LANE, build_gate
from services.whisper_service import TranscriptionResult

REAL_TENANT = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
AUDIO = {"file": ("a.wav", b"RIFFfake-audio", "audio/wav")}


class _BlockingBackend:
    """Stub WhisperService: every decode blocks until ``gate_open`` is set."""

    is_loaded = True
    name = "stub"
    model_size = "stub-model"
    device = "cpu"
    compute_type = "fp16"

    def __init__(self, tmp_dir: Path, *, block: bool = True, raises: Exception | None = None) -> None:
        self.gate_open = threading.Event()
        if not block:
            self.gate_open.set()
        self.raises = raises
        self.tmp_dir = tmp_dir
        self.calls = 0
        self.running = 0
        self.max_running = 0
        self.seen_paths: list[Path] = []
        self._lock = threading.Lock()

    def transcribe(self, path: Path, *_a: object, **_kw: object) -> TranscriptionResult:
        assert path.exists(), "decode must receive a real on-disk temp file"
        with self._lock:
            self.calls += 1
            self.running += 1
            self.max_running = max(self.max_running, self.running)
            self.seen_paths.append(path)
        try:
            if not self.gate_open.wait(timeout=30):
                raise AssertionError("test never released the stub decode")
            if self.raises is not None:
                raise self.raises
            return TranscriptionResult(
                backend="stub", model="stub-model", language="en", duration=60.0, text="hi", segments=[]
            )
        finally:
            with self._lock:
                self.running -= 1


class _TenantForwarder:
    """ASGI wrapper that forwards a real tenant exactly as AuthMiddleware would.

    ``request.state`` is backed by ``scope["state"]``, so seeding it here is what
    the auth layer does in production (auth itself is off in the test env).
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Callable[[], Awaitable[MutableMapping[str, Any]]],
        send: Callable[[MutableMapping[str, Any]], Awaitable[None]],
    ) -> None:
        if scope["type"] == "http":
            scope.setdefault("state", {})["tenant_id"] = REAL_TENANT
        await self.app(scope, receive, send)


@pytest.fixture
def tmp_audio_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point tempfile at an isolated dir so the test can count temp audio files."""
    import tempfile

    d = tmp_path / "tmp-audio"
    d.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(d))
    return d


@pytest.fixture
def wired_app(tmp_audio_dir: Path) -> Iterator[tuple[Any, Any]]:
    """main.app with a fresh gate (code defaults, env ignored) + blocking stub backend."""
    from main import app

    saved = {k: getattr(app.state, k, None) for k in ("whisper_service", "metering_redis", "gpu_gate")}
    backend = _BlockingBackend(tmp_audio_dir)
    app.state.whisper_service = backend
    app.state.metering_redis = None
    app.state.gpu_gate = build_gate(environ={})
    try:
        yield app, backend
    finally:
        backend.gate_open.set()
        for k, v in saved.items():
            setattr(app.state, k, v)


@pytest.fixture
async def http(wired_app: tuple[Any, Any]) -> AsyncIterator[httpx.AsyncClient]:
    app, _ = wired_app
    transport = httpx.ASGITransport(app=_TenantForwarder(app))
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as ac:
        yield ac


async def _wait_for(pred: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.01)


def test_code_defaults_are_the_issue_numbers() -> None:
    assert TRANSCRIBE_LANE == "transcribe"
    assert (TRANSCRIBE_DEFAULTS.max_active, TRANSCRIBE_DEFAULTS.max_waiting, TRANSCRIBE_DEFAULTS.max_wait_s) == (
        1,
        8,
        60.0,
    )
    cfg = build_gate(environ={}).lane("transcribe").config
    assert (cfg.max_active, cfg.max_waiting, cfg.max_wait_s) == (1, 8, 60.0)


def test_env_overrides_the_lane() -> None:
    env = {"GPU_LANE_TRANSCRIBE_MAX_WAITING": "64", "GPU_LANE_TRANSCRIBE_MAX_WAIT_S": "120"}
    cfg = build_gate(environ=env).lane("transcribe").config
    assert (cfg.max_active, cfg.max_waiting, cfg.max_wait_s) == (1, 64, 120.0)


def test_malformed_env_fails_loud() -> None:
    with pytest.raises(ValueError, match="GPU_LANE_TRANSCRIBE_MAX_ACTIVE"):
        build_gate(environ={"GPU_LANE_TRANSCRIBE_MAX_ACTIVE": "lots"})


async def test_burst_of_20_admits_1_queues_8_rejects_11(
    wired_app: tuple[Any, _BlockingBackend], http: httpx.AsyncClient, tmp_audio_dir: Path
) -> None:
    app, backend = wired_app
    lane = app.state.gpu_gate.lane("transcribe")

    tasks = [asyncio.create_task(http.post("/api/transcribe", files=AUDIO)) for _ in range(20)]

    # Saturation: 1 decode running, 8 parked in the queue.
    await _wait_for(lambda: backend.running == 1 and lane.waiting == 8 and sum(t.done() for t in tasks) == 11)

    rejected = [t.result() for t in tasks if t.done()]
    assert len(rejected) == 11
    for r in rejected:
        assert r.status_code == 503, r.text
        assert r.json() == {"error": "gpu_busy", "lane": "transcribe", "retry_after": r.json()["retry_after"]}
        assert 1 <= int(r.headers["Retry-After"]) <= 120
        assert r.headers["Retry-After"] == str(r.json()["retry_after"])

    # Only the RUNNING decode has a temp file; the 8 queued + 11 rejected wrote none.
    assert len(list(tmp_audio_dir.iterdir())) == 1

    # /health answers fast while the lane is saturated (decode is off the loop).
    t0 = time.perf_counter()
    h = await http.get("/health")
    assert h.status_code == 200
    assert time.perf_counter() - t0 < 0.2

    # /metrics renders the gate's series from the default registry.
    m = await http.get("/metrics")
    assert 'gpu_admission_active{lane="transcribe"} 1.0' in m.text
    assert 'gpu_admission_waiting{lane="transcribe"} 8.0' in m.text
    assert 'gpu_admission_rejected_total{lane="transcribe",reason="full"}' in m.text

    backend.gate_open.set()
    done = await asyncio.gather(*tasks)
    assert sorted(r.status_code for r in done) == [200] * 9 + [503] * 11
    assert backend.calls == 9
    assert backend.max_running == 1
    assert list(tmp_audio_dir.iterdir()) == []
    assert lane.active == 0 and lane.waiting == 0


async def test_queued_request_past_max_wait_gets_503_timeout(
    wired_app: tuple[Any, _BlockingBackend], tmp_audio_dir: Path
) -> None:
    app, backend = wired_app
    app.state.gpu_gate = build_gate(environ={"GPU_LANE_TRANSCRIBE_MAX_WAIT_S": "0.2"})
    transport = httpx.ASGITransport(app=_TenantForwarder(app))
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as ac:
        first = asyncio.create_task(ac.post("/api/transcribe", files=AUDIO))
        await _wait_for(lambda: backend.running == 1)
        r = await ac.post("/api/transcribe", files=AUDIO)
        assert r.status_code == 503
        assert r.json()["error"] == "gpu_busy"
        assert "Retry-After" in r.headers
        backend.gate_open.set()
        assert (await first).status_code == 200
    assert backend.calls == 1
    assert list(tmp_audio_dir.iterdir()) == []


_BOUNDARY = "xBOUNDARYx"
_MULTIPART = (
    f'--{_BOUNDARY}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
    f"Content-Type: audio/wav\r\n\r\nRIFFfake-audio\r\n--{_BOUNDARY}--\r\n"
).encode()


async def _raw_transcribe(app: Any, gone: asyncio.Event) -> list[MutableMapping[str, Any]]:
    """POST /api/transcribe as a raw ASGI call; the client disconnects once ``gone`` is set.

    Like a real server: after the body, ``receive()`` blocks until the client
    leaves, then returns ``http.disconnect``. Goes through the REAL ``main.app``
    stack (DisconnectProbe + RequestId + SecurityHeaders + RequestLogging, three
    of them BaseHTTPMiddleware) — the layers that hide a disconnect from
    ``Request.is_disconnected()``.
    """
    messages: list[dict[str, Any]] = [{"type": "http.request", "body": _MULTIPART, "more_body": False}]

    async def receive() -> dict[str, Any]:
        if messages:
            return messages.pop(0)
        await gone.wait()
        return {"type": "http.disconnect"}

    sent: list[MutableMapping[str, Any]] = []

    async def send(message: MutableMapping[str, Any]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/transcribe",
        "raw_path": b"/api/transcribe",
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"host", b"test"),
            (b"content-type", f"multipart/form-data; boundary={_BOUNDARY}".encode()),
            (b"content-length", str(len(_MULTIPART)).encode()),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    }
    await _TenantForwarder(app)(scope, receive, send)
    return sent


async def test_client_gone_before_admission_skips_the_decode(
    wired_app: tuple[Any, _BlockingBackend], tmp_audio_dir: Path
) -> None:
    """Client already gone once the body is read, lane free: skipped, 499."""
    app, backend = wired_app
    backend.gate_open.set()
    gone = asyncio.Event()
    gone.set()
    sent = await _raw_transcribe(app, gone)

    assert backend.calls == 0, "decode ran for a client that was already gone"
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 499
    assert list(tmp_audio_dir.iterdir()) == []
    assert app.state.gpu_gate.lane("transcribe").active == 0


async def test_client_gone_while_queued_skips_the_decode(
    wired_app: tuple[Any, _BlockingBackend], http: httpx.AsyncClient, tmp_audio_dir: Path
) -> None:
    """Client leaves while parked behind a running decode: on admission it is skipped (499)."""
    app, backend = wired_app
    lane = app.state.gpu_gate.lane("transcribe")
    first = asyncio.create_task(http.post("/api/transcribe", files=AUDIO))
    await _wait_for(lambda: backend.running == 1)

    gone = asyncio.Event()
    queued = asyncio.create_task(_raw_transcribe(app, gone))
    await _wait_for(lambda: lane.waiting == 1)
    gone.set()  # client hangs up while queued
    await asyncio.sleep(0.05)
    backend.gate_open.set()  # running decode ends -> the queued request is admitted

    sent = await asyncio.wait_for(queued, timeout=10)
    assert (await first).status_code == 200
    assert backend.calls == 1, "decode ran for a client that left while queued"
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 499
    assert list(tmp_audio_dir.iterdir()) == []
    assert lane.active == 0 and lane.waiting == 0


async def test_failed_decode_still_unlinks_temp_and_frees_slot(
    wired_app: tuple[Any, _BlockingBackend], http: httpx.AsyncClient, tmp_audio_dir: Path
) -> None:
    app, backend = wired_app
    backend.raises = RuntimeError("metal oom")
    backend.gate_open.set()
    r = await http.post("/api/transcribe", files=AUDIO)
    assert r.status_code == 500
    assert backend.calls == 1
    assert not backend.seen_paths[0].exists()
    assert list(tmp_audio_dir.iterdir()) == []
    assert app.state.gpu_gate.lane("transcribe").active == 0


async def test_oversize_upload_is_413_before_the_gate(
    wired_app: tuple[Any, _BlockingBackend], http: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from routers import asr_routes

    app, backend = wired_app
    monkeypatch.setattr(asr_routes, "_MAX_AUDIO_BYTES", 4)
    r = await http.post("/api/transcribe", files=AUDIO)
    assert r.status_code == 413
    assert backend.calls == 0
    assert app.state.gpu_gate.lane("transcribe").active == 0


async def test_closed_gate_is_503_not_500(wired_app: tuple[Any, _BlockingBackend], http: httpx.AsyncClient) -> None:
    app, backend = wired_app
    await app.state.gpu_gate.shutdown(timeout=1)
    r = await http.post("/api/transcribe", files=AUDIO)
    assert r.status_code == 503
    assert "shutting down" in r.text
    assert backend.calls == 0


def test_lifespan_builds_and_shuts_down_the_gate() -> None:
    from main import app

    saved = getattr(app.state, "gpu_gate", None)
    try:
        with TestClient(app):
            gate = app.state.gpu_gate
            assert not gate.lane("transcribe").closed
        assert gate.lane("transcribe").closed
    finally:
        app.state.gpu_gate = saved
