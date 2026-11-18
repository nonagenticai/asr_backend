"""ASR API routes — preserves the transcription_worker_service contract."""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from w2v_common.admission import AdmissionGate, GateClosedError
from w2v_common.auth import LICENSE_GATE
from w2v_common.auth.identity_context import (
    get_license_id_from_context,
    get_subject_id_from_context,
)

from auth.dependencies import get_current_user
from auth.models import User
from models import (
    ModelInfo,
    ModelsResponse,
    TranscriptionResponse,
    TranscriptSegmentResponse,
    TranscriptWordResponse,
)
from routers.responses import ASR_ENDPOINT_RESPONSES, AUTH_RESPONSES
from services.admission import TRANSCRIBE_LANE
from services.metering import (
    COST_SOURCE,
    SERVICE_KEY,
    MissingTenantError,
    require_tenant,
    schedule_cost_event,
)
from services.srt_formatter import format_srt
from services.whisper_service import (
    BackendUnavailableError,
    TranscriptionResult,
    WhisperService,
)

logger = logging.getLogger(__name__)
# Entitlement gate, mounted router-wide — the same mount as tts_backend's
# `tts_routes.py` (w2v_common#222). ``AuthMiddleware`` answers "who are you" and
# stops there: a tenant whose licence is expired or revoked still presents a
# syntactically valid token, and every transcription it asked for was served AND
# billed. LICENSE_GATE is the "may you". It ships DORMANT — `require_license` is a
# no-op until `W2V_LICENSE_GATE_ENABLED` is true in this service's env — so
# merging this changes nothing until the flag is armed, and it never applies to
# the platform's probe paths (`w2v_common.auth.is_license_exempt_path`); the
# health routes also live on `main.app`, not this router.
router = APIRouter(prefix="/api", tags=["Transcription"], dependencies=list(LICENSE_GATE))

# ASR is billed per minute of audio. This advisory cost_usd hint must match the
# metering `provider_pricing` row for provider "mlx-whisper" ($0.017/min); the
# metering service recomputes the authoritative cost from that row regardless.
_ASR_USD_PER_MINUTE = float(os.getenv("ASR_USD_PER_MINUTE", "0.017"))

_AVAILABLE_MODEL_SIZES = [
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v1",
    "large-v2",
    "large-v3",
]

_MAX_AUDIO_BYTES = int(os.getenv("ASR_MAX_AUDIO_BYTES", str(100 * 1024 * 1024)))


def _backend_from_request(request: Request) -> WhisperService:
    backend = getattr(request.app.state, "whisper_service", None)
    if backend is None or not backend.is_loaded:
        raise HTTPException(status_code=503, detail="ASR backend not initialised")
    return backend


def _gate_from_request(request: Request) -> AdmissionGate:
    gate = getattr(request.app.state, "gpu_gate", None)
    if gate is None:
        raise HTTPException(status_code=503, detail="GPU admission gate not initialised")
    return gate


def _transcribe_in_lane(
    backend: WhisperService,
    raw: bytes,
    suffix: str,
    language: str | None,
    word_timestamps: bool,
) -> TranscriptionResult:
    """Runs on the ``transcribe`` lane's executor thread, only once admitted.

    The temp file exists only for the life of this call: a request that is
    rejected (503), times out in the queue, or whose client left before
    admission (499) never touches disk, and an admitted one is unlinked even if
    the decode raises or the awaiting coroutine was cancelled mid-decode.
    """
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(raw)
        return backend.transcribe(tmp_path, language, word_timestamps)
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Failed to delete temp audio file %s", tmp_path)


def _serialize_result(result: TranscriptionResult, include_srt: bool) -> TranscriptionResponse:
    segments = [
        TranscriptSegmentResponse(
            id=seg.id,
            start=seg.start,
            end=seg.end,
            text=seg.text,
            words=[
                TranscriptWordResponse(start=w.start, end=w.end, word=w.word, probability=w.probability)
                for w in seg.words
            ],
            avg_logprob=seg.avg_logprob,
            no_speech_prob=seg.no_speech_prob,
        )
        for seg in result.segments
    ]
    srt_text = format_srt(result.segments) if include_srt else None
    return TranscriptionResponse(
        backend=result.backend,
        model=result.model,
        language=result.language,
        duration=result.duration,
        text=result.text,
        segments=segments,
        srt=srt_text,
    )


def _get_tenant_id(request: Request) -> uuid.UUID:
    """Return the caller's REAL forwarded tenant, or 401 — never a sentinel.

    PRODUCT-OWNER RULING: a ``tenant_id`` may never hold a sentinel. This used
    to be ``getattr(state, "tenant_id", "") or DEFAULT_TENANT_ID``, substituting
    the nil UUID — a VALID uuid that passes every cast, which billing-service
    then buckets as unattributed and NEVER CHARGES. Real transcription minutes,
    silently unbilled. Do not re-add a fallback here.

    ``require_tenant`` also rejects the nil UUID BY VALUE, not just a missing
    one: ``w2v_common.auth.middleware`` still writes it onto
    ``request.state.tenant_id`` on the legacy trusted-network path, so a
    ``None``-only guard would leave the hole open from upstream.

    401 (not 400): ``request.state.tenant_id`` is populated exclusively by
    ``AuthMiddleware`` from the ``X-Tenant-Id: Bearer <license_token>``
    handshake, so its absence means authentication resolved no identity, not
    that the multipart payload was malformed. Same choice as ``slide_service``
    and the ``tts_service`` twin. (``video_compiler_service`` returns 400
    because there the tenant is a uuid COLUMN it must parse and store;
    asr-backend stores nothing and only forwards the value onto the cost bus.)
    """
    try:
        return require_tenant(getattr(request.state, "tenant_id", None))
    except MissingTenantError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def _emit_asr_cost_event(request: Request, result: TranscriptionResult, latency_ms: int, tenant_id: uuid.UUID) -> None:
    """Fire-and-forget a per-transcription cost event to the cost-bus stream.

    Bills per minute of audio: input_tokens = whole seconds of audio,
    output_tokens = 0, provider = "mlx-whisper" (must match the pricing row).
    No-op when metering is disabled (METERING_REDIS_URL unset). ``tenant_id`` is
    the already-validated real tenant from :func:`_get_tenant_id`; this function
    never resolves or substitutes one itself.
    """
    metering_redis = getattr(request.app.state, "metering_redis", None)
    if metering_redis is None:
        return
    audio_seconds = max(0.0, float(result.duration))
    cost_usd = (audio_seconds / 60.0) * _ASR_USD_PER_MINUTE
    # GC-safe fire-and-forget: schedule_cost_event owns the task lifecycle and
    # never raises, so a Redis hiccup never blocks the transcription response.
    schedule_cost_event(
        metering_redis,
        service_key=SERVICE_KEY,
        cost_source=COST_SOURCE,
        tenant_id=tenant_id,
        # license_id / subject_id are NOT flat request.state attrs under
        # w2v_common — read them from the identity ContextVar so metering
        # attribution stays correct.
        license_id=get_license_id_from_context() or "",
        subject_id=get_subject_id_from_context() or "",
        app_id="",
        provider="mlx-whisper",
        model=result.model,
        input_tokens=round(audio_seconds),
        output_tokens=0,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        usage_extra={
            "unit": "seconds",
            "duration_s": audio_seconds,
            "language": result.language or "",
        },
    )


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    operation_id="transcribe_audio",
    summary="Transcribe an audio file to text + segments (+ optional SRT)",
    responses=ASR_ENDPOINT_RESPONSES,
)
async def transcribe_endpoint(
    request: Request,
    file: Annotated[UploadFile, File(..., description="Audio file to transcribe")],
    _user: Annotated[User, Depends(get_current_user)],
    language: Annotated[str | None, Form()] = None,
    word_timestamps: Annotated[bool, Form()] = True,
    return_srt: Annotated[bool, Form()] = False,
) -> TranscriptionResponse:
    # Resolve the tenant BEFORE any work: a tenant-less request must cost
    # nothing and emit nothing, not transcribe and then fail to attribute.
    tenant_id = _get_tenant_id(request)
    backend = _backend_from_request(request)

    gate = _gate_from_request(request)

    try:
        raw = await file.read()
    finally:
        # Drop the multipart spool now: the bytes are in memory, and a request
        # parked in the queue (or rejected) must not keep a spooled file open.
        await file.close()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio payload")
    if len(raw) > _MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio payload exceeds size limit")

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    started = time.monotonic()
    try:
        # GpuBusy (503 + Retry-After) and ClientGone (499) propagate to the
        # handlers installed by install_gpu_busy_handler; neither ran a decode.
        result = await gate.run(
            TRANSCRIBE_LANE,
            _transcribe_in_lane,
            backend,
            raw,
            suffix,
            language,
            word_timestamps,
            request=request,
        )
    except GateClosedError as exc:
        # Shutting down: no decode ran. Must precede the RuntimeError arm below
        # (GateClosedError subclasses RuntimeError) or it would read as a 500.
        raise HTTPException(status_code=503, detail="ASR backend is shutting down") from exc
    except BackendUnavailableError as exc:
        logger.error("Backend unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
    latency_ms = int((time.monotonic() - started) * 1000)

    _emit_asr_cost_event(request, result, latency_ms, tenant_id)
    return _serialize_result(result, include_srt=return_srt)


@router.get(
    "/models",
    response_model=ModelsResponse,
    operation_id="list_asr_models",
    summary="List the currently-loaded ASR model and available sizes",
    responses={**AUTH_RESPONSES, 503: {"description": "Backend not initialised"}},
)
async def list_models(
    request: Request,
    _user: Annotated[User, Depends(get_current_user)],
) -> ModelsResponse:
    backend = _backend_from_request(request)
    info = ModelInfo(
        backend=backend.name,
        model=backend.model_size,
        device=backend.device,
        compute_type=backend.compute_type,
        loaded=bool(backend.is_loaded),
    )
    return ModelsResponse(current=info, available_sizes=_AVAILABLE_MODEL_SIZES)
