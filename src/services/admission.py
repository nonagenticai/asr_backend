"""GPU admission control for asr-backend (nonagenticai/asr_backend#34).

One lane, ``transcribe``: the single mlx-whisper model on the host GPU. Every
decode goes through :meth:`AdmissionGate.run`, which runs it in the lane's own
executor (never on the event loop) and holds the slot until the decode thread
really ends. Beyond ``max_active + max_waiting`` callers get an immediate 503 +
``Retry-After`` (see ``w2v_common.admission``).

Code defaults are the issue's initial limits (1 active / 8 waiting / 60 s).
``GPU_LANE_TRANSCRIBE_MAX_ACTIVE|_MAX_WAITING|_MAX_WAIT_S|_SEED_S`` override them,
read once when the gate is built (app startup).
"""

from __future__ import annotations

from collections.abc import Mapping

from w2v_common.admission import AdmissionGate, LaneConfig

TRANSCRIBE_LANE = "transcribe"

TRANSCRIBE_DEFAULTS = LaneConfig(max_active=1, max_waiting=8, max_wait_s=60.0)

# Bound on how long shutdown waits for an in-flight decode before the process exits.
SHUTDOWN_TIMEOUT_S = 30.0


def build_gate(environ: Mapping[str, str] | None = None) -> AdmissionGate:
    """Build the service's gate; env overrides are applied here."""
    return AdmissionGate({TRANSCRIBE_LANE: TRANSCRIBE_DEFAULTS}, environ=environ)
