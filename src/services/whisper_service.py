"""mlx-whisper transcription service.

Loads a Whisper model via mlx-whisper and exposes a thin synchronous transcribe()
that returns a :class:`TranscriptionResult` matching the on-the-wire contract
preserved from w2v/transcription_worker_service.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TranscriptWord:
    start: float
    end: float
    word: str
    probability: float | None = None


@dataclass
class TranscriptSegment:
    id: int
    start: float
    end: float
    text: str
    words: list[TranscriptWord] = field(default_factory=list)
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


@dataclass
class TranscriptionResult:
    backend: str
    model: str
    language: str | None
    duration: float
    text: str
    segments: list[TranscriptSegment]


class BackendUnavailableError(RuntimeError):
    """Raised when the underlying ML library failed to load."""


class WhisperService:
    """mlx-whisper backed transcription service.

    Lazy-loads the model on first transcribe() call. Designed to be used as a
    process-singleton from FastAPI app.state.
    """

    name: str = "mlx-whisper"

    def __init__(self) -> None:
        self.model_repo: str = os.getenv("ASR_MODEL", "mlx-community/whisper-large-v3-mlx")
        self.compute_type: str = os.getenv("ASR_COMPUTE_TYPE", "fp16")
        self.default_language: str = os.getenv("ASR_LANGUAGE_DEFAULT", "")
        self.device: str = "unified"
        self._loaded: bool = False
        self._mlx_whisper: Any = None
        self._mx: Any = None

    @property
    def model_size(self) -> str:
        # Best-effort size string for /models endpoint compatibility.
        repo = self.model_repo.rsplit("/", 1)[-1]
        for size in ("large-v3", "large-v2", "large-v1", "large", "medium", "small", "base", "tiny"):
            if size in repo:
                return size
        return repo

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def initialize(self) -> None:
        """Eagerly import the runtime so the first request doesn't pay the cold-load tax."""
        try:
            import mlx.core as mx
            import mlx_whisper
        except ImportError as exc:
            raise BackendUnavailableError("mlx_whisper is not installed; this service requires Apple Silicon.") from exc
        self._mlx_whisper = mlx_whisper
        self._mx = mx
        self._loaded = True
        logger.info("mlx-whisper ready (model_repo=%s)", self.model_repo)

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        word_timestamps: bool = True,
    ) -> TranscriptionResult:
        """Run synchronous transcription. Call from a worker thread."""
        if not self._loaded or self._mlx_whisper is None:
            raise BackendUnavailableError("WhisperService not initialised")

        lang = language if language is not None else (self.default_language or None)

        # MLX streams are thread-local; pin to the GPU stream so worker threads
        # (e.g. asyncio.to_thread) can access the model. Without this we hit
        # "There is no Stream(gpu, N) in current thread."
        with self._mx.stream(self._mx.gpu):
            result: dict[str, Any] = self._mlx_whisper.transcribe(
                str(audio_path),
                path_or_hf_repo=self.model_repo,
                word_timestamps=word_timestamps,
                language=lang,
                verbose=False,
            )

        segments_raw = result.get("segments") or []
        segments: list[TranscriptSegment] = []
        max_end = 0.0
        for idx, seg in enumerate(segments_raw):
            words = [
                TranscriptWord(
                    start=float(w.get("start", 0.0)),
                    end=float(w.get("end", 0.0)),
                    word=str(w.get("word", "")).strip(),
                    probability=_optional_float(w.get("probability")),
                )
                for w in (seg.get("words") or [])
            ]
            seg_end = float(seg.get("end", 0.0))
            max_end = max(max_end, seg_end)
            segments.append(
                TranscriptSegment(
                    id=int(seg.get("id", idx)),
                    start=float(seg.get("start", 0.0)),
                    end=seg_end,
                    text=str(seg.get("text", "")).strip(),
                    words=words,
                    avg_logprob=_optional_float(seg.get("avg_logprob")),
                    no_speech_prob=_optional_float(seg.get("no_speech_prob")),
                )
            )

        return TranscriptionResult(
            backend=self.name,
            model=self.model_repo,
            language=result.get("language"),
            duration=float(result.get("duration", max_end)),
            text=str(result.get("text", "")).strip(),
            segments=segments,
        )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except TypeError, ValueError:
        return None


_singleton: WhisperService | None = None


def get_whisper_service() -> WhisperService:
    global _singleton
    if _singleton is None:
        _singleton = WhisperService()
    return _singleton
