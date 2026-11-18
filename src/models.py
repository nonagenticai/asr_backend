"""Shared response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health/readiness/liveness response."""

    status: str
    service: str
    version: str
    device: str
    models_loaded: list[str] = Field(default_factory=list)


class TranscriptWordResponse(BaseModel):
    start: float
    end: float
    word: str
    probability: float | None = None


class TranscriptSegmentResponse(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: list[TranscriptWordResponse] = Field(default_factory=list)
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


class TranscriptionResponse(BaseModel):
    """Preserves the transcription_worker_service contract so consumers don't change."""

    backend: str
    model: str
    language: str | None = None
    duration: float
    text: str
    segments: list[TranscriptSegmentResponse]
    srt: str | None = Field(default=None, description="SRT-formatted subtitles")


class ModelInfo(BaseModel):
    backend: str
    model: str
    device: str
    compute_type: str | None = None
    loaded: bool


class ModelsResponse(BaseModel):
    current: ModelInfo
    available_sizes: list[str]
