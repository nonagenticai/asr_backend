//! Response models — field-for-field with `src/models.py`.
//!
//! `TranscriptionResponse` deliberately preserves the
//! `transcription_worker_service` contract, exactly as the Python model's
//! docstring says, so consumers do not change when the backend does.

use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub service: String,
    pub version: String,
    pub device: String,
    pub models_loaded: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct TranscriptWord {
    pub start: f64,
    pub end: f64,
    pub word: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probability: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct TranscriptSegment {
    pub id: usize,
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub words: Vec<TranscriptWord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_logprob: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub no_speech_prob: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct TranscriptionResponse {
    pub backend: String,
    pub model: String,
    pub language: Option<String>,
    pub duration: f64,
    pub text: String,
    pub segments: Vec<TranscriptSegment>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub srt: Option<String>,
}

/// OpenAI-compatible transcription, `response_format=json` (the default).
///
/// `usage` is what the OpenAI *audio* models return and what this service's
/// metering already knows (seconds of audio, billed per minute), so a caller
/// switching between the local service and a commercial one sees the same
/// fields on both.
#[derive(Debug, Serialize)]
pub struct OpenAiTranscription {
    pub text: String,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub seconds: f64,
}

/// `response_format=verbose_json`.
///
/// `usage` is not decoration: strata-ingest bills a transcribe step from
/// `usage.seconds` and records `usage_unknown` with 0 units when it is absent
/// (`strata_ingest/src/providers/transcribe.rs::billable_seconds`). The Python
/// provider this replaced (`mlx_api/src/mlx_api/app.py`) emits it on
/// verbose_json too — dropping it here silently zeroed a money path.
#[derive(Debug, Serialize)]
pub struct OpenAiVerboseTranscription {
    pub task: &'static str,
    pub language: Option<String>,
    pub duration: f64,
    pub text: String,
    pub segments: Vec<OpenAiSegment>,
    pub usage: Usage,
}

/// The OpenAI segment shape is a strict subset of ours: `id/start/end/text`
/// only. Word timings are the w2v contract (`/api/transcribe`), not this one.
#[derive(Debug, Serialize)]
pub struct OpenAiSegment {
    pub id: usize,
    pub start: f64,
    pub end: f64,
    pub text: String,
}

impl From<&TranscriptSegment> for OpenAiSegment {
    fn from(s: &TranscriptSegment) -> Self {
        Self {
            id: s.id,
            start: s.start,
            end: s.end,
            text: s.text.trim().to_string(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub backend: String,
    pub model: String,
    pub device: String,
    pub compute_type: Option<String>,
    pub loaded: bool,
}

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub current: ModelInfo,
    pub available_sizes: Vec<String>,
}
