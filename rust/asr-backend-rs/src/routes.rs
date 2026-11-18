//! HTTP surface.
//!
//! Three faces, one engine:
//!
//! * `/api/transcribe`, `/api/models` — the w2v contract the Python service
//!   serves today (`src/routers/asr_routes.py` mounts its router at `/api`).
//!   Every live caller uses these paths, so they are the drop-in surface.
//! * `/transcribe`, `/models` — the same handlers, un-prefixed. Kept because
//!   this crate shipped them and a probe that hard-codes them should not break.
//! * `/v1/audio/transcriptions` — the OpenAI shape, so an app can point at this
//!   service or at a commercial vendor with the same client code. This is the
//!   route the service was missing entirely.

use std::sync::Arc;

use axum::extract::{DefaultBodyLimit, Multipart, State};
use axum::http::{HeaderMap, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};

use crate::auth::LicenceGate;
use crate::config::Config;
use crate::error::AppError;
use crate::metrics::Metrics;
use crate::models::{
    HealthResponse, ModelInfo, ModelsResponse, OpenAiSegment, OpenAiTranscription,
    OpenAiVerboseTranscription, TranscriptionResponse, Usage,
};
use crate::srt::to_srt;
use crate::whisper::{Engine, TranscribeOptions, Transcript};

pub struct AppState {
    pub engine: Engine,
    pub gate: LicenceGate,
    pub config: Config,
    /// Scrape counters — see `metrics.rs` for why this route must exist.
    pub metrics: Metrics,
    /// For `process_start_time_seconds`.
    pub started: std::time::Instant,
}

pub fn router(state: Arc<AppState>) -> Router {
    let max = state.config.max_audio_bytes;
    Router::new()
        .route("/health", get(health))
        .route("/health/live", get(health_live))
        .route("/health/ready", get(health_ready))
        .route("/metrics", get(metrics))
        .route("/models", get(models))
        .route("/api/models", get(models))
        .route("/transcribe", post(transcribe_api))
        .route("/api/transcribe", post(transcribe_api))
        .route("/v1/audio/transcriptions", post(transcribe_openai))
        // The Python route enforced the cap itself and answered 413; axum's
        // body limit answers 413 too, so the caller sees the same code for the
        // same input.
        .layer(DefaultBodyLimit::max(max))
        .with_state(state)
}

async fn health(State(st): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(health_body(&st, "healthy"))
}

async fn health_live(State(st): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(health_body(&st, "healthy"))
}

async fn health_ready(State(st): State<Arc<AppState>>) -> Json<HealthResponse> {
    // The engine is loaded in the startup path before the listener binds, so
    // reaching this handler at all means the model is resident.
    Json(health_body(&st, "ready"))
}

/// `GET /metrics` — Prometheus scrape, unauthenticated (see `metrics.rs`).
async fn metrics(State(st): State<Arc<AppState>>) -> impl IntoResponse {
    let body = st.metrics.render(
        st.started,
        &st.engine.model_name,
        &st.config.service_version,
        &st.config.device,
    );
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
}

fn health_body(st: &AppState, status: &str) -> HealthResponse {
    HealthResponse {
        status: status.to_string(),
        service: st.config.service_name.clone(),
        version: st.config.service_version.clone(),
        device: st.config.device.clone(),
        models_loaded: vec![st.engine.model_name.clone()],
    }
}

async fn models(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<ModelsResponse>, AppError> {
    // Python gates this route with the same middleware as transcription
    // (measured: 401 without a token), so a caller cannot enumerate the model
    // surface anonymously.
    let _identity = st.gate.authenticate(&headers).await?;
    Ok(Json(ModelsResponse {
        current: ModelInfo {
            backend: "whisper-rs".to_string(),
            model: st.engine.model_name.clone(),
            device: st.config.device.clone(),
            compute_type: None,
            loaded: true,
        },
        available_sizes: vec![
            "tiny".into(),
            "base".into(),
            "small".into(),
            "medium".into(),
            "large-v2".into(),
            "large-v3".into(),
            "large-v3-turbo".into(),
        ],
    }))
}

/// Everything the multipart body can carry, across both surfaces.
#[derive(Default)]
struct Form {
    file: Option<(Vec<u8>, Option<String>)>,
    language: Option<String>,
    prompt: Option<String>,
    word_timestamps: Option<bool>,
    return_srt: Option<bool>,
    response_format: Option<String>,
}

impl Form {
    fn take_audio(self) -> Result<(Vec<u8>, Option<String>), AppError> {
        let (bytes, ext) = self
            .file
            .ok_or_else(|| AppError::BadRequest("Empty audio payload".into()))?;
        if bytes.is_empty() {
            return Err(AppError::BadRequest("Empty audio payload".into()));
        }
        Ok((bytes, ext))
    }
}

async fn read_form(mut multipart: Multipart) -> Result<Form, AppError> {
    let mut form = Form::default();
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::BadRequest(format!("malformed multipart body: {e}")))?
    {
        let Some(name) = field.name().map(str::to_string) else {
            continue;
        };
        match name.as_str() {
            "file" => {
                let ext = field
                    .file_name()
                    .and_then(|f| std::path::Path::new(f).extension())
                    .and_then(|e| e.to_str())
                    .map(str::to_string);
                let data = field
                    .bytes()
                    .await
                    .map_err(|e| AppError::BadRequest(format!("could not read upload: {e}")))?;
                form.file = Some((data.to_vec(), ext));
            }
            "language" => {
                let v = text_field(field).await?;
                if !v.trim().is_empty() {
                    form.language = Some(v);
                }
            }
            "prompt" => {
                let v = text_field(field).await?;
                if !v.trim().is_empty() {
                    form.prompt = Some(v);
                }
            }
            "word_timestamps" => {
                form.word_timestamps = Some(parse_bool(&text_field(field).await?, true))
            }
            "return_srt" => form.return_srt = Some(parse_bool(&text_field(field).await?, false)),
            "response_format" => {
                let v = text_field(field).await?;
                if !v.trim().is_empty() {
                    form.response_format = Some(v);
                }
            }
            _ => {
                // Drain unknown fields (`model`, `timestamp_granularities[]`,
                // ...) so the parser does not stall. `model` is deliberately
                // ignored: the engine serves exactly one checkpoint, and
                // echoing a caller-supplied name back would claim we loaded
                // something we did not.
                let _ = field.bytes().await;
            }
        }
    }
    Ok(form)
}

async fn text_field(field: axum::extract::multipart::Field<'_>) -> Result<String, AppError> {
    field
        .text()
        .await
        .map_err(|e| AppError::BadRequest(format!("bad form field: {e}")))
}

/// `POST /api/transcribe` — the w2v contract, unchanged.
async fn transcribe_api(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    multipart: Multipart,
) -> Result<Json<TranscriptionResponse>, AppError> {
    // Authenticate BEFORE reading the body: a request that will be rejected
    // must cost nothing, the same ordering the Python route documents.
    let identity = st.gate.authenticate(&headers).await?;
    let form = read_form(multipart).await?;

    // Form default matches the Python signature; `false` skips token
    // timestamps entirely, which is most of the decode cost on a long file.
    let word_timestamps = form.word_timestamps.unwrap_or(true);
    let return_srt = form.return_srt.unwrap_or(false);
    let language = form.language.clone();
    let prompt = form.prompt.clone();

    let result = run(
        &st,
        &identity,
        form,
        TranscribeOptions {
            language,
            word_timestamps,
            prompt,
        },
    )
    .await?;

    let srt = return_srt.then(|| to_srt(&result.segments));

    Ok(Json(TranscriptionResponse {
        backend: "whisper-rs".to_string(),
        model: st.engine.model_name.clone(),
        language: result.language,
        duration: result.duration,
        text: result.text,
        segments: result.segments,
        srt,
    }))
}

/// `POST /v1/audio/transcriptions` — the OpenAI shape.
async fn transcribe_openai(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    multipart: Multipart,
) -> Result<Response, AppError> {
    let identity = st.gate.authenticate(&headers).await?;
    let form = read_form(multipart).await?;

    let format = form
        .response_format
        .clone()
        .unwrap_or_else(|| "json".to_string());
    let format = format.trim().to_ascii_lowercase();
    match format.as_str() {
        "json" | "text" | "verbose_json" | "srt" => {}
        other => {
            return Err(AppError::BadRequest(format!(
                "response_format '{other}' is not supported; use json, text, verbose_json or srt"
            )));
        }
    }

    let language = form.language.clone();
    let prompt = form.prompt.clone();
    let result = run(
        &st,
        &identity,
        form,
        TranscribeOptions {
            language,
            word_timestamps: false,
            prompt,
        },
    )
    .await?;

    let body = match format.as_str() {
        "text" => (
            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            result.text.clone(),
        )
            .into_response(),
        "srt" => (
            [(header::CONTENT_TYPE, "application/x-subrip; charset=utf-8")],
            to_srt(&result.segments),
        )
            .into_response(),
        "verbose_json" => Json(OpenAiVerboseTranscription {
            task: "transcribe",
            language: result.language.clone(),
            duration: result.duration,
            text: result.text.clone(),
            segments: result.segments.iter().map(OpenAiSegment::from).collect(),
            usage: Usage {
                kind: "duration",
                seconds: result.duration,
            },
        })
        .into_response(),
        _ => Json(OpenAiTranscription {
            text: result.text.clone(),
            usage: Usage {
                kind: "duration",
                seconds: result.duration,
            },
        })
        .into_response(),
    };
    Ok(body)
}

/// Decode, transcribe, meter, log — identical for both surfaces.
async fn run(
    st: &AppState,
    identity: &crate::auth::Identity,
    form: Form,
    opts: TranscribeOptions,
) -> Result<Transcript, AppError> {
    let (raw, ext) = form.take_audio()?;

    let audio =
        tokio::task::spawn_blocking(move || crate::audio::decode_to_16k_mono(&raw, ext.as_deref()))
            .await
            .map_err(|e| AppError::Internal(format!("decode task failed: {e}")))??;

    let started = std::time::Instant::now();
    let result = st.engine.transcribe(audio, opts).await?;
    let latency_ms = started.elapsed().as_millis() as u64;

    st.metrics.record(result.duration);

    tracing::info!(
        tenant_id = %identity.tenant_id,
        licence_id = identity.licence_id.as_deref().unwrap_or(""),
        duration_s = result.duration,
        latency_ms,
        segments = result.segments.len(),
        "asr.transcribed"
    );

    // Fire-and-forget metering, mirroring w2v_common.metering: a metering
    // failure must never fail a transcription that already succeeded.
    if let Some(url) = st.config.metering_redis_url.clone() {
        let tenant = identity.tenant_id.clone();
        let secs = result.duration;
        tokio::spawn(async move {
            crate::metering::emit_cost_event(&url, &tenant, secs).await;
        });
    }

    Ok(result)
}

fn parse_bool(v: &str, default: bool) -> bool {
    match v.trim().to_ascii_lowercase().as_str() {
        "" => default,
        "1" | "true" | "yes" | "on" => true,
        "0" | "false" | "no" | "off" => false,
        _ => default,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_form_booleans_the_way_pydantic_does() {
        assert!(parse_bool("true", false));
        assert!(parse_bool(" TRUE ", false));
        assert!(parse_bool("1", false));
        assert!(parse_bool("yes", false));
        assert!(!parse_bool("false", true));
        assert!(!parse_bool("0", true));
        // An unparseable value falls back to the field default rather than a
        // 422 — the Python form field has a default for exactly this case.
        assert!(parse_bool("maybe", true));
    }

    #[test]
    fn an_openai_segment_is_the_subset_of_ours() {
        use crate::models::{TranscriptSegment, TranscriptWord};
        let s = TranscriptSegment {
            id: 3,
            start: 1.5,
            end: 2.25,
            text: " hello world ".into(),
            words: vec![TranscriptWord {
                start: 1.5,
                end: 1.8,
                word: "hello".into(),
                probability: Some(0.9),
            }],
            avg_logprob: None,
            no_speech_prob: Some(0.01),
        };
        let o: OpenAiSegment = (&s).into();
        assert_eq!(o.id, 3);
        assert_eq!(o.text, "hello world");
        let json = serde_json::to_value(&o).unwrap();
        assert_eq!(
            json.as_object().unwrap().keys().collect::<Vec<_>>(),
            vec!["end", "id", "start", "text"]
        );
    }

    /// The shape strata-ingest decodes: it bills from `usage.seconds`, and a
    /// verbose_json body without it records `usage_unknown` with 0 units. The
    /// Python provider (`mlx_api`) emits it, so this is parity, not an extra.
    #[test]
    fn verbose_json_carries_the_usage_block_a_billing_caller_reads() {
        let body = OpenAiVerboseTranscription {
            task: "transcribe",
            language: Some("en".into()),
            duration: 4.25,
            text: "hello".into(),
            segments: vec![OpenAiSegment {
                id: 0,
                start: 0.0,
                end: 4.25,
                text: "hello".into(),
            }],
            usage: Usage {
                kind: "duration",
                seconds: 4.25,
            },
        };
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(
            json["usage"],
            serde_json::json!({"type": "duration", "seconds": 4.25})
        );
        assert!(json["usage"]["seconds"].as_f64().unwrap() > 0.0);
        assert_eq!(json["segments"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn the_usage_block_matches_the_openai_audio_shape() {
        let json = serde_json::to_value(Usage {
            kind: "duration",
            seconds: 12.5,
        })
        .unwrap();
        assert_eq!(
            json,
            serde_json::json!({"type": "duration", "seconds": 12.5})
        );
    }
}
