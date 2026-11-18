//! Whisper engine: one loaded `WhisperContext`, one state per request.
//!
//! `WhisperContext` is `Send + Sync` (the model weights are immutable after
//! load) but `WhisperState` is not — it carries the decoder's KV cache. So the
//! context lives in shared state and a fresh state is built inside the blocking
//! worker for each request. `full()` is CPU/Metal-bound and MUST NOT run on a
//! tokio worker thread.
//!
//! Concurrency is bounded by a semaphore rather than by the runtime: Metal will
//! happily accept several concurrent decodes and then thrash, which on a 16 GiB
//! mini is an OOM, not a slowdown.

use std::path::Path;
use std::sync::Arc;

use tokio::sync::Semaphore;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::error::AppError;
use crate::models::{TranscriptSegment, TranscriptWord};
use crate::srt::TICKS_PER_SECOND;

pub struct Engine {
    ctx: Arc<WhisperContext>,
    // An `Arc` rather than a bare `Semaphore` so `transcribe` can take an
    // *owned* permit: the permit is released inside the blocking task, and a
    // borrowed permit would tie that task to `&self` instead of `'static`.
    gate: Arc<Semaphore>,
    pub model_name: String,
    pub n_threads: i32,
}

/// Options the route passes through from the multipart form.
#[derive(Debug, Clone)]
pub struct TranscribeOptions {
    pub language: Option<String>,
    pub word_timestamps: bool,
    /// OpenAI's `prompt`: whisper.cpp takes it as an initial text context, the
    /// same way mlx-whisper's `initial_prompt` did.
    pub prompt: Option<String>,
}

#[derive(Debug)]
pub struct Transcript {
    pub language: Option<String>,
    pub duration: f64,
    pub text: String,
    pub segments: Vec<TranscriptSegment>,
}

impl Engine {
    pub fn load(model_path: &Path, n_threads: i32, max_concurrent: usize) -> Result<Self, String> {
        if !model_path.exists() {
            return Err(format!(
                "Whisper model not found at {} — set ASR_MODEL_PATH to a whisper.cpp GGUF",
                model_path.display()
            ));
        }
        let ctx = WhisperContext::new_with_params(
            model_path
                .to_str()
                .ok_or("ASR_MODEL_PATH is not valid UTF-8")?,
            WhisperContextParameters::default(),
        )
        .map_err(|e| format!("could not load Whisper model: {e}"))?;

        Ok(Self {
            ctx: Arc::new(ctx),
            gate: Arc::new(Semaphore::new(max_concurrent.max(1))),
            model_name: model_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("whisper")
                .to_string(),
            n_threads,
        })
    }

    /// Runs blocking Whisper inference on a blocking thread.
    pub async fn transcribe(
        &self,
        audio: Vec<f32>,
        opts: TranscribeOptions,
    ) -> Result<Transcript, AppError> {
        // `acquire()` is held for the whole decode, so the permit count IS the
        // concurrency limit. try_acquire first: a caller that would queue
        // forever is better told 429 + Retry-After, matching the Python
        // admission lane's contract. The permit must be *owned* (hence the
        // `Arc`) because it moves into the blocking task below.
        let gate = Arc::clone(&self.gate);
        let permit = match gate.clone().try_acquire_owned() {
            Ok(p) => p,
            Err(_) => {
                let waited =
                    tokio::time::timeout(std::time::Duration::from_secs(30), gate.acquire_owned())
                        .await;
                match waited {
                    Ok(Ok(p)) => p,
                    _ => {
                        return Err(AppError::TooManyRequests {
                            detail: "ASR backend is at capacity".into(),
                            retry_after_s: 5,
                        });
                    }
                }
            }
        };

        let ctx = Arc::clone(&self.ctx);
        let n_threads = self.n_threads;
        let duration = audio.len() as f64 / 16_000.0;

        let joined = tokio::task::spawn_blocking(move || {
            let out = run_blocking(&ctx, &audio, &opts, n_threads);
            drop(permit);
            out
        })
        .await
        .map_err(|e| AppError::Internal(format!("inference task failed: {e}")))??;

        Ok(Transcript {
            language: joined.language,
            duration,
            text: joined.text,
            segments: joined.segments,
        })
    }
}

fn run_blocking(
    ctx: &WhisperContext,
    audio: &[f32],
    opts: &TranscribeOptions,
    n_threads: i32,
) -> Result<Transcript, AppError> {
    let mut state = ctx
        .create_state()
        .map_err(|e| AppError::Internal(format!("could not create Whisper state: {e}")))?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(n_threads);
    // whisper.cpp prints segments to stdout by default; this is an HTTP
    // service and its stdout is the launchd log.
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_token_timestamps(opts.word_timestamps);

    // `set_language` borrows from the params struct, so the value has to be
    // owned for at least as long as `params`.
    let language_owned = opts.language.clone();
    match language_owned.as_deref() {
        Some(lang) if !lang.is_empty() && lang != "auto" => params.set_language(Some(lang)),
        _ => params.set_language(None), // None = autodetect, as mlx-whisper did
    }

    // Same borrow rule as the language above.
    if let Some(prompt) = opts.prompt.as_deref()
        && !prompt.is_empty()
    {
        params.set_initial_prompt(prompt);
    }

    state
        .full(params, audio)
        .map_err(|e| AppError::Internal(format!("transcription failed: {e}")))?;

    let language = language_from_state(&state);
    let n_segments = state.full_n_segments();

    let mut segments = Vec::with_capacity(n_segments.max(0) as usize);
    let mut text_parts: Vec<String> = Vec::with_capacity(n_segments.max(0) as usize);

    for i in 0..n_segments {
        let Some(seg) = state.get_segment(i) else {
            continue;
        };
        let text = seg.to_str_lossy().unwrap_or_default().to_string();
        let start = seg.start_timestamp() as f64 / TICKS_PER_SECOND;
        let end = seg.end_timestamp() as f64 / TICKS_PER_SECOND;

        let words = if opts.word_timestamps {
            collect_words(&seg)
        } else {
            Vec::new()
        };

        text_parts.push(text.clone());
        segments.push(TranscriptSegment {
            id: i as usize,
            start,
            end,
            text,
            words,
            // whisper.cpp does not expose these through whisper-rs 0.16's
            // segment API. They are optional in the contract (Python always
            // sent them), so they are omitted rather than faked with 0.0,
            // which a caller could mistake for "certain".
            avg_logprob: None,
            no_speech_prob: Some(seg.no_speech_probability() as f64),
        });
    }

    Ok(Transcript {
        language,
        duration: 0.0, // replaced by the caller, which knows the true audio length
        text: text_parts.join("").trim().to_string(),
        segments,
    })
}

/// whisper.cpp emits BPE-style tokens; a new word starts at a leading space.
/// Whisper's tokens carry `t0`/`t1` in centiseconds when token timestamps are on.
fn collect_words(seg: &whisper_rs::WhisperSegment<'_>) -> Vec<TranscriptWord> {
    let mut words: Vec<TranscriptWord> = Vec::new();
    for t in 0..seg.n_tokens() {
        let Some(tok) = seg.get_token(t) else {
            continue;
        };
        let Ok(piece) = tok.to_str_lossy() else {
            continue;
        };
        let data = tok.token_data();
        let start = data.t0 as f64 / TICKS_PER_SECOND;
        let end = data.t1 as f64 / TICKS_PER_SECOND;

        match words.last_mut() {
            Some(w) if !piece.starts_with(' ') => {
                w.word.push_str(&piece);
                w.end = end;
            }
            _ => words.push(TranscriptWord {
                start,
                end,
                word: piece.trim_start().to_string(),
                probability: Some(data.p as f64),
            }),
        }
    }
    words.retain(|w| !w.word.is_empty());
    words
}

fn language_from_state(state: &whisper_rs::WhisperState) -> Option<String> {
    let id = state.full_lang_id_from_state();
    if id < 0 {
        return None;
    }
    whisper_rs::get_lang_str(id).map(|s| s.to_string())
}
