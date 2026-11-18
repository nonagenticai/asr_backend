//! Runtime configuration, mirroring the Python service's pydantic settings.
//!
//! Every value that the Python service read from the environment is read here
//! under the SAME name, so the existing `host-config-sync` ConfigMap
//! (`asr-backend-env`) keeps working unchanged. Only the two path settings
//! differ, because they name an mlx-whisper repo and a whisper.cpp GGUF
//! respectively and cannot be the same string.

use std::env;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Config {
    pub bind_addr: String,
    /// whisper.cpp GGUF checkpoint. The Python service pointed `ASR_MODEL` at a
    /// HuggingFace *mlx* repo (`mlx-community/whisper-large-v3-mlx`), which
    /// whisper.cpp cannot load — a GGUF path is required instead.
    pub model_path: PathBuf,
    pub n_threads: i32,
    /// Same cap the Python route enforced before reading the body.
    pub max_audio_bytes: usize,
    pub auth_enabled: bool,
    pub license_introspect_url: String,
    pub metering_redis_url: Option<String>,
    pub service_name: String,
    pub service_version: String,
    /// Reported by `/health`. whisper.cpp on these hosts is Metal-backed, which
    /// is what the field means; the Python service reported `unified` because
    /// MLX shares memory with the GPU.
    pub device: String,
    /// `AUTH_DELEGATING_SUBJECTS`: comma-separated credential subjects allowed
    /// to stamp `X-Effective-Tenant-Id`. Empty = nobody may delegate.
    pub auth_delegating_subjects: Vec<String>,
    /// `W2V_LICENSE_GATE_ENABLED` — the *entitlement* gate (app keys, the
    /// `/licenses/active` lookup). Dormant on this host today: the env keys are
    /// present but the flag is not set, so the service resolves identity and
    /// stops there. Read so the log line can say which mode it is in.
    pub entitlement_gate_enabled: bool,
}

fn var(name: &str) -> Option<String> {
    env::var(name).ok().filter(|v| !v.trim().is_empty())
}

impl Config {
    pub fn from_env() -> Result<Self, String> {
        // Port 8812-equivalent discipline from the w2v rules: never default to
        // a port that something else on the box owns. 9201 is the ASR port the
        // CoreDNS/ServiceEntry wiring already points at.
        let bind_addr = var("ASR_BIND_ADDR").unwrap_or_else(|| "0.0.0.0:9201".to_string());

        let model_path = var("ASR_MODEL_PATH")
            .or_else(|| var("ASR_MODEL"))
            .ok_or_else(|| {
                "ASR_MODEL_PATH is required: path to a whisper.cpp GGUF checkpoint \
                 (whisper.cpp cannot load the Python service's mlx-whisper repo id)"
                    .to_string()
            })?;

        let license_introspect_url = var("LICENSE_INTROSPECT_URL").unwrap_or_default();
        let auth_enabled = var("ENABLE_AUTH")
            .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(true);

        // The Python service refused to boot on this exact pair rather than
        // serving unauthenticated (asr_backend src/app.py). Keep that: an
        // auth-enabled ASR node that silently allows everything is a licence
        // bypass, and metering would attribute the work to nobody.
        if auth_enabled && license_introspect_url.is_empty() {
            return Err(
                "ENABLE_AUTH=true requires LICENSE_INTROSPECT_URL; refusing to boot \
                        rather than serve unauthenticated"
                    .to_string(),
            );
        }

        Ok(Self {
            bind_addr,
            model_path: PathBuf::from(model_path),
            n_threads: var("ASR_N_THREADS")
                .and_then(|v| v.parse().ok())
                .unwrap_or(4),
            max_audio_bytes: var("ASR_MAX_AUDIO_BYTES")
                .and_then(|v| v.parse().ok())
                .unwrap_or(100 * 1024 * 1024),
            auth_enabled,
            license_introspect_url,
            metering_redis_url: var("METERING_REDIS_URL"),
            service_name: var("ASR_SERVICE_NAME").unwrap_or_else(|| "asr-backend".to_string()),
            service_version: var("ASR_VERSION").unwrap_or_else(|| "0.1.0".to_string()),
            device: var("ASR_DEVICE").unwrap_or_else(|| "metal".to_string()),
            auth_delegating_subjects: var("AUTH_DELEGATING_SUBJECTS")
                .map(|v| {
                    v.split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default(),
            entitlement_gate_enabled: var("W2V_LICENSE_GATE_ENABLED")
                .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
                .unwrap_or(false),
        })
    }
}
