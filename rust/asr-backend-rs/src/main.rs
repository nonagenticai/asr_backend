//! asr-backend-rs — Whisper ASR over HTTP.
//!
//! Rust rewrite of the Python `asr_backend` launchd service (mini2, `:9201`,
//! reached in-cluster as `asr.test`). Same routes, same response shapes,
//! same licence/metering semantics. The engine changes from `mlx-whisper`
//! (Apple MLX) to whisper.cpp via `whisper-rs` with the Metal backend.

mod audio;
mod auth;
mod config;
mod error;
mod metering;
mod metrics;
mod models;
mod routes;
mod srt;
mod whisper;

use std::sync::Arc;

use tracing_subscriber::EnvFilter;

use crate::auth::LicenceGate;
use crate::config::Config;
use crate::routes::AppState;
use crate::whisper::Engine;

/// Whisper decodes are serialised behind this. Metal will accept several
/// concurrent decodes and then thrash; on a 16 GiB mini that is an OOM rather
/// than a slowdown. 2 matches the Python admission lane's max_active.
const MAX_CONCURRENT_DECODES: usize = 2;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .json()
        .init();

    let config = match Config::from_env() {
        Ok(c) => c,
        Err(e) => {
            // Refusing to boot is deliberate for the ENABLE_AUTH + missing
            // LICENSE_INTROSPECT_URL case: serving unauthenticated would be a
            // licence bypass, and a restart loop is more visible than a silent
            // one.
            tracing::error!(error = %e, "asr-backend-rs: configuration rejected");
            std::process::exit(2);
        }
    };

    tracing::info!(
        model = %config.model_path.display(),
        addr = %config.bind_addr,
        auth_enabled = config.auth_enabled,
        entitlement_gate = config.entitlement_gate_enabled,
        "loading Whisper model"
    );

    let engine = Engine::load(&config.model_path, config.n_threads, MAX_CONCURRENT_DECODES)
        .map_err(|e| {
            tracing::error!(error = %e, "asr-backend-rs: engine load failed");
            e
        })?;

    let state = Arc::new(AppState {
        metrics: crate::metrics::Metrics::default(),
        started: std::time::Instant::now(),
        engine,
        gate: LicenceGate::new(
            config.license_introspect_url.clone(),
            config.auth_enabled,
            config.auth_delegating_subjects.clone(),
        ),
        config,
    });

    let app = routes::router(Arc::clone(&state));
    let listener = tokio::net::TcpListener::bind(&state.config.bind_addr).await?;
    tracing::info!(addr = %state.config.bind_addr, "asr-backend-rs listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    tracing::info!("shutdown signal received");
}
