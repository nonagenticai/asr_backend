//! `/metrics` — the scrape target Prometheus already has for this host.
//!
//! `clusters/strato/services/prometheus/configmap.yaml` scrapes
//! `192.0.2.20:9201/metrics` since 2026-09-12 (measured then: 200, 16
//! families, 35 samples — every one of them a generic `python_*`/`process_*`
//! family; no ASR-specific metric existed). Serving nothing here does not
//! merely lose those families: the target goes DOWN, which is the alert
//! condition, and that is exactly what the cutover to this binary would have
//! caused.
//!
//! These are the numbers this service can honestly produce without a client
//! library — what was transcribed, how much audio, and since when — plus the
//! identity series a dashboard needs to tell a fresh process from a wedged one.
//! Text format 0.0.4, which any Prometheus accepts.
//!
//! Unauthenticated on purpose: Prometheus presents no licence token, and the
//! Python route this replaces was equally open.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Default)]
pub struct Metrics {
    /// Completed transcriptions, any surface.
    pub transcriptions: AtomicU64,
    /// Audio seconds accepted for transcription, in milliseconds (integer
    /// atomics; the exposition divides).
    pub audio_millis: AtomicU64,
}

impl Metrics {
    pub fn record(&self, audio_seconds: f64) {
        self.transcriptions.fetch_add(1, Ordering::Relaxed);
        self.audio_millis
            .fetch_add((audio_seconds.max(0.0) * 1000.0) as u64, Ordering::Relaxed);
    }

    pub fn render(&self, started: Instant, model: &str, version: &str, device: &str) -> String {
        let since_epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        let process_start = since_epoch - started.elapsed().as_secs_f64();
        let transcriptions = self.transcriptions.load(Ordering::Relaxed);
        let audio_seconds = self.audio_millis.load(Ordering::Relaxed) as f64 / 1000.0;

        let mut out = String::with_capacity(1024);
        out.push_str("# HELP asr_backend_info Build identity of the running service.\n");
        out.push_str("# TYPE asr_backend_info gauge\n");
        out.push_str(&format!(
            "asr_backend_info{{version=\"{version}\",model=\"{model}\",device=\"{device}\"}} 1\n"
        ));

        out.push_str(
            "# HELP process_start_time_seconds Start time of the process since unix epoch in seconds.\n",
        );
        out.push_str("# TYPE process_start_time_seconds gauge\n");
        out.push_str(&format!("process_start_time_seconds {process_start:.3}\n"));

        out.push_str("# HELP process_uptime_seconds Seconds since this process started.\n");
        out.push_str("# TYPE process_uptime_seconds gauge\n");
        out.push_str(&format!(
            "process_uptime_seconds {:.3}\n",
            started.elapsed().as_secs_f64()
        ));

        out.push_str("# HELP asr_transcriptions_total Transcriptions completed.\n");
        out.push_str("# TYPE asr_transcriptions_total counter\n");
        out.push_str(&format!("asr_transcriptions_total {transcriptions}\n"));

        out.push_str("# HELP asr_audio_seconds_total Audio seconds transcribed.\n");
        out.push_str("# TYPE asr_audio_seconds_total counter\n");
        out.push_str(&format!("asr_audio_seconds_total {audio_seconds:.3}\n"));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_prometheus_text_with_the_service_identity() {
        let m = Metrics::default();
        m.record(2.5);
        m.record(1.5);
        let body = m.render(Instant::now(), "ggml-large-v3-turbo-q5_0", "0.1.0", "metal");
        assert!(body.contains(
            "asr_backend_info{version=\"0.1.0\",model=\"ggml-large-v3-turbo-q5_0\",device=\"metal\"} 1"
        ));
        assert!(body.contains("asr_transcriptions_total 2"));
        assert!(body.contains("asr_audio_seconds_total 4.000"));
        // Every series needs its TYPE line: an exposition a scraper has to
        // guess at is one it will mis-type.
        for name in [
            "asr_transcriptions_total",
            "asr_audio_seconds_total",
            "process_uptime_seconds",
        ] {
            assert!(
                body.contains(&format!("# TYPE {name} ")),
                "{name} has no TYPE"
            );
        }
        assert!(body.ends_with('\n'));
    }
}
