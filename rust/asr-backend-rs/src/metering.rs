//! Cost-bus emitter.
//!
//! Matches `w2v_common.metering.schedule_cost_event`: a fire-and-forget XADD to
//! the canonical cost-bus Redis stream carrying `cost_usd = 0.0` (the compute is
//! ours; the event is usage telemetry, not an invoice). A metering failure is
//! logged and swallowed — billing must never fail a request that already
//! produced audio.

/// The canonical cost-bus stream, mirroring `w2v_common.metering.CANONICAL_COST_STREAM`.
///
/// History: `metering:costs` → `improvement-loop:costs` (2026-06-24) → `billing:costs`
/// (the billing-bus rename). This port was written from the *first* name and kept it after
/// both renames, so every ASR cost event landed on a stream with no consumer group and was
/// silently dropped (measured 2026-11-25: a single 0.1 s transcription wrote
/// `metering:costs` and raised `MeteringDeadStreamWritten`).
const CANONICAL_STREAM: &str = "billing:costs";

/// The stream to write: the deployment's `METERING_STREAM` when set, else the canonical name.
///
/// Mirrors `w2v_common.metering.resolve_stream()`. The override exists so a bus rename is a
/// config change on the host, not a rebuild of this binary.
fn stream_name(override_value: Option<&str>) -> String {
    override_value.unwrap_or(CANONICAL_STREAM).to_string()
}

pub async fn emit_cost_event(redis_url: &str, tenant_id: &str, audio_seconds: f64) {
    let client = match redis::Client::open(redis_url) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(error = %e, "metering: could not build redis client");
            return;
        }
    };
    let mut conn = match client.get_multiplexed_async_connection().await {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(error = %e, "metering: redis unavailable");
            return;
        }
    };

    let stream = stream_name(std::env::var("METERING_STREAM").ok().as_deref());
    let fields: Vec<(&str, String)> = vec![
        ("service", "asr-backend".to_string()),
        ("tenant_id", tenant_id.to_string()),
        ("cost_usd", "0.0".to_string()),
        ("unit", "audio_second".to_string()),
        ("quantity", format!("{audio_seconds:.3}")),
    ];

    let res: redis::RedisResult<String> = redis::cmd("XADD")
        .arg(&stream)
        .arg("*")
        .arg(&fields)
        .query_async(&mut conn)
        .await;

    if let Err(e) = res {
        tracing::warn!(error = %e, stream = %stream, "metering: XADD failed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_stream_is_the_one_billing_consumes() {
        assert_eq!(stream_name(None), "billing:costs");
    }

    #[test]
    fn a_deployment_may_override_the_stream() {
        assert_eq!(
            stream_name(Some("improvement-loop:costs")),
            "improvement-loop:costs"
        );
    }

    #[test]
    fn the_dead_stream_is_never_the_default() {
        // The regression this module was fixed for: no configuration may fall back to the
        // renamed-away stream, because it has no consumer group and silently drops billing
        // evidence.
        assert_ne!(stream_name(None), "metering:costs");
    }
}
