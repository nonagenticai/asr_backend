//! Cost-bus emitter.
//!
//! Matches `w2v_common.metering.schedule_cost_event`: a fire-and-forget XADD to
//! the `metering:costs` Redis stream carrying `cost_usd = 0.0` (the compute is
//! ours; the event is usage telemetry, not an invoice). A metering failure is
//! logged and swallowed — billing must never fail a request that already
//! produced audio.

const STREAM: &str = "metering:costs";

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

    let fields: Vec<(&str, String)> = vec![
        ("service", "asr-backend".to_string()),
        ("tenant_id", tenant_id.to_string()),
        ("cost_usd", "0.0".to_string()),
        ("unit", "audio_second".to_string()),
        ("quantity", format!("{audio_seconds:.3}")),
    ];

    let res: redis::RedisResult<String> = redis::cmd("XADD")
        .arg(STREAM)
        .arg("*")
        .arg(&fields)
        .query_async(&mut conn)
        .await;

    if let Err(e) = res {
        tracing::warn!(error = %e, "metering: XADD failed");
    }
}
