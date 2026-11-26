//! Cost-bus emitter.
//!
//! Matches `w2v_common.metering.schedule_cost_event`: a fire-and-forget XADD to
//! the canonical cost-bus Redis stream carrying `cost_usd = 0.0` (the compute is
//! ours; the event is usage telemetry, not an invoice). A metering failure is
//! logged and swallowed — billing must never fail a request that already
//! produced audio.
//!
//! WHAT BILLING REQUIRES. `billing-service` validates every cost event before it
//! prices anything (`services/_cost_schema.validate_cost_event`): `provider` is
//! mandatory (it anchors the pricing lookup) and `tenant_id` must resolve to a
//! REAL tenant. Anything else is quarantined on `billing:costs:dead` and never
//! billed. Measured 2026-11-26: the single standing dead letter was this
//! emitter's own output — `tenant_id=unauthenticated`, no `provider`, no
//! `service_key` — so this module neither emits a payload billing will reject
//! nor sends fields billing does not read.

/// The canonical cost-bus stream, mirroring `w2v_common.metering.CANONICAL_COST_STREAM`.
///
/// History: `metering:costs` → `improvement-loop:costs` (2026-06-24) → `billing:costs`
/// (the billing-bus rename). This port was written from the *first* name and kept it after
/// both renames, so every ASR cost event landed on a stream with no consumer group and was
/// silently dropped (measured 2026-11-25: a single 0.1 s transcription wrote
/// `metering:costs` and raised `MeteringDeadStreamWritten`).
const CANONICAL_STREAM: &str = "billing:costs";

/// The billing service's `service_key` for this emitter.
///
/// NOT the HTTP service name. billing's consumer reads `service_key` (it ignores
/// a `service` field entirely), and its pricing notes, markup rules and
/// per-service metrics are keyed on `w2v-asr` — the Python provider's
/// SERVICE_KEY. The port emitted `service = "asr-backend"`, so even a reachable
/// event would have been counted as an unmapped service key.
const SERVICE_KEY: &str = "w2v-asr";

/// The `provider_pricing` row this usage prices against: mlx-whisper,
/// per-second, `input_per_1m_usd = 283.33333333`, output zero.
const PROVIDER: &str = "mlx-whisper";

/// The stream to write: the deployment's `METERING_STREAM` when set, else the canonical name.
///
/// Mirrors `w2v_common.metering.resolve_stream()`. The override exists so a bus rename is a
/// config change on the host, not a rebuild of this binary.
fn stream_name(override_value: Option<&str>) -> String {
    override_value.unwrap_or(CANONICAL_STREAM).to_string()
}

/// True when `raw` is a real, non-nil tenant uuid.
///
/// The licence gate writes the literal `unauthenticated` when it is disabled
/// (`auth::LicenceGate::authenticate`); billing rejects an unresolvable tenant,
/// so emitting one only creates a dead-letter record that nothing can drain.
/// Skip it at the source instead.
fn is_billable_tenant(raw: &str) -> bool {
    match uuid::Uuid::parse_str(raw.trim()) {
        Ok(id) => !id.is_nil(),
        Err(_) => false,
    }
}

/// The Redis field set.
///
/// `input_tokens` is whole seconds of audio because that is the unit the
/// mlx-whisper pricing row converts (`283.33333333 / 1M` seconds, output 0);
/// `unit`/`quantity` stay as human-readable context for the logs and the DLQ.
fn cost_fields(tenant_id: &str, audio_seconds: f64) -> Vec<(&'static str, String)> {
    vec![
        ("service_key", SERVICE_KEY.to_string()),
        ("provider", PROVIDER.to_string()),
        ("tenant_id", tenant_id.trim().to_string()),
        // Advisory only: billing recomputes the authoritative cost from
        // provider_pricing, exactly as it does for the Python provider.
        ("cost_usd", "0.0".to_string()),
        ("input_tokens", format!("{:.0}", audio_seconds.round())),
        ("output_tokens", "0".to_string()),
        ("unit", "audio_second".to_string()),
        ("quantity", format!("{audio_seconds:.3}")),
    ]
}

pub async fn emit_cost_event(redis_url: &str, tenant_id: &str, audio_seconds: f64) {
    if !is_billable_tenant(tenant_id) {
        tracing::debug!(
            tenant_id,
            "metering: no real tenant on this request; cost event not emitted"
        );
        return;
    }

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
    let fields = cost_fields(tenant_id, audio_seconds);

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

    #[test]
    fn a_sentinel_or_non_uuid_tenant_is_not_billable() {
        // The exact value the gate-disabled licence check writes (auth.rs:111).
        assert!(!is_billable_tenant("unauthenticated"));
        assert!(!is_billable_tenant(""));
        assert!(!is_billable_tenant("some-slug"));
        // The nil uuid is a well-formed uuid, so it needs the explicit check:
        // billing rejects it as `nil_tenant_id`.
        assert!(!is_billable_tenant("00000000-0000-0000-0000-000000000000"));
    }

    #[test]
    fn a_real_tenant_uuid_is_billable() {
        assert!(is_billable_tenant("0190c3a1-6a4e-7f0a-9c1d-2b3e4f5a6b7c"));
    }

    #[test]
    fn the_payload_carries_what_billing_requires() {
        let fields = cost_fields("0190c3a1-6a4e-7f0a-9c1d-2b3e4f5a6b7c", 60.0);
        let get = |name: &str| {
            fields
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, v)| v.as_str())
        };

        // validate_cost_event rejects a payload without either of these.
        assert_eq!(get("provider"), Some("mlx-whisper"));
        assert_eq!(
            get("tenant_id"),
            Some("0190c3a1-6a4e-7f0a-9c1d-2b3e4f5a6b7c")
        );
        // The service key billing reads (it ignores a `service` field).
        assert_eq!(get("service_key"), Some("w2v-asr"));
        // 60 s of audio prices as 60 whole seconds against the mlx-whisper row.
        assert_eq!(get("input_tokens"), Some("60"));
        assert_eq!(get("output_tokens"), Some("0"));
        assert_eq!(get("cost_usd"), Some("0.0"));
    }
}
