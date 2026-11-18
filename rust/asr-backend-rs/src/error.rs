//! Error envelope.
//!
//! The Python service answers every error with the shared w2v shape (see
//! `tests/test_error_envelope.py`):
//!
//! ```json
//! {"error": {"code": "...", "message": "...", "status": 401,
//!            "request_id": "<uuid7>", "timestamp": "YYYY-MM-DDTHH:MM:SSZ"}}
//! ```
//!
//! and sets the same id on the `X-Request-ID` response header. A caller that
//! logs that header today can quote it to support tomorrow; a `{"detail": ...}`
//! body (what this module used to emit) silently loses that.

use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// Request id on the response header, mirroring the Python middleware.
pub const REQUEST_ID_HEADER: &str = "x-request-id";

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    /// 400 — empty payload and other caller mistakes.
    #[error("{0}")]
    BadRequest(String),

    /// 401 — no credential was presented at all.
    #[error("Missing authentication token")]
    AuthenticationRequired,

    /// 401 — a credential was presented and it is not valid.
    #[error("{0}")]
    InvalidToken(String),

    // NOTE: there is deliberately no 413 variant. The upload cap is enforced by
    // axum's `DefaultBodyLimit` layer, which renders its own 413 before a
    // handler runs — the Python service also answered 413 itself rather than
    // raising. A variant here would be dead code that reads as reachable.
    /// 429 — admission lane full. Carries Retry-After so callers back off.
    #[error("{detail}")]
    TooManyRequests { detail: String, retry_after_s: u64 },

    /// 503 — model not loaded, shutting down, or licence service unreachable.
    #[error("{0}")]
    ServiceUnavailable(String),

    /// 500 — anything else.
    #[error("{0}")]
    Internal(String),
}

impl AppError {
    pub fn status(&self) -> StatusCode {
        match self {
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::AuthenticationRequired | Self::InvalidToken(_) => StatusCode::UNAUTHORIZED,
            Self::TooManyRequests { .. } => StatusCode::TOO_MANY_REQUESTS,
            Self::ServiceUnavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// The machine-readable `error.code` the Python service emits.
    pub fn code(&self) -> &str {
        match self {
            Self::BadRequest(_) => "VALIDATION_ERROR",
            Self::AuthenticationRequired => "AUTHENTICATION_REQUIRED",
            Self::InvalidToken(_) => "INVALID_TOKEN",
            Self::TooManyRequests { .. } => "RATE_LIMITED",
            Self::ServiceUnavailable(_) => "SERVICE_UNAVAILABLE",
            Self::Internal(_) => "INTERNAL_SERVER_ERROR",
        }
    }

    fn retry_after(&self) -> Option<u64> {
        match self {
            Self::TooManyRequests { retry_after_s, .. } => Some(*retry_after_s),
            Self::ServiceUnavailable(_) => Some(30),
            _ => None,
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = self.status();
        let request_id = uuid::Uuid::now_v7().to_string();
        let body = json!({
            "error": {
                "code": self.code(),
                "message": self.to_string(),
                "status": status.as_u16(),
                "request_id": request_id,
                "timestamp": format_rfc3339_utc(SystemTime::now()),
            }
        });

        let mut resp = (status, axum::Json(body)).into_response();
        if let Ok(v) = HeaderValue::from_str(&request_id) {
            resp.headers_mut().insert(REQUEST_ID_HEADER, v);
        }
        if let Some(secs) = self.retry_after()
            && let Ok(v) = HeaderValue::from_str(&secs.to_string())
        {
            resp.headers_mut().insert(header::RETRY_AFTER, v);
        }
        resp
    }
}

/// `YYYY-MM-DDTHH:MM:SSZ` in UTC — the Python handler's `strftime` format.
///
/// Hand-rolled rather than pulling a date crate: the only consumer is this
/// envelope, the whole conversion is civil-from-days, and it is unit-tested
/// against fixed instants.
fn format_rfc3339_utc(at: SystemTime) -> String {
    let secs = at
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// Howard Hinnant's `civil_from_days`, days since 1970-01-01.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_a_known_instant() {
        // 2026-11-13T22:11:38Z — the `timestamp` the live Python service
        // returned on a 401 during this port's parity probes.
        let at = UNIX_EPOCH + std::time::Duration::from_secs(1_794_607_898);
        assert_eq!(format_rfc3339_utc(at), "2026-11-13T22:11:38Z");
    }

    #[test]
    fn formats_the_epoch() {
        assert_eq!(format_rfc3339_utc(UNIX_EPOCH), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn auth_errors_are_401_with_the_python_codes() {
        assert_eq!(AppError::AuthenticationRequired.status().as_u16(), 401);
        assert_eq!(
            AppError::AuthenticationRequired.code(),
            "AUTHENTICATION_REQUIRED"
        );
        assert_eq!(
            AppError::AuthenticationRequired.to_string(),
            "Missing authentication token"
        );
        assert_eq!(AppError::InvalidToken("bad".into()).code(), "INVALID_TOKEN");
    }

    #[test]
    fn a_503_tells_the_caller_to_retry() {
        assert_eq!(
            AppError::ServiceUnavailable("x".into()).retry_after(),
            Some(30)
        );
        assert_eq!(AppError::BadRequest("x".into()).retry_after(), None);
    }
}
