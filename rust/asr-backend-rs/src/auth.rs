//! Identity gate.
//!
//! Inbound auth on this service is **attribution for billing and a deny gate**:
//! the live Python service answers `401 AUTHENTICATION_REQUIRED` with no
//! credential and `401 INVALID_TOKEN` with a bad one (measured on mini2,
//! 2026-11-13), and w2v runs everything behind a licence token. The tenant is
//! resolved from `X-Tenant-Id: Bearer <licence-token>` (the estate's east-west
//! convention) or `Authorization: Bearer <token>`, onto the request, so a
//! metering event can name who spent the compute.
//!
//! ⚠️ An unauthenticated transcription is worse than a rejected one: it burns
//! GPU time that nobody is billed for. So a licence service that cannot be
//! reached is a 503 — never a free pass — and a token the service rejects is a
//! 401, never an anonymous request.
//!
//! The *entitlement* gate (`W2V_LICENSE_GATE_ENABLED`, the app-key/lookup
//! machinery in `w2v_common.auth.entitlement`) is DORMANT on this host today —
//! its env keys are present but the flag is not set — so this module resolves
//! identity and stops there. Arming it is a separate decision.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use axum::http::HeaderMap;
use serde::Deserialize;
use tokio::sync::Mutex;

use crate::error::AppError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identity {
    pub tenant_id: String,
    pub licence_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct IntrospectResponse {
    #[serde(default)]
    active: bool,
    #[serde(default)]
    tenant_id: Option<String>,
    // The wire name is `license_id`, not the British spelling used everywhere
    // in this crate — a `licence_id` field here silently deserialises to None
    // and the metering event then carries no licence at all.
    #[serde(default, rename = "license_id")]
    licence_id: Option<String>,
    #[serde(default)]
    state: Option<String>,
    /// Whose credential this is. `subject_actor_sub` is the SA that presented
    /// the token; a delegating caller stamps the real tenant separately.
    #[serde(default)]
    subject_id: Option<String>,
    #[serde(default)]
    subject_actor_sub: Option<String>,
    #[serde(default)]
    status: Option<String>,
    /// Why an inactive token is inactive (`invalid_token`, `token_not_found`,
    /// ...). The Python gate puts this after the colon in its 401 message, so a
    /// caller that greps for it sees no change.
    #[serde(default)]
    reason: Option<String>,
}

struct Cached {
    identity: Identity,
    /// The credential's own subject, kept so per-request delegation can be
    /// decided AFTER the cache: the effective tenant differs by header, so it
    /// must never be part of the cached value.
    subject: Option<String>,
    expires_at: Instant,
}

pub struct LicenceGate {
    client: reqwest::Client,
    url: String,
    enabled: bool,
    /// 60 s positive cache — the same window w2v_common uses. Keyed by the
    /// presented token: an unkeyed cache would attribute tenant A's work to
    /// tenant B for up to a minute, which is a billing defect, not a
    /// performance tweak.
    cache: Mutex<HashMap<String, Cached>>,
    /// `AUTH_DELEGATING_SUBJECTS`: the service accounts allowed to stamp
    /// `X-Effective-Tenant-Id` on a request. Empty means nobody may.
    delegating_subjects: Vec<String>,
}

const POSITIVE_TTL: Duration = Duration::from_secs(60);

/// Cache ceiling. The estate has a few dozen callers; anything beyond this is
/// churn, and dropping the map costs one introspect per caller to refill.
const CACHE_MAX: usize = 256;

impl LicenceGate {
    pub fn new(url: String, enabled: bool, delegating_subjects: Vec<String>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .expect("reqwest client"),
            url,
            enabled,
            cache: Mutex::new(HashMap::new()),
            delegating_subjects,
        }
    }

    /// Resolves the caller's identity from the request headers.
    pub async fn authenticate(&self, headers: &HeaderMap) -> Result<Identity, AppError> {
        if !self.enabled {
            return Ok(Identity {
                tenant_id: "unauthenticated".to_string(),
                licence_id: None,
            });
        }

        let token = presented_token(headers).ok_or(AppError::AuthenticationRequired)?;

        let (identity, subject) = {
            let cached = {
                let cache = self.cache.lock().await;
                cache
                    .get(&token)
                    .filter(|c| c.expires_at > Instant::now())
                    .map(|c| (c.identity.clone(), c.subject.clone()))
            };
            match cached {
                Some(hit) => hit,
                None => {
                    let resolution = self.introspect(&token).await?;
                    let mut cache = self.cache.lock().await;
                    if cache.len() >= CACHE_MAX {
                        cache.clear();
                    }
                    cache.insert(
                        token,
                        Cached {
                            identity: resolution.0.clone(),
                            subject: resolution.1.clone(),
                            expires_at: Instant::now() + POSITIVE_TTL,
                        },
                    );
                    resolution
                }
            }
        };

        Ok(effective_identity(
            headers,
            identity,
            subject.as_deref(),
            &self.delegating_subjects,
        ))
    }

    async fn introspect(&self, token: &str) -> Result<(Identity, Option<String>), AppError> {
        let resp = self
            .client
            .post(&self.url)
            // The token travels in the BODY (`{"token": ...}`), not in an
            // Authorization header: the endpoint is public and the token in
            // the body IS the credential (w2v_common middleware does exactly
            // this). Sending a bearer header answers 422.
            .json(&serde_json::json!({ "token": token }))
            .send()
            .await
            // A licence service we cannot reach is NOT a licence we may skip:
            // that is the one failure mode that turns a gate into a bypass.
            .map_err(|e| {
                AppError::ServiceUnavailable(format!("licence service unreachable: {e}"))
            })?;

        if resp.status() == reqwest::StatusCode::UNAUTHORIZED
            || resp.status() == reqwest::StatusCode::FORBIDDEN
        {
            let detail = introspect_error(resp).await;
            return Err(AppError::InvalidToken(format!(
                "Invalid or expired license token: {detail}"
            )));
        }
        if !resp.status().is_success() {
            return Err(AppError::ServiceUnavailable(format!(
                "licence introspection failed: HTTP {}",
                resp.status().as_u16()
            )));
        }

        let body: IntrospectResponse = resp.json().await.map_err(|e| {
            AppError::ServiceUnavailable(format!("licence response unparseable: {e}"))
        })?;

        if !body.active {
            // A token the licences service does not recognise is a CREDENTIAL
            // problem, not an entitlement one: the live Python service answers
            // 401 INVALID_TOKEN here (measured 2026-11-13), and 402 is reserved
            // for the entitlement gate, which is dormant on this host.
            let detail = body
                .reason
                .as_deref()
                .or(body.state.as_deref())
                .or(body.status.as_deref())
                .unwrap_or("invalid_token");
            return Err(AppError::InvalidToken(format!(
                "Invalid or expired license token: {detail}"
            )));
        }

        let identity = Identity {
            tenant_id: body.tenant_id.ok_or_else(|| {
                AppError::ServiceUnavailable("licence response had no tenant".into())
            })?,
            licence_id: body.licence_id,
        };

        let subject = body.subject_actor_sub.or(body.subject_id);
        Ok((identity, subject))
    }
}

/// The `detail` the licences service gives for a rejected token, so the 401
/// message reads like the Python one (`... invalid_token`).
async fn introspect_error(resp: reqwest::Response) -> String {
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    serde_json::from_str::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| {
            v.get("detail")
                .and_then(|d| d.as_str().map(str::to_string))
                .or_else(|| {
                    v.get("error")
                        .and_then(|e| e.get("code"))
                        .and_then(|c| c.as_str())
                        .map(str::to_string)
                })
        })
        .unwrap_or_else(|| status.as_u16().to_string())
}

/// A service account may act *for* a tenant: `X-Effective-Tenant-Id` is
/// honoured ONLY when the credential's own subject is listed in
/// `AUTH_DELEGATING_SUBJECTS` — the same rule w2v_common applies, and the
/// reason an ASR cost event can name the run's tenant instead of the caller's
/// service-account tenant.
///
/// A customer subject is never on that list, so a tenant cannot mint an
/// effective-tenant header and spend as somebody else.
fn effective_identity(
    headers: &HeaderMap,
    mut identity: Identity,
    subject: Option<&str>,
    allow: &[String],
) -> Identity {
    let Some(subject) = subject else {
        return identity;
    };
    if !allow.iter().any(|s| s.eq_ignore_ascii_case(subject)) {
        return identity;
    }
    let Some(effective) = headers
        .get("x-effective-tenant-id")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|v| !v.is_empty())
    else {
        return identity;
    };
    identity.tenant_id = effective.to_string();
    identity
}

/// The licence token on a request: `X-Tenant-Id: Bearer <token>` first (the
/// estate's east-west convention, and what the w2v callers send), else
/// `Authorization: Bearer <token>`. Mirrors `licence.rs` in mlx_inference.
pub fn presented_token(headers: &HeaderMap) -> Option<String> {
    for name in ["x-tenant-id", "authorization"] {
        if let Some(raw) = headers.get(name).and_then(|v| v.to_str().ok()) {
            let token = raw
                .trim()
                .strip_prefix("Bearer ")
                .or_else(|| raw.trim().strip_prefix("bearer "))
                .unwrap_or(raw)
                .trim();
            if !token.is_empty() {
                return Some(token.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(*k, HeaderValue::from_str(v).unwrap());
        }
        h
    }

    #[test]
    fn x_tenant_id_wins_over_authorization() {
        let h = headers(&[
            ("x-tenant-id", "Bearer estate-token"),
            ("authorization", "Bearer keycloak-jwt"),
        ]);
        assert_eq!(presented_token(&h).as_deref(), Some("estate-token"));
    }

    #[test]
    fn authorization_is_the_fallback() {
        let h = headers(&[("authorization", "Bearer jwt")]);
        assert_eq!(presented_token(&h).as_deref(), Some("jwt"));
    }

    #[test]
    fn a_bare_token_is_accepted_without_the_bearer_prefix() {
        let h = headers(&[("x-tenant-id", "opaque")]);
        assert_eq!(presented_token(&h).as_deref(), Some("opaque"));
    }

    #[test]
    fn no_header_is_no_token() {
        assert_eq!(presented_token(&HeaderMap::new()), None);
        assert_eq!(presented_token(&headers(&[("x-tenant-id", "  ")])), None);
    }

    #[tokio::test]
    async fn a_disabled_gate_does_not_require_a_token() {
        let gate = LicenceGate::new(String::new(), false, Vec::new());
        let id = gate.authenticate(&HeaderMap::new()).await.unwrap();
        assert_eq!(id.tenant_id, "unauthenticated");
    }

    #[test]
    fn a_listed_service_account_may_act_for_another_tenant() {
        let allow = vec!["8382f0b3-c082-4f97-8d47-120ce5a18e05".to_string()];
        let base = Identity {
            tenant_id: "sa-tenant".into(),
            licence_id: Some("lic".into()),
        };
        let h = headers(&[("x-effective-tenant-id", "run-tenant")]);
        let id = effective_identity(
            &h,
            base,
            Some("8382f0b3-c082-4f97-8d47-120ce5a18e05"),
            &allow,
        );
        assert_eq!(id.tenant_id, "run-tenant");
        // The licence is untouched — only the attribution moved.
        assert_eq!(id.licence_id.as_deref(), Some("lic"));
    }

    #[test]
    fn an_unlisted_subject_cannot_stamp_a_tenant() {
        let allow = vec!["8382f0b3-c082-4f97-8d47-120ce5a18e05".to_string()];
        let base = Identity {
            tenant_id: "customer-tenant".into(),
            licence_id: None,
        };
        let h = headers(&[("x-effective-tenant-id", "someone-else")]);
        let id = effective_identity(&h, base.clone(), Some("customer-subject"), &allow);
        assert_eq!(id.tenant_id, "customer-tenant");

        // No allowlist at all is the default and delegates to nobody.
        let id = effective_identity(&h, base, Some("8382f0b3-c082-4f97-8d47-120ce5a18e05"), &[]);
        assert_eq!(id.tenant_id, "customer-tenant");
    }

    #[tokio::test]
    async fn an_armed_gate_answers_401_without_a_token() {
        let gate = LicenceGate::new("http://127.0.0.1:1/introspect".into(), true, Vec::new());
        let err = gate.authenticate(&HeaderMap::new()).await.unwrap_err();
        assert_eq!(err.code(), "AUTHENTICATION_REQUIRED");
        assert_eq!(err.status().as_u16(), 401);
    }

    #[tokio::test]
    async fn an_unreachable_licence_service_is_a_503_never_a_pass() {
        let gate = LicenceGate::new("http://127.0.0.1:1/introspect".into(), true, Vec::new());
        let err = gate
            .authenticate(&headers(&[("x-tenant-id", "Bearer t")]))
            .await
            .unwrap_err();
        assert_eq!(err.status().as_u16(), 503);
    }
}
