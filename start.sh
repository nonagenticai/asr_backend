#!/bin/bash
# Start ASR Backend service
set -euo pipefail

cd "$(dirname "$0")"

PORT="${ASR_PORT:-9201}"

# Source synced env file if present (populated by gitops sync-env.sh).
# Prefer the user-writable XDG path (no sudo needed on mini2); fall back to
# /etc/ for backwards-compat with hosts that have privileged provisioning.
ENV_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/asr-backend/env"
if [ ! -f "$ENV_FILE" ] && [ -f /etc/asr-backend/env ]; then
  ENV_FILE=/etc/asr-backend/env
fi
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ENV_FILE"
  set +a
fi

# Kill any existing instance on the port
lsof -ti :"${PORT}" | xargs kill -9 2>/dev/null || true
sleep 1

# Rust implementation (whisper-rs + whisper.cpp on Metal). The binary is
# PREBUILT on a dev Mac and copied here: neither mini has a Rust toolchain, and
# a host-side cargo build would not be the binary that was tested.
#
# Falls back to the Python service when the binary or its checkpoint is absent,
# so a repo pull can never leave the unit dead — the fallback is loud in the log.
BIN="${ASR_RS_BIN:-$(pwd)/asr-backend-rs}"
MODEL="${ASR_MODEL_PATH:-$HOME/models/whisper/ggml-large-v3-turbo-q5_0.bin}"
if [ "${ASR_IMPL:-auto}" != "python" ] && [ -x "$BIN" ] && [ -f "$MODEL" ]; then
  export ASR_MODEL_PATH="$MODEL"
  export ASR_BIND_ADDR="${ASR_BIND_ADDR:-0.0.0.0:${PORT}}"
  echo "Starting ASR Backend (rust) on ${ASR_BIND_ADDR} model=${MODEL}..."
  exec "$BIN"
fi
echo "ASR Backend: Rust binary or checkpoint absent (bin=$BIN model=$MODEL) — falling back to Python" >&2

mkdir -p logs

# System CA bundle for HuggingFace model downloads
export SSL_CERT_FILE="${SSL_CERT_FILE:-/opt/homebrew/etc/ca-certificates/cert.pem}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-/opt/homebrew/etc/ca-certificates/cert.pem}"

echo "Starting ASR Backend on port ${PORT}..."
echo "Docs: http://localhost:${PORT}/docs"

PYTHONPATH=src exec uv run uvicorn main:app --host 0.0.0.0 --port "${PORT}"
