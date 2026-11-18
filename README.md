# asr-backend

Speech-to-text inference service for Apple Silicon. Replaces an earlier in-cluster transcription deployment.

| | |
|---|---|
| Engine | `mlx-whisper` (ANE-accelerated) — requires `brew install ffmpeg` on the host |
| Default model | `mlx-community/whisper-large-v3-mlx` |
| Port | `9201` |
| Hostname | `asr.test` (platform-internal hostname) |
| Auth | Istio header pass-through (`x-user-id`, `x-tenant-id`, ...) — license-authz validates upstream |
| Host | launchd service on an Apple Silicon Mac (not a Kubernetes pod) |

## API

| Method | Path | Notes |
|---|---|---|
| `POST` | `/api/transcribe` | multipart audio in; `file` + optional `language`, `word_timestamps`, `return_srt` form fields. Response shape preserved from the old `transcription-worker-service`. |
| `GET` | `/api/models` | currently loaded model + available size strings |
| `GET` | `/health` | liveness-style; always 200 if process is up |
| `GET` | `/health/ready` | reports `degraded` until model is loaded |
| `GET` | `/health/live` | minimal liveness |

## Configuration

Every knob is environment-variable-driven; defaults below are the baked-in fallbacks. Deployments render the production values into a host env file (outside this repository) that `start.sh` sources.

| Var | Default | Purpose |
|---|---|---|
| `ASR_MODEL` | `mlx-community/whisper-large-v3-mlx` | HF repo or local path |
| `ASR_COMPUTE_TYPE` | `fp16` | reserved for future int8 MLX builds |
| `ASR_LANGUAGE_DEFAULT` | `` (auto-detect) | default language code |
| `ASR_MAX_AUDIO_BYTES` | `104857600` | reject larger audio uploads |
| `LOG_LEVEL` | `INFO` | uvicorn + app logging |
| `CORS_ORIGINS` | `https://api.nonagentic.ai,https://w2v.nonagentic.ai` | comma-separated |

`ASR_VAD_ENABLED` was removed on 2026-11-14. It was parsed into
`WhisperService.vad_enabled` and never read: there has never been a
voice-activity-detection implementation in this service, nor in the Rust binary
`start.sh` runs. "No VAD" is the design, not a gap — do not re-open without a
new owner decision.

## Local dev

```bash
uv sync
PYTHONPATH=src uv run pytest                                  # CI-safe (no MLX needed)
PYTHONPATH=src uv run uvicorn main:app --host 0.0.0.0 --port 9201   
```

## Production install on the host

The launchd plist and env-sync wiring are provisioned by the operator and are not part of this repository.

## Licensing

This service contains no licence enforcement of its own and runs no licence
server. It runs with no licence switch set; unset means off.

## License

Apache 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
