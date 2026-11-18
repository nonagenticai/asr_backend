# Operation ID Plan — asr_backend

Source of truth for FastAPI `operation_id=` values. Mirrored in
`tests/test_openapi.py::EXPECTED_OPERATION_IDS`. Generators consume these IDs
verbatim, so renames must be coordinated across the fleet.

Naming convention: `verb_noun` snake_case, globally unique across the
`mini1`/`mini2` host-services fleet.

## `src/main.py`

| Method | Path           | Function | operation_id              |
| ------ | -------------- | -------- | ------------------------- |
| GET    | /              | root     | service_root              |
| GET    | /health        | health   | health_check              |
| GET    | /health/ready  | ready    | health_ready              |
| GET    | /health/live   | live     | health_live               |
| GET    | /me            | me       | get_current_identity      |

## `src/routers/asr_routes.py` (prefix `/api`, tag `Transcription`)

| Method | Path             | Function            | operation_id        |
| ------ | ---------------- | ------------------- | ------------------- |
| POST   | /api/transcribe  | transcribe_endpoint | transcribe_audio    |
| GET    | /api/models      | list_models         | list_asr_models     |

Total: 7 explicit `operation_id` values.
