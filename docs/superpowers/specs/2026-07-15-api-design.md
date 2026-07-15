# Phase 5 — FastAPI Backend Design

## What this is

The backend is the fifth stage of SurveyIQ. It wraps the detection pipeline
(normalize → extract → predict) in a small, stateless HTTP API so a website —
or any client — can upload a raw survey CSV plus a column mapping and get back
scored, explained results.

```
HTTP upload (CSV + mapping)  →  [API]  →  normalize → extract → predict  →  JSON (summary + per-respondent)
```

The API is a thin wrapper: all real work lives in the already-built and tested
Phase 1–4 library. This phase adds only HTTP plumbing and a startup-trained
model.

## Decisions locked in (from brainstorming)

1. **Model trained on synthetic data at startup, held in memory.** Real uploads
   have no labels, so the API cannot train on them. On startup it generates
   labelled synthetic data, runs the full pipeline, trains the Phase 4 model,
   and keeps it in memory. This is a deliberate v1 stopgap — production would
   load a persisted, research-calibrated model (ties to the G1 gate and the
   deferred model-persistence work).
2. **Three endpoints:** `/health`, `/columns`, `/analyze`.
3. **Stateless / confidentiality-first.** No uploaded data is written to disk or
   logged. Data is processed in memory and discarded after the response; the
   normalizer strips PII before anything is scored.

## Dependencies

Add to `requirements.txt`: `fastapi`, `uvicorn`, `python-multipart` (multipart
file uploads). Tests use `fastapi.testclient.TestClient` (needs `httpx`, already
installed). All are already present in the environment.

## Endpoints

### `GET /health`
Liveness check. Returns `{"status": "ok"}` with 200.

### `POST /columns`
Multipart upload of a CSV file (field name `file`). Reads only the header and
returns the column names, to help a client build a mapping.

Response: `{"columns": ["Response ID", "Start Time", ...]}` (200).
Errors: unparseable CSV → 400.

### `POST /analyze`
Multipart upload: a CSV `file` plus a `mapping` form field containing a JSON
string. Runs the pipeline and returns scored results.

The `mapping` JSON is the Phase 3 mapping object with one optional addition:

```json
{
  "columns": {"Response ID": "respondent_id", "...[Q1]": "question", "...": "..."},
  "scale": [1, 5],
  "attention_check_answers": {"...[AC1]": 5},
  "contradiction_pairs": [["...[Q1]", "...[Q2]"]]
}
```

`contradiction_pairs` is optional and expressed as **column-name** pairs; the
API converts them to internal `q`-keys (both columns must be tagged
`question`). Omitting it means `contradiction_score` is 0 for all respondents
(graceful degradation).

Response (200):

```json
{
  "summary": {"total": 200, "flagged": 50, "reliable": 150, "overall_quality_pct": 75.0},
  "respondents": [
    {"respondent_id": "R1", "reliability_score": 0.08, "flag_reason": "Answered very fast (...)."},
    {"respondent_id": "R2", "reliability_score": 0.97, "flag_reason": ""}
  ]
}
```

Errors: malformed CSV, invalid mapping (normalizer `ValueError`), or a
`contradiction_pairs` entry naming a non-question column → **400** with the
error message. Missing `file`/`mapping` fields → FastAPI **422**.

## Architecture

Keep HTTP thin and the logic pure/testable.

- **`src/api/pipeline_service.py`** — the brains, no HTTP:
  - `train_startup_model(n_respondents=300, n_questions=16, seed=42) -> model` —
    generates synthetic data in a `TemporaryDirectory`, runs
    normalize→extract→train, returns the fitted model. Writes only to the temp
    dir, which is cleaned up.
  - `analyze(raw_df, mapping, model) -> dict` — pure function: applies the
    mapping, extracts features (converting `contradiction_pairs` column pairs to
    `q`-keys), predicts, and assembles the `{summary, respondents}` dict. Raises
    `ValueError` on bad input (mapping problems, bad contradiction columns).
  - `_pairs_to_qkeys(mapping) -> list[tuple[str, str]]` — helper mapping
    question columns (in `columns` order) to `q1..qn` and translating the
    optional `contradiction_pairs` column pairs; raises `ValueError` if a named
    column is not a `question`.
  - `_build_training_mapping(columns) -> dict` — programmatic mapping for the
    generator's known column names (`[Q` → question, `[AC` → attention_check,
    `Response ID`/`Start Time`/`Timestamp`/`Email Address` → their roles, else
    demographic), reused at startup.
- **`src/api/main.py`** — the FastAPI app:
  - Trains the startup model via a lifespan handler and stores it on
    `app.state.model`.
  - Defines the three routes; `/analyze` reads the upload, parses the mapping
    JSON, calls `analyze`, and maps `ValueError` → `HTTPException(400)`.
- **`src/api/__init__.py`** — package marker.

## Confidentiality

- Uploaded bytes are read into a pandas DataFrame in memory and never written to
  disk.
- No request body or respondent data is logged.
- The normalizer's PII safety net runs before scoring; the response contains
  only `respondent_id`, scores, and reasons — never raw responses or
  demographics.

## Testing

Pure-logic tests (no HTTP):
- `_build_training_mapping` tags generator columns correctly.
- `_pairs_to_qkeys` converts column pairs to `q`-keys and raises on a
  non-question column.
- `analyze(df, mapping, model)` on a small hand-built DataFrame + a tiny model
  returns the right `summary` counts and one row per respondent, with
  `reliability_score` in [0, 1] and empty `flag_reason` for reliable rows.
- `train_startup_model()` returns a fitted model (uses a small synthetic set for
  speed).

HTTP tests (`TestClient`):
- `GET /health` → 200 `{"status": "ok"}`.
- `POST /columns` with a generated CSV → 200 and the expected column list.
- `POST /analyze` with a generated CSV + programmatic mapping → 200; response has
  `summary` with `total` equal to the row count and a `respondents` list of the
  same length; flagged respondents carry a non-empty `flag_reason`.
- `POST /analyze` with an invalid mapping (e.g. no `respondent_id` role) → 400.
- `POST /analyze` missing the `mapping` field → 422.

The `TestClient` triggers the lifespan handler, so the startup model trains once
per test app instance; a small synthetic size keeps this fast.

## Out of scope

- Authentication / rate limiting / CORS config (deployment concerns, not v1
  logic).
- Model persistence, email delivery, result history (v2 backlog).
- The React dashboard (Phase 6) — this API is what it will call.
- Async/streaming for very large files — v1 handles a normal in-memory upload.
