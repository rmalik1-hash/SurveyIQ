# Phase 6 — Dashboard (and the `question_stats` API gap)

## What this is

The final v1 phase. It delivers the administrator-facing dashboard and closes a
gap between the API and CLAUDE.md's definition of "done".

CLAUDE.md says a v1 administrator can:
1. Upload a CSV from their survey tool
2. Map their columns once (~2 minutes)
3. See a dashboard: overall data-quality score, flagged respondents with scores
   and plain-English explanations, and **which questions generated the most
   inconsistencies**
4. **Download a cleaned dataset** with unreliable responses removed or marked

Items 3 (question-level view) and 4 (download) are not supported by the current
API, so this phase is split into two parts.

## Part A — `question_stats` in the API

### Why
`POST /analyze` returns only `summary` and `respondents`. The dashboard's
"which questions caused the most inconsistencies" panel needs per-question data.
We compute it in Python rather than re-deriving it in JavaScript, so the
analysis logic has a single source of truth.

### Targeted refactor (justified, not scope creep)
The rule for "what counts as a contradiction" currently lives inline inside
`extract.contradiction_score`. Part A extracts it into a public helper so both
the feature and the new stats share one definition:

```python
# src/features/extract.py
def pair_contradicts(responses, scale_min, scale_max, a_key, b_key, tolerance=1) -> bool:
    """True if a reverse-coded pair's answers are inconsistent."""
```

`contradiction_score` is rewritten to call it. Behaviour is unchanged (the
existing Phase 2 tests must keep passing untouched).

### New response field
`/analyze` gains `question_stats`: a list sorted by `affected` descending.

```json
"question_stats": [
  {"label": "...[Q1] / ...[Q2]", "type": "contradiction_pair", "affected": 34},
  {"label": "...[AC1]", "type": "attention_check", "affected": 8}
]
```

- **contradiction_pair** — one entry per configured pair; `affected` = number of
  respondents whose answers contradict on it. `label` is `"<colA> / <colB>"`
  using the client's original column names.
- **attention_check** — one entry per attention-check column; `affected` =
  number of respondents whose given answer != the correct answer. `label` is the
  original column name.
- Empty list when a survey has neither pairs nor attention checks (graceful).

Computed by `_question_stats(respondents, mapping, qkey_pairs)` in
`src/api/pipeline_service.py` and included by `analyze`.

## Part B — The React dashboard (`frontend/`)

### Stack
Vite + React + Plotly (per CLAUDE.md's tech stack) + Vitest/Testing-Library.
Plus **PapaParse**: the generator's headers contain commas inside quotes, so
naive CSV splitting corrupts data — a real parser is required for the download
feature.

### Flow (three steps, matching CLAUDE.md's "done")
1. **Upload** — choose a CSV. The file is sent to `POST /columns`; the parsed
   file is also kept in browser memory for the later download.
2. **Map** — per-column role dropdown (`respondent_id`, `start_time`,
   `end_time`, `question`, `attention_check`, `demographic`, `ignore`), scale
   min/max inputs, a correct-answer input per attention-check column, and an
   optional list of reverse-coded question pairs. Submitting sends the file +
   mapping JSON to `POST /analyze`.
3. **Results** — renders the response:
   - summary cards (total / flagged / reliable / overall quality %)
   - a Plotly gauge of overall quality
   - the flagged-respondent table with `reliability_score` and `flag_reason`
   - a Plotly horizontal bar chart of `question_stats`
   - a **Download cleaned CSV** control

### Download (client-side, server stays stateless)
The browser already holds the original parsed rows. It joins the API's
per-respondent scores by `respondent_id` and offers two modes:
- **Marked** — original columns plus `reliability_score` and `flag_reason`.
- **Cleaned** — flagged rows removed.

No re-upload; nothing extra is stored server-side.

### Code layout (logic separated from UI so it is testable)
```
frontend/
  index.html
  package.json
  vite.config.js
  src/
    main.jsx
    App.jsx                  # step state machine: upload -> map -> results
    lib/api.js               # fetchColumns(file), analyze(file, mapping)
    lib/mapping.js           # buildMapping(state) -> mapping object; validateMapping(state)
    lib/cleanedCsv.js        # buildCleanedRows(rows, respondents, mode), toCsv(rows)
    components/UploadStep.jsx
    components/MappingStep.jsx
    components/ResultsDashboard.jsx
  tests/
    mapping.test.js
    cleanedCsv.test.js
    UploadStep.test.jsx
```

### Configuration
The API base URL comes from `VITE_API_BASE` (default `http://127.0.0.1:8000`).
The FastAPI app gains permissive CORS for local development so the Vite dev
server (a different origin) can call it.

### Error handling
- API errors (400/422) surface the server's `detail` message in the UI.
- `validateMapping` blocks submission client-side when there is no
  `respondent_id`, no `question` column, an attention check missing its correct
  answer, or an invalid scale — the same rules the backend enforces, surfaced
  early for a better experience. The backend remains the authority.

### Testing
- **Pure logic (properly unit-tested):** `buildMapping` produces the exact
  mapping object shape the API expects; `validateMapping` catches each invalid
  case; `buildCleanedRows` marks and filters correctly and joins by
  `respondent_id`; `toCsv` quotes fields containing commas/quotes.
- **Light component tests:** `UploadStep` renders and enables its action once a
  file is chosen.
- Python side: Part A is TDD'd (`pair_contradicts`, `_question_stats`, and the
  `/analyze` response including `question_stats`).

## Out of scope

- Authentication, hosting/deployment, and the v2 backlog (email delivery,
  history comparison) — see `docs/roadmap-and-backlog.md`.
- Server-side generation of the cleaned file (client-side keeps the API
  stateless).
- Polished visual design beyond the existing mockup's structure; the mockup
  (`mockups/dashboard-mockup.html`) is the layout reference.
