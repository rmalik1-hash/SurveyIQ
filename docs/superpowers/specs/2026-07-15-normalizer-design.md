# Phase 3 — Normalizer + Column Mapper Design

## What this is

The normalizer is the third stage of the SurveyIQ pipeline. It converts a raw,
messy survey DataFrame plus a caller-supplied column mapping into the internal
respondent schema that the Phase 2 feature extractor consumes.

```
raw messy DataFrame + mapping  →  [NORMALIZER]  →  list[respondent dict]  →  feature extractor
```

This is the "Option B" case deferred from Phase 2: handling exports as they
actually arrive from Google Forms / Qualtrics / SurveyMonkey, with verbose
headers, stray columns, and PII, without forcing schools to reformat.

## Decisions locked in (from brainstorming)

1. **Responses stay raw ints + scale.** The normalizer outputs raw Likert ints
   plus `scale_min`/`scale_max`, matching the internal schema and the Phase 2
   extractor. Normalization to 0–1 lives inside the extractor. This resolves a
   self-contradiction in CLAUDE.md (its prose says "normalize to 0–1 at
   ingestion" but its schema and the built extractor use raw ints) in favor of
   the working pipeline. Single source of truth.
2. **PII: explicit mapping + safety net.** Only mapped columns are kept; a
   heuristic drops obvious PII even if mis-tagged.
3. **Mapping persistence deferred.** Phase 3 builds the pure
   `apply_mapping(raw_df, mapping)` function only. Save/load-per-school waits
   for the API/frontend (Phase 5/6).

## Input contract

`apply_mapping(raw_df: pd.DataFrame, mapping: dict) -> list[dict]`

The `mapping` dict:

```python
{
    "columns": {                       # raw column name -> role
        "Response ID": "respondent_id",
        "Start Time": "start_time",
        "Timestamp": "end_time",
        "On a scale of 1 to 5 ... [Q1]": "question",
        "On a scale of 1 to 5 ... [Q2]": "question",
        "For quality control ... [AC1]": "attention_check",
        "Email Address": "ignore",
        "grade level": "demographic",
        "School Name": "demographic",
    },
    "scale": [1, 5],                   # [scale_min, scale_max]
    "attention_check_answers": {       # correct answer per attention-check column
        "For quality control ... [AC1]": 5,
    },
}
```

Roles: `respondent_id` (exactly one required), `start_time`, `end_time`,
`question` (at least one required), `attention_check`, `demographic`, `ignore`.
Any column not present in `columns` is treated as `ignore` (dropped).

## Output contract

A list of respondent dicts, one per input row, in row order:

```python
{
    "respondent_id": str,
    "duration_seconds": int | None,    # end - start; None if either timestamp absent/unparseable
    "responses": {"q1": int, "q2": int, ...},   # raw ints, keyed q1..qN in column order
    "attention_checks": {              # {} if no attention-check columns
        "ac1_given": int, "ac1_correct": int,
        "ac2_given": int, "ac2_correct": int,
    },
    "demographics": {"grade level": <value>, ...},  # passed through, minus PII
    "scale_min": int,
    "scale_max": int,
}
```

Question columns map to keys `q1, q2, ...` in the left-to-right order they
appear in `mapping["columns"]`. Attention-check columns map to `ac1, ac2, ...`
in the same order; `acN_given` is the respondent's answer, `acN_correct` comes
from `attention_check_answers`.

## Architecture

Pure helper functions plus a thin orchestrator, in one focused module.

- **File:** `src/ingestion/normalize.py`
- **Package:** `src/ingestion/__init__.py`

### Functions

| Function | Responsibility |
|---|---|
| `apply_mapping(raw_df, mapping)` | Orchestrator: validate, then build one dict per row |
| `_validate_mapping(raw_df, mapping)` | Raise `ValueError` on structural problems (see below) |
| `_columns_by_role(mapping)` | Group column names by role, preserving order |
| `_compute_duration(start, end)` | `(end - start)` seconds; `None` if either missing/unparseable |
| `_extract_responses(row, question_cols)` | `{q1: int, ...}` from a row |
| `_extract_attention_checks(row, ac_cols, answers)` | `{acN_given, acN_correct}` |
| `_extract_demographics(row, demo_cols)` | pass-through dict, PII columns already excluded |
| `_is_pii_column(col_name, values)` | True if header or values look like name/email/phone/SSN |

### PII safety net

`demographic` columns are the only ones whose raw values pass through to output.
Before including one, `_is_pii_column` checks:
- header matches (case-insensitive) any of: `name`, `email`, `e-mail`, `phone`,
  `ssn`, `social security`, `address`;
- OR sampled values match an email regex.

If flagged, the column is dropped from `demographics` and a `warnings.warn`
fires (observable, testable via `pytest.warns`). The `respondent_id` column is
kept regardless (it is the anonymous token) but is checked and warned if its
values look like emails.

## Error handling

`_validate_mapping` raises `ValueError` when:
- a column named in `mapping["columns"]` is not in `raw_df`;
- no column has role `respondent_id`, or more than one does;
- no column has role `question`;
- `mapping["scale"]` is missing, malformed, or `scale_min >= scale_max`;
- a column tagged `attention_check` has no entry in `attention_check_answers`.

Malformed cell values within a valid mapping (e.g. a non-numeric answer in a
question cell) surface as the natural `int()`/parse error — the normalizer does
not silently coerce bad data.

## Testing

Hand-crafted small DataFrames built inline with pandas.

Unit tests (per helper):
- `_compute_duration`: normal delta; `None` when start or end is `NaT`/missing.
- `_extract_responses`: raw ints, keyed `q1..qN` in column order.
- `_extract_attention_checks`: `given` from row, `correct` from answers; `{}` when none.
- `_is_pii_column`: flags an `Email Address` header and email-valued column;
  does not flag `grade level`.
- `_validate_mapping`: each `ValueError` trigger above.

Orchestrator tests:
- A 2–3 row DataFrame produces correctly-shaped respondent dicts.
- Missing `end_time` mapping → `duration_seconds is None`.
- A `demographic` column of emails is dropped + warns.
- `respondent_id` preserved; `scale_min`/`scale_max` set.

Payoff integration test (ties Phases 1→3→2 together):
- Use the Phase 1 generator to write a small survey to a `tmp_path`.
- Build a mapping programmatically from its column names (name contains `[Q` →
  `question`, `[AC` → `attention_check`, etc.).
- `apply_mapping` → `extract_features`.
- Assert the features separate archetypes (e.g. a straightliner row has
  `straightlining_score == 1.0`; a speeder has `completion_time_ratio < 0.5`),
  cross-referencing the generator's labels sidecar.

## Out of scope

- Reading files from disk / file upload (that's the API, Phase 5). `apply_mapping`
  takes an already-loaded DataFrame.
- The interactive column-tagging UI (Phase 6).
- Persisting/reloading a school's mapping (deferred; Phase 5/6).
- Any feature computation or scoring (Phases 2 and 4).
