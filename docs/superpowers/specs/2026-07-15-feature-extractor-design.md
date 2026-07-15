# Phase 2 — Feature Extractor Design

## What this is

The feature extractor is the second stage of the SurveyIQ pipeline. It takes
normalized respondent records and computes six behavioral signals per
respondent — the numeric features the Phase 4 decision-tree model will use to
flag careless responding. It does **not** classify; it only measures.

```
normalized respondents  →  [FEATURE EXTRACTOR]  →  feature DataFrame (1 row/respondent)
```

## Input contract (Option A — clean structured data)

The extractor consumes a list of respondent dicts in CLAUDE.md's internal
schema. It does **not** read raw CSVs — converting messy exports into this
schema is Phase 3's job (the normalizer). Testing uses hand-crafted schema
dicts where the correct feature values are obvious.

```python
{
    "respondent_id": str,
    "duration_seconds": int | None,   # None if timestamps were absent
    "responses": {"q1": int, "q2": int, ...},  # raw Likert ints
    "attention_checks": {              # may be empty
        "ac1_given": int, "ac1_correct": int,
        "ac2_given": int, "ac2_correct": int,
    },
    "scale_min": int,                  # e.g. 1
    "scale_max": int,                  # e.g. 5
    # start_time / end_time / demographics may be present; not used here
}
```

Plus an optional argument to the orchestrator:

```python
contradiction_pairs: list[tuple[str, str]] | None  # e.g. [("q1","q2"), ("q3","q4")]
```

If `None` or empty, `contradiction_score` is `0.0` for every respondent (per
CLAUDE.md's "0 if none defined").

## Output contract

A pandas DataFrame, one row per respondent, in input order, with columns:

| Column | Type | Range |
|---|---|---|
| `respondent_id` | str | — |
| `completion_time_ratio` | float or NaN | ≥ 0 |
| `straightlining_score` | float | 0–1 |
| `response_variance` | float | 0–1 |
| `contradiction_score` | float | 0–1 |
| `attention_check_pass_rate` | float | 0–1 |
| `extreme_response_rate` | float | 0–1 |

Missing/uncomputable features are represented as `np.nan` (the pandas-native
form of CLAUDE.md's "None"). Only `completion_time_ratio` can be `NaN` (when
`duration_seconds is None`); the other five always compute from the answer grid
or default sensibly.

## Architecture

One small pure function per feature plus a thin orchestrator — mirroring the
Phase 1 archetypes structure (independently testable units). All live in a
single focused module.

- **File:** `src/features/extract.py`
- **Packages:** `src/__init__.py`, `src/features/__init__.py`
- **Constant:** `AVG_SECONDS_PER_QUESTION = 8` (from CLAUDE.md)

### Feature functions

Each takes the minimal data it needs and returns a float (or `np.nan`).

| Function | Signature | Rule |
|---|---|---|
| `completion_time_ratio` | `(duration_seconds, num_questions)` | `duration / (num_questions * 8)`; `np.nan` if `duration is None` |
| `straightlining_score` | `(responses)` | `count(modal value) / num_questions` |
| `response_variance` | `(responses, scale_min, scale_max)` | population `std` of answers normalized to 0–1 via `(v - min)/(max - min)` |
| `contradiction_score` | `(responses, scale_min, scale_max, pairs, tolerance=1)` | fraction of pairs that contradict; `0.0` if no pairs |
| `attention_check_pass_rate` | `(attention_checks)` | `correct / total`; `1.0` if no checks |
| `extreme_response_rate` | `(responses, scale_min, scale_max)` | `count(v == min or v == max) / num_questions` |

**Contradiction rule (detail):** a pair `(a, b)` is reverse-coded, so a
consistent respondent satisfies `answer_a + answer_b == scale_min + scale_max`.
The expected consistent value of `b` is `mirror(a) = scale_min + scale_max -
answer_a`. A pair counts as a contradiction when
`abs(answer_b - mirror(answer_a)) > tolerance`. Default `tolerance = 1` scale
point — exact for our synthetic (perfectly mirrored) data, forgiving for noisy
real data later. `contradiction_score = contradicting_pairs / total_pairs`.

**Normalization note:** the schema stores raw Likert ints plus
`scale_min`/`scale_max`. Features that need a 0–1 view (`response_variance`)
normalize internally; features that key on raw endpoints
(`extreme_response_rate`) use the raw ints. Responses are not mutated.

### Orchestrator

```python
def extract_features(
    respondents: list[dict],
    contradiction_pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    ...
```

Iterates respondents, calls each feature function, assembles one dict per row,
returns `pd.DataFrame(rows)` with a stable column order (respondent_id first).

## Edge cases (handled explicitly)

- `duration_seconds is None` → `completion_time_ratio = np.nan`.
- Empty `attention_checks` → `attention_check_pass_rate = 1.0`.
- `contradiction_pairs` `None`/empty → `contradiction_score = 0.0`.
- `scale_max == scale_min` → guard against divide-by-zero in normalization;
  treat variance/extreme as `0.0` (degenerate single-value scale).
- Empty `responses` → raise `ValueError` (a respondent with no answers is a
  normalizer bug, not something to silently score).
- A contradiction pair naming a question key not in `responses` → raise
  `ValueError` (caller passed pairs that don't match the data).

## Testing (TDD, hand-crafted inputs)

Per CLAUDE.md: small inputs where the expected output is obvious. Each feature
function gets direct unit tests, then the orchestrator gets an integration test.

Representative cases:
- Answered `3` to every question → `straightlining_score = 1.0`,
  `response_variance = 0.0`.
- Alternating `1` and `5` on a 1–5 scale → `extreme_response_rate = 1.0`,
  high variance.
- `duration_seconds = 20` over 20 questions → `completion_time_ratio ≈ 0.125`
  (well under 0.5); `None` → `np.nan`.
- Passed 1 of 2 attention checks → `0.5`; no checks → `1.0`.
- Pair perfectly mirrored (`q1=1, q2=5` on 1–5) → contradiction `0.0`;
  matched pair (`q1=1, q2=1`) → contradiction `1.0`; no pairs → `0.0`.
- Orchestrator: a 2–3 respondent list → DataFrame with the right shape, column
  order, respondent order, and one `NaN` where a respondent lacks timestamps.
- Degenerate scale (`scale_min == scale_max`) does not raise.
- Empty responses / mismatched pair key → raises `ValueError`.

Generator-CSV integration is deferred to Phase 3 (that stage builds the schema
from raw exports). Phase 2 is validated entirely on hand-crafted schema inputs.

## Out of scope

- Reading CSVs or the messy generator output (Phase 3).
- Any classification, scoring thresholds, or flag reasons (Phase 4).
- Feature scaling/selection for the model (Phase 4 concern).
