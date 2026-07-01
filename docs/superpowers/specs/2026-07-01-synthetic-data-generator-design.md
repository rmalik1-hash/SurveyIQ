# SurveyIQ — Design Spec: Synthetic Data Generator (Phase 1)

## Context

SurveyIQ detects careless survey responding (straightlining, speeding, random
answering, logical contradictions) and scores respondent reliability using a
decision tree, per `CLAUDE.md`. Every downstream phase (feature extraction,
normalization, model training, API, dashboard) depends on having realistic
survey data with known ground truth. Real IMSA survey data is not available
yet (requires administrative approval), so the synthetic generator is the
first thing to build.

A sample file was dropped into the project root
(`School Start Time Survey (Responses).xlsx`) — a 73-row survey of Florida
school districts about HB 733 start-time compliance. It does not match the
per-respondent Likert/attention-check shape this system is designed around
(it's district-level, Yes/No/free-text, single timestamp, contains real
names/emails). Decision: **it is used only as a reference for what a messy,
real-world school export looks like** (inconsistent headers, stray PII
columns, mixed types) when designing the CSV output style below. It does not
change the feature/model design, and it is not itself ingested by the
pipeline in this phase.

## Overall Roadmap (confirmed, from CLAUDE.md)

1. **Synthetic data generator** — this spec.
2. **Feature extractor** (`src/features/`) — six behavioral features per
   respondent.
3. **Normalizer + column mapper** (`src/ingestion/`) — raw CSV + user mapping
   → internal schema.
4. **Decision tree trainer/scorer** (`src/models/`) — binary reliable/flagged
   classifier with human-readable rule paths.
5. **FastAPI backend** (`src/api/`) — upload → map → normalize → extract →
   score → return JSON.
6. **React dashboard** (`frontend/`) — last, only once the pipeline works
   end-to-end via the API.

Each phase gets its own spec → plan → build cycle. This document covers
Phase 1 only.

## Phase 1: Synthetic Data Generator

### Location

`data/synthetic/generator.py`, with supporting modules alongside it in
`data/synthetic/`.

### Public interface

```python
generate_survey_csv(
    n_respondents=200,
    n_questions=20,
    scale=(1, 5),
    contamination_rate=0.25,   # fraction of careless respondents
    seed=42,
    output_path="data/synthetic/survey_001.csv",
)
```

This matches the interface already specified in `CLAUDE.md` — no changes.

### Outputs

Calling `generate_survey_csv` writes **three** files (the CSV plus two
sidecars used only for development/evaluation, never fed into the pipeline
as features):

- `survey_001.csv` — the messy raw export (see "CSV style" below). Contains
  no ground-truth label column.
- `survey_001_labels.csv` — columns `respondent_id, is_careless, archetype`.
  `archetype` is one of `reliable`, `straightliner`, `speeder`,
  `random_responder`, `contradictor`.
- `survey_001_pairs.json` — the list of reverse-coded question-index pairs
  used for contradiction scoring (see "Contradiction pairs" below), so the
  feature extractor (Phase 2) can consume the same source of truth the
  generator used to create contradictions.

### Archetype assignment

Careless respondents (`contamination_rate × n_respondents`, rounded) are
split **evenly** across the four archetypes (straightliner, speeder,
random_responder, contradictor) — roughly 1/4 each, with any remainder
distributed deterministically (e.g. first archetypes in iteration order get
the extra respondent). The split ratio is itself a parameter with this even
split as the default, so it can be skewed later for testing specific
detectors without changing the generator's core logic. All other respondents
are `reliable`.

### Internal components

- **`archetypes.py`** — one simulation function per behavior:
  `simulate_reliable`, `simulate_straightliner`, `simulate_speeder`,
  `simulate_random_responder`, `simulate_contradictor`. Each takes
  `(n_questions, scale, contradiction_pairs, rng)` and returns raw Likert
  answers plus start/end timestamps.
    - *Reliable*: varied answers within a plausible range, reasonable
      per-question timing (~8s avg), passes attention checks, respects
      contradiction pairs (answers are logically consistent).
    - *Straightliner*: same value for (nearly) every question.
    - *Speeder*: total duration far below `n_questions × 8s` (implausibly
      fast), answers otherwise unremarkable.
    - *Random responder*: high-variance answers with no logical structure,
      fails attention checks at chance rate.
    - *Contradictor*: normal-looking answers except the designated
      reverse-coded pairs are answered inconsistently (both high, or both
      low, when one should be the inverse of the other).
- **`contradiction_pairs.py`** — for a given `n_questions`, designates
  roughly 2 reverse-coded pairs per 20 questions (e.g. Q3 "I enjoy school" /
  Q14 "I dislike school"). Returns a list of `(index_a, index_b)` pairs.
  This is written out as `survey_001_pairs.json` and is the single source of
  truth both the contradictor archetype and the Phase 2 feature extractor
  use.
- **`template.py`** — renders the single Google-Forms-style messy CSV
  format: verbose auto-generated headers (e.g. "On a scale of 1-5, how much
  do you enjoy school? [Q3]"), a stray `Timestamp` column and an
  `Email Address` column (synthetic, non-identifying), inconsistent header
  capitalization. Takes clean internal rows and renders the messy
  `DataFrame`. This is the intended swap point if additional export styles
  (Qualtrics, SurveyMonkey) are added in a later phase — not built now.
- **`generator.py`** — orchestrates the full flow (see Data Flow below).

### Attention checks and demographics

By default, the generator includes 1-2 attention-check items interspersed
among the questions (e.g. "Please select Strongly Agree") and 1-2
demographic columns (grade level, school name) passed through untouched —
enough to exercise `attention_check_pass_rate` without bloating the survey.
Both are configurable but present by default so Phase 2's feature extractor
has real signal to test against.

### Data flow

```
generate_survey_csv(params)
  → validate params (raise ValueError on invalid input, no silent clamping)
  → assign archetype per respondent (even split across careless types)
  → for each respondent: simulate_<archetype>() → raw answers + timestamps
  → inject attention checks + demographic columns
  → render via template.py → messy DataFrame
  → write survey_NNN.csv (messy, no ground truth)
  → write survey_NNN_labels.csv (respondent_id, is_careless, archetype)
  → write survey_NNN_pairs.json (question pair indices)
```

### Error handling

This is a developer/testing tool, not user-facing. Invalid parameters
(`contamination_rate` outside `[0, 1]`, `n_questions` too small to fit the
requested contradiction pairs, `n_respondents <= 0`) raise `ValueError`
immediately. Silent clamping is explicitly avoided because it would produce
misleading ground truth that downstream accuracy measurements would trust.

### Testing

Unit tests with small, fixed seeds asserting:

- Straightliner rows have identical (or near-identical) answers across all
  questions.
- Speeder rows have `duration_seconds` far below the expected reading time.
- Contradictor rows violate their designated pairs (both high or both low
  where the pair is reverse-coded).
- `survey_NNN_labels.csv` row count matches `survey_NNN.csv` row count, and
  `respondent_id` values line up 1:1.
- `contamination_rate` is honored within rounding for a range of
  `n_respondents`.

## Out of scope for this phase

- Multiple export templates (Qualtrics, SurveyMonkey styles) — deferred,
  `template.py` is structured to make this a later addition.
- Feature extraction, normalization, modeling, API, frontend — later phases,
  each with their own spec.
- Ingesting the real district survey sample — reference only, not consumed
  by code in this phase.
