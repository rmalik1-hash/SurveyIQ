# SurveyIQ — Roadmap & Backlog

Living document. Tracks build order, parallel risk-retirement work, and deferred
(v2) scope. The authoritative per-component design lives in
`docs/superpowers/specs/`; this file is the higher-level "what next and why."

## Guiding principle: value scales with data, it isn't all-or-nothing

SurveyIQ analyzes the survey data a school **already has**. Detection quality
scales with how rich that data is — it does not collapse when signals are
missing. This is a deliberate design commitment (see Graceful degradation).

## Build order (v1)

| Phase | Component | Status |
|---|---|---|
| 1 | Synthetic data generator (`data/synthetic/`) | Done |
| 2 | Feature extractor (`src/features/`) | Done |
| 3 | Normalizer + column mapper (`src/ingestion/`) | Done |
| 4 | Decision-tree trainer + scorer (`src/models/`) | Done — gated (see G1) |
| 5 | FastAPI backend (`src/api/`) | Done |
| 6 | React dashboard (`frontend/`) | Done |

**v1 is complete.** 129 Python tests + 22 frontend tests. An administrator can
upload a CSV, map columns once, see the quality score and every flag explained
in plain English, and download a marked or cleaned dataset — verified end to end
in a browser against the live API.

The mockup (`mockups/dashboard-mockup.html`) was the visual target and is now
superseded by the real dashboard; keep it only as a reference/talking piece.

## What matters most now

The build is done; the open work is **trust**, not features. G1 below still
stands: nothing about the model's accuracy is a real-world claim until R2 and
ideally R1 are done. R1 is the only item the team (not code) must drive, and it
has a long approval lead time — start it first.

## Graceful degradation (design commitment)

Of the 6 features, **3 need only the answer grid** and work on ANY survey:
`straightlining_score`, `response_variance`, `extreme_response_rate`. The other
3 need extra data (`attention_check_pass_rate` needs check items,
`completion_time_ratio` needs start+end timestamps, `contradiction_score` needs
reverse-coded pairs). The model must handle any subset of features being
present — missing signals reduce power, they do not break the product.

## Risk-retirement tracks (run in parallel with the build)

These address the project's two deepest risks. They are NOT blockers for
Phases 2–3, but they gate Phase 4 (see G1).

### R1. Secure a real validation survey  ← start now, long lead time
Get one real IMSA survey (with administrative approval) to validate against.
- We only need a **small** set (~100–200 responses), used for validation, not
  training. Establish ground truth via included attention checks / bogus items
  or a human-reviewed sample.
- This is a team/administrative action, not a code task. Highest-leverage thing
  we can do — start the approval process immediately because it is slow.

### R2. Calibrate the generator to published research  ← checkpoint before Phase 4
Tune generator parameters (careless prevalence ~5–15%, speeding thresholds,
genuine-respondent variance) to empirical distributions from the careless- /
insufficient-effort-responding literature.
- Changes the story from "we invented this data" to "we modeled it on research."
- Mostly reading + parameter-tuning; no new architecture.

### R3. Cross-check against established detectors
On real (even unlabeled) data, compare our model's flags against standard
statistical measures (LongString, inter-item variance, Mahalanobis distance).
Agreement is convergent evidence we detect something real.

## Gates

### G1. Do not trust or report model accuracy until…
Before presenting any Phase 4 accuracy numbers as meaningful: R2 (generator
calibrated to research) must be done, and ideally R1 (validated against real
data) as well. Synthetic-only accuracy is for debugging the pipeline, not for
claims about real-world performance.

## v2 backlog (deferred — from mentor meeting 2026-07-02)

Recorded so they aren't lost. NOT built in v1; revisit after the Phase 1–6
pipeline works end to end. Each carries a confidentiality constraint.

### B1. Email delivery of results
Consumer uploads a CSV and receives results by email.
- **Privacy-safe design:** email a *link* to a report, or an *aggregate*
  summary only — never raw student responses in the email body/attachment.
- Depends on: Phase 5 (API) + a report-hosting/link mechanism.

### B2. Compare to previous results
Let an administrator compare a new upload against past uploads (trend over time).
- **Privacy-safe design:** persist only *anonymized, aggregate* results
  (e.g. batch date + quality score + flag-type counts) — never row-level
  responses.
- Depends on: some persistence layer + a batch/run identity concept.

### B3. Public, consumer-usable website
A hosted site (beyond the local dev dashboard) an administrator can actually use.
- Depends on: Phases 4–6, plus hosting + confidentiality review.

### B4. SurveyIQ-optimized survey template + timing guide  (NOT a survey platform)
An optional add-on that maximizes data quality WITHOUT us collecting data:
a ready-made question set with attention checks and reverse-coded pairs baked
in, plus a one-page guide to enabling response-timing in Google Forms /
Qualtrics. Schools import it into *their existing* tool.
- **Rationale:** captures ~80% of the "ideal data" benefit for ~5% of the
  effort, and reinforces the core value prop ("works on the data you already
  have"). A full survey *platform* is explicitly rejected — it is a bigger
  product than SurveyIQ, would swallow the team's capacity, and would break the
  no-reformatting adoption advantage.

## Confidentiality constraints (reinforced by mentors 2026-07-02)

Carried from CLAUDE.md "Hard constraints", elevated here because the v2 asks
increase data-handling risk:

- **No identifiable student data** ever persists. The normalizer strips/ignores
  any column resembling a name, email, or non-anonymous ID.
- **Email is not a secure channel** — never send raw responses over it (see B1).
- **Comparison requires storage** — store only anonymized aggregates (see B2).
- Any feature that stores or transmits data must have its privacy design
  reviewed before implementation.
