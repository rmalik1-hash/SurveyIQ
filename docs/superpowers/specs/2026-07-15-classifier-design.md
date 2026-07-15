# Phase 4 — Decision-Tree Classifier Design

## What this is

The classifier is the fourth stage of the SurveyIQ pipeline and the product's
core. It trains an interpretable decision tree on the per-respondent feature
table, then scores each respondent's reliability and — for every flagged one —
states in plain English the exact rule that flagged them.

```
feature DataFrame + labels  →  [train]  →  model
model + feature DataFrame    →  [predict]  →  scored DataFrame (id, reliability_score, flag_reason)
```

## Decisions locked in (from brainstorming)

1. **Native NaN handling.** scikit-learn decision trees handle missing values
   natively (since v1.3; environment has 1.8). The `completion_time_ratio`
   feature is `NaN` when a survey lacks timestamps; the tree learns which way to
   send "missing." No imputation, no fabricated values. `requirements.txt`
   gains `scikit-learn>=1.3`.
2. **Full decision-path reasons.** Each flagged respondent's `flag_reason` is
   the plain-English translation of the actual root-to-leaf rule the tree used,
   a conjunction of clauses — not a single-feature summary.
3. **Shallow tree for readability.** `max_depth=4` by default so rules stay
   short and auditable (CLAUDE.md hard constraint: decision trees only, every
   flag explainable).

## Interface

```python
train(feature_df, labels, max_depth=4, random_state=42) -> DecisionTreeClassifier
predict(model, feature_df) -> pd.DataFrame
```

- `feature_df`: exactly the Phase 2 extractor's output — a `respondent_id`
  column plus the six feature columns. `train` drops `respondent_id` and fits on
  the six features (`FEATURE_NAMES`, imported/derived from
  `extract.FEATURE_COLUMNS` minus `respondent_id`).
- `labels`: array-like aligned to `feature_df` rows; `0` = reliable, `1` =
  flagged. Sourced from the generator's `is_careless` (bool → int) during
  development.
- `train` returns a fitted `DecisionTreeClassifier(max_depth=max_depth,
  random_state=random_state)`.

### `predict` output

A DataFrame, one row per respondent, in input order:

| Column | Type | Meaning |
|---|---|---|
| `respondent_id` | str | passthrough |
| `reliability_score` | float 0–1 | `P(reliable)` = `predict_proba` column for class `0` |
| `flag_reason` | str | plain-English rule for flagged rows; `""` for reliable rows |

A respondent is flagged when the model predicts class `1` (equivalently
`reliability_score < 0.5`). Reliable rows get an empty `flag_reason`.

## Architecture

- **File:** `src/models/classifier.py`
- **Package:** `src/models/__init__.py`

### Functions

| Function | Responsibility |
|---|---|
| `train(feature_df, labels, max_depth, random_state)` | drop id, fit tree |
| `predict(model, feature_df)` | score + build reason column |
| `_feature_matrix(feature_df)` | return the 6-feature matrix in `FEATURE_NAMES` order |
| `_describe_path(model, x_row)` | translate one sample's decision path to a sentence |
| `_clause(feature_name, threshold, went_left)` | one human clause for a split |

### `FEATURE_NAMES`

The six model inputs, in fixed order (derived from `extract.FEATURE_COLUMNS`
excluding `respondent_id`):
`completion_time_ratio, straightlining_score, response_variance,
contradiction_score, attention_check_pass_rate, extreme_response_rate`.

### Reason generation

For a flagged sample, walk the tree's decision path (`model.decision_path` /
`model.tree_`). For each internal node on the path, emit one clause from
`_clause`, using a per-feature template keyed on direction:

| Feature | went_left (`value <= threshold`) | went_right (`value > threshold`) |
|---|---|---|
| `completion_time_ratio` | "answered very fast" | "took adequate time" |
| `straightlining_score` | "varied their answers" | "gave the same answer repeatedly" |
| `response_variance` | "answers had little spread" | "answers were highly erratic" |
| `contradiction_score` | "answers were consistent" | "contradicted on paired questions" |
| `attention_check_pass_rate` | "failed attention checks" | "passed attention checks" |
| `extreme_response_rate` | "used the scale moderately" | "overused the scale endpoints" |

Each clause appends the auditable numeric condition, e.g.
`"answered very fast (completion_time_ratio ≤ 0.30)"`. Clauses join with
" and "; the sentence ends with a period. Missing-value branches (sklearn sends
NaN one way) are described as `"had no timing data"` for the timing feature and
generically otherwise. Reliable rows return `""`.

## Error handling

- `train` raises `ValueError` if `feature_df` is empty or `labels` length does
  not match row count.
- `predict` raises `ValueError` if `feature_df` is missing any `FEATURE_NAMES`
  column.
- Single-class training data (all reliable or all flagged) is allowed — sklearn
  fits a degenerate tree; `predict` still returns valid columns (reasons empty
  when nothing is flagged).

## Testing

- `train` returns a fitted `DecisionTreeClassifier` with `max_depth <= 4`.
- `predict` output has the three columns, correct row count/order,
  `reliability_score` within [0, 1].
- **NaN handling:** a feature_df with `NaN` in `completion_time_ratio` trains
  and predicts with no error and valid scores.
- **Reason helper:** on a trained model, a clearly-careless sample yields a
  non-empty `flag_reason` containing an expected phrase; a clearly-reliable
  sample yields `""`.
- **Label/shape guards:** mismatched `labels` length and empty `feature_df`
  raise `ValueError`; missing feature column in `predict` raises.
- **End-to-end (Phases 1→3→2→4):** generate a labelled synthetic survey →
  normalize → extract → `train` on part → `predict`, and assert the model
  recovers careless respondents with high accuracy (e.g. ≥ 0.9 on held-out
  synthetic rows) and that a straightliner's `flag_reason` mentions the same
  answer / straightlining. **This accuracy is a structural check that the
  pipeline learns — not a real-world performance claim (G1 gate: no accuracy
  claims until the generator is calibrated to research / validated on real
  data).**

## Out of scope

- Model persistence (save/load) — deferred to the API phase (Phase 5).
- Hyperparameter tuning / cross-validation — the model runs on synthetic data
  under the G1 gate; tuning waits for calibrated/real data.
- Ensembles, boosting, neural nets — explicitly forbidden by CLAUDE.md v1.
- Any UI — Phase 6.
