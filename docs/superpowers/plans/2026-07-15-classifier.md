# Decision-Tree Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 4 decision-tree classifier: `train` fits an interpretable tree on the feature table; `predict` returns a per-respondent reliability score plus a plain-English reason for each flag.

**Architecture:** `train`/`predict` plus helpers in `src/models/classifier.py`. Uses `sklearn.tree.DecisionTreeClassifier` with native NaN handling (verified in sklearn 1.8). Reasons come from translating the tree's actual decision path (via `model.decision_path`) into templated clauses. Built with TDD, capped by an end-to-end Phases 1→3→2→4 test.

**Tech Stack:** Python 3, numpy, pandas, scikit-learn (>=1.3), pytest.

**Environment notes (Windows):** Repo root is `C:\Users\rayan\OneDrive\Documents`; project is `SurveyIQ`; branch is `surveyiq-classifier`. The repo has thousands of unrelated files — NEVER `git add -A`/`git add .`. Stage only the files each task names, by explicit path. Run tests with `python -m pytest` from the `SurveyIQ` directory. `conftest.py` puts the project root on `sys.path`. scikit-learn is already installed (1.8.0).

**Verified facts (do not second-guess):** `DecisionTreeClassifier` in this environment accepts `NaN` in `fit`/`predict`/`predict_proba`/`decision_path` without error. `model.classes_` is `[0 1]` when both labels present; `predict_proba` column `0` is `P(reliable)`. `model.tree_.feature[node] == -2` marks a leaf. Direction at a node is "went left" iff the next node on the path equals `tree_.children_left[node]`.

---

### Task 0: Scaffolding + dependency

**Files:**
- Create: `src/models/__init__.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Create the package marker** (`src/models/__init__.py`, empty).

- [ ] **Step 2: Add scikit-learn to `requirements.txt`.** The file currently contains `pandas`, `numpy`, `pytest`. Add a line so it reads:

```
pandas
numpy
pytest
scikit-learn>=1.3
```

- [ ] **Step 3: Commit**

```bash
git add SurveyIQ/src/models/__init__.py SurveyIQ/requirements.txt
git commit -m "chore: scaffold src/models package and add scikit-learn dependency"
```

---

### Task 1: FEATURE_NAMES, _feature_matrix, train (creates the module)

**Files:**
- Create: `src/models/classifier.py`
- Test: `tests/test_classifier.py`

- [ ] **Step 1: Write the failing test** — create `tests/test_classifier.py`:

```python
import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier
from src.models.classifier import train, FEATURE_NAMES, _feature_matrix


def _feature_df(n=6):
    data = {
        "respondent_id": [f"R{i}" for i in range(n)],
        "completion_time_ratio": [1.0, 1.1, 0.1, 1.2, 0.12, 1.0][:n],
        "straightlining_score": [0.3, 0.25, 1.0, 0.2, 0.9, 0.3][:n],
        "response_variance": [0.4, 0.45, 0.0, 0.5, 0.05, 0.4][:n],
        "contradiction_score": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0][:n],
        "attention_check_pass_rate": [1.0, 1.0, 1.0, 1.0, 0.0, 1.0][:n],
        "extreme_response_rate": [0.3, 0.3, 0.2, 0.3, 0.9, 0.2][:n],
    }
    return pd.DataFrame(data)


def test_feature_names_excludes_respondent_id():
    assert "respondent_id" not in FEATURE_NAMES
    assert len(FEATURE_NAMES) == 6


def test_feature_matrix_shape():
    assert _feature_matrix(_feature_df()).shape == (6, 6)


def test_feature_matrix_missing_column_raises():
    df = _feature_df().drop(columns=["response_variance"])
    with pytest.raises(ValueError):
        _feature_matrix(df)


def test_train_returns_fitted_tree_with_depth_cap():
    model = train(_feature_df(), [0, 0, 1, 0, 1, 0], max_depth=4)
    assert isinstance(model, DecisionTreeClassifier)
    assert model.get_depth() <= 4
    assert hasattr(model, "tree_")


def test_train_empty_raises():
    with pytest.raises(ValueError):
        train(_feature_df(0), [])


def test_train_label_mismatch_raises():
    with pytest.raises(ValueError):
        train(_feature_df(), [0, 1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_classifier.py -v`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Write minimal implementation** — create `src/models/classifier.py`:

```python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.features.extract import FEATURE_COLUMNS

FEATURE_NAMES = [c for c in FEATURE_COLUMNS if c != "respondent_id"]


def _feature_matrix(feature_df):
    missing = [c for c in FEATURE_NAMES if c not in feature_df.columns]
    if missing:
        raise ValueError(f"feature_df missing columns: {missing}")
    return feature_df[FEATURE_NAMES].to_numpy(dtype=float)


def train(feature_df, labels, max_depth=4, random_state=42):
    if len(feature_df) == 0:
        raise ValueError("feature_df is empty")
    if len(labels) != len(feature_df):
        raise ValueError("labels length does not match feature_df row count")
    X = _feature_matrix(feature_df)
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(X, list(labels))
    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_classifier.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add SurveyIQ/src/models/classifier.py SurveyIQ/tests/test_classifier.py
git commit -m "feat: add classifier train and feature matrix"
```

---

### Task 2: _clause and _describe_path (reason generator)

**Files:**
- Modify: `src/models/classifier.py`
- Test: `tests/test_classifier.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_classifier.py`:

```python
from src.models.classifier import _clause, _describe_path


def test_clause_low_and_high_phrases():
    low = _clause("straightlining_score", 0.8, went_left=True, is_nan=False)
    high = _clause("straightlining_score", 0.8, went_left=False, is_nan=False)
    assert "varied" in low
    assert "same answer" in high
    assert "0.80" in high


def test_clause_nan_timing():
    c = _clause("completion_time_ratio", 0.3, went_left=True, is_nan=True)
    assert "no timing data" in c


def test_describe_path_flagged_is_capitalized_sentence():
    model = train(_feature_df(), [0, 0, 1, 0, 1, 0])
    X = _feature_matrix(_feature_df())
    reason = _describe_path(model, X[2])
    assert isinstance(reason, str)
    assert reason.endswith(".")
    assert reason[0].isupper()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_classifier.py -k "clause or describe_path" -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation** — append to `src/models/classifier.py`:

```python
_CLAUSE_TEMPLATES = {
    "completion_time_ratio": ("answered very fast", "took adequate time"),
    "straightlining_score": ("varied their answers", "gave the same answer repeatedly"),
    "response_variance": ("answers had little spread", "answers were highly erratic"),
    "contradiction_score": ("answers were consistent", "contradicted on paired questions"),
    "attention_check_pass_rate": ("failed attention checks", "passed attention checks"),
    "extreme_response_rate": ("used the scale moderately", "overused the scale endpoints"),
}


def _clause(feature_name, threshold, went_left, is_nan):
    if is_nan and feature_name == "completion_time_ratio":
        return "had no timing data"
    low_phrase, high_phrase = _CLAUSE_TEMPLATES[feature_name]
    phrase = low_phrase if went_left else high_phrase
    op = "≤" if went_left else ">"
    return f"{phrase} ({feature_name} {op} {threshold:.2f})"


def _describe_path(model, x_row):
    x_row = np.asarray(x_row, dtype=float)
    dp = model.decision_path(x_row.reshape(1, -1))
    node_ids = dp.indices[dp.indptr[0]:dp.indptr[1]]
    tree = model.tree_
    clauses = []
    for i in range(len(node_ids) - 1):
        node = node_ids[i]
        feature_idx = tree.feature[node]
        if feature_idx == -2:
            continue
        went_left = node_ids[i + 1] == tree.children_left[node]
        value = x_row[feature_idx]
        clauses.append(_clause(
            FEATURE_NAMES[feature_idx], tree.threshold[node], went_left, np.isnan(value)
        ))
    if not clauses:
        return ""
    sentence = " and ".join(clauses)
    return sentence[0].upper() + sentence[1:] + "."
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_classifier.py -k "clause or describe_path" -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add SurveyIQ/src/models/classifier.py SurveyIQ/tests/test_classifier.py
git commit -m "feat: add plain-English decision-path reason generator"
```

---

### Task 3: predict

**Files:**
- Modify: `src/models/classifier.py`
- Test: `tests/test_classifier.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_classifier.py`:

```python
from src.models.classifier import predict


def test_predict_columns_order_and_score_range():
    df = _feature_df()
    model = train(df, [0, 0, 1, 0, 1, 0])
    out = predict(model, df)
    assert list(out.columns) == ["respondent_id", "reliability_score", "flag_reason"]
    assert len(out) == 6
    assert list(out["respondent_id"]) == list(df["respondent_id"])
    assert ((out["reliability_score"] >= 0) & (out["reliability_score"] <= 1)).all()


def test_predict_flagged_have_reason_reliable_empty():
    df = _feature_df()
    model = train(df, [0, 0, 1, 0, 1, 0])
    out = predict(model, df)
    for _, r in out.iterrows():
        if r["reliability_score"] < 0.5:
            assert r["flag_reason"] != ""
        else:
            assert r["flag_reason"] == ""


def test_predict_handles_nan_feature():
    df = _feature_df()
    df.loc[0, "completion_time_ratio"] = np.nan
    model = train(df, [0, 0, 1, 0, 1, 0])
    out = predict(model, df)
    assert len(out) == 6
    assert out["reliability_score"].notna().all()


def test_predict_missing_feature_column_raises():
    df = _feature_df()
    model = train(df, [0, 0, 1, 0, 1, 0])
    with pytest.raises(ValueError):
        predict(model, df.drop(columns=["contradiction_score"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_classifier.py -k predict -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation** — append to `src/models/classifier.py`:

```python
def predict(model, feature_df):
    X = _feature_matrix(feature_df)
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 0 in classes:
        reliability = proba[:, classes.index(0)]
    else:
        reliability = np.zeros(len(X))
    predictions = model.predict(X)
    ids = feature_df["respondent_id"].tolist()
    rows = []
    for i, rid in enumerate(ids):
        flagged = predictions[i] == 1
        reason = _describe_path(model, X[i]) if flagged else ""
        rows.append({
            "respondent_id": rid,
            "reliability_score": float(reliability[i]),
            "flag_reason": reason,
        })
    return pd.DataFrame(rows, columns=["respondent_id", "reliability_score", "flag_reason"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_classifier.py -k predict -v`
Expected: 4 passed.

- [ ] **Step 5: Run the classifier file fully**

Run: `python -m pytest tests/test_classifier.py -v`
Expected: 13 passed.

- [ ] **Step 6: Commit**

```bash
git add SurveyIQ/src/models/classifier.py SurveyIQ/tests/test_classifier.py
git commit -m "feat: add predict with reliability score and flag reasons"
```

---

### Task 4: End-to-end pipeline test (generator -> ... -> model)

**Files:**
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write the test** — append to `tests/test_pipeline.py` (it already imports `json`, `pandas as pd`, `generate_survey_csv`, `apply_mapping`, `extract_features`, and defines `_build_mapping`):

```python
def test_full_pipeline_model_recovers_careless(tmp_path):
    from src.models.classifier import train, predict, FEATURE_NAMES

    out = tmp_path / "s.csv"
    generate_survey_csv(
        n_respondents=200, n_questions=12, scale=(1, 5),
        contamination_rate=0.3, seed=11, output_path=str(out),
    )
    df = pd.read_csv(out)
    labels_df = pd.read_csv(tmp_path / "s_labels.csv")
    pairs_idx = json.load(open(tmp_path / "s_pairs.json"))["pairs"]

    respondents = apply_mapping(df, _build_mapping(list(df.columns)))
    pair_keys = [(f"q{a + 1}", f"q{b + 1}") for a, b in pairs_idx]
    features = extract_features(respondents, contradiction_pairs=pair_keys)

    merged = features.merge(labels_df, on="respondent_id")
    y = merged["is_careless"].astype(int).tolist()
    feat = merged[["respondent_id"] + FEATURE_NAMES].reset_index(drop=True)

    split = 140
    model = train(feat.iloc[:split].reset_index(drop=True), y[:split])
    preds = predict(model, feat.iloc[split:].reset_index(drop=True))

    y_test = y[split:]
    pred_flagged = [1 if s < 0.5 else 0 for s in preds["reliability_score"]]
    accuracy = sum(a == b for a, b in zip(pred_flagged, y_test)) / len(y_test)
    assert accuracy >= 0.85  # structural check that the pipeline learns (G1 gate: not a real-world claim)

    for _, r in preds.iterrows():
        if r["reliability_score"] < 0.5:
            assert r["flag_reason"] != "" and r["flag_reason"].endswith(".")
        else:
            assert r["flag_reason"] == ""
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: 2 passed (the Phase 3 pipeline test + this one).

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest -v` from the SurveyIQ directory.
Expected: all pass (92 prior + 13 classifier + 1 new pipeline test).

- [ ] **Step 4: Commit**

```bash
git add SurveyIQ/tests/test_pipeline.py
git commit -m "test: add end-to-end pipeline test through the classifier"
```
