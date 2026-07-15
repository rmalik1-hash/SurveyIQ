# Normalizer + Column Mapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 3 normalizer that converts a raw messy survey DataFrame + a column mapping into the internal respondent schema the Phase 2 extractor consumes.

**Architecture:** Pure helper functions plus an `apply_mapping` orchestrator, all in `src/ingestion/normalize.py`. Outputs raw Likert ints + scale (matching Phase 2). A PII safety net drops obvious personal-data columns and warns. Built with TDD on hand-crafted DataFrames, capped by an end-to-end test through Phases 1→3→2.

**Tech Stack:** Python 3, pandas, pytest. (All already installed.)

**Environment notes (Windows):** Repo root is `C:\Users\rayan\OneDrive\Documents` (the whole Documents folder); project is the `SurveyIQ` subfolder; branch is `surveyiq-normalizer`. The repo has thousands of unrelated files — NEVER `git add -A`/`git add .`. Stage only the files each task names, by explicit path. Run tests with `python -m pytest` from the `SurveyIQ` directory. `conftest.py` already puts the project root on `sys.path`.

---

### Task 0: Scaffolding

**Files:**
- Create: `src/ingestion/__init__.py`

- [ ] **Step 1: Create the empty package marker** (`src/ingestion/__init__.py`, empty).

- [ ] **Step 2: Commit**

```bash
git add SurveyIQ/src/ingestion/__init__.py
git commit -m "chore: scaffold src/ingestion package"
```

---

### Task 1: _compute_duration (creates the module)

**Files:**
- Create: `src/ingestion/normalize.py`
- Test: `tests/test_normalize.py`

- [ ] **Step 1: Write the failing test** — create `tests/test_normalize.py`:

```python
import pandas as pd
import pytest
from src.ingestion.normalize import _compute_duration


def test_compute_duration_normal():
    assert _compute_duration("2024-03-01T08:00:00", "2024-03-01T08:02:00") == 120


def test_compute_duration_missing_returns_none():
    assert _compute_duration(None, "2024-03-01T08:02:00") is None
    assert _compute_duration("2024-03-01T08:00:00", None) is None


def test_compute_duration_unparseable_returns_none():
    assert _compute_duration("not a date", "2024-03-01T08:02:00") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_normalize.py -v`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Write minimal implementation** — create `src/ingestion/normalize.py`:

```python
import re
import warnings

import pandas as pd


def _compute_duration(start, end):
    if start is None or end is None:
        return None
    start_ts = pd.to_datetime(start, errors="coerce")
    end_ts = pd.to_datetime(end, errors="coerce")
    if pd.isna(start_ts) or pd.isna(end_ts):
        return None
    return int((end_ts - start_ts).total_seconds())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_normalize.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add SurveyIQ/src/ingestion/normalize.py SurveyIQ/tests/test_normalize.py
git commit -m "feat: add _compute_duration to normalizer"
```

---

### Task 2: _is_pii_column

**Files:**
- Modify: `src/ingestion/normalize.py`
- Test: `tests/test_normalize.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_normalize.py`:

```python
from src.ingestion.normalize import _is_pii_column


def test_is_pii_flags_email_header():
    assert _is_pii_column("Email Address", ["a", "b"]) is True


def test_is_pii_flags_person_name_headers():
    assert _is_pii_column("Student Name", ["Alice", "Bob"]) is True
    assert _is_pii_column("Name", ["Alice"]) is True


def test_is_pii_flags_email_values():
    assert _is_pii_column("contact", ["x@y.com", "z@w.org"]) is True


def test_is_pii_does_not_flag_school_name_or_grade():
    assert _is_pii_column("School Name", ["Lincoln High"]) is False
    assert _is_pii_column("grade level", ["9", "10"]) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_normalize.py -k is_pii -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation** — add to `src/ingestion/normalize.py` (after the imports, before `_compute_duration`):

```python
PII_HEADER_SUBSTRINGS = ["email", "e-mail", "phone", "ssn", "social security", "address"]
PII_NAME_HEADERS = ["first name", "last name", "full name", "student name", "your name", "surname"]
EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


def _is_pii_column(col_name, values):
    lowered = col_name.strip().lower()
    if lowered == "name":
        return True
    if any(sub in lowered for sub in PII_HEADER_SUBSTRINGS):
        return True
    if any(pat in lowered for pat in PII_NAME_HEADERS):
        return True
    for value in values:
        if isinstance(value, str) and EMAIL_RE.search(value):
            return True
    return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_normalize.py -k is_pii -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add SurveyIQ/src/ingestion/normalize.py SurveyIQ/tests/test_normalize.py
git commit -m "feat: add PII column detection to normalizer"
```

---

### Task 3: _extract_responses and _extract_attention_checks

**Files:**
- Modify: `src/ingestion/normalize.py`
- Test: `tests/test_normalize.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_normalize.py`:

```python
from src.ingestion.normalize import _extract_responses, _extract_attention_checks


def test_extract_responses_keys_in_order():
    row = pd.Series({"Q_a": 3, "Q_b": 5})
    assert _extract_responses(row, ["Q_a", "Q_b"]) == {"q1": 3, "q2": 5}


def test_extract_attention_checks_pairs_given_and_correct():
    row = pd.Series({"AC_1": 4})
    assert _extract_attention_checks(row, ["AC_1"], {"AC_1": 5}) == {
        "ac1_given": 4, "ac1_correct": 5,
    }


def test_extract_attention_checks_empty():
    row = pd.Series({"Q_a": 3})
    assert _extract_attention_checks(row, [], {}) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_normalize.py -k extract_ -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation** — append to `src/ingestion/normalize.py`:

```python
def _extract_responses(row, question_cols):
    return {f"q{i + 1}": int(row[col]) for i, col in enumerate(question_cols)}


def _extract_attention_checks(row, ac_cols, answers):
    checks = {}
    for i, col in enumerate(ac_cols):
        checks[f"ac{i + 1}_given"] = int(row[col])
        checks[f"ac{i + 1}_correct"] = int(answers[col])
    return checks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_normalize.py -k extract_ -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add SurveyIQ/src/ingestion/normalize.py SurveyIQ/tests/test_normalize.py
git commit -m "feat: add response and attention-check extraction to normalizer"
```

---

### Task 4: _validate_mapping and _columns_by_role

**Files:**
- Modify: `src/ingestion/normalize.py`
- Test: `tests/test_normalize.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_normalize.py`:

```python
from src.ingestion.normalize import _validate_mapping, _columns_by_role


def _valid_df():
    return pd.DataFrame({"id": ["R1"], "Q1": [3], "AC1": [5]})


def _valid_mapping():
    return {
        "columns": {"id": "respondent_id", "Q1": "question", "AC1": "attention_check"},
        "scale": [1, 5],
        "attention_check_answers": {"AC1": 5},
    }


def test_validate_mapping_ok():
    _validate_mapping(_valid_df(), _valid_mapping())  # must not raise


def test_validate_mapping_unknown_column():
    m = _valid_mapping()
    m["columns"]["ghost"] = "question"
    with pytest.raises(ValueError):
        _validate_mapping(_valid_df(), m)


def test_validate_mapping_no_respondent_id():
    m = _valid_mapping()
    m["columns"]["id"] = "demographic"
    with pytest.raises(ValueError):
        _validate_mapping(_valid_df(), m)


def test_validate_mapping_two_respondent_ids():
    df = pd.DataFrame({"id": ["R1"], "id2": ["X"], "Q1": [3], "AC1": [5]})
    m = _valid_mapping()
    m["columns"]["id2"] = "respondent_id"
    with pytest.raises(ValueError):
        _validate_mapping(df, m)


def test_validate_mapping_no_question():
    m = _valid_mapping()
    m["columns"]["Q1"] = "demographic"
    with pytest.raises(ValueError):
        _validate_mapping(_valid_df(), m)


def test_validate_mapping_bad_scale():
    m = _valid_mapping()
    m["scale"] = [5, 1]
    with pytest.raises(ValueError):
        _validate_mapping(_valid_df(), m)


def test_validate_mapping_attention_without_answer():
    m = _valid_mapping()
    m["attention_check_answers"] = {}
    with pytest.raises(ValueError):
        _validate_mapping(_valid_df(), m)


def test_columns_by_role_groups_and_preserves_order():
    m = {
        "columns": {"id": "respondent_id", "Qa": "question", "Qb": "question"},
        "scale": [1, 5],
    }
    grouped = _columns_by_role(m)
    assert grouped["question"] == ["Qa", "Qb"]
    assert grouped["respondent_id"] == ["id"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_normalize.py -k "validate_mapping or columns_by_role" -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation** — add `VALID_ROLES` near the other constants and append the functions to `src/ingestion/normalize.py`:

```python
VALID_ROLES = {
    "respondent_id", "start_time", "end_time", "question",
    "attention_check", "demographic", "ignore",
}


def _columns_by_role(mapping):
    grouped = {role: [] for role in VALID_ROLES}
    for col, role in mapping["columns"].items():
        if role in grouped:
            grouped[role].append(col)
    return grouped


def _validate_mapping(raw_df, mapping):
    columns = mapping.get("columns", {})
    for col in columns:
        if col not in raw_df.columns:
            raise ValueError(f"mapping references column not in DataFrame: {col!r}")
    roles = list(columns.values())
    if roles.count("respondent_id") != 1:
        raise ValueError("mapping must have exactly one respondent_id column")
    if "question" not in roles:
        raise ValueError("mapping must have at least one question column")
    scale = mapping.get("scale")
    if not scale or len(scale) != 2 or scale[0] >= scale[1]:
        raise ValueError("mapping needs a valid scale [min, max] with min < max")
    answers = mapping.get("attention_check_answers", {})
    for col, role in columns.items():
        if role == "attention_check" and col not in answers:
            raise ValueError(f"attention_check column has no correct answer: {col!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_normalize.py -k "validate_mapping or columns_by_role" -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add SurveyIQ/src/ingestion/normalize.py SurveyIQ/tests/test_normalize.py
git commit -m "feat: add mapping validation and role grouping to normalizer"
```

---

### Task 5: apply_mapping orchestrator (and _extract_demographics)

**Files:**
- Modify: `src/ingestion/normalize.py`
- Test: `tests/test_normalize.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_normalize.py`:

```python
import warnings
from src.ingestion.normalize import apply_mapping


def test_apply_mapping_basic_shape():
    df = pd.DataFrame({
        "Response ID": ["R1", "R2"],
        "Start Time": ["2024-03-01T08:00:00", "2024-03-01T08:00:00"],
        "Timestamp": ["2024-03-01T08:02:00", "2024-03-01T08:01:00"],
        "Q1": [3, 1], "Q2": [3, 5],
        "AC1": [5, 1],
        "grade level": ["9", "10"],
    })
    mapping = {
        "columns": {
            "Response ID": "respondent_id", "Start Time": "start_time",
            "Timestamp": "end_time", "Q1": "question", "Q2": "question",
            "AC1": "attention_check", "grade level": "demographic",
        },
        "scale": [1, 5],
        "attention_check_answers": {"AC1": 5},
    }
    out = apply_mapping(df, mapping)
    assert len(out) == 2
    r0 = out[0]
    assert r0["respondent_id"] == "R1"
    assert r0["duration_seconds"] == 120
    assert r0["responses"] == {"q1": 3, "q2": 3}
    assert r0["attention_checks"] == {"ac1_given": 5, "ac1_correct": 5}
    assert r0["demographics"] == {"grade level": "9"}
    assert r0["scale_min"] == 1 and r0["scale_max"] == 5


def test_apply_mapping_missing_end_time_gives_none_duration():
    df = pd.DataFrame({"Response ID": ["R1"], "Q1": [3]})
    mapping = {
        "columns": {"Response ID": "respondent_id", "Q1": "question"},
        "scale": [1, 5],
    }
    out = apply_mapping(df, mapping)
    assert out[0]["duration_seconds"] is None


def test_apply_mapping_drops_pii_demographic_and_warns():
    df = pd.DataFrame({
        "Response ID": ["R1"], "Q1": [3], "Email Address": ["a@b.com"],
    })
    mapping = {
        "columns": {
            "Response ID": "respondent_id", "Q1": "question",
            "Email Address": "demographic",
        },
        "scale": [1, 5],
    }
    with pytest.warns(UserWarning):
        out = apply_mapping(df, mapping)
    assert "Email Address" not in out[0]["demographics"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_normalize.py -k apply_mapping -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation** — append to `src/ingestion/normalize.py`:

```python
def _extract_demographics(row, demo_cols):
    return {col: row[col] for col in demo_cols}


def apply_mapping(raw_df, mapping):
    _validate_mapping(raw_df, mapping)
    grouped = _columns_by_role(mapping)
    scale_min, scale_max = mapping["scale"]
    answers = mapping.get("attention_check_answers", {})

    id_col = grouped["respondent_id"][0]
    start_col = grouped["start_time"][0] if grouped["start_time"] else None
    end_col = grouped["end_time"][0] if grouped["end_time"] else None
    question_cols = grouped["question"]
    ac_cols = grouped["attention_check"]

    if _is_pii_column(id_col, raw_df[id_col].tolist()):
        warnings.warn(f"respondent_id column {id_col!r} looks like it may contain PII")

    demo_cols = []
    for col in grouped["demographic"]:
        if _is_pii_column(col, raw_df[col].tolist()):
            warnings.warn(f"dropping demographic column that looks like PII: {col!r}")
        else:
            demo_cols.append(col)

    respondents = []
    for _, row in raw_df.iterrows():
        start = row[start_col] if start_col else None
        end = row[end_col] if end_col else None
        respondents.append({
            "respondent_id": str(row[id_col]),
            "duration_seconds": _compute_duration(start, end),
            "responses": _extract_responses(row, question_cols),
            "attention_checks": _extract_attention_checks(row, ac_cols, answers),
            "demographics": _extract_demographics(row, demo_cols),
            "scale_min": scale_min,
            "scale_max": scale_max,
        })
    return respondents
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_normalize.py -k apply_mapping -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add SurveyIQ/src/ingestion/normalize.py SurveyIQ/tests/test_normalize.py
git commit -m "feat: add apply_mapping orchestrator to normalizer"
```

---

### Task 6: End-to-end pipeline test (generator -> normalizer -> extractor)

**Files:**
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the test** — create `tests/test_pipeline.py`:

```python
import json
import pandas as pd
from data.synthetic.generator import generate_survey_csv
from src.ingestion.normalize import apply_mapping
from src.features.extract import extract_features


def _build_mapping(columns):
    col_roles = {}
    ac_answers = {}
    for c in columns:
        if "[Q" in c:
            col_roles[c] = "question"
        elif "[AC" in c:
            col_roles[c] = "attention_check"
            ac_answers[c] = 5
        elif c == "Response ID":
            col_roles[c] = "respondent_id"
        elif c == "Start Time":
            col_roles[c] = "start_time"
        elif c == "Timestamp":
            col_roles[c] = "end_time"
        elif c == "Email Address":
            col_roles[c] = "ignore"
        else:
            col_roles[c] = "demographic"
    return {"columns": col_roles, "scale": [1, 5], "attention_check_answers": ac_answers}


def test_full_pipeline_separates_archetypes(tmp_path):
    out = tmp_path / "s.csv"
    generate_survey_csv(
        n_respondents=40, n_questions=10, scale=(1, 5),
        contamination_rate=0.4, seed=5, output_path=str(out),
    )
    df = pd.read_csv(out)
    labels = pd.read_csv(tmp_path / "s_labels.csv")
    pairs_idx = json.load(open(tmp_path / "s_pairs.json"))["pairs"]

    respondents = apply_mapping(df, _build_mapping(list(df.columns)))
    pair_keys = [(f"q{a + 1}", f"q{b + 1}") for a, b in pairs_idx]
    features = extract_features(respondents, contradiction_pairs=pair_keys)

    merged = features.merge(labels, on="respondent_id")

    straight = merged[merged["archetype"] == "straightliner"]
    assert len(straight) > 0
    assert (straight["straightlining_score"] == 1.0).all()

    speed = merged[merged["archetype"] == "speeder"]
    assert len(speed) > 0
    assert (speed["completion_time_ratio"] < 0.5).all()

    reliable = merged[merged["archetype"] == "reliable"]
    assert (reliable["contradiction_score"] == 0.0).all()
    assert (reliable["attention_check_pass_rate"] == 1.0).all()
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: 1 passed (this proves Phases 1→3→2 work together).

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest -v` from the SurveyIQ directory.
Expected: all pass (66 prior + new normalizer/pipeline tests).

- [ ] **Step 4: Commit**

```bash
git add SurveyIQ/tests/test_pipeline.py
git commit -m "test: add end-to-end generator-to-features pipeline test"
```
