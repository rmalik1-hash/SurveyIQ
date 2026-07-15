# Feature Extractor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 2 feature extractor that turns normalized respondent records into a per-respondent DataFrame of six behavioral features for the Phase 4 model.

**Architecture:** One small pure function per feature plus a thin orchestrator, all in `src/features/extract.py`. Consumes CLAUDE.md's internal respondent schema (Option A — clean structured dicts, not raw CSV). Missing values are `np.nan`. Built with TDD on hand-crafted inputs.

**Tech Stack:** Python 3, numpy, pandas, pytest. (All already in `requirements.txt`.)

**Environment notes (Windows):** Repo root is `C:\Users\rayan\OneDrive\Documents` (the whole Documents folder); the project is the `SurveyIQ` subfolder; branch is `surveyiq-feature-extractor`. The repo has thousands of unrelated files — NEVER `git add -A` or `git add .`. Stage only the specific files each task names, by explicit path. Run tests with `python -m pytest` from the `SurveyIQ` directory. `conftest.py` already puts the project root on `sys.path` so `from src.features.extract import ...` resolves.

---

### Task 0: Scaffolding

**Files:**
- Create: `src/__init__.py`
- Create: `src/features/__init__.py`

- [ ] **Step 1: Create the package markers**

Both files are empty.

- [ ] **Step 2: Commit**

```bash
git add SurveyIQ/src/__init__.py SurveyIQ/src/features/__init__.py
git commit -m "chore: scaffold src/features package"
```

---

### Task 1: completion_time_ratio (creates the module)

**Files:**
- Create: `src/features/extract.py`
- Test: `tests/test_extract.py`

- [ ] **Step 1: Write the failing test** — create `tests/test_extract.py`:

```python
import numpy as np
import pytest
from src.features.extract import completion_time_ratio, AVG_SECONDS_PER_QUESTION


def test_completion_time_ratio_normal():
    assert completion_time_ratio(duration_seconds=20, num_questions=20) == pytest.approx(0.125)


def test_completion_time_ratio_uses_eight_seconds_per_question():
    assert AVG_SECONDS_PER_QUESTION == 8
    assert completion_time_ratio(duration_seconds=160, num_questions=20) == pytest.approx(1.0)


def test_completion_time_ratio_none_duration_is_nan():
    assert np.isnan(completion_time_ratio(duration_seconds=None, num_questions=20))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_extract.py -v`
Expected: FAIL with ModuleNotFoundError / ImportError.

- [ ] **Step 3: Write minimal implementation** — create `src/features/extract.py`:

```python
import numpy as np
import pandas as pd

AVG_SECONDS_PER_QUESTION = 8


def completion_time_ratio(duration_seconds, num_questions):
    if duration_seconds is None:
        return np.nan
    return duration_seconds / (num_questions * AVG_SECONDS_PER_QUESTION)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_extract.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add SurveyIQ/src/features/extract.py SurveyIQ/tests/test_extract.py
git commit -m "feat: add completion_time_ratio feature"
```

---

### Task 2: straightlining_score

**Files:**
- Modify: `src/features/extract.py`
- Test: `tests/test_extract.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_extract.py`:

```python
from src.features.extract import straightlining_score


def test_straightlining_all_same_is_one():
    assert straightlining_score({"q1": 3, "q2": 3, "q3": 3, "q4": 3}) == 1.0


def test_straightlining_all_different():
    assert straightlining_score({"q1": 1, "q2": 2, "q3": 3, "q4": 4}) == 0.25


def test_straightlining_partial():
    # modal value 5 appears 3 of 4 times
    assert straightlining_score({"q1": 5, "q2": 5, "q3": 5, "q4": 1}) == 0.75


def test_straightlining_empty_raises():
    with pytest.raises(ValueError):
        straightlining_score({})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_extract.py -k straightlining -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation** — append to `src/features/extract.py`:

```python
def straightlining_score(responses):
    values = list(responses.values())
    n = len(values)
    if n == 0:
        raise ValueError("responses must not be empty")
    modal_count = max(values.count(v) for v in set(values))
    return modal_count / n
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_extract.py -k straightlining -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add SurveyIQ/src/features/extract.py SurveyIQ/tests/test_extract.py
git commit -m "feat: add straightlining_score feature"
```

---

### Task 3: response_variance

**Files:**
- Modify: `src/features/extract.py`
- Test: `tests/test_extract.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_extract.py`:

```python
from src.features.extract import response_variance


def test_response_variance_all_same_is_zero():
    assert response_variance({"q1": 3, "q2": 3, "q3": 3}, scale_min=1, scale_max=5) == 0.0


def test_response_variance_alternating_endpoints():
    # normalized values are 0.0 and 1.0 -> population std 0.5
    r = {"q1": 1, "q2": 5, "q3": 1, "q4": 5}
    assert response_variance(r, scale_min=1, scale_max=5) == pytest.approx(0.5)


def test_response_variance_degenerate_scale_is_zero():
    # scale_min == scale_max must not divide by zero
    assert response_variance({"q1": 3, "q2": 3}, scale_min=3, scale_max=3) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_extract.py -k response_variance -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation** — append to `src/features/extract.py`:

```python
def response_variance(responses, scale_min, scale_max):
    values = list(responses.values())
    if len(values) == 0:
        raise ValueError("responses must not be empty")
    if scale_max == scale_min:
        return 0.0
    normalized = [(v - scale_min) / (scale_max - scale_min) for v in values]
    return float(np.std(normalized))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_extract.py -k response_variance -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add SurveyIQ/src/features/extract.py SurveyIQ/tests/test_extract.py
git commit -m "feat: add response_variance feature"
```

---

### Task 4: extreme_response_rate

**Files:**
- Modify: `src/features/extract.py`
- Test: `tests/test_extract.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_extract.py`:

```python
from src.features.extract import extreme_response_rate


def test_extreme_response_rate_all_endpoints():
    r = {"q1": 1, "q2": 5, "q3": 1, "q4": 5}
    assert extreme_response_rate(r, scale_min=1, scale_max=5) == 1.0


def test_extreme_response_rate_none_extreme():
    r = {"q1": 2, "q2": 3, "q3": 4}
    assert extreme_response_rate(r, scale_min=1, scale_max=5) == 0.0


def test_extreme_response_rate_half():
    r = {"q1": 1, "q2": 3, "q3": 5, "q4": 3}
    assert extreme_response_rate(r, scale_min=1, scale_max=5) == 0.5


def test_extreme_response_rate_degenerate_scale_is_zero():
    assert extreme_response_rate({"q1": 3, "q2": 3}, scale_min=3, scale_max=3) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_extract.py -k extreme -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation** — append to `src/features/extract.py`:

```python
def extreme_response_rate(responses, scale_min, scale_max):
    values = list(responses.values())
    n = len(values)
    if n == 0:
        raise ValueError("responses must not be empty")
    if scale_max == scale_min:
        return 0.0
    extreme = sum(1 for v in values if v == scale_min or v == scale_max)
    return extreme / n
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_extract.py -k extreme -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add SurveyIQ/src/features/extract.py SurveyIQ/tests/test_extract.py
git commit -m "feat: add extreme_response_rate feature"
```

---

### Task 5: attention_check_pass_rate

**Files:**
- Modify: `src/features/extract.py`
- Test: `tests/test_extract.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_extract.py`:

```python
from src.features.extract import attention_check_pass_rate


def test_attention_all_pass():
    ac = {"ac1_given": 5, "ac1_correct": 5, "ac2_given": 5, "ac2_correct": 5}
    assert attention_check_pass_rate(ac) == 1.0


def test_attention_half_pass():
    ac = {"ac1_given": 5, "ac1_correct": 5, "ac2_given": 1, "ac2_correct": 5}
    assert attention_check_pass_rate(ac) == 0.5


def test_attention_no_checks_is_one():
    assert attention_check_pass_rate({}) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_extract.py -k attention -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation** — append to `src/features/extract.py`:

```python
def attention_check_pass_rate(attention_checks):
    given_keys = [k for k in attention_checks if k.endswith("_given")]
    total = 0
    correct = 0
    for given_key in given_keys:
        prefix = given_key[: -len("_given")]
        correct_key = prefix + "_correct"
        if correct_key in attention_checks:
            total += 1
            if attention_checks[given_key] == attention_checks[correct_key]:
                correct += 1
    if total == 0:
        return 1.0
    return correct / total
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_extract.py -k attention -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add SurveyIQ/src/features/extract.py SurveyIQ/tests/test_extract.py
git commit -m "feat: add attention_check_pass_rate feature"
```

---

### Task 6: contradiction_score (and _mirror helper)

**Files:**
- Modify: `src/features/extract.py`
- Test: `tests/test_extract.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_extract.py`:

```python
from src.features.extract import contradiction_score


def test_contradiction_mirrored_pair_is_consistent():
    # q1=1, q2=5 on a 1-5 scale: 1+5 == min+max, perfectly reverse-coded
    r = {"q1": 1, "q2": 5}
    assert contradiction_score(r, 1, 5, [("q1", "q2")]) == 0.0


def test_contradiction_matched_pair_contradicts():
    # q1=1, q2=1: expected mirror of q1 is 5, actual 1 -> gap 4 > tolerance
    r = {"q1": 1, "q2": 1}
    assert contradiction_score(r, 1, 5, [("q1", "q2")]) == 1.0


def test_contradiction_no_pairs_is_zero():
    r = {"q1": 1, "q2": 5}
    assert contradiction_score(r, 1, 5, None) == 0.0
    assert contradiction_score(r, 1, 5, []) == 0.0


def test_contradiction_half_of_two_pairs():
    # pair1 mirrored (consistent), pair2 matched (contradiction)
    r = {"q1": 1, "q2": 5, "q3": 2, "q4": 2}
    assert contradiction_score(r, 1, 5, [("q1", "q2"), ("q3", "q4")]) == 0.5


def test_contradiction_within_tolerance_is_consistent():
    # mirror of q1(2) is 4; q2=3 -> gap 1, not > tolerance(1) -> consistent
    r = {"q1": 2, "q2": 3}
    assert contradiction_score(r, 1, 5, [("q1", "q2")]) == 0.0


def test_contradiction_pair_key_missing_raises():
    r = {"q1": 1, "q2": 5}
    with pytest.raises(ValueError):
        contradiction_score(r, 1, 5, [("q1", "q9")])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_extract.py -k contradiction -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation** — append to `src/features/extract.py`:

```python
def _mirror(value, scale_min, scale_max):
    return scale_min + scale_max - value


def contradiction_score(responses, scale_min, scale_max, pairs, tolerance=1):
    if not pairs:
        return 0.0
    contradicting = 0
    for a_key, b_key in pairs:
        if a_key not in responses or b_key not in responses:
            raise ValueError(
                f"contradiction pair ({a_key}, {b_key}) references a missing response"
            )
        gap = abs(responses[b_key] - _mirror(responses[a_key], scale_min, scale_max))
        if gap > tolerance:
            contradicting += 1
    return contradicting / len(pairs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_extract.py -k contradiction -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add SurveyIQ/src/features/extract.py SurveyIQ/tests/test_extract.py
git commit -m "feat: add contradiction_score feature"
```

---

### Task 7: extract_features orchestrator

**Files:**
- Modify: `src/features/extract.py`
- Test: `tests/test_extract.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_extract.py`:

```python
import pandas as pd
from src.features.extract import extract_features


def _respondent(rid, responses, duration, attention=None):
    return {
        "respondent_id": rid,
        "duration_seconds": duration,
        "responses": responses,
        "attention_checks": attention or {},
        "scale_min": 1,
        "scale_max": 5,
    }


def test_extract_features_shape_and_columns():
    respondents = [
        _respondent("R1", {"q1": 3, "q2": 3, "q3": 3, "q4": 3}, 160,
                    {"ac1_given": 5, "ac1_correct": 5}),
        _respondent("R2", {"q1": 1, "q2": 5, "q3": 1, "q4": 5}, 20,
                    {"ac1_given": 1, "ac1_correct": 5}),
    ]
    df = extract_features(respondents)
    assert list(df.columns) == [
        "respondent_id", "completion_time_ratio", "straightlining_score",
        "response_variance", "contradiction_score", "attention_check_pass_rate",
        "extreme_response_rate",
    ]
    assert list(df["respondent_id"]) == ["R1", "R2"]
    assert len(df) == 2


def test_extract_features_values_for_known_respondents():
    respondents = [
        _respondent("R1", {"q1": 3, "q2": 3, "q3": 3, "q4": 3}, 160,
                    {"ac1_given": 5, "ac1_correct": 5}),
    ]
    df = extract_features(respondents)
    row = df.iloc[0]
    assert row["straightlining_score"] == 1.0
    assert row["response_variance"] == 0.0
    assert row["completion_time_ratio"] == pytest.approx(160 / (4 * 8))
    assert row["attention_check_pass_rate"] == 1.0
    assert row["extreme_response_rate"] == 0.0
    assert row["contradiction_score"] == 0.0


def test_extract_features_missing_duration_is_nan():
    respondents = [_respondent("R1", {"q1": 2, "q2": 4}, None)]
    df = extract_features(respondents)
    assert np.isnan(df.iloc[0]["completion_time_ratio"])


def test_extract_features_uses_contradiction_pairs():
    respondents = [
        _respondent("R1", {"q1": 1, "q2": 1, "q3": 2, "q4": 4}, 100),
    ]
    df = extract_features(respondents, contradiction_pairs=[("q1", "q2"), ("q3", "q4")])
    # q1/q2 matched -> contradiction; q3/q4 mirrored (2+4==6) -> consistent
    assert df.iloc[0]["contradiction_score"] == 0.5


def test_extract_features_empty_responses_raises():
    respondents = [_respondent("R1", {}, 100)]
    with pytest.raises(ValueError):
        extract_features(respondents)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_extract.py -k extract_features -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation** — append to `src/features/extract.py`:

```python
FEATURE_COLUMNS = [
    "respondent_id",
    "completion_time_ratio",
    "straightlining_score",
    "response_variance",
    "contradiction_score",
    "attention_check_pass_rate",
    "extreme_response_rate",
]


def extract_features(respondents, contradiction_pairs=None):
    rows = []
    for r in respondents:
        responses = r["responses"]
        if len(responses) == 0:
            raise ValueError(f"respondent {r.get('respondent_id')} has no responses")
        num_questions = len(responses)
        scale_min = r["scale_min"]
        scale_max = r["scale_max"]
        rows.append({
            "respondent_id": r["respondent_id"],
            "completion_time_ratio": completion_time_ratio(
                r.get("duration_seconds"), num_questions),
            "straightlining_score": straightlining_score(responses),
            "response_variance": response_variance(responses, scale_min, scale_max),
            "contradiction_score": contradiction_score(
                responses, scale_min, scale_max, contradiction_pairs),
            "attention_check_pass_rate": attention_check_pass_rate(
                r.get("attention_checks", {})),
            "extreme_response_rate": extreme_response_rate(
                responses, scale_min, scale_max),
        })
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_extract.py -k extract_features -v`
Expected: 5 passed.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -v` from the SurveyIQ directory.
Expected: all pass (37 from Phase 1 + the new extractor tests).

- [ ] **Step 6: Commit**

```bash
git add SurveyIQ/src/features/extract.py SurveyIQ/tests/test_extract.py
git commit -m "feat: add extract_features orchestrator"
```
