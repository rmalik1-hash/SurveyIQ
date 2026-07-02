# Synthetic Data Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `generate_survey_csv()`, a synthetic survey generator that produces a messy, Google-Forms-style raw CSV plus ground-truth sidecar files (labels + contradiction pairs), covering 5 respondent archetypes (reliable, straightliner, speeder, random_responder, contradictor).

**Architecture:** Four small modules under `data/synthetic/`: `contradiction_pairs.py` (pure function, picks reverse-coded question index pairs), `archetypes.py` (one simulation function per archetype, each returns `{"answers": [...], "duration_seconds": int}`), `template.py` (messy header naming + DataFrame assembly), and `generator.py` (orchestrates: assign archetypes → simulate → inject attention checks/demographics → render → write 3 files). No classes — plain functions, a shared `rng` (`numpy.random.Generator`) threaded through for reproducibility.

**Tech Stack:** Python 3.11, pandas, numpy, pytest.

Reference spec: [docs/superpowers/specs/2026-07-01-synthetic-data-generator-design.md](../specs/2026-07-01-synthetic-data-generator-design.md)

---

## Task 0: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `conftest.py`
- Create: `.gitignore`
- Create: `data/__init__.py`
- Create: `data/synthetic/__init__.py`
- Create: `tests/test_placeholder.py` (deleted at end of task, just to prove the harness works)

- [ ] **Step 1: Create `requirements.txt`**

```
pandas
numpy
pytest
```

- [ ] **Step 2: Create empty package `__init__.py` files**

Create `data/__init__.py` with empty content, and `data/synthetic/__init__.py` with empty content. These make `data.synthetic.generator` importable as `import data.synthetic.generator`.

- [ ] **Step 3: Create root `conftest.py`**

```python
# Ensures the project root is on sys.path so tests can `import data.synthetic...`
```

An empty (comment-only) `conftest.py` at the repo root is enough — pytest inserts the directory containing the rootdir's `conftest.py` onto `sys.path`.

- [ ] **Step 4: Create `.gitignore`**

```
__pycache__/
*.pyc
data/synthetic/*.csv
data/synthetic/*.json
```

Generated survey CSVs/labels/pairs files are regenerable artifacts — do not commit them.

- [ ] **Step 5: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: pandas, numpy, and pytest install without errors.

- [ ] **Step 6: Verify pytest harness works**

Create `tests/test_placeholder.py`:

```python
def test_placeholder():
    assert True
```

Run: `pytest tests/test_placeholder.py -v`
Expected: `1 passed`

- [ ] **Step 7: Remove the placeholder test**

Delete `tests/test_placeholder.py` (its only purpose was proving the harness works).

- [ ] **Step 8: Commit**

```bash
git add requirements.txt conftest.py .gitignore data/__init__.py data/synthetic/__init__.py
git commit -m "chore: scaffold project structure for synthetic data generator"
```

---

## Task 1: Contradiction Pairs

**Files:**
- Create: `data/synthetic/contradiction_pairs.py`
- Test: `tests/test_contradiction_pairs.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from data.synthetic.contradiction_pairs import get_contradiction_pairs


def test_returns_one_pair_per_ten_questions():
    pairs = get_contradiction_pairs(20)
    assert pairs == [(0, 1), (2, 3)]


def test_minimum_one_pair_for_small_surveys():
    pairs = get_contradiction_pairs(4)
    assert pairs == [(0, 1)]


def test_pairs_never_reuse_or_exceed_question_indices():
    pairs = get_contradiction_pairs(100)
    seen = set()
    for a, b in pairs:
        assert a < 100 and b < 100
        assert a not in seen and b not in seen
        seen.add(a)
        seen.add(b)


def test_raises_for_too_few_questions():
    with pytest.raises(ValueError):
        get_contradiction_pairs(3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_contradiction_pairs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.synthetic.contradiction_pairs'`

- [ ] **Step 3: Implement `get_contradiction_pairs`**

```python
def get_contradiction_pairs(n_questions: int) -> list[tuple[int, int]]:
    """Designate reverse-coded question index pairs for contradiction scoring.

    Roughly 1 pair per 10 questions, minimum 1 pair. Pairs are the first
    2 * n_pairs question indices, taken in order: (0,1), (2,3), ...
    """
    if n_questions < 4:
        raise ValueError("n_questions must be at least 4 to generate contradiction pairs")

    n_pairs = max(1, n_questions // 10)
    if n_pairs * 2 > n_questions:
        n_pairs = n_questions // 2

    return [(i * 2, i * 2 + 1) for i in range(n_pairs)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_contradiction_pairs.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add data/synthetic/contradiction_pairs.py tests/test_contradiction_pairs.py
git commit -m "feat: add contradiction pair selection for synthetic surveys"
```

---

## Task 2: Archetypes — `reliable`

**Files:**
- Create: `data/synthetic/archetypes.py`
- Test: `tests/test_archetypes.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from data.synthetic.archetypes import simulate_reliable


def test_reliable_answers_are_within_scale():
    rng = np.random.default_rng(1)
    result = simulate_reliable(n_questions=10, scale=(1, 5), contradiction_pairs=[], rng=rng)
    assert len(result["answers"]) == 10
    assert all(1 <= a <= 5 for a in result["answers"])
    assert isinstance(result["duration_seconds"], int)


def test_reliable_respects_contradiction_pairs():
    rng = np.random.default_rng(1)
    result = simulate_reliable(
        n_questions=4, scale=(1, 5), contradiction_pairs=[(0, 1)], rng=rng
    )
    a, b = result["answers"][0], result["answers"][1]
    assert a + b == 6  # mirrored: scale_min + scale_max = 1 + 5


def test_reliable_duration_is_near_expected_reading_time():
    rng = np.random.default_rng(1)
    result = simulate_reliable(n_questions=20, scale=(1, 5), contradiction_pairs=[], rng=rng)
    expected = 20 * 8
    assert 0.8 * expected <= result["duration_seconds"] <= 1.5 * expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_archetypes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.synthetic.archetypes'`

- [ ] **Step 3: Implement `_mirror` and `simulate_reliable`**

```python
import numpy as np

AVG_SECONDS_PER_QUESTION = 8


def _mirror(value: int, scale_min: int, scale_max: int) -> int:
    return scale_min + scale_max - value


def simulate_reliable(n_questions, scale, contradiction_pairs, rng):
    scale_min, scale_max = scale
    baseline = rng.uniform(scale_min, scale_max)
    spread = (scale_max - scale_min) * 0.15
    raw = rng.normal(loc=baseline, scale=spread, size=n_questions)
    answers = np.clip(np.round(raw), scale_min, scale_max).astype(int).tolist()

    for a_idx, b_idx in contradiction_pairs:
        answers[b_idx] = _mirror(answers[a_idx], scale_min, scale_max)

    duration = n_questions * AVG_SECONDS_PER_QUESTION * rng.uniform(0.9, 1.4)
    return {"answers": answers, "duration_seconds": int(round(duration))}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_archetypes.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add data/synthetic/archetypes.py tests/test_archetypes.py
git commit -m "feat: add reliable respondent simulation"
```

---

## Task 3: Archetypes — `straightliner`

**Files:**
- Modify: `data/synthetic/archetypes.py`
- Modify: `tests/test_archetypes.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_archetypes.py`:

```python
from data.synthetic.archetypes import simulate_straightliner


def test_straightliner_gives_identical_answer_to_every_question():
    rng = np.random.default_rng(2)
    result = simulate_straightliner(
        n_questions=15, scale=(1, 5), contradiction_pairs=[], rng=rng
    )
    assert len(set(result["answers"])) == 1
    assert len(result["answers"]) == 15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_archetypes.py::test_straightliner_gives_identical_answer_to_every_question -v`
Expected: FAIL with `ImportError: cannot import name 'simulate_straightliner'`

- [ ] **Step 3: Implement `simulate_straightliner`**

Add to `data/synthetic/archetypes.py`:

```python
def simulate_straightliner(n_questions, scale, contradiction_pairs, rng):
    scale_min, scale_max = scale
    value = int(rng.integers(scale_min, scale_max + 1))
    answers = [value] * n_questions
    duration = n_questions * AVG_SECONDS_PER_QUESTION * rng.uniform(0.8, 1.3)
    return {"answers": answers, "duration_seconds": int(round(duration))}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_archetypes.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add data/synthetic/archetypes.py tests/test_archetypes.py
git commit -m "feat: add straightliner respondent simulation"
```

---

## Task 4: Archetypes — `speeder`

**Files:**
- Modify: `data/synthetic/archetypes.py`
- Modify: `tests/test_archetypes.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_archetypes.py`:

```python
from data.synthetic.archetypes import simulate_speeder


def test_speeder_duration_is_implausibly_short():
    rng = np.random.default_rng(3)
    result = simulate_speeder(n_questions=20, scale=(1, 5), contradiction_pairs=[], rng=rng)
    expected = 20 * 8
    assert result["duration_seconds"] < 0.5 * expected


def test_speeder_still_respects_contradiction_pairs():
    rng = np.random.default_rng(3)
    result = simulate_speeder(
        n_questions=4, scale=(1, 5), contradiction_pairs=[(0, 1)], rng=rng
    )
    a, b = result["answers"][0], result["answers"][1]
    assert a + b == 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_archetypes.py -v -k speeder`
Expected: FAIL with `ImportError: cannot import name 'simulate_speeder'`

- [ ] **Step 3: Implement `simulate_speeder`**

Add to `data/synthetic/archetypes.py`:

```python
def simulate_speeder(n_questions, scale, contradiction_pairs, rng):
    scale_min, scale_max = scale
    baseline = rng.uniform(scale_min, scale_max)
    spread = (scale_max - scale_min) * 0.15
    raw = rng.normal(loc=baseline, scale=spread, size=n_questions)
    answers = np.clip(np.round(raw), scale_min, scale_max).astype(int).tolist()

    for a_idx, b_idx in contradiction_pairs:
        answers[b_idx] = _mirror(answers[a_idx], scale_min, scale_max)

    duration = n_questions * AVG_SECONDS_PER_QUESTION * rng.uniform(0.1, 0.3)
    return {"answers": answers, "duration_seconds": int(round(duration))}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_archetypes.py -v`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add data/synthetic/archetypes.py tests/test_archetypes.py
git commit -m "feat: add speeder respondent simulation"
```

---

## Task 5: Archetypes — `random_responder`

**Files:**
- Modify: `data/synthetic/archetypes.py`
- Modify: `tests/test_archetypes.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_archetypes.py`:

```python
from data.synthetic.archetypes import simulate_random_responder


def test_random_responder_has_high_variance():
    rng = np.random.default_rng(4)
    result = simulate_random_responder(
        n_questions=1000, scale=(1, 5), contradiction_pairs=[], rng=rng
    )
    assert np.std(result["answers"]) > 1.0  # true uniform std over {1..5} is ~1.41


def test_random_responder_answers_within_scale():
    rng = np.random.default_rng(4)
    result = simulate_random_responder(
        n_questions=20, scale=(1, 5), contradiction_pairs=[], rng=rng
    )
    assert all(1 <= a <= 5 for a in result["answers"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_archetypes.py -v -k random_responder`
Expected: FAIL with `ImportError: cannot import name 'simulate_random_responder'`

- [ ] **Step 3: Implement `simulate_random_responder`**

Add to `data/synthetic/archetypes.py`:

```python
def simulate_random_responder(n_questions, scale, contradiction_pairs, rng):
    scale_min, scale_max = scale
    answers = rng.integers(scale_min, scale_max + 1, size=n_questions).tolist()
    duration = n_questions * AVG_SECONDS_PER_QUESTION * rng.uniform(0.5, 1.5)
    return {"answers": answers, "duration_seconds": int(round(duration))}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_archetypes.py -v`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add data/synthetic/archetypes.py tests/test_archetypes.py
git commit -m "feat: add random responder simulation"
```

---

## Task 6: Archetypes — `contradictor`

**Files:**
- Modify: `data/synthetic/archetypes.py`
- Modify: `tests/test_archetypes.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_archetypes.py`:

```python
from data.synthetic.archetypes import simulate_contradictor


def test_contradictor_violates_contradiction_pairs():
    rng = np.random.default_rng(5)
    result = simulate_contradictor(
        n_questions=4, scale=(1, 5), contradiction_pairs=[(0, 1), (2, 3)], rng=rng
    )
    answers = result["answers"]
    for a_idx, b_idx in [(0, 1), (2, 3)]:
        # A reliable respondent would have answers[a] + answers[b] == 6 (mirrored).
        # The contradictor instead gives the *same* value to both, breaking that.
        assert answers[a_idx] == answers[b_idx]
        assert answers[a_idx] + answers[b_idx] != 6


def test_contradictor_answers_within_scale():
    rng = np.random.default_rng(5)
    result = simulate_contradictor(
        n_questions=4, scale=(1, 5), contradiction_pairs=[(0, 1)], rng=rng
    )
    assert all(1 <= a <= 5 for a in result["answers"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_archetypes.py -v -k contradictor`
Expected: FAIL with `ImportError: cannot import name 'simulate_contradictor'`

- [ ] **Step 3: Implement `simulate_contradictor`**

Add to `data/synthetic/archetypes.py`:

```python
def simulate_contradictor(n_questions, scale, contradiction_pairs, rng):
    scale_min, scale_max = scale
    baseline = rng.uniform(scale_min, scale_max)
    spread = (scale_max - scale_min) * 0.15
    raw = rng.normal(loc=baseline, scale=spread, size=n_questions)
    answers = np.clip(np.round(raw), scale_min, scale_max).astype(int).tolist()

    midpoint = (scale_min + scale_max) / 2
    non_midpoint_values = [v for v in range(scale_min, scale_max + 1) if v != midpoint]

    for a_idx, b_idx in contradiction_pairs:
        value = int(rng.choice(non_midpoint_values))
        answers[a_idx] = value
        answers[b_idx] = value  # same value instead of mirrored -> contradiction

    duration = n_questions * AVG_SECONDS_PER_QUESTION * rng.uniform(0.9, 1.4)
    return {"answers": answers, "duration_seconds": int(round(duration))}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_archetypes.py -v`
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add data/synthetic/archetypes.py tests/test_archetypes.py
git commit -m "feat: add contradictor respondent simulation"
```

---

## Task 7: Messy CSV Template

**Files:**
- Create: `data/synthetic/template.py`
- Test: `tests/test_template.py`

- [ ] **Step 1: Write the failing tests**

```python
import pandas as pd
from data.synthetic.template import (
    question_header,
    attention_check_header,
    render_messy_csv,
)


def test_question_header_contains_question_tag_and_scale():
    header = question_header(index=2, scale_min=1, scale_max=5)
    assert "[Q3]" in header
    assert "1" in header and "5" in header


def test_attention_check_header_contains_tag_and_target():
    header = attention_check_header(ac_number=1, scale_max=5)
    assert "[AC1]" in header
    assert "5" in header


def test_render_messy_csv_preserves_rows_and_column_order():
    rows = [
        {"Response ID": "R0001", "Start Time": "2024-01-01T08:00:00", "Timestamp": "2024-01-01T08:05:00"},
        {"Response ID": "R0002", "Start Time": "2024-01-01T08:10:00", "Timestamp": "2024-01-01T08:15:00"},
    ]
    df = render_messy_csv(rows)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Response ID", "Start Time", "Timestamp"]
    assert df.iloc[0]["Response ID"] == "R0001"
    assert len(df) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_template.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.synthetic.template'`

- [ ] **Step 3: Implement `template.py`**

```python
import pandas as pd


def question_header(index: int, scale_min: int, scale_max: int) -> str:
    return (
        f"On a scale of {scale_min} to {scale_max}, how much do you agree "
        f"with statement {index + 1}? [Q{index + 1}]"
    )


def attention_check_header(ac_number: int, scale_max: int) -> str:
    return (
        f"For quality control, please select {scale_max} for this item. "
        f"[AC{ac_number}]"
    )


def render_messy_csv(rows: list[dict]) -> pd.DataFrame:
    """Assemble per-respondent flat dicts into a DataFrame, preserving the
    column order of the first row (all rows are expected to share the same
    keys in the same order, as produced by generator.py)."""
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_template.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add data/synthetic/template.py tests/test_template.py
git commit -m "feat: add messy CSV header templates and renderer"
```

---

## Task 8: Generator Helpers

**Files:**
- Create: `data/synthetic/generator.py`
- Test: `tests/test_generator.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest
from data.synthetic.generator import (
    _validate_params,
    _even_split,
    _assign_archetypes,
    _attention_check_value,
    CARELESS_ARCHETYPES,
)


def test_validate_params_rejects_zero_respondents():
    with pytest.raises(ValueError):
        _validate_params(n_respondents=0, n_questions=20, scale=(1, 5), contamination_rate=0.25)


def test_validate_params_rejects_too_few_questions():
    with pytest.raises(ValueError):
        _validate_params(n_respondents=10, n_questions=2, scale=(1, 5), contamination_rate=0.25)


def test_validate_params_rejects_bad_scale():
    with pytest.raises(ValueError):
        _validate_params(n_respondents=10, n_questions=20, scale=(5, 1), contamination_rate=0.25)


def test_validate_params_rejects_bad_contamination_rate():
    with pytest.raises(ValueError):
        _validate_params(n_respondents=10, n_questions=20, scale=(1, 5), contamination_rate=1.5)


def test_validate_params_accepts_valid_input():
    _validate_params(n_respondents=10, n_questions=20, scale=(1, 5), contamination_rate=0.25)


def test_even_split_distributes_remainder_to_earlier_groups():
    assert _even_split(total=10, n_groups=4) == [3, 3, 2, 2]
    assert _even_split(total=8, n_groups=4) == [2, 2, 2, 2]
    assert _even_split(total=0, n_groups=4) == [0, 0, 0, 0]


def test_assign_archetypes_honors_contamination_rate():
    rng = np.random.default_rng(6)
    archetypes = _assign_archetypes(n_respondents=100, contamination_rate=0.25, rng=rng)
    assert len(archetypes) == 100
    n_careless = sum(1 for a in archetypes if a != "reliable")
    assert n_careless == 25
    # 25 careless split across 4 archetypes -> [7, 6, 6, 6] (remainder to earlier groups)
    counts = [archetypes.count(archetype) for archetype in CARELESS_ARCHETYPES]
    assert sum(counts) == 25
    assert all(c in (6, 7) for c in counts)


def test_attention_check_value_matches_target_for_non_random_archetypes():
    rng = np.random.default_rng(7)
    assert _attention_check_value("reliable", 1, 5, rng) == 5
    assert _attention_check_value("straightliner", 1, 5, rng) == 5
    assert _attention_check_value("speeder", 1, 5, rng) == 5
    assert _attention_check_value("contradictor", 1, 5, rng) == 5


def test_attention_check_value_is_drawn_from_scale_for_random_responder():
    rng = np.random.default_rng(7)
    value = _attention_check_value("random_responder", 1, 5, rng)
    assert 1 <= value <= 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_generator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.synthetic.generator'`

- [ ] **Step 3: Implement the helper functions in `generator.py`**

```python
import numpy as np

from data.synthetic.archetypes import (
    simulate_reliable,
    simulate_straightliner,
    simulate_speeder,
    simulate_random_responder,
    simulate_contradictor,
)

CARELESS_ARCHETYPES = ["straightliner", "speeder", "random_responder", "contradictor"]

ARCHETYPE_SIMULATORS = {
    "reliable": simulate_reliable,
    "straightliner": simulate_straightliner,
    "speeder": simulate_speeder,
    "random_responder": simulate_random_responder,
    "contradictor": simulate_contradictor,
}


def _validate_params(n_respondents, n_questions, scale, contamination_rate):
    if n_respondents <= 0:
        raise ValueError("n_respondents must be positive")
    if n_questions < 4:
        raise ValueError("n_questions must be at least 4")
    scale_min, scale_max = scale
    if scale_min >= scale_max:
        raise ValueError("scale must have scale_min < scale_max")
    if not (0 <= contamination_rate <= 1):
        raise ValueError("contamination_rate must be between 0 and 1")


def _even_split(total: int, n_groups: int) -> list[int]:
    base = total // n_groups
    remainder = total % n_groups
    return [base + 1 if i < remainder else base for i in range(n_groups)]


def _assign_archetypes(n_respondents: int, contamination_rate: float, rng) -> list[str]:
    n_careless = round(n_respondents * contamination_rate)
    counts = _even_split(n_careless, len(CARELESS_ARCHETYPES))

    archetypes = []
    for archetype, count in zip(CARELESS_ARCHETYPES, counts):
        archetypes.extend([archetype] * count)
    archetypes.extend(["reliable"] * (n_respondents - len(archetypes)))

    rng.shuffle(archetypes)
    return archetypes


def _attention_check_value(archetype: str, scale_min: int, scale_max: int, rng) -> int:
    if archetype == "random_responder":
        return int(rng.integers(scale_min, scale_max + 1))
    return scale_max
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_generator.py -v`
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add data/synthetic/generator.py tests/test_generator.py
git commit -m "feat: add generator parameter validation and archetype assignment"
```

---

## Task 9: `generate_survey_csv` Orchestration

**Files:**
- Modify: `data/synthetic/generator.py`
- Modify: `tests/test_generator.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_generator.py`:

```python
import json
import pandas as pd
from data.synthetic.generator import generate_survey_csv


def test_generate_survey_csv_writes_three_files(tmp_path):
    output_path = tmp_path / "survey_001.csv"
    generate_survey_csv(
        n_respondents=20,
        n_questions=10,
        scale=(1, 5),
        contamination_rate=0.25,
        seed=42,
        output_path=str(output_path),
    )

    labels_path = tmp_path / "survey_001_labels.csv"
    pairs_path = tmp_path / "survey_001_pairs.json"

    assert output_path.exists()
    assert labels_path.exists()
    assert pairs_path.exists()

    survey_df = pd.read_csv(output_path)
    labels_df = pd.read_csv(labels_path)
    assert len(survey_df) == 20
    assert len(labels_df) == 20
    assert set(labels_df.columns) == {"respondent_id", "is_careless", "archetype"}

    with open(pairs_path) as f:
        pairs_data = json.load(f)
    assert pairs_data["n_questions"] == 10
    assert isinstance(pairs_data["pairs"], list)


def test_generate_survey_csv_has_no_ground_truth_columns_in_raw_csv(tmp_path):
    output_path = tmp_path / "survey_001.csv"
    generate_survey_csv(
        n_respondents=10,
        n_questions=8,
        scale=(1, 5),
        contamination_rate=0.25,
        seed=1,
        output_path=str(output_path),
    )
    survey_df = pd.read_csv(output_path)
    for forbidden in ("is_careless", "archetype"):
        assert forbidden not in survey_df.columns


def test_generate_survey_csv_includes_expected_messy_columns(tmp_path):
    output_path = tmp_path / "survey_001.csv"
    generate_survey_csv(
        n_respondents=5,
        n_questions=8,
        scale=(1, 5),
        contamination_rate=0.0,
        seed=1,
        output_path=str(output_path),
    )
    survey_df = pd.read_csv(output_path)
    assert "Response ID" in survey_df.columns
    assert "Start Time" in survey_df.columns
    assert "Timestamp" in survey_df.columns
    assert "Email Address" in survey_df.columns
    assert "grade level" in survey_df.columns
    assert "School Name" in survey_df.columns
    assert any("[Q1]" in col for col in survey_df.columns)
    assert any("[AC1]" in col for col in survey_df.columns)


def test_generate_survey_csv_is_reproducible_with_same_seed(tmp_path):
    path_a = tmp_path / "a.csv"
    path_b = tmp_path / "b.csv"
    generate_survey_csv(
        n_respondents=15, n_questions=10, scale=(1, 5),
        contamination_rate=0.25, seed=99, output_path=str(path_a),
    )
    generate_survey_csv(
        n_respondents=15, n_questions=10, scale=(1, 5),
        contamination_rate=0.25, seed=99, output_path=str(path_b),
    )
    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)
    pd.testing.assert_frame_equal(df_a, df_b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_generator.py -v -k generate_survey_csv`
Expected: FAIL with `ImportError: cannot import name 'generate_survey_csv'`

- [ ] **Step 3: Implement `generate_survey_csv`**

Add to `data/synthetic/generator.py`:

```python
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from data.synthetic.contradiction_pairs import get_contradiction_pairs
from data.synthetic.template import question_header, attention_check_header, render_messy_csv

GRADE_LEVELS = ["9", "10", "11", "12"]
SCHOOL_NAMES = ["Lincoln High", "Washington High", "Roosevelt High"]
BASE_START_TIME = datetime(2024, 3, 1, 8, 0, 0)


def generate_survey_csv(
    n_respondents=200,
    n_questions=20,
    scale=(1, 5),
    contamination_rate=0.25,
    seed=42,
    output_path="data/synthetic/survey_001.csv",
):
    _validate_params(n_respondents, n_questions, scale, contamination_rate)

    rng = np.random.default_rng(seed)
    scale_min, scale_max = scale
    pairs = get_contradiction_pairs(n_questions)
    archetypes = _assign_archetypes(n_respondents, contamination_rate, rng)

    rows = []
    labels = []
    for i, archetype in enumerate(archetypes):
        respondent_id = f"R{i + 1:04d}"
        sim = ARCHETYPE_SIMULATORS[archetype](n_questions, scale, pairs, rng)

        start_time = BASE_START_TIME + timedelta(minutes=i)
        end_time = start_time + timedelta(seconds=sim["duration_seconds"])

        row = {
            "Response ID": respondent_id,
            "Start Time": start_time.isoformat(),
            "Timestamp": end_time.isoformat(),
        }
        for q_idx, answer in enumerate(sim["answers"]):
            row[question_header(q_idx, scale_min, scale_max)] = answer

        ac1_value = _attention_check_value(archetype, scale_min, scale_max, rng)
        ac2_value = _attention_check_value(archetype, scale_min, scale_max, rng)
        row[attention_check_header(1, scale_max)] = ac1_value
        row[attention_check_header(2, scale_max)] = ac2_value

        row["Email Address"] = f"respondent{i + 1}@example.com"
        row["grade level"] = str(rng.choice(GRADE_LEVELS))
        row["School Name"] = str(rng.choice(SCHOOL_NAMES))

        rows.append(row)
        labels.append({
            "respondent_id": respondent_id,
            "is_careless": archetype != "reliable",
            "archetype": archetype,
        })

    survey_df = render_messy_csv(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    survey_df.to_csv(output_path, index=False)

    labels_path = output_path.with_name(output_path.stem + "_labels.csv")
    pd.DataFrame(labels).to_csv(labels_path, index=False)

    pairs_path = output_path.with_name(output_path.stem + "_pairs.json")
    with open(pairs_path, "w") as f:
        json.dump({"n_questions": n_questions, "pairs": pairs}, f, indent=2)

    return output_path
```

Add `import numpy as np` to the top of `data/synthetic/generator.py` if not already present from Task 8.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_generator.py -v`
Expected: `13 passed`

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: all tests across `test_contradiction_pairs.py`, `test_archetypes.py`, `test_template.py`, `test_generator.py` pass (29 total).

- [ ] **Step 6: Commit**

```bash
git add data/synthetic/generator.py tests/test_generator.py
git commit -m "feat: implement generate_survey_csv end-to-end orchestration"
```

---

## Task 10: Manual Verification

**Files:** none (verification only)

- [ ] **Step 1: Generate a default-sized sample survey**

Run:

```bash
python -c "from data.synthetic.generator import generate_survey_csv; print(generate_survey_csv())"
```

Expected: prints a path ending in `data/synthetic/survey_001.csv`; the command exits with no error.

- [ ] **Step 2: Inspect the generated CSV by eye**

Run: `python -c "import pandas as pd; df = pd.read_csv('data/synthetic/survey_001.csv'); print(df.shape); print(list(df.columns))"`

Expected: 200 rows; columns include `Response ID`, `Start Time`, `Timestamp`, several `[Q#]`-tagged question columns, two `[AC#]`-tagged attention-check columns, `Email Address`, `grade level`, `School Name`.

- [ ] **Step 3: Inspect the labels sidecar**

Run: `python -c "import pandas as pd; df = pd.read_csv('data/synthetic/survey_001_labels.csv'); print(df['archetype'].value_counts())"`

Expected: `reliable` count is 150 (75% of 200), and each of the 4 careless archetypes has 12 or 13 (25% of 200 split across 4 groups).

- [ ] **Step 4: Confirm generated files are gitignored**

Run: `git status --short`
Expected: `data/synthetic/survey_001.csv`, `_labels.csv`, and `_pairs.json` do NOT appear (they're covered by the `.gitignore` entry from Task 0).

No commit for this task — it's manual verification only, and the generated files should not be checked in.

---

## Self-Review Notes

- **Spec coverage:** all Phase 1 spec sections are covered — public interface (Task 9), archetype simulation for all 5 types (Tasks 2-6), contradiction pairs (Task 1), messy template (Task 7), attention checks + demographics (Task 9), ground-truth sidecar files (Task 9), error handling via `_validate_params` (Task 8), reproducibility via seeded `rng` (Task 9 test).
- **No placeholders:** every step includes complete, runnable code.
- **Type/name consistency checked:** `simulate_*` functions all take `(n_questions, scale, contradiction_pairs, rng)` and return `{"answers": [...], "duration_seconds": int}` consistently from Task 2 through Task 9's orchestration; `ARCHETYPE_SIMULATORS` dict keys match the archetype name strings used in `CARELESS_ARCHETYPES` and returned by `_assign_archetypes`.
