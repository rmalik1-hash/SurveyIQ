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


def test_generate_survey_csv_sidecars_are_reproducible_with_same_seed(tmp_path):
    path_a = tmp_path / "a.csv"
    path_b = tmp_path / "b.csv"
    for path in (path_a, path_b):
        generate_survey_csv(
            n_respondents=15, n_questions=10, scale=(1, 5),
            contamination_rate=0.25, seed=99, output_path=str(path),
        )

    labels_a = pd.read_csv(path_a.with_name("a_labels.csv"))
    labels_b = pd.read_csv(path_b.with_name("b_labels.csv"))
    pd.testing.assert_frame_equal(labels_a, labels_b)

    with open(path_a.with_name("a_pairs.json")) as f:
        pairs_a = json.load(f)
    with open(path_b.with_name("b_pairs.json")) as f:
        pairs_b = json.load(f)
    assert pairs_a == pairs_b


@pytest.mark.parametrize(
    "n_respondents, contamination_rate, expected_careless",
    [
        (100, 0.25, 25),
        (200, 0.25, 50),
        (10, 0.25, 2),   # round(2.5) -> 2 (banker's rounding)
        (33, 0.30, 10),  # round(9.9) -> 10
        (50, 0.0, 0),
        (50, 1.0, 50),
    ],
)
def test_generate_survey_csv_honors_contamination_rate_across_sizes(
    tmp_path, n_respondents, contamination_rate, expected_careless
):
    output_path = tmp_path / "survey_001.csv"
    generate_survey_csv(
        n_respondents=n_respondents,
        n_questions=10,
        scale=(1, 5),
        contamination_rate=contamination_rate,
        seed=3,
        output_path=str(output_path),
    )
    labels_df = pd.read_csv(output_path.with_name("survey_001_labels.csv"))
    assert len(labels_df) == n_respondents
    assert int(labels_df["is_careless"].sum()) == expected_careless
