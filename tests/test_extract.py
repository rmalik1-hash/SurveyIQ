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


from src.features.extract import attention_check_pass_rate


def test_attention_all_pass():
    ac = {"ac1_given": 5, "ac1_correct": 5, "ac2_given": 5, "ac2_correct": 5}
    assert attention_check_pass_rate(ac) == 1.0


def test_attention_half_pass():
    ac = {"ac1_given": 5, "ac1_correct": 5, "ac2_given": 1, "ac2_correct": 5}
    assert attention_check_pass_rate(ac) == 0.5


def test_attention_no_checks_is_one():
    assert attention_check_pass_rate({}) == 1.0


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


def test_extract_features_empty_list_returns_empty_framed_columns():
    df = extract_features([])
    assert len(df) == 0
    assert list(df.columns) == [
        "respondent_id", "completion_time_ratio", "straightlining_score",
        "response_variance", "contradiction_score", "attention_check_pass_rate",
        "extreme_response_rate",
    ]
