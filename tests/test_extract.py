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
