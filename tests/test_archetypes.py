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
