import numpy as np
from data.synthetic.archetypes import simulate_reliable
from data.synthetic.archetypes import simulate_straightliner
from data.synthetic.archetypes import simulate_speeder
from data.synthetic.archetypes import simulate_random_responder
from data.synthetic.archetypes import simulate_contradictor


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


def test_straightliner_gives_identical_answer_to_every_question():
    rng = np.random.default_rng(2)
    result = simulate_straightliner(
        n_questions=15, scale=(1, 5), contradiction_pairs=[], rng=rng
    )
    assert len(set(result["answers"])) == 1
    assert len(result["answers"]) == 15


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


def test_contradictor_violates_contradiction_pairs():
    rng = np.random.default_rng(5)
    result = simulate_contradictor(
        n_questions=4, scale=(1, 5), contradiction_pairs=[(0, 1), (2, 3)], rng=rng
    )
    answers = result["answers"]
    for a_idx, b_idx in [(0, 1), (2, 3)]:
        assert answers[a_idx] == answers[b_idx]
        assert answers[a_idx] + answers[b_idx] != 6


def test_contradictor_answers_within_scale():
    rng = np.random.default_rng(5)
    result = simulate_contradictor(
        n_questions=4, scale=(1, 5), contradiction_pairs=[(0, 1)], rng=rng
    )
    assert all(1 <= a <= 5 for a in result["answers"])
