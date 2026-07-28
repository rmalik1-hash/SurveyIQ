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


from src.features.extract import straightlining_score
from data.synthetic.archetypes import simulate_fatiguer


def test_fatiguer_answers_uniformly_only_after_the_switch():
    rng = np.random.default_rng(9)
    result = simulate_fatiguer(n_questions=20, scale=(1, 5), contradiction_pairs=[], rng=rng)
    answers = result["answers"]
    assert len(answers) == 20
    # the tail collapses to a single repeated value
    tail = answers[-6:]
    assert len(set(tail)) == 1
    # the opening section does not
    assert len(set(answers[:6])) > 1


def test_fatiguer_respects_pairs_that_fall_before_the_switch():
    rng = np.random.default_rng(9)
    result = simulate_fatiguer(
        n_questions=20, scale=(1, 5), contradiction_pairs=[(0, 1), (2, 3)], rng=rng
    )
    answers = result["answers"]
    # early pairs stay reverse-coded, so contradiction_score will not catch this archetype
    assert answers[0] + answers[1] == 6
    assert answers[2] + answers[3] == 6


def test_fatiguer_timing_is_plausible_so_speed_does_not_give_it_away():
    rng = np.random.default_rng(9)
    result = simulate_fatiguer(n_questions=20, scale=(1, 5), contradiction_pairs=[], rng=rng)
    expected = 20 * 8
    assert result["duration_seconds"] > 0.5 * expected


def test_fatiguer_is_invisible_to_global_straightlining():
    """The point of this archetype: existing features cannot see it.

    Reliable respondents already average ~0.55 global straightlining, so a
    fatiguer sitting in the same range is genuinely indistinguishable without a
    change-point detector.
    """
    rng = np.random.default_rng(3)
    scores = []
    for _ in range(30):
        result = simulate_fatiguer(n_questions=20, scale=(1, 5), contradiction_pairs=[], rng=rng)
        responses = {f"q{i + 1}": v for i, v in enumerate(result["answers"])}
        scores.append(straightlining_score(responses))
    assert np.mean(scores) < 0.75
