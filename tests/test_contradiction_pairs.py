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
