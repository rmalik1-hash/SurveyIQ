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
