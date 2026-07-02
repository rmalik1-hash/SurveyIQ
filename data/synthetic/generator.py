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
