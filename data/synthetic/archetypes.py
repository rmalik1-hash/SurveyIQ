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


def simulate_straightliner(n_questions, scale, contradiction_pairs, rng):
    scale_min, scale_max = scale
    value = int(rng.integers(scale_min, scale_max + 1))
    answers = [value] * n_questions
    duration = n_questions * AVG_SECONDS_PER_QUESTION * rng.uniform(0.8, 1.3)
    return {"answers": answers, "duration_seconds": int(round(duration))}
