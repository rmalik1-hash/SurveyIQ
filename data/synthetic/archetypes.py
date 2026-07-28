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


def simulate_fatiguer(n_questions, scale, contradiction_pairs, rng):
    """A respondent who starts carefully and gives up partway through.

    Deliberately built to be invisible to the other five features: timing stays
    plausible, early contradiction pairs stay consistent, and global
    straightlining lands in the same range as a genuine respondent. The only
    thing that betrays this archetype is *where* the behaviour changes, which is
    what the change-point detector exists to find.
    """
    scale_min, scale_max = scale

    # Give up somewhere in the middle: never so early that there is nothing to
    # compare against, never so late that the careless stretch is trivial.
    switch = int(rng.integers(max(2, int(n_questions * 0.4)),
                              max(3, int(n_questions * 0.75)) + 1))

    baseline = rng.uniform(scale_min, scale_max)
    spread = (scale_max - scale_min) * 0.15
    raw = rng.normal(loc=baseline, scale=spread, size=switch)
    answers = np.clip(np.round(raw), scale_min, scale_max).astype(int).tolist()

    # From the switch onward, the same answer every time.
    tired_value = int(rng.integers(scale_min, scale_max + 1))
    answers.extend([tired_value] * (n_questions - switch))

    # Pairs that sit entirely in the careful stretch are still answered properly.
    for a_idx, b_idx in contradiction_pairs:
        if a_idx < switch and b_idx < switch:
            answers[b_idx] = _mirror(answers[a_idx], scale_min, scale_max)

    # Drawn from exactly the same range as a genuine respondent. Giving up on
    # the questions does not necessarily mean finishing early, and if this
    # distribution were shifted at all the classifier would learn to spot
    # fatigue from the clock instead of from the answers -- making the archetype
    # a test of timing rather than of change detection.
    duration = n_questions * AVG_SECONDS_PER_QUESTION * rng.uniform(0.9, 1.4)
    return {"answers": answers, "duration_seconds": int(round(duration))}


def simulate_straightliner(n_questions, scale, contradiction_pairs, rng):
    scale_min, scale_max = scale
    value = int(rng.integers(scale_min, scale_max + 1))
    answers = [value] * n_questions
    duration = n_questions * AVG_SECONDS_PER_QUESTION * rng.uniform(0.8, 1.3)
    return {"answers": answers, "duration_seconds": int(round(duration))}


def simulate_speeder(n_questions, scale, contradiction_pairs, rng):
    scale_min, scale_max = scale
    baseline = rng.uniform(scale_min, scale_max)
    spread = (scale_max - scale_min) * 0.15
    raw = rng.normal(loc=baseline, scale=spread, size=n_questions)
    answers = np.clip(np.round(raw), scale_min, scale_max).astype(int).tolist()

    for a_idx, b_idx in contradiction_pairs:
        answers[b_idx] = _mirror(answers[a_idx], scale_min, scale_max)

    duration = n_questions * AVG_SECONDS_PER_QUESTION * rng.uniform(0.1, 0.3)
    return {"answers": answers, "duration_seconds": int(round(duration))}


def simulate_random_responder(n_questions, scale, contradiction_pairs, rng):
    scale_min, scale_max = scale
    answers = rng.integers(scale_min, scale_max + 1, size=n_questions).tolist()
    duration = n_questions * AVG_SECONDS_PER_QUESTION * rng.uniform(0.5, 1.5)
    return {"answers": answers, "duration_seconds": int(round(duration))}


def simulate_contradictor(n_questions, scale, contradiction_pairs, rng):
    scale_min, scale_max = scale
    baseline = rng.uniform(scale_min, scale_max)
    spread = (scale_max - scale_min) * 0.15
    raw = rng.normal(loc=baseline, scale=spread, size=n_questions)
    answers = np.clip(np.round(raw), scale_min, scale_max).astype(int).tolist()

    midpoint = (scale_min + scale_max) / 2
    non_midpoint_values = [v for v in range(scale_min, scale_max + 1) if v != midpoint]

    for a_idx, b_idx in contradiction_pairs:
        value = int(rng.choice(non_midpoint_values))
        answers[a_idx] = value
        answers[b_idx] = value  # same value instead of mirrored -> contradiction

    duration = n_questions * AVG_SECONDS_PER_QUESTION * rng.uniform(0.9, 1.4)
    return {"answers": answers, "duration_seconds": int(round(duration))}
