import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data.synthetic.archetypes import (
    simulate_reliable,
    simulate_straightliner,
    simulate_speeder,
    simulate_random_responder,
    simulate_contradictor,
    simulate_fatiguer,
)
from data.synthetic.contradiction_pairs import get_contradiction_pairs
from data.synthetic.template import question_header, attention_check_header, render_messy_csv

CARELESS_ARCHETYPES = [
    "straightliner", "speeder", "random_responder", "contradictor", "fatiguer",
]

ARCHETYPE_SIMULATORS = {
    "reliable": simulate_reliable,
    "straightliner": simulate_straightliner,
    "speeder": simulate_speeder,
    "random_responder": simulate_random_responder,
    "contradictor": simulate_contradictor,
    "fatiguer": simulate_fatiguer,
}

GRADE_LEVELS = ["9", "10", "11", "12"]
SCHOOL_NAMES = ["Lincoln High", "Washington High", "Roosevelt High"]
BASE_START_TIME = datetime(2024, 3, 1, 8, 0, 0)


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


def _apply_question_shifts(answers, shifts, pairs, scale_min, scale_max):
    """Move answers up or down the scale, leaving reverse-coded pairs untouched."""
    paired = {index for pair in pairs for index in pair}
    shifted = list(answers)
    for index, shift in enumerate(shifts):
        if index in paired or not shift:
            continue
        moved = round(shifted[index] + shift)
        shifted[index] = int(max(scale_min, min(scale_max, moved)))
    return shifted


def _attention_check_value(archetype: str, scale_min: int, scale_max: int, rng) -> int:
    if archetype == "random_responder":
        return int(rng.integers(scale_min, scale_max + 1))
    return scale_max


def generate_survey_csv(
    n_respondents=200,
    n_questions=20,
    scale=(1, 5),
    contamination_rate=0.25,
    seed=42,
    output_path="data/synthetic/survey_001.csv",
    question_shifts=None,
):
    """Write a synthetic survey plus its ground-truth sidecars.

    `question_shifts` nudges individual questions up or down the scale for
    genuine respondents, so a series of surveys can show opinion moving over
    time -- "students liked this class more each term" -- rather than every
    survey looking identical. Questions in a contradiction pair are left alone,
    since shifting one half of a reverse-coded pair would make honest
    respondents look like they were contradicting themselves.
    """
    _validate_params(n_respondents, n_questions, scale, contamination_rate)
    if question_shifts is not None and len(question_shifts) != n_questions:
        raise ValueError("question_shifts must have one entry per question")

    rng = np.random.default_rng(seed)
    scale_min, scale_max = scale
    pairs = get_contradiction_pairs(n_questions)
    archetypes = _assign_archetypes(n_respondents, contamination_rate, rng)

    rows = []
    labels = []
    for i, archetype in enumerate(archetypes):
        respondent_id = f"R{i + 1:04d}"
        sim = ARCHETYPE_SIMULATORS[archetype](n_questions, scale, pairs, rng)

        # Opinion drift only applies to genuine respondents: a straightliner is
        # not expressing a view that could shift between terms.
        if question_shifts is not None and archetype == "reliable":
            sim["answers"] = _apply_question_shifts(
                sim["answers"], question_shifts, pairs, scale_min, scale_max
            )

        start_time = BASE_START_TIME + timedelta(minutes=i)
        end_time = start_time + timedelta(seconds=sim["duration_seconds"])

        row = {
            "Response ID": respondent_id,
            "Start Time": start_time.isoformat(),
            "Timestamp": end_time.isoformat(),
        }
        for q_idx, answer in enumerate(sim["answers"]):
            row[question_header(q_idx, scale_min, scale_max)] = answer

        ac1_value = _attention_check_value(archetype, scale_min, scale_max, rng)
        ac2_value = _attention_check_value(archetype, scale_min, scale_max, rng)
        row[attention_check_header(1, scale_max)] = ac1_value
        row[attention_check_header(2, scale_max)] = ac2_value

        row["Email Address"] = f"respondent{i + 1}@example.com"
        row["grade level"] = str(rng.choice(GRADE_LEVELS))
        row["School Name"] = str(rng.choice(SCHOOL_NAMES))

        rows.append(row)
        labels.append({
            "respondent_id": respondent_id,
            "is_careless": archetype != "reliable",
            "archetype": archetype,
        })

    survey_df = render_messy_csv(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    survey_df.to_csv(output_path, index=False)

    labels_path = output_path.with_name(output_path.stem + "_labels.csv")
    pd.DataFrame(labels).to_csv(labels_path, index=False)

    pairs_path = output_path.with_name(output_path.stem + "_pairs.json")
    with open(pairs_path, "w") as f:
        json.dump({"n_questions": n_questions, "pairs": pairs}, f, indent=2)

    return output_path
