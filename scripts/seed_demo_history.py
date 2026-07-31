"""Generate a series of surveys and analyse them, so trends have real data.

Produces five terms of the same survey where opinion genuinely moves: one
question climbs, another falls, the rest hold steady. Each is analysed through
the running API exactly as an uploaded file would be, so nothing here fakes a
result -- only the survey dates are supplied, because a school analysing last
autumn's survey should see it plotted at autumn.

Usage (with the API running on port 8000):

    python scripts/seed_demo_history.py
    python scripts/seed_demo_history.py --api http://127.0.0.1:8000 --clear
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.synthetic.generator import generate_survey_csv  # noqa: E402
from src.api.pipeline_service import _build_training_mapping  # noqa: E402

SURVEY_LABEL = "Student experience survey"
N_QUESTIONS = 12

# One entry per term: the date it ran, and how far each question has drifted
# from where it started. Question 3 improves steadily, question 7 declines,
# everything else stays put -- so a trend chart has something honest to show.
TERMS = [
    ("2025-09-15", 0.0, 0.0),
    ("2025-12-15", 0.4, -0.3),
    ("2026-03-15", 0.9, -0.7),
    ("2026-06-15", 1.4, -1.1),
    ("2026-09-15", 1.8, -1.5),
]

RISING_QUESTION = 2    # zero-based: question 3
FALLING_QUESTION = 6   # zero-based: question 7


def shifts_for(rise, fall):
    shifts = [0.0] * N_QUESTIONS
    shifts[RISING_QUESTION] = rise
    shifts[FALLING_QUESTION] = fall
    return shifts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--clear", action="store_true",
                        help="clear existing history first")
    args = parser.parse_args()

    try:
        requests.get(f"{args.api}/health", timeout=5).raise_for_status()
    except Exception as exc:
        raise SystemExit(f"API not reachable at {args.api}: {exc}")

    if args.clear:
        removed = requests.delete(f"{args.api}/history", timeout=30).json()["removed"]
        print(f"cleared {removed} previous run(s)")

    with tempfile.TemporaryDirectory() as tmp:
        for index, (date, rise, fall) in enumerate(TERMS):
            out = Path(tmp) / f"term_{index}.csv"
            generate_survey_csv(
                n_respondents=180,
                n_questions=N_QUESTIONS,
                scale=(1, 5),
                # Data quality improves as the school gets better at running the
                # survey, which is its own trend worth seeing.
                contamination_rate=round(0.34 - index * 0.045, 3),
                seed=100 + index,
                output_path=str(out),
                question_shifts=shifts_for(rise, fall),
            )

            columns = list(__import__("pandas").read_csv(out, nrows=0).columns)
            mapping = _build_training_mapping(columns)

            with open(out, "rb") as handle:
                response = requests.post(
                    f"{args.api}/analyze",
                    files={"file": (out.name, handle, "text/csv")},
                    data={
                        "mapping": json.dumps(mapping),
                        "survey_label": SURVEY_LABEL,
                        "survey_date": f"{date}T09:00:00+00:00",
                    },
                    timeout=120,
                )
            response.raise_for_status()
            summary = response.json()["summary"]
            print(
                f"{date}  quality {summary['overall_quality_pct']:>5}%  "
                f"flagged {summary['flagged']:>3}/{summary['total']}"
            )

    runs = requests.get(f"{args.api}/history", timeout=30).json()["runs"]
    print(f"\n{len(runs)} run(s) now in history for trend charts")


if __name__ == "__main__":
    main()
