"""Aggregate-only history of past analyses, for quality trends over time.

Deliberately narrow. A school tracking whether its data quality is improving
needs totals, not responses -- so this module stores totals and nothing else.
`record_run` copies a fixed allow-list of numeric fields off the summary and
discards everything else, which means respondent ids, answers, demographics and
flag reasons cannot reach disk even if a caller passes them in.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_HISTORY_PATH = Path(
    os.environ.get("SURVEYIQ_HISTORY_PATH", "data/history.json")
)

# The complete set of keys ever written. Enforced by test, not just documented.
ALLOWED_FIELDS = {
    "recorded_at",
    "survey_label",
    "total",
    "flagged",
    "reliable",
    "overall_quality_pct",
    "mean_response",
    "question_means",
}

_NUMERIC_FIELDS = [
    "total", "flagged", "reliable", "overall_quality_pct", "mean_response",
]


def _read(path):
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        # A missing or damaged history file must never break an analysis --
        # trends are a convenience, scoring is the product.
        return []
    return data if isinstance(data, list) else []


def _write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write via a temp file so an interrupted write cannot corrupt the store.
    fd, temp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(rows, handle, indent=2)
        os.replace(temp_path, path)
    except Exception:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise


def record_run(summary, survey_label, path=None, question_means=None, recorded_at=None):
    """Append one run's aggregate totals. Returns the row that was stored.

    `question_means` carries the average answer per question so trends can show
    opinion moving, not just how many responses were flagged. It is a mapping of
    question text to a number -- no respondent is identifiable from an average.

    `recorded_at` lets a caller supply the date the survey was actually run,
    which is rarely the day it gets uploaded.
    """
    label = (survey_label or "").strip()
    if not label:
        raise ValueError("survey_label is required so trends can be grouped")

    path = Path(path or DEFAULT_HISTORY_PATH)
    row = {
        "recorded_at": (recorded_at or datetime.now(timezone.utc)
                        .isoformat(timespec="seconds")),
        "survey_label": label,
    }
    for field in _NUMERIC_FIELDS:
        value = summary.get(field)
        row[field] = float(value) if isinstance(value, float) else int(value or 0)

    # Question averages only -- keys are question text, values are numbers.
    row["question_means"] = {
        str(label): round(float(value), 3)
        for label, value in (question_means or {}).items()
        if isinstance(value, (int, float))
    }

    assert set(row) == ALLOWED_FIELDS, "history row must stay aggregate-only"

    rows = _read(path)
    rows.append(row)
    _write(path, rows)
    return row


def list_runs(survey_label=None, path=None):
    """Past runs, oldest first, optionally narrowed to one survey."""
    rows = _read(Path(path or DEFAULT_HISTORY_PATH))
    rows = [r for r in rows if set(r) == ALLOWED_FIELDS]
    if survey_label:
        rows = [r for r in rows if r["survey_label"] == survey_label]
    return rows


def clear_runs(path=None):
    """Delete all stored history. Returns how many runs were removed."""
    path = Path(path or DEFAULT_HISTORY_PATH)
    removed = len(_read(path))
    _write(path, [])
    return removed
