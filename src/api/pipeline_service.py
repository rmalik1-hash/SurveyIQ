import json
import statistics
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

from data.synthetic.generator import generate_survey_csv
from src.ingestion.normalize import apply_mapping
from src.features.extract import extract_features, pair_contradicts
from src.models.classifier import train, predict, FEATURE_NAMES


def _build_training_mapping(columns):
    col_roles = {}
    ac_answers = {}
    for c in columns:
        if "[Q" in c:
            col_roles[c] = "question"
        elif "[AC" in c:
            col_roles[c] = "attention_check"
            ac_answers[c] = 5
        elif c == "Response ID":
            col_roles[c] = "respondent_id"
        elif c == "Start Time":
            col_roles[c] = "start_time"
        elif c == "Timestamp":
            col_roles[c] = "end_time"
        elif c == "Email Address":
            col_roles[c] = "ignore"
        else:
            col_roles[c] = "demographic"
    return {"columns": col_roles, "scale": [1, 5], "attention_check_answers": ac_answers}


def _pairs_to_qkeys(mapping):
    question_cols = [c for c, role in mapping["columns"].items() if role == "question"]
    col_to_qkey = {c: f"q{i + 1}" for i, c in enumerate(question_cols)}
    pairs = mapping.get("contradiction_pairs") or []
    qkey_pairs = []
    for a_col, b_col in pairs:
        if a_col not in col_to_qkey or b_col not in col_to_qkey:
            raise ValueError(
                f"contradiction pair references a non-question column: {(a_col, b_col)}"
            )
        qkey_pairs.append((col_to_qkey[a_col], col_to_qkey[b_col]))
    return qkey_pairs


# Survey shapes the startup model trains across, as multipliers on the base
# question count and absolute contamination rates. Real uploads never match one
# fixed shape, and a model fitted to a single configuration puts far more
# respondents in the borderline zone when the shape differs -- which shows up as
# flags the explanation logic can only describe vaguely.
_TRAINING_SHAPES = [
    (0.65, 0.20),
    (1.00, 0.30),
    (1.35, 0.40),
]


def _training_frame(n_respondents, n_questions, contamination_rate, seed, tmp, tag):
    """Generate one labelled survey and return its (features, labels)."""
    out = Path(tmp) / f"train_{tag}.csv"
    generate_survey_csv(
        n_respondents=n_respondents, n_questions=n_questions, scale=(1, 5),
        contamination_rate=contamination_rate, seed=seed, output_path=str(out),
    )
    df = pd.read_csv(out)
    labels_df = pd.read_csv(Path(tmp) / f"train_{tag}_labels.csv")
    with open(Path(tmp) / f"train_{tag}_pairs.json") as pairs_file:
        pairs_idx = json.load(pairs_file)["pairs"]

    mapping = _build_training_mapping(list(df.columns))
    respondents = apply_mapping(df, mapping)
    pair_keys = [(f"q{a + 1}", f"q{b + 1}") for a, b in pairs_idx]
    features = extract_features(respondents, contradiction_pairs=pair_keys)

    merged = features.merge(labels_df, on="respondent_id")
    # Respondent ids repeat across surveys, so make them unique before stacking.
    merged["respondent_id"] = merged["respondent_id"].astype(str) + f"_{tag}"
    return merged[["respondent_id"] + FEATURE_NAMES], merged["is_careless"].astype(int)


def train_startup_model(n_respondents=300, n_questions=16, seed=42):
    """Fit the model the API scores uploads with.

    Trains across several survey lengths and contamination rates rather than one,
    so the thresholds still fit when a school uploads something shaped
    differently from any single training run.
    """
    frames, labels = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for index, (length_factor, contamination) in enumerate(_TRAINING_SHAPES):
            questions = max(4, round(n_questions * length_factor))
            feat, y = _training_frame(
                n_respondents=n_respondents,
                n_questions=questions,
                contamination_rate=contamination,
                seed=seed + index,
                tmp=tmp,
                tag=f"s{index}",
            )
            frames.append(feat)
            labels.append(y)

    combined = pd.concat(frames, ignore_index=True)
    y = pd.concat(labels, ignore_index=True).tolist()
    return train(combined, y)


def _question_stats(respondents, mapping, qkey_pairs):
    """Per-question trouble counts: how many respondents each pair/check caught."""
    stats = []

    pair_cols = mapping.get("contradiction_pairs") or []
    for (a_col, b_col), (a_key, b_key) in zip(pair_cols, qkey_pairs):
        affected = sum(
            1 for r in respondents
            if pair_contradicts(r["responses"], r["scale_min"], r["scale_max"], a_key, b_key)
        )
        stats.append({
            "label": f"{a_col} / {b_col}",
            "type": "contradiction_pair",
            "affected": affected,
        })

    ac_cols = [c for c, role in mapping["columns"].items() if role == "attention_check"]
    for i, col in enumerate(ac_cols, start=1):
        given_key, correct_key = f"ac{i}_given", f"ac{i}_correct"
        affected = sum(
            1 for r in respondents
            if r["attention_checks"].get(given_key) != r["attention_checks"].get(correct_key)
        )
        stats.append({"label": col, "type": "attention_check", "affected": affected})

    stats.sort(key=lambda s: s["affected"], reverse=True)
    return stats


def _question_summary(respondents, mapping, flagged_ids):
    """Per-question answer distributions, for the question-level report.

    Counts are taken from the normalized responses, so they reflect exactly what
    was scored. Trustworthy counts are reported separately so an administrator
    can see whether a question looks different once careless responses are set
    aside.
    """
    question_cols = [c for c, role in mapping["columns"].items() if role == "question"]
    scale_min, scale_max = mapping["scale"]
    scale_points = [str(v) for v in range(scale_min, scale_max + 1)]

    summary = []
    for index, column in enumerate(question_cols):
        qkey = f"q{index + 1}"
        counts = {point: 0 for point in scale_points}
        counts_trustworthy = {point: 0 for point in scale_points}
        values = []

        for respondent in respondents:
            if qkey not in respondent["responses"]:
                continue
            answer = str(respondent["responses"][qkey])
            values.append(respondent["responses"][qkey])
            if answer in counts:
                counts[answer] += 1
                if respondent["respondent_id"] not in flagged_ids:
                    counts_trustworthy[answer] += 1

        summary.append({
            "label": column,
            "position": index + 1,
            "counts": counts,
            "counts_trustworthy": counts_trustworthy,
            "mean": round(statistics.fmean(values), 2) if values else None,
            "median": statistics.median(values) if values else None,
            "mode": _mode(values),
            "concern": _question_concern(values, scale_min, scale_max),
        })
    return summary


def _mode(values):
    """Most frequent answer. Ties resolve to the lowest, so the result is stable."""
    if not values:
        return None
    counts = Counter(values)
    highest = max(counts.values())
    return min(v for v, count in counts.items() if count == highest)


def _question_concern(values, scale_min, scale_max):
    """A plain-English note about a question, or None when nothing stands out."""
    if not values:
        return None
    distinct = set(values)
    if len(distinct) == 1:
        return (
            "Every respondent gave the same answer, so this question separates "
            "nobody. Check the wording, or whether it belongs on this scale."
        )
    at_ends = sum(1 for v in values if v in (scale_min, scale_max)) / len(values)
    if at_ends > 0.9:
        return (
            "Almost every answer sits at one end of the scale, so the middle "
            "options are doing no work here."
        )
    return None


def analyze(raw_df, mapping, model):
    respondents = apply_mapping(raw_df, mapping)
    qkey_pairs = _pairs_to_qkeys(mapping)
    features = extract_features(respondents, contradiction_pairs=qkey_pairs)
    preds = predict(model, features)

    total = len(preds)
    flagged = int((preds["reliability_score"] < 0.5).sum())
    reliable = total - flagged
    overall = round(100.0 * reliable / total, 1) if total else 0.0

    flagged_ids = {
        str(row["respondent_id"])
        for _, row in preds.iterrows()
        if row["reliability_score"] < 0.5
    }

    respondents_out = [
        {
            "respondent_id": row["respondent_id"],
            "reliability_score": round(float(row["reliability_score"]), 4),
            "flag_reason": row["flag_reason"],
        }
        for _, row in preds.iterrows()
    ]
    question_summary = _question_summary(respondents, mapping, flagged_ids)

    # Average answer across every question, so trend tracking can show whether
    # responses themselves drift over time, not just how many were flagged.
    all_answers = [v for r in respondents for v in r["responses"].values()]
    mean_response = round(statistics.fmean(all_answers), 2) if all_answers else 0.0

    return {
        "summary": {
            "total": total, "flagged": flagged, "reliable": reliable,
            "overall_quality_pct": overall,
            "mean_response": mean_response,
        },
        "respondents": respondents_out,
        "question_stats": _question_stats(respondents, mapping, qkey_pairs),
        "question_summary": question_summary,
    }
