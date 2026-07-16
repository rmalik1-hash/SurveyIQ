import json
import tempfile
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


def train_startup_model(n_respondents=300, n_questions=16, seed=42):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "train.csv"
        generate_survey_csv(
            n_respondents=n_respondents, n_questions=n_questions, scale=(1, 5),
            contamination_rate=0.3, seed=seed, output_path=str(out),
        )
        df = pd.read_csv(out)
        labels_df = pd.read_csv(Path(tmp) / "train_labels.csv")
        with open(Path(tmp) / "train_pairs.json") as pairs_file:
            pairs_idx = json.load(pairs_file)["pairs"]

        mapping = _build_training_mapping(list(df.columns))
        respondents = apply_mapping(df, mapping)
        pair_keys = [(f"q{a + 1}", f"q{b + 1}") for a, b in pairs_idx]
        features = extract_features(respondents, contradiction_pairs=pair_keys)

        merged = features.merge(labels_df, on="respondent_id")
        y = merged["is_careless"].astype(int).tolist()
        feat = merged[["respondent_id"] + FEATURE_NAMES]
        return train(feat, y)


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


def analyze(raw_df, mapping, model):
    respondents = apply_mapping(raw_df, mapping)
    qkey_pairs = _pairs_to_qkeys(mapping)
    features = extract_features(respondents, contradiction_pairs=qkey_pairs)
    preds = predict(model, features)

    total = len(preds)
    flagged = int((preds["reliability_score"] < 0.5).sum())
    reliable = total - flagged
    overall = round(100.0 * reliable / total, 1) if total else 0.0

    respondents_out = [
        {
            "respondent_id": row["respondent_id"],
            "reliability_score": round(float(row["reliability_score"]), 4),
            "flag_reason": row["flag_reason"],
        }
        for _, row in preds.iterrows()
    ]
    return {
        "summary": {
            "total": total, "flagged": flagged, "reliable": reliable,
            "overall_quality_pct": overall,
        },
        "respondents": respondents_out,
        "question_stats": _question_stats(respondents, mapping, qkey_pairs),
    }
