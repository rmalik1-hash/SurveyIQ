import json
import tempfile
from pathlib import Path

import pandas as pd

from data.synthetic.generator import generate_survey_csv
from src.ingestion.normalize import apply_mapping
from src.features.extract import extract_features
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
