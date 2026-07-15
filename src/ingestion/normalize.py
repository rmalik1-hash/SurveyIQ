import re
import warnings

import pandas as pd


PII_HEADER_SUBSTRINGS = ["email", "e-mail", "phone", "ssn", "social security", "address"]
PII_NAME_HEADERS = ["first name", "last name", "full name", "student name", "your name", "surname"]
EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


def _is_pii_column(col_name, values):
    lowered = col_name.strip().lower()
    if lowered == "name":
        return True
    if any(sub in lowered for sub in PII_HEADER_SUBSTRINGS):
        return True
    if any(pat in lowered for pat in PII_NAME_HEADERS):
        return True
    for value in values:
        if isinstance(value, str) and EMAIL_RE.search(value):
            return True
    return False


def _compute_duration(start, end):
    if start is None or end is None:
        return None
    start_ts = pd.to_datetime(start, errors="coerce")
    end_ts = pd.to_datetime(end, errors="coerce")
    if pd.isna(start_ts) or pd.isna(end_ts):
        return None
    return int((end_ts - start_ts).total_seconds())


def _extract_responses(row, question_cols):
    return {f"q{i + 1}": int(row[col]) for i, col in enumerate(question_cols)}


def _extract_attention_checks(row, ac_cols, answers):
    checks = {}
    for i, col in enumerate(ac_cols):
        checks[f"ac{i + 1}_given"] = int(row[col])
        checks[f"ac{i + 1}_correct"] = int(answers[col])
    return checks
