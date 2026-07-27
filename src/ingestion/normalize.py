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
    responses = {}
    for i, col in enumerate(question_cols):
        value = row[col]
        try:
            responses[f"q{i + 1}"] = int(value)
        except (ValueError, TypeError):
            # Surface a message an administrator can act on, naming the column
            # and what to do, rather than a bare pandas error. Blanks are not
            # silently dropped: that would change the denominator behind the
            # timing and straightlining scores without telling anyone.
            raise ValueError(
                f"Column {col!r} contains a non-numeric answer: {value!r}. "
                f"SurveyIQ only scores numeric scale questions (e.g. 1-5). "
                f"Map this column as 'demographic' or 'ignore', or clean the "
                f"blank/invalid cells."
            )
    return responses


def _extract_attention_checks(row, ac_cols, answers):
    checks = {}
    for i, col in enumerate(ac_cols):
        checks[f"ac{i + 1}_given"] = int(row[col])
        checks[f"ac{i + 1}_correct"] = int(answers[col])
    return checks


VALID_ROLES = {
    "respondent_id", "start_time", "end_time", "question",
    "attention_check", "demographic", "ignore",
}


def _columns_by_role(mapping):
    grouped = {role: [] for role in VALID_ROLES}
    for col, role in mapping["columns"].items():
        if role in grouped:
            grouped[role].append(col)
    return grouped


def _validate_mapping(raw_df, mapping):
    columns = mapping.get("columns", {})
    for col, role in columns.items():
        if col not in raw_df.columns:
            raise ValueError(f"mapping references column not in DataFrame: {col!r}")
        if role not in VALID_ROLES:
            raise ValueError(f"unknown role {role!r} for column {col!r}")
    roles = list(columns.values())
    if roles.count("respondent_id") != 1:
        raise ValueError("mapping must have exactly one respondent_id column")
    if "question" not in roles:
        raise ValueError("mapping must have at least one question column")
    scale = mapping.get("scale")
    if not scale or len(scale) != 2 or scale[0] >= scale[1]:
        raise ValueError("mapping needs a valid scale [min, max] with min < max")
    answers = mapping.get("attention_check_answers", {})
    for col, role in columns.items():
        if role == "attention_check" and col not in answers:
            raise ValueError(f"attention_check column has no correct answer: {col!r}")


def _extract_demographics(row, demo_cols):
    return {col: row[col] for col in demo_cols}


def apply_mapping(raw_df, mapping):
    _validate_mapping(raw_df, mapping)
    grouped = _columns_by_role(mapping)
    scale_min, scale_max = mapping["scale"]
    answers = mapping.get("attention_check_answers", {})

    id_col = grouped["respondent_id"][0]
    start_col = grouped["start_time"][0] if grouped["start_time"] else None
    end_col = grouped["end_time"][0] if grouped["end_time"] else None
    question_cols = grouped["question"]
    ac_cols = grouped["attention_check"]

    if _is_pii_column(id_col, raw_df[id_col].tolist()):
        warnings.warn(f"respondent_id column {id_col!r} looks like it may contain PII")

    demo_cols = []
    for col in grouped["demographic"]:
        if _is_pii_column(col, raw_df[col].tolist()):
            warnings.warn(f"dropping demographic column that looks like PII: {col!r}")
        else:
            demo_cols.append(col)

    respondents = []
    for _, row in raw_df.iterrows():
        start = row[start_col] if start_col else None
        end = row[end_col] if end_col else None
        respondents.append({
            "respondent_id": str(row[id_col]),
            "duration_seconds": _compute_duration(start, end),
            "responses": _extract_responses(row, question_cols),
            "attention_checks": _extract_attention_checks(row, ac_cols, answers),
            "demographics": _extract_demographics(row, demo_cols),
            "scale_min": scale_min,
            "scale_max": scale_max,
        })
    return respondents
