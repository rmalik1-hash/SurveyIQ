import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.features.extract import FEATURE_COLUMNS, AVG_SECONDS_PER_QUESTION

FEATURE_NAMES = [c for c in FEATURE_COLUMNS if c != "respondent_id"]


def _feature_matrix(feature_df):
    missing = [c for c in FEATURE_NAMES if c not in feature_df.columns]
    if missing:
        raise ValueError(f"feature_df missing columns: {missing}")
    return feature_df[FEATURE_NAMES].to_numpy(dtype=float)


def train(feature_df, labels, max_depth=4, random_state=42):
    if len(feature_df) == 0:
        raise ValueError("feature_df is empty")
    if len(labels) != len(feature_df):
        raise ValueError("labels length does not match feature_df row count")
    X = _feature_matrix(feature_df)
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(X, list(labels))
    return model


# Which direction of each feature indicates careless responding. True means a
# HIGH value is the suspicious one; False means a LOW value is.
_HIGH_IS_SUSPICIOUS = {
    "completion_time_ratio": False,      # too fast is suspicious
    "straightlining_score": True,        # same answer over and over
    "response_variance": True,           # wildly erratic answers
    "contradiction_score": True,         # inconsistent on paired questions
    "attention_check_pass_rate": False,  # failing checks is suspicious
    "extreme_response_rate": True,       # only ever picking the endpoints
}


def _pct(value):
    return f"{round(value * 100)}%"


def _describe_split(feature_name, value, went_left, is_nan):
    """Describe one split in plain English.

    Returns (text, suspicious). `suspicious` marks whether this split pushed the
    respondent toward being flagged, so the caller can lead with the clauses
    that actually explain the flag. Every split still gets text, which is what
    guarantees a flagged respondent is never left without a reason.
    """
    if is_nan:
        return "had no timing data recorded", False

    suspicious = went_left != _HIGH_IS_SUSPICIOUS[feature_name]

    if feature_name == "completion_time_ratio":
        seconds = value * AVG_SECONDS_PER_QUESTION
        if suspicious:
            return (
                f"spent about {seconds:.0f}s per question, far less than a careful "
                f"reader needs"
            ), True
        return f"spent about {seconds:.0f}s per question, a plausible pace", False

    if feature_name == "straightlining_score":
        if suspicious:
            return f"gave the same answer to {_pct(value)} of questions", True
        return "varied their answers across the survey", False

    if feature_name == "response_variance":
        if suspicious:
            return "answers swung erratically with no consistent pattern", True
        return "answers stayed within a narrow range", False

    if feature_name == "contradiction_score":
        if suspicious:
            return f"contradicted themselves on {_pct(value)} of paired questions", True
        return "stayed consistent on the paired questions", False

    if feature_name == "attention_check_pass_rate":
        if suspicious:
            return f"passed only {_pct(value)} of the attention checks", True
        return "passed the attention checks", False

    # extreme_response_rate
    if suspicious:
        return f"picked the highest or lowest option on {_pct(value)} of questions", True
    return "used the middle of the scale, not just the endpoints", False


def _sentence(parts):
    joined = " and ".join(parts)
    return joined[0].upper() + joined[1:] + "."


def _describe_path(model, x_row):
    """Translate the tree's actual decision path into a plain-English reason.

    Leads with the clauses that pushed toward "flagged" so administrators read
    signal rather than boilerplate. If a path somehow contains no suspicious
    split, the deepest (most specific) clause is used instead -- a flagged
    respondent must never come back without a reason.
    """
    x_row = np.asarray(x_row, dtype=float)
    dp = model.decision_path(x_row.reshape(1, -1))
    node_ids = dp.indices[dp.indptr[0]:dp.indptr[1]]
    tree = model.tree_

    described = []
    for i in range(len(node_ids) - 1):
        node = node_ids[i]
        feature_idx = tree.feature[node]
        if feature_idx == -2:
            continue
        went_left = node_ids[i + 1] == tree.children_left[node]
        value = x_row[feature_idx]
        described.append(_describe_split(
            FEATURE_NAMES[feature_idx], value, went_left, bool(np.isnan(value))
        ))

    if not described:
        return ""

    suspicious = [text for text, is_suspicious in described if is_suspicious]
    if suspicious:
        return _sentence(suspicious)
    return _sentence([described[-1][0]])


def predict(model, feature_df):
    if "respondent_id" not in feature_df.columns:
        raise ValueError("feature_df missing column: 'respondent_id'")
    X = _feature_matrix(feature_df)
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 0 in classes:
        reliability = proba[:, classes.index(0)]
    else:
        reliability = np.zeros(len(X))
    predictions = model.predict(X)
    ids = feature_df["respondent_id"].tolist()
    rows = []
    for i, rid in enumerate(ids):
        flagged = predictions[i] == 1
        reason = ""
        if flagged:
            reason = _describe_path(model, X[i]) or "Flagged by the model with no single distinguishing rule."
        rows.append({
            "respondent_id": rid,
            "reliability_score": float(reliability[i]),
            "flag_reason": reason,
        })
    return pd.DataFrame(rows, columns=["respondent_id", "reliability_score", "flag_reason"])
