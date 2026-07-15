import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.features.extract import FEATURE_COLUMNS

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


_CLAUSE_TEMPLATES = {
    "completion_time_ratio": ("answered very fast", "took adequate time"),
    "straightlining_score": ("varied their answers", "gave the same answer repeatedly"),
    "response_variance": ("answers had little spread", "answers were highly erratic"),
    "contradiction_score": ("answers were consistent", "contradicted on paired questions"),
    "attention_check_pass_rate": ("failed attention checks", "passed attention checks"),
    "extreme_response_rate": ("used the scale moderately", "overused the scale endpoints"),
}


def _clause(feature_name, threshold, went_left, is_nan):
    if is_nan and feature_name == "completion_time_ratio":
        return "had no timing data"
    low_phrase, high_phrase = _CLAUSE_TEMPLATES[feature_name]
    phrase = low_phrase if went_left else high_phrase
    op = "<=" if went_left else ">"
    return f"{phrase} ({feature_name} {op} {threshold:.2f})"


def _describe_path(model, x_row):
    x_row = np.asarray(x_row, dtype=float)
    dp = model.decision_path(x_row.reshape(1, -1))
    node_ids = dp.indices[dp.indptr[0]:dp.indptr[1]]
    tree = model.tree_
    clauses = []
    for i in range(len(node_ids) - 1):
        node = node_ids[i]
        feature_idx = tree.feature[node]
        if feature_idx == -2:
            continue
        went_left = node_ids[i + 1] == tree.children_left[node]
        value = x_row[feature_idx]
        clauses.append(_clause(
            FEATURE_NAMES[feature_idx], tree.threshold[node], went_left, np.isnan(value)
        ))
    if not clauses:
        return ""
    sentence = " and ".join(clauses)
    return sentence[0].upper() + sentence[1:] + "."


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
