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
