import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier
from src.models.classifier import train, FEATURE_NAMES, _feature_matrix


def _feature_df(n=6):
    data = {
        "respondent_id": [f"R{i}" for i in range(n)],
        "completion_time_ratio": [1.0, 1.1, 0.1, 1.2, 0.12, 1.0][:n],
        "straightlining_score": [0.3, 0.25, 1.0, 0.2, 0.9, 0.3][:n],
        "response_variance": [0.4, 0.45, 0.0, 0.5, 0.05, 0.4][:n],
        "contradiction_score": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0][:n],
        "attention_check_pass_rate": [1.0, 1.0, 1.0, 1.0, 0.0, 1.0][:n],
        "extreme_response_rate": [0.3, 0.3, 0.2, 0.3, 0.9, 0.2][:n],
    }
    return pd.DataFrame(data)


def test_feature_names_excludes_respondent_id():
    assert "respondent_id" not in FEATURE_NAMES
    assert len(FEATURE_NAMES) == 6


def test_feature_matrix_shape():
    assert _feature_matrix(_feature_df()).shape == (6, 6)


def test_feature_matrix_missing_column_raises():
    df = _feature_df().drop(columns=["response_variance"])
    with pytest.raises(ValueError):
        _feature_matrix(df)


def test_train_returns_fitted_tree_with_depth_cap():
    model = train(_feature_df(), [0, 0, 1, 0, 1, 0], max_depth=4)
    assert isinstance(model, DecisionTreeClassifier)
    assert model.get_depth() <= 4
    assert hasattr(model, "tree_")


def test_train_empty_raises():
    with pytest.raises(ValueError):
        train(_feature_df(0), [])


def test_train_label_mismatch_raises():
    with pytest.raises(ValueError):
        train(_feature_df(), [0, 1])
