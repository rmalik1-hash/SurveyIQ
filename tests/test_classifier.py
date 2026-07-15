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


from src.models.classifier import _clause, _describe_path


def test_clause_low_and_high_phrases():
    low = _clause("straightlining_score", 0.8, went_left=True, is_nan=False)
    high = _clause("straightlining_score", 0.8, went_left=False, is_nan=False)
    assert "varied" in low
    assert "same answer" in high
    assert "0.80" in high


def test_clause_nan_timing():
    c = _clause("completion_time_ratio", 0.3, went_left=True, is_nan=True)
    assert "no timing data" in c


def test_describe_path_flagged_is_capitalized_sentence():
    model = train(_feature_df(), [0, 0, 1, 0, 1, 0])
    X = _feature_matrix(_feature_df())
    reason = _describe_path(model, X[2])
    assert isinstance(reason, str)
    assert reason.endswith(".")
    assert reason[0].isupper()


from src.models.classifier import predict


def test_predict_columns_order_and_score_range():
    df = _feature_df()
    model = train(df, [0, 0, 1, 0, 1, 0])
    out = predict(model, df)
    assert list(out.columns) == ["respondent_id", "reliability_score", "flag_reason"]
    assert len(out) == 6
    assert list(out["respondent_id"]) == list(df["respondent_id"])
    assert ((out["reliability_score"] >= 0) & (out["reliability_score"] <= 1)).all()


def test_predict_flagged_have_reason_reliable_empty():
    df = _feature_df()
    model = train(df, [0, 0, 1, 0, 1, 0])
    out = predict(model, df)
    for _, r in out.iterrows():
        if r["reliability_score"] < 0.5:
            assert r["flag_reason"] != ""
        else:
            assert r["flag_reason"] == ""


def test_predict_handles_nan_feature():
    df = _feature_df()
    df.loc[0, "completion_time_ratio"] = np.nan
    model = train(df, [0, 0, 1, 0, 1, 0])
    out = predict(model, df)
    assert len(out) == 6
    assert out["reliability_score"].notna().all()


def test_predict_missing_feature_column_raises():
    df = _feature_df()
    model = train(df, [0, 0, 1, 0, 1, 0])
    with pytest.raises(ValueError):
        predict(model, df.drop(columns=["contradiction_score"]))
