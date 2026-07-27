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


from src.models.classifier import _describe_split, _describe_path


def test_describe_split_marks_suspicious_direction():
    # high straightlining is the suspicious direction
    text, suspicious = _describe_split("straightlining_score", 0.94, went_left=False, is_nan=False)
    assert suspicious is True
    assert "same answer" in text
    assert "94%" in text  # concrete value, not a raw threshold

    text, suspicious = _describe_split("straightlining_score", 0.3, went_left=True, is_nan=False)
    assert suspicious is False
    assert "varied" in text


def test_describe_split_low_attention_is_suspicious():
    text, suspicious = _describe_split(
        "attention_check_pass_rate", 0.0, went_left=True, is_nan=False
    )
    assert suspicious is True
    assert "0%" in text


def test_describe_split_fast_timing_reports_seconds_per_question():
    # ratio 0.125 on an 8s-per-question baseline is about 1s per question
    text, suspicious = _describe_split(
        "completion_time_ratio", 0.125, went_left=True, is_nan=False
    )
    assert suspicious is True
    assert "1s per question" in text


def test_describe_split_nan_timing_is_not_suspicious():
    text, suspicious = _describe_split(
        "completion_time_ratio", float("nan"), went_left=True, is_nan=True
    )
    assert "no timing data" in text
    assert suspicious is False


def test_describe_split_never_exposes_raw_feature_names():
    # reasons are for administrators; internal feature names must not leak
    for name in [
        "completion_time_ratio", "straightlining_score", "response_variance",
        "contradiction_score", "attention_check_pass_rate", "extreme_response_rate",
    ]:
        for went_left in (True, False):
            text, _ = _describe_split(name, 0.5, went_left=went_left, is_nan=False)
            assert name not in text
            assert "<=" not in text and ">" not in text


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


def test_predict_missing_respondent_id_raises():
    df = _feature_df()
    model = train(df, [0, 0, 1, 0, 1, 0])
    with pytest.raises(ValueError):
        predict(model, df.drop(columns=["respondent_id"]))


def test_predict_flagged_reason_nonempty_even_for_degenerate_tree():
    # all-flagged training -> single-class root-leaf tree with no split nodes
    df = _feature_df()
    model = train(df, [1, 1, 1, 1, 1, 1])
    out = predict(model, df)
    assert (out["reliability_score"] < 0.5).all()  # all flagged
    assert (out["flag_reason"] != "").all()  # invariant: flagged -> non-empty reason


def test_reasons_are_ascii_only():
    # reasons get written to locale-encoded files on Windows; keep them ASCII
    df = _feature_df()
    model = train(df, [0, 0, 1, 0, 1, 0])
    for reason in predict(model, df)["flag_reason"]:
        reason.encode("ascii")  # raises UnicodeEncodeError if not ASCII


def test_every_flagged_respondent_gets_a_specific_reason():
    """The product's core promise: no flag without an explanation.

    Runs the full pipeline over several seeded surveys and asserts that every
    single flagged respondent gets a concrete reason -- never blank, never a
    vague catch-all.
    """
    import json
    import tempfile
    from pathlib import Path
    from data.synthetic.generator import generate_survey_csv
    from src.ingestion.normalize import apply_mapping
    from src.features.extract import extract_features
    from src.api.pipeline_service import _build_training_mapping

    VAGUE = ["overall response pattern", "no single distinguishing rule"]
    total_flagged = 0

    with tempfile.TemporaryDirectory() as tmp:
        for seed in (11, 22, 33, 44):
            out = Path(tmp) / f"s{seed}.csv"
            generate_survey_csv(
                n_respondents=120, n_questions=12, scale=(1, 5),
                contamination_rate=0.3, seed=seed, output_path=str(out),
            )
            raw = pd.read_csv(out)
            labels = pd.read_csv(Path(tmp) / f"s{seed}_labels.csv")
            with open(Path(tmp) / f"s{seed}_pairs.json") as fh:
                pairs_idx = json.load(fh)["pairs"]

            mapping = _build_training_mapping(list(raw.columns))
            respondents = apply_mapping(raw, mapping)
            pair_keys = [(f"q{a + 1}", f"q{b + 1}") for a, b in pairs_idx]
            features = extract_features(respondents, contradiction_pairs=pair_keys)

            merged = features.merge(labels, on="respondent_id")
            y = merged["is_careless"].astype(int).tolist()
            feat = merged[["respondent_id"] + FEATURE_NAMES].reset_index(drop=True)

            preds = predict(train(feat, y), feat)
            flagged = preds[preds["reliability_score"] < 0.5]
            total_flagged += len(flagged)

            for _, row in flagged.iterrows():
                reason = row["flag_reason"]
                assert reason != "", f"{row['respondent_id']} flagged with no reason"
                assert reason.endswith(".")
                for vague in VAGUE:
                    assert vague not in reason, f"vague reason: {reason}"

    assert total_flagged > 50  # the surveys really were contaminated
