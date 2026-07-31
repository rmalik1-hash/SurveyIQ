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
        "behavior_shift_score": [0.0, 0.1, 0.0, 0.0, 0.2, 0.0][:n],
    }
    return pd.DataFrame(data)


def test_feature_names_excludes_respondent_id():
    assert "respondent_id" not in FEATURE_NAMES
    assert len(FEATURE_NAMES) == 7


def test_feature_matrix_shape():
    assert _feature_matrix(_feature_df()).shape == (6, 7)


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


def test_describe_split_behavior_shift_names_the_question():
    text, suspicious = _describe_split(
        "behavior_shift_score", 0.8, went_left=False, is_nan=False, shift_at=18
    )
    assert suspicious is True
    assert "question 18" in text


def test_describe_split_behavior_shift_without_position_still_reads_well():
    text, suspicious = _describe_split(
        "behavior_shift_score", 0.8, went_left=False, is_nan=False, shift_at=None
    )
    assert suspicious is True
    assert "repetitive" in text
    assert "None" not in text


def test_behavior_shift_reason_appears_for_a_fatiguer_end_to_end():
    """Fatiguers must be caught and explained by where their answers changed.

    Averaged over several seeds rather than asserted on one, so this measures
    the detector rather than a lucky draw.
    """
    import json
    import tempfile
    from pathlib import Path
    from data.synthetic.generator import generate_survey_csv
    from src.ingestion.normalize import apply_mapping
    from src.features.extract import extract_features
    from src.api.pipeline_service import _build_training_mapping

    caught_rates = []
    reasons_seen = []

    with tempfile.TemporaryDirectory() as tmp:
        for seed in (8, 21, 42):
            out = Path(tmp) / f"s{seed}.csv"
            generate_survey_csv(
                n_respondents=300, n_questions=20, scale=(1, 5),
                contamination_rate=0.4, seed=seed, output_path=str(out),
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
            feat = merged[
                ["respondent_id"] + FEATURE_NAMES + ["behavior_shift_at"]
            ].reset_index(drop=True)

            preds = predict(train(feat, y), feat).merge(labels, on="respondent_id")
            fatiguers = preds[preds["archetype"] == "fatiguer"]
            assert len(fatiguers) > 10
            caught_rates.append((fatiguers["reliability_score"] < 0.5).mean())
            reasons_seen.append(" ".join(fatiguers["flag_reason"].tolist()))

    mean_caught = sum(caught_rates) / len(caught_rates)
    assert mean_caught > 0.7, f"only caught {mean_caught:.0%} of fatiguers on average"

    # at least some explanations should name where the behaviour changed
    assert any("from question" in r for r in reasons_seen)


def test_zero_variance_is_suspicious_in_both_directions():
    """Almost no spread means the same answer went down the page.

    Only treating *high* variance as suspicious left straightliners with no
    incriminating clause at all, so they fell through to the vague catch-all.
    """
    flat, suspicious = _describe_split(
        "response_variance", 0.0, went_left=True, is_nan=False
    )
    assert suspicious is True
    assert "same answer" in flat

    erratic, suspicious = _describe_split(
        "response_variance", 0.45, went_left=False, is_nan=False
    )
    assert suspicious is True
    assert "erratic" in erratic

    # a normal middling spread is not itself evidence of anything
    _, suspicious = _describe_split(
        "response_variance", 0.22, went_left=True, is_nan=False
    )
    assert suspicious is False


def test_vague_reasons_stay_rare_on_surveys_unlike_the_training_data():
    """Real uploads never match the training shape; explanations must survive that.

    Scores surveys of varying length and contamination with the startup model and
    asserts the catch-all reason stays rare.
    """
    import tempfile
    from pathlib import Path
    from data.synthetic.generator import generate_survey_csv
    from src.api.pipeline_service import (
        _build_training_mapping, analyze, train_startup_model,
    )

    model = train_startup_model()
    vague_total = flagged_total = 0

    with tempfile.TemporaryDirectory() as tmp:
        for n, questions, contamination, seed in [
            (150, 15, 0.25, 1), (200, 12, 0.20, 2), (120, 20, 0.35, 3),
        ]:
            out = Path(tmp) / f"u{seed}.csv"
            generate_survey_csv(
                n_respondents=n, n_questions=questions, scale=(1, 5),
                contamination_rate=contamination, seed=seed, output_path=str(out),
            )
            raw = pd.read_csv(out)
            result = analyze(raw, _build_training_mapping(list(raw.columns)), model)
            flagged = [r for r in result["respondents"] if r["reliability_score"] < 0.5]
            flagged_total += len(flagged)
            vague_total += sum(1 for r in flagged if "No single signal" in r["flag_reason"])

    assert flagged_total > 50
    assert vague_total / flagged_total < 0.15, (
        f"{vague_total}/{flagged_total} flags could not be explained specifically"
    )
