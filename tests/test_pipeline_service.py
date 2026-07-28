import pytest
from src.api.pipeline_service import _build_training_mapping, _pairs_to_qkeys


def test_build_training_mapping_roles():
    cols = ["Response ID", "Start Time", "Timestamp", "x [Q1]", "y [AC1]",
            "Email Address", "grade level"]
    m = _build_training_mapping(cols)
    assert m["columns"]["Response ID"] == "respondent_id"
    assert m["columns"]["Start Time"] == "start_time"
    assert m["columns"]["Timestamp"] == "end_time"
    assert m["columns"]["x [Q1]"] == "question"
    assert m["columns"]["y [AC1]"] == "attention_check"
    assert m["attention_check_answers"]["y [AC1]"] == 5
    assert m["columns"]["Email Address"] == "ignore"
    assert m["columns"]["grade level"] == "demographic"
    assert m["scale"] == [1, 5]


def test_pairs_to_qkeys_converts_column_pairs():
    mapping = {
        "columns": {"id": "respondent_id", "A": "question", "B": "question", "C": "question"},
        "contradiction_pairs": [["A", "B"]],
    }
    assert _pairs_to_qkeys(mapping) == [("q1", "q2")]


def test_pairs_to_qkeys_absent_is_empty():
    mapping = {"columns": {"id": "respondent_id", "A": "question"}}
    assert _pairs_to_qkeys(mapping) == []


def test_pairs_to_qkeys_non_question_column_raises():
    mapping = {
        "columns": {"id": "respondent_id", "A": "question"},
        "contradiction_pairs": [["A", "Z"]],
    }
    with pytest.raises(ValueError):
        _pairs_to_qkeys(mapping)


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from src.api.pipeline_service import train_startup_model, analyze
from src.models.classifier import train, FEATURE_NAMES


def _tiny_model():
    df = pd.DataFrame({
        "respondent_id": [f"R{i}" for i in range(6)],
        "completion_time_ratio": [1.0, 1.1, 0.1, 1.2, 0.12, 1.0],
        "straightlining_score": [0.3, 0.25, 1.0, 0.2, 0.9, 0.3],
        "response_variance": [0.4, 0.45, 0.0, 0.5, 0.05, 0.4],
        "contradiction_score": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "attention_check_pass_rate": [1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
        "extreme_response_rate": [0.3, 0.3, 0.2, 0.3, 0.9, 0.2],
        "behavior_shift_score": [0.0, 0.1, 0.0, 0.0, 0.2, 0.0],
    })
    return train(df, [0, 0, 1, 0, 1, 0])


def test_train_startup_model_returns_fitted():
    model = train_startup_model(n_respondents=60, n_questions=8, seed=1)
    assert isinstance(model, DecisionTreeClassifier)
    assert hasattr(model, "tree_")


def test_analyze_returns_summary_and_respondents():
    raw = pd.DataFrame({
        "Response ID": ["R1", "R2"],
        "Q1": [3, 1], "Q2": [3, 5], "Q3": [3, 1], "Q4": [3, 5],
    })
    mapping = {
        "columns": {
            "Response ID": "respondent_id", "Q1": "question", "Q2": "question",
            "Q3": "question", "Q4": "question",
        },
        "scale": [1, 5],
    }
    result = analyze(raw, mapping, _tiny_model())
    assert result["summary"]["total"] == 2
    assert result["summary"]["flagged"] + result["summary"]["reliable"] == 2
    assert len(result["respondents"]) == 2
    assert set(result["respondents"][0].keys()) == {
        "respondent_id", "reliability_score", "flag_reason",
    }
    for r in result["respondents"]:
        assert 0.0 <= r["reliability_score"] <= 1.0


def test_analyze_invalid_mapping_raises():
    raw = pd.DataFrame({"Response ID": ["R1"], "Q1": [3]})
    bad_mapping = {"columns": {"Response ID": "demographic", "Q1": "demographic"}, "scale": [1, 5]}
    with pytest.raises(ValueError):
        analyze(raw, bad_mapping, _tiny_model())


def _stats_respondents():
    def r(rid, responses, checks):
        return {"respondent_id": rid, "duration_seconds": 100, "responses": responses,
                "attention_checks": checks, "scale_min": 1, "scale_max": 5}
    return [
        # consistent pair (1+5), passes AC
        r("R1", {"q1": 1, "q2": 5}, {"ac1_given": 5, "ac1_correct": 5}),
        # contradicts pair (1,1), fails AC
        r("R2", {"q1": 1, "q2": 1}, {"ac1_given": 2, "ac1_correct": 5}),
        # contradicts pair (5,5), passes AC
        r("R3", {"q1": 5, "q2": 5}, {"ac1_given": 5, "ac1_correct": 5}),
    ]


def _stats_mapping():
    return {
        "columns": {"id": "respondent_id", "A": "question", "B": "question", "AC": "attention_check"},
        "scale": [1, 5],
        "attention_check_answers": {"AC": 5},
        "contradiction_pairs": [["A", "B"]],
    }


def test_question_stats_counts_contradictions_and_ac_failures():
    from src.api.pipeline_service import _question_stats
    stats = _question_stats(_stats_respondents(), _stats_mapping(), [("q1", "q2")])
    by_label = {s["label"]: s for s in stats}
    assert by_label["A / B"]["type"] == "contradiction_pair"
    assert by_label["A / B"]["affected"] == 2  # R2 and R3
    assert by_label["AC"]["type"] == "attention_check"
    assert by_label["AC"]["affected"] == 1  # R2 only


def test_question_stats_sorted_by_affected_desc():
    from src.api.pipeline_service import _question_stats
    stats = _question_stats(_stats_respondents(), _stats_mapping(), [("q1", "q2")])
    affected = [s["affected"] for s in stats]
    assert affected == sorted(affected, reverse=True)


def test_question_stats_empty_when_no_pairs_or_checks():
    from src.api.pipeline_service import _question_stats
    respondents = [{"respondent_id": "R1", "duration_seconds": 10, "responses": {"q1": 3},
                    "attention_checks": {}, "scale_min": 1, "scale_max": 5}]
    mapping = {"columns": {"id": "respondent_id", "A": "question"}, "scale": [1, 5]}
    assert _question_stats(respondents, mapping, []) == []


def test_analyze_response_includes_question_stats():
    raw = pd.DataFrame({"Response ID": ["R1", "R2"], "Q1": [1, 1], "Q2": [5, 1],
                        "Q3": [2, 3], "Q4": [4, 3]})
    mapping = {
        "columns": {"Response ID": "respondent_id", "Q1": "question", "Q2": "question",
                    "Q3": "question", "Q4": "question"},
        "scale": [1, 5],
        "contradiction_pairs": [["Q1", "Q2"]],
    }
    result = analyze(raw, mapping, _tiny_model())
    assert "question_stats" in result
    stats = {s["label"]: s["affected"] for s in result["question_stats"]}
    assert stats["Q1 / Q2"] == 1  # only R2 contradicts
