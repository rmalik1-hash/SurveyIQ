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
