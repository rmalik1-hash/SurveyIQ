import pandas as pd
import pytest
from src.ingestion.normalize import _compute_duration


def test_compute_duration_normal():
    assert _compute_duration("2024-03-01T08:00:00", "2024-03-01T08:02:00") == 120


def test_compute_duration_missing_returns_none():
    assert _compute_duration(None, "2024-03-01T08:02:00") is None
    assert _compute_duration("2024-03-01T08:00:00", None) is None


def test_compute_duration_unparseable_returns_none():
    assert _compute_duration("not a date", "2024-03-01T08:02:00") is None


from src.ingestion.normalize import _is_pii_column


def test_is_pii_flags_email_header():
    assert _is_pii_column("Email Address", ["a", "b"]) is True


def test_is_pii_flags_person_name_headers():
    assert _is_pii_column("Student Name", ["Alice", "Bob"]) is True
    assert _is_pii_column("Name", ["Alice"]) is True


def test_is_pii_flags_email_values():
    assert _is_pii_column("contact", ["x@y.com", "z@w.org"]) is True


def test_is_pii_does_not_flag_school_name_or_grade():
    assert _is_pii_column("School Name", ["Lincoln High"]) is False
    assert _is_pii_column("grade level", ["9", "10"]) is False


from src.ingestion.normalize import _extract_responses, _extract_attention_checks


def test_extract_responses_keys_in_order():
    row = pd.Series({"Q_a": 3, "Q_b": 5})
    assert _extract_responses(row, ["Q_a", "Q_b"]) == {"q1": 3, "q2": 5}


def test_extract_attention_checks_pairs_given_and_correct():
    row = pd.Series({"AC_1": 4})
    assert _extract_attention_checks(row, ["AC_1"], {"AC_1": 5}) == {
        "ac1_given": 4, "ac1_correct": 5,
    }


def test_extract_attention_checks_empty():
    row = pd.Series({"Q_a": 3})
    assert _extract_attention_checks(row, [], {}) == {}


from src.ingestion.normalize import _validate_mapping, _columns_by_role


def _valid_df():
    return pd.DataFrame({"id": ["R1"], "Q1": [3], "AC1": [5]})


def _valid_mapping():
    return {
        "columns": {"id": "respondent_id", "Q1": "question", "AC1": "attention_check"},
        "scale": [1, 5],
        "attention_check_answers": {"AC1": 5},
    }


def test_validate_mapping_ok():
    _validate_mapping(_valid_df(), _valid_mapping())  # must not raise


def test_validate_mapping_unknown_column():
    m = _valid_mapping()
    m["columns"]["ghost"] = "question"
    with pytest.raises(ValueError):
        _validate_mapping(_valid_df(), m)


def test_validate_mapping_no_respondent_id():
    m = _valid_mapping()
    m["columns"]["id"] = "demographic"
    with pytest.raises(ValueError):
        _validate_mapping(_valid_df(), m)


def test_validate_mapping_two_respondent_ids():
    df = pd.DataFrame({"id": ["R1"], "id2": ["X"], "Q1": [3], "AC1": [5]})
    m = _valid_mapping()
    m["columns"]["id2"] = "respondent_id"
    with pytest.raises(ValueError):
        _validate_mapping(df, m)


def test_validate_mapping_no_question():
    m = _valid_mapping()
    m["columns"]["Q1"] = "demographic"
    with pytest.raises(ValueError):
        _validate_mapping(_valid_df(), m)


def test_validate_mapping_bad_scale():
    m = _valid_mapping()
    m["scale"] = [5, 1]
    with pytest.raises(ValueError):
        _validate_mapping(_valid_df(), m)


def test_validate_mapping_attention_without_answer():
    m = _valid_mapping()
    m["attention_check_answers"] = {}
    with pytest.raises(ValueError):
        _validate_mapping(_valid_df(), m)


def test_columns_by_role_groups_and_preserves_order():
    m = {
        "columns": {"id": "respondent_id", "Qa": "question", "Qb": "question"},
        "scale": [1, 5],
    }
    grouped = _columns_by_role(m)
    assert grouped["question"] == ["Qa", "Qb"]
    assert grouped["respondent_id"] == ["id"]


import warnings
from src.ingestion.normalize import apply_mapping


def test_apply_mapping_basic_shape():
    df = pd.DataFrame({
        "Response ID": ["R1", "R2"],
        "Start Time": ["2024-03-01T08:00:00", "2024-03-01T08:00:00"],
        "Timestamp": ["2024-03-01T08:02:00", "2024-03-01T08:01:00"],
        "Q1": [3, 1], "Q2": [3, 5],
        "AC1": [5, 1],
        "grade level": ["9", "10"],
    })
    mapping = {
        "columns": {
            "Response ID": "respondent_id", "Start Time": "start_time",
            "Timestamp": "end_time", "Q1": "question", "Q2": "question",
            "AC1": "attention_check", "grade level": "demographic",
        },
        "scale": [1, 5],
        "attention_check_answers": {"AC1": 5},
    }
    out = apply_mapping(df, mapping)
    assert len(out) == 2
    r0 = out[0]
    assert r0["respondent_id"] == "R1"
    assert r0["duration_seconds"] == 120
    assert r0["responses"] == {"q1": 3, "q2": 3}
    assert r0["attention_checks"] == {"ac1_given": 5, "ac1_correct": 5}
    assert r0["demographics"] == {"grade level": "9"}
    assert r0["scale_min"] == 1 and r0["scale_max"] == 5


def test_apply_mapping_missing_end_time_gives_none_duration():
    df = pd.DataFrame({"Response ID": ["R1"], "Q1": [3]})
    mapping = {
        "columns": {"Response ID": "respondent_id", "Q1": "question"},
        "scale": [1, 5],
    }
    out = apply_mapping(df, mapping)
    assert out[0]["duration_seconds"] is None


def test_apply_mapping_drops_pii_demographic_and_warns():
    df = pd.DataFrame({
        "Response ID": ["R1"], "Q1": [3], "Email Address": ["a@b.com"],
    })
    mapping = {
        "columns": {
            "Response ID": "respondent_id", "Q1": "question",
            "Email Address": "demographic",
        },
        "scale": [1, 5],
    }
    with pytest.warns(UserWarning):
        out = apply_mapping(df, mapping)
    assert "Email Address" not in out[0]["demographics"]
