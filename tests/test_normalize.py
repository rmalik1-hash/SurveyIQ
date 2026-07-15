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
