import json

import pytest

from src.api.history import (
    ALLOWED_FIELDS,
    clear_runs,
    list_runs,
    record_run,
)


@pytest.fixture
def store(tmp_path):
    return tmp_path / "history.json"


def _summary(total=100, flagged=25):
    return {
        "total": total,
        "flagged": flagged,
        "reliable": total - flagged,
        "overall_quality_pct": round(100.0 * (total - flagged) / total, 1),
        "mean_response": 3.1,
    }


def test_record_run_returns_the_stored_row(store):
    row = record_run(_summary(), survey_label="Wellbeing survey", path=store)
    assert row["survey_label"] == "Wellbeing survey"
    assert row["total"] == 100
    assert row["flagged"] == 25
    assert row["overall_quality_pct"] == 75.0
    assert row["recorded_at"]


def test_runs_come_back_oldest_first(store):
    record_run(_summary(100, 40), survey_label="Wellbeing", path=store)
    record_run(_summary(100, 20), survey_label="Wellbeing", path=store)
    runs = list_runs(path=store)
    assert [r["flagged"] for r in runs] == [40, 20]


def test_runs_can_be_filtered_by_survey(store):
    record_run(_summary(), survey_label="Wellbeing", path=store)
    record_run(_summary(), survey_label="Climate", path=store)
    assert len(list_runs(path=store)) == 2
    assert len(list_runs(survey_label="Climate", path=store)) == 1


def test_listing_an_absent_store_is_empty_not_an_error(store):
    assert list_runs(path=store) == []


def test_clear_removes_everything(store):
    record_run(_summary(), survey_label="Wellbeing", path=store)
    assert clear_runs(path=store) == 1
    assert list_runs(path=store) == []


def test_only_aggregate_fields_are_ever_written(store):
    """The privacy guarantee, enforced rather than documented.

    History exists to show quality trends. Nothing that could identify a
    respondent -- ids, answers, demographics, flag reasons -- may reach disk.
    """
    contaminated = {
        **_summary(),
        "respondents": [{"respondent_id": "R1", "flag_reason": "Gave the same answer"}],
        "respondent_id": "R1",
        "responses": {"q1": 3},
        "demographics": {"grade level": "9"},
        "email": "student@school.org",
    }
    record_run(contaminated, survey_label="Wellbeing", path=store)

    written = json.loads(store.read_text())
    assert len(written) == 1
    assert set(written[0]) == ALLOWED_FIELDS

    blob = json.dumps(written)
    for forbidden in ["R1", "respondents", "responses", "demographics",
                      "student@school.org", "grade level", "flag_reason"]:
        assert forbidden not in blob


def test_survey_label_is_required_so_trends_have_something_to_group_by(store):
    with pytest.raises(ValueError):
        record_run(_summary(), survey_label="", path=store)


def test_a_corrupt_store_does_not_take_the_api_down(store):
    store.write_text("this is not json")
    assert list_runs(path=store) == []
    # and recording recovers rather than failing
    record_run(_summary(), survey_label="Wellbeing", path=store)
    assert len(list_runs(path=store)) == 1


def test_question_means_are_stored_for_opinion_trends(store):
    row = record_run(
        _summary(), survey_label="Wellbeing", path=store,
        question_means={"I like this class [Q1]": 3.2, "I feel safe [Q2]": 4.1},
    )
    assert row["question_means"]["I like this class [Q1]"] == 3.2
    stored = list_runs(path=store)[0]
    assert stored["question_means"]["I feel safe [Q2]"] == 4.1


def test_question_means_reject_non_numeric_values(store):
    """Only averages belong here -- never a stray answer or identifier."""
    record_run(
        _summary(), survey_label="Wellbeing", path=store,
        question_means={"Q1": 3.2, "Q2": "R0001", "Q3": {"nested": "data"}},
    )
    stored = list_runs(path=store)[0]["question_means"]
    assert stored == {"Q1": 3.2}


def test_a_survey_date_can_be_supplied(store):
    """Surveys are usually analysed some time after they were actually run."""
    row = record_run(
        _summary(), survey_label="Wellbeing", path=store,
        recorded_at="2026-03-01T00:00:00+00:00",
    )
    assert row["recorded_at"] == "2026-03-01T00:00:00+00:00"
