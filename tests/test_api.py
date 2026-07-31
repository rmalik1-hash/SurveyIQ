import io
import json
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from data.synthetic.generator import generate_survey_csv
from src.api.main import app
from src.api.pipeline_service import _build_training_mapping


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _generated_csv(tmp_path_factory):
    d = tmp_path_factory.mktemp("api")
    out = d / "s.csv"
    generate_survey_csv(
        n_respondents=40, n_questions=10, scale=(1, 5),
        contamination_rate=0.3, seed=7, output_path=str(out),
    )
    return out.read_bytes(), pd.read_csv(out)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_columns(client, tmp_path_factory):
    data, df = _generated_csv(tmp_path_factory)
    r = client.post("/columns", files={"file": ("s.csv", data, "text/csv")})
    assert r.status_code == 200
    assert r.json()["columns"] == list(df.columns)


def test_analyze_ok(client, tmp_path_factory):
    data, df = _generated_csv(tmp_path_factory)
    mapping = _build_training_mapping(list(df.columns))
    r = client.post(
        "/analyze",
        files={"file": ("s.csv", data, "text/csv")},
        data={"mapping": json.dumps(mapping)},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["total"] == len(df)
    assert len(body["respondents"]) == len(df)
    assert body["summary"]["flagged"] + body["summary"]["reliable"] == len(df)

    # the generated survey is 30% careless, so the model must flag someone
    flagged = [r for r in body["respondents"] if r["reliability_score"] < 0.5]
    assert len(flagged) == body["summary"]["flagged"] > 0
    # every flag must be explainable -- the product's core promise
    for r in flagged:
        assert r["flag_reason"] != ""
        assert r["flag_reason"].endswith(".")
    for r in body["respondents"]:
        if r["reliability_score"] >= 0.5:
            assert r["flag_reason"] == ""


def test_analyze_bad_mapping_returns_400(client, tmp_path_factory):
    data, df = _generated_csv(tmp_path_factory)
    bad = {"columns": {c: "demographic" for c in df.columns}, "scale": [1, 5]}
    r = client.post(
        "/analyze",
        files={"file": ("s.csv", data, "text/csv")},
        data={"mapping": json.dumps(bad)},
    )
    assert r.status_code == 400


def test_analyze_missing_mapping_returns_422(client, tmp_path_factory):
    data, df = _generated_csv(tmp_path_factory)
    r = client.post("/analyze", files={"file": ("s.csv", data, "text/csv")})
    assert r.status_code == 422


def test_generate_returns_a_usable_csv(client):
    r = client.post(
        "/generate",
        data={"n_respondents": "30", "n_questions": "8", "contamination_rate": "0.3"},
    )
    assert r.status_code == 200
    assert "text/csv" in r.headers["content-type"]
    df = pd.read_csv(io.BytesIO(r.content))
    assert len(df) == 30
    assert "Response ID" in df.columns
    # ground truth must never be served alongside the survey
    assert "is_careless" not in df.columns
    assert "archetype" not in df.columns


def test_generate_output_flows_through_analyze(client):
    gen = client.post("/generate", data={"n_respondents": "40", "n_questions": "8"})
    assert gen.status_code == 200
    df = pd.read_csv(io.BytesIO(gen.content))
    mapping = _build_training_mapping(list(df.columns))
    r = client.post(
        "/analyze",
        files={"file": ("synthetic_survey.csv", gen.content, "text/csv")},
        data={"mapping": json.dumps(mapping)},
    )
    assert r.status_code == 200
    assert r.json()["summary"]["total"] == 40


def test_generate_rejects_invalid_parameters(client):
    r = client.post("/generate", data={"n_respondents": "0", "n_questions": "8"})
    assert r.status_code == 400


def test_columns_accepts_xlsx_upload(client, tmp_path_factory):
    data, df = _generated_csv(tmp_path_factory)
    xlsx_path = tmp_path_factory.mktemp("xlsx") / "survey.xlsx"
    pd.read_csv(io.BytesIO(data)).to_excel(xlsx_path, index=False)
    r = client.post(
        "/columns",
        files={
            "file": (
                "survey.xlsx",
                xlsx_path.read_bytes(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )
    assert r.status_code == 200
    assert r.json()["columns"] == list(df.columns)


def test_analyze_accepts_xlsx_upload(client, tmp_path_factory):
    data, df = _generated_csv(tmp_path_factory)
    xlsx_path = tmp_path_factory.mktemp("xlsx2") / "survey.xlsx"
    pd.read_csv(io.BytesIO(data)).to_excel(xlsx_path, index=False)
    mapping = _build_training_mapping(list(df.columns))
    r = client.post(
        "/analyze",
        files={
            "file": (
                "survey.xlsx",
                xlsx_path.read_bytes(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
        data={"mapping": json.dumps(mapping)},
    )
    assert r.status_code == 200
    assert r.json()["summary"]["total"] == len(df)


@pytest.fixture(autouse=True)
def _isolated_history(tmp_path, monkeypatch):
    """Keep every test's history in its own file, never the real store."""
    from src.api import history
    monkeypatch.setattr(history, "DEFAULT_HISTORY_PATH", tmp_path / "history.json")


def test_analyze_without_a_label_records_nothing(client, tmp_path_factory):
    data, df = _generated_csv(tmp_path_factory)
    mapping = _build_training_mapping(list(df.columns))
    client.post(
        "/analyze",
        files={"file": ("s.csv", data, "text/csv")},
        data={"mapping": json.dumps(mapping)},
    )
    assert client.get("/history").json()["runs"] == []


def test_analyze_with_a_label_records_one_run(client, tmp_path_factory):
    data, df = _generated_csv(tmp_path_factory)
    mapping = _build_training_mapping(list(df.columns))
    r = client.post(
        "/analyze",
        files={"file": ("s.csv", data, "text/csv")},
        data={"mapping": json.dumps(mapping), "survey_label": "Wellbeing survey"},
    )
    assert r.status_code == 200

    runs = client.get("/history").json()["runs"]
    assert len(runs) == 1
    assert runs[0]["survey_label"] == "Wellbeing survey"
    assert runs[0]["total"] == len(df)


def test_history_response_carries_no_respondent_data(client, tmp_path_factory):
    data, df = _generated_csv(tmp_path_factory)
    mapping = _build_training_mapping(list(df.columns))
    client.post(
        "/analyze",
        files={"file": ("s.csv", data, "text/csv")},
        data={"mapping": json.dumps(mapping), "survey_label": "Wellbeing survey"},
    )
    blob = json.dumps(client.get("/history").json())
    for forbidden in ["R0001", "flag_reason", "respondents", "Email", "grade level"]:
        assert forbidden not in blob


def test_history_can_be_filtered_and_cleared(client, tmp_path_factory):
    data, df = _generated_csv(tmp_path_factory)
    mapping = _build_training_mapping(list(df.columns))
    for label in ("Wellbeing survey", "Climate survey"):
        client.post(
            "/analyze",
            files={"file": ("s.csv", data, "text/csv")},
            data={"mapping": json.dumps(mapping), "survey_label": label},
        )
    assert len(client.get("/history").json()["runs"]) == 2
    filtered = client.get("/history", params={"survey_label": "Climate survey"}).json()
    assert len(filtered["runs"]) == 1

    assert client.delete("/history").json()["removed"] == 2
    assert client.get("/history").json()["runs"] == []


def test_analyze_returns_question_summary(client, tmp_path_factory):
    data, df = _generated_csv(tmp_path_factory)
    mapping = _build_training_mapping(list(df.columns))
    r = client.post(
        "/analyze",
        files={"file": ("s.csv", data, "text/csv")},
        data={"mapping": json.dumps(mapping)},
    )
    summary = r.json()["question_summary"]
    n_questions = sum(1 for role in mapping["columns"].values() if role == "question")
    assert len(summary) == n_questions
    assert summary[0]["position"] == 1
    assert sum(summary[0]["counts"].values()) == len(df)
