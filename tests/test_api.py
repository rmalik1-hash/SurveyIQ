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
