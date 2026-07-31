import io
import json
import os
import tempfile
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from data.synthetic.generator import generate_survey_csv
from src.api import history
from src.api.pipeline_service import train_startup_model, analyze


@asynccontextmanager
async def lifespan(app):
    app.state.model = train_startup_model()
    yield


app = FastAPI(title="SurveyIQ API", lifespan=lifespan)

# Permissive CORS for local development: the Vite dev server runs on a different
# origin. Tighten this before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_upload(raw_bytes, filename, nrows=None):
    """Read an uploaded survey. Schools export .xlsx as often as .csv."""
    try:
        if (filename or "").lower().endswith((".xlsx", ".xlsm")):
            return pd.read_excel(io.BytesIO(raw_bytes), nrows=nrows)
        return pd.read_csv(io.BytesIO(raw_bytes), nrows=nrows)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not parse file: {exc}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/columns")
async def columns(file: UploadFile = File(...)):
    # headers only; respondent rows are not needed to build a mapping
    df = _read_upload(await file.read(), file.filename, nrows=0)
    return {"columns": list(df.columns)}


@app.post("/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...),
    mapping: str = Form(...),
    survey_label: str = Form(""),
):
    df = _read_upload(await file.read(), file.filename)
    try:
        mapping_obj = json.loads(mapping)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid mapping JSON: {exc}")
    try:
        result = analyze(df, mapping_obj, app.state.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Recording is opt-in: without a survey label there is nothing to trend
    # against, so nothing is written. Only aggregate totals are ever stored.
    if survey_label.strip():
        try:
            history.record_run(result["summary"], survey_label)
        except OSError:
            # Trend tracking is a convenience; never fail an analysis over it.
            pass
    return result


@app.get("/history")
def history_endpoint(survey_label: str = ""):
    return {"runs": history.list_runs(survey_label or None)}


@app.delete("/history")
def clear_history_endpoint():
    return {"removed": history.clear_runs()}


def _cleanup_generated(path):
    """Remove a generated survey and its sidecars once the response is sent."""
    base = path[: -len(".csv")] if path.endswith(".csv") else path
    for candidate in (path, f"{base}_labels.csv", f"{base}_pairs.json"):
        try:
            os.remove(candidate)
        except OSError:
            pass


@app.post("/generate")
async def generate_endpoint(
    background_tasks: BackgroundTasks,
    n_respondents: int = Form(200),
    n_questions: int = Form(20),
    contamination_rate: float = Form(0.25),
):
    """Hand back a synthetic survey CSV, so the dashboard can be tried without
    supplying real data. The ground-truth sidecars are never served."""
    fd, temp_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        generate_survey_csv(
            n_respondents=n_respondents,
            n_questions=n_questions,
            contamination_rate=contamination_rate,
            output_path=temp_path,
        )
    except ValueError as exc:
        _cleanup_generated(temp_path)
        raise HTTPException(status_code=400, detail=str(exc))

    background_tasks.add_task(_cleanup_generated, temp_path)
    return FileResponse(
        temp_path, media_type="text/csv", filename="synthetic_survey.csv"
    )
