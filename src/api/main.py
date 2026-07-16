import io
import json
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.api.pipeline_service import train_startup_model, analyze


@asynccontextmanager
async def lifespan(app):
    app.state.model = train_startup_model()
    yield


app = FastAPI(title="SurveyIQ API", lifespan=lifespan)


def _read_csv(raw_bytes, nrows=None):
    try:
        return pd.read_csv(io.BytesIO(raw_bytes), nrows=nrows)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not parse CSV: {exc}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/columns")
async def columns(file: UploadFile = File(...)):
    df = _read_csv(await file.read(), nrows=0)  # headers only; respondent rows are not needed
    return {"columns": list(df.columns)}


@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...), mapping: str = Form(...)):
    df = _read_csv(await file.read())
    try:
        mapping_obj = json.loads(mapping)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid mapping JSON: {exc}")
    try:
        return analyze(df, mapping_obj, app.state.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
