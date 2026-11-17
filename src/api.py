"""HTTP API for scholarship application outcome predictions."""

from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data_pipeline import build_features


DEFAULT_ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "scholarship_model.joblib"
ARTIFACT_PATH = Path(os.environ.get("MODEL_ARTIFACT", str(DEFAULT_ARTIFACT)))

app = FastAPI(
    title="Scholarship Outcome API",
    description="Research-use predictions for scholarship application outcomes.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    mean_grade: str = Field(description="Secondary-school mean grade, for example B+.")
    o_level_year: int = Field(description="Year the applicant completed O-level studies.")
    county: str = Field(description="Kenyan county of residence.")
    date_applied: datetime = Field(description="Application submission date and time.")
    high_school: str = Field(description="Name of the applicant's high school.")
    parent_status: str = Field(default="Both Parents")


class PredictionResponse(BaseModel):
    prediction: str
    approved_probability: float
    not_approved_probability: float
    model: str


@lru_cache(maxsize=1)
def _load_artifact() -> dict[str, Any]:
    if not ARTIFACT_PATH.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at {ARTIFACT_PATH}. Run `python -m src.train` first."
        )
    artifact = joblib.load(ARTIFACT_PATH)
    required = {"model", "county_rates", "school_categories", "model_name"}
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"Model artifact is missing fields: {', '.join(sorted(missing))}")
    return artifact


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"status": "ok", "model_available": ARTIFACT_PATH.is_file()}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        artifact = _load_artifact()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    row = pd.DataFrame(
        [
            {
                "Mean Grade": request.mean_grade,
                "O LEVEL YEAR": request.o_level_year,
                "County": request.county,
                "Parent Status": request.parent_status,
                "High School": request.high_school,
                "Apply_Year": request.date_applied.year,
            }
        ]
    )
    features = build_features(
        row, artifact["county_rates"], artifact["school_categories"]
    )
    probabilities = artifact["model"].predict_proba(features)[0]
    class_probabilities = dict(
        zip(artifact["model"].classes_, probabilities, strict=True)
    )
    approved_probability = float(class_probabilities[1])
    return PredictionResponse(
        prediction="Approved" if approved_probability >= 0.5 else "Not Approved",
        approved_probability=approved_probability,
        not_approved_probability=float(class_probabilities[0]),
        model=artifact["model_name"],
    )
