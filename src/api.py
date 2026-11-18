"""Research-only student-level prediction API, version 3 (group-aware, School-free)."""
from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data_pipeline import (FEATURE_COLUMNS, GRADE_SCORES,
                               PARENT_STATUS_SCORES, build_features,
                               classify_programme, normalized_key)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = Path(os.environ.get("MODEL_ARTIFACT", str(ROOT / "artifacts" / "scholarship_model.joblib")))
app = FastAPI(title="FAO Scholarship Decision-Support API",
              description="Human-review research predictions, separate from programme eligibility.",
              version="3.0.0")


class PredictionRequest(BaseModel):
    student_id: str | None = Field(default=None, description="Optional reference identifier")
    mean_grade: str = Field(min_length=1, description="KCSE mean grade, e.g. B+")
    o_level_year: int = Field(ge=1950, le=2100)
    county: str = Field(min_length=1)
    date_applied: datetime
    high_school: str = Field(min_length=1)
    parent_status: str = Field(default="Both Parents")
    course: str = Field(min_length=1, description="Used to classify programme eligibility and display faculty, not a model feature")


class PredictionResponse(BaseModel):
    student_id: str | None
    university_school: str
    programme_level: str
    programme_eligible: int | None
    prediction: str
    approved_probability: float | None
    not_approved_probability: float | None
    model: str
    note: str


@lru_cache(maxsize=1)
def _load_artifact():
    if not ARTIFACT_PATH.is_file():
        raise FileNotFoundError(f"Missing artifact at {ARTIFACT_PATH}; run python -m src.train")
    model_data = joblib.load(ARTIFACT_PATH)  # NEVER load untrusted joblib bytes.
    expected = {"schema_version", "model", "feature_columns", "model_name",
                "county_rates", "school_categories", "faculty_schools"}
    if not isinstance(model_data, dict) or not expected.issubset(model_data):
        raise ValueError("Artifact is not the v3 school-free model. Run python -m src.train")
    if model_data["schema_version"] != "3.0" or list(model_data["feature_columns"]) != FEATURE_COLUMNS:
        raise ValueError("Outdated artifact (School was a predictive feature). Retrain with v3 train.py")
    return model_data


@app.get("/health")
def health():
    try:
        a = _load_artifact()
        return {"status": "ok", "model_ready": True, "schema_version": a["schema_version"]}
    except (FileNotFoundError, ValueError):
        return {"status": "model_not_ready", "model_ready": False, "schema_version": "3.0"}


@app.get("/model-info")
def model_info():
    try:
        a = _load_artifact()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"model": a["model_name"], "schema_version": a["schema_version"],
            "feature_columns": a["feature_columns"], "threshold": a["threshold"],
            "university_school_is_predictive": False,
            "programme_eligible_is_predictive": False,
            "programme_eligible_is_a_separate_rule": True}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        artifact = _load_artifact()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    grade = request.mean_grade.strip().upper()
    parent = request.parent_status.strip().upper()
    if grade not in GRADE_SCORES:
        raise HTTPException(status_code=422, detail=f"Unknown mean grade: {grade}")
    if parent not in PARENT_STATUS_SCORES:
        raise HTTPException(status_code=422, detail=f"Unknown parent status: {request.parent_status}")
    if request.o_level_year > request.date_applied.year:
        raise HTTPException(status_code=422, detail="O-level year cannot follow application year")
    programme_level, eligible = classify_programme(request.course)
    university_school = artifact["faculty_schools"].get(normalized_key(request.course), "UNKNOWN")
    common = {"student_id": request.student_id, "university_school": str(university_school),
              "programme_level": programme_level, "programme_eligible": eligible,
              "model": artifact["model_name"]}
    if eligible != 1:
        status = "Not Eligible" if eligible == 0 else "Manual Review"
        return PredictionResponse(**common, prediction=status,
                                  approved_probability=None, not_approved_probability=None,
                                  note="Programme eligibility rule: no ML approval probability was calculated.")

    input_frame = pd.DataFrame([{"Mean Grade": grade, "O LEVEL YEAR": request.o_level_year,
                                 "County": request.county, "Apply_Year": request.date_applied.year,
                                 "High School": request.high_school,
                                 "Parent Status": request.parent_status, "Course": request.course}])
    X = build_features(input_frame, artifact["county_rates"], artifact["school_categories"])
    try:
        p = float(artifact["model"].predict_proba(X)[0, list(artifact["model"].classes_).index(1)])
    except Exception as error:
        # The terminal log retains the exception; avoid leaking internals through API.
        import logging
        logging.exception("Model inference failed")
        raise HTTPException(status_code=500, detail="Inference failed; see application logs") from error
    threshold = float(artifact.get("threshold", 0.5))
    return PredictionResponse(**common,
                              prediction="Approved" if p >= threshold else "Not Approved",
                              approved_probability=p, not_approved_probability=1 - p,
                              note="Historical approval prediction only; human review required.")
