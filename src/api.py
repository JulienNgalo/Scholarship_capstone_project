"""Research-only scholarship outcome prediction API (human-review decision support)."""
from __future__ import annotations
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.data_pipeline import build_features

DEFAULT_ARTIFACT = Path(__file__).resolve().parents[1]/'artifacts'/'scholarship_model.joblib'
ARTIFACT_PATH=Path(os.environ.get('MODEL_ARTIFACT',str(DEFAULT_ARTIFACT)))
app=FastAPI(title='Scholarship Outcome API',description='Research-use predictions; not automated award decisions.',version='2.0.0')

class PredictionRequest(BaseModel):
    mean_grade: str = Field(description='Secondary-school mean grade, e.g. B+')
    o_level_year: int
    county: str
    date_applied: datetime
    high_school: str
    parent_status: str = 'Both Parents'
    course: str | None = Field(default=None, description='SU course, used to derive faculty/school')

class PredictionResponse(BaseModel):
    prediction: str
    approved_probability: float
    not_approved_probability: float
    model: str
    note: str = 'Historical approval prediction only; human review required.'

@lru_cache(maxsize=1)
def _load_artifact():
    if not ARTIFACT_PATH.is_file():
        raise FileNotFoundError(f'Model not found at {ARTIFACT_PATH}; run python -m src.train')
    artifact=joblib.load(ARTIFACT_PATH)  # Load only artifacts produced by your own trusted training code.
    required={'model','county_rates','school_categories','faculty_schools','model_name','feature_columns'}
    if not required.issubset(artifact):
        raise ValueError(f'Missing artifact keys: {sorted(required-set(artifact))}')
    return artifact

@app.get('/health')
def health():
    return {'status':'ok','model_available':ARTIFACT_PATH.is_file()}

@app.post('/predict',response_model=PredictionResponse)
def predict(request:PredictionRequest):
    try:
        artifact=_load_artifact()
        row=pd.DataFrame([{'Mean Grade':request.mean_grade,'O LEVEL YEAR':request.o_level_year,
            'County':request.county,'Parent Status':request.parent_status,
            'High School':request.high_school,'Apply_Year':request.date_applied.year,
            'Course':request.course}])
        X=build_features(row,artifact['county_rates'],artifact['school_categories'],artifact['faculty_schools'])
        X=X[artifact['feature_columns']]
        probabilities=dict(zip(artifact['model'].classes_,artifact['model'].predict_proba(X)[0]))
        p=float(probabilities[1]);threshold=float(artifact.get('threshold',.5))
        return PredictionResponse(prediction='Approved' if p>=threshold else 'Not Approved',
            approved_probability=p,not_approved_probability=float(probabilities[0]),
            model=artifact['model_name'])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503,detail=str(exc)) from exc
    except (ValueError,KeyError) as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc
