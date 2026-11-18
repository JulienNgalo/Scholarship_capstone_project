"""Model scoring and SHAP services for FAO dashboard.

This module uses the EXISTING deployment model and src.data_pipeline.build_features.
It does not train a new model or impose new school-category penalties.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data_pipeline import (GRADE_SCORES, PARENT_STATUS_SCORES,
                               build_features, normalized_key)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "scholarship_model.joblib"
TRAINING_PATH = PROJECT_ROOT / "artifacts" / "training_data.csv"

REQUIRED_INPUTS = (
    "mean_grade", "o_level_year", "county", "date_applied", "high_school",
    "parent_status", "course",
)

# The CSV can use the same snake_case names as the API, or notebook column names.
ALIASES = {
    "id": "student_id", "student id": "student_id", "student_id": "student_id",
    "mean grade": "mean_grade", "mean_grade": "mean_grade",
    "o level year": "o_level_year", "o_level_year": "o_level_year",
    "county": "county", "date applied": "date_applied", "date_applied": "date_applied",
    "high school": "high_school", "high_school": "high_school",
    "parent status": "parent_status", "parent_status": "parent_status",
    "course": "course",
}


@dataclass
class ScoringResult:
    scores: pd.DataFrame  # contains _row_number: 1-based original CSV row number
    features: pd.DataFrame  # indexed by _row_number for SHAP retrieval
    errors: pd.DataFrame


def load_artifact(path: Path = ARTIFACT_PATH) -> dict[str, Any]:
    """Load a trusted artifact created by src.train (joblib is not safe for untrusted files)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}. Run python -m src.train first.")
    artifact = joblib.load(path)
    required = {"model", "model_name", "feature_columns", "county_rates",
                "school_categories", "faculty_schools"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError("Model artifact does not match the current scholarship pipeline")
    if not all(hasattr(artifact["model"], method) for method in ("predict_proba", "named_steps")):
        raise ValueError("Expected a fitted preprocessing + classification pipeline")
    return artifact


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize API/notebook CSV names and reject ambiguous duplicate columns."""
    renamed = df.copy()
    mapped = []
    for c in renamed.columns:
        key = " ".join(str(c).strip().casefold().replace("_", " ").split())
        mapped.append(ALIASES.get(key, str(c).strip()))
    if len(mapped) != len(set(mapped)):
        raise ValueError("Duplicate input columns after normalization; keep one naming style")
    renamed.columns = mapped
    missing = set(REQUIRED_INPUTS).difference(renamed.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")
    return renamed


def _text(value: Any, field: str) -> str:
    if pd.isna(value) or not str(value).strip():
        raise ValueError(f"Missing {field}")
    return str(value).strip()


def _record_to_model_row(record: pd.Series, artifact: dict[str, Any]) -> tuple[dict, list[str]]:
    grade = _text(record["mean_grade"], "mean_grade").upper()
    if grade not in GRADE_SCORES:
        raise ValueError(f"Unsupported mean_grade: {grade}")
    try:
        numeric_year = float(record["o_level_year"])
        o_level_year = int(numeric_year)
        if numeric_year != o_level_year:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError("o_level_year must be a whole year") from None
    try:
        dt = pd.to_datetime(_text(record["date_applied"], "date_applied"), errors="raise")
        application_year = int(dt.year)
    except (ValueError, OverflowError, TypeError):
        raise ValueError("date_applied must be a valid date (e.g. 2026-11-18)") from None
    if not 1950 <= o_level_year <= application_year <= 2100:
        raise ValueError("o_level_year must be 1950–application year; check date_applied")
    county = _text(record["county"], "county")
    high_school = _text(record["high_school"], "high_school")
    course = _text(record["course"], "course")
    parent = _text(record["parent_status"], "parent_status")
    if parent.upper() not in PARENT_STATUS_SCORES:
        raise ValueError(f"Unsupported parent_status: {parent}")
    notes = []
    if normalized_key(county) not in artifact["county_rates"]:
        notes.append("County not mapped; model will impute County Rate")
    if normalized_key(high_school) not in artifact["school_categories"]:
        notes.append("High school not mapped; model uses UNKNOWN category")
    if normalized_key(course) not in artifact["faculty_schools"]:
        notes.append("Course not mapped; model uses UNKNOWN university school")
    model_row = {
        "Mean Grade": grade, "O LEVEL YEAR": o_level_year,
        "County": county, "Parent Status": parent,
        "High School": high_school, "Apply_Year": application_year,
        "Course": course,
    }
    return model_row, notes


def score_applicants(df: pd.DataFrame, artifact: dict[str, Any]) -> ScoringResult:
    """Validate and score rows. Bad rows go to errors; good rows keep stable row numbers."""
    clean = canonicalize_columns(df)
    good_rows: list[dict] = []
    metadata: list[dict] = []
    failures: list[dict] = []
    for i, record in enumerate(clean.to_dict(orient="records"), start=1):
        s = pd.Series(record)
        student_id = record.get("student_id", "")
        if pd.isna(student_id) or not str(student_id).strip():
            student_id = f"Row {i}"
        else:
            student_id = str(student_id).strip()
        try:
            transformed_row, notes = _record_to_model_row(s, artifact)
            good_rows.append({"_row_number": i, **transformed_row})
            metadata.append({
                "_row_number": i, "Student ID": student_id,
                "Mean Grade": transformed_row["Mean Grade"],
                "County": transformed_row["County"],
                "High School": transformed_row["High School"],
                "Course": transformed_row["Course"],
                "Check notes": "; ".join(notes),
            })
        except (ValueError, TypeError) as error:
            failures.append({"CSV row": i, "Student ID": student_id, "Error": str(error)})

    empty_scores = pd.DataFrame(columns=["_row_number", "Student ID", "Approval probability",
                                          "Predicted status", "Not approved probability"])
    empty_features = pd.DataFrame(columns=artifact["feature_columns"])
    errors = pd.DataFrame(failures, columns=["CSV row", "Student ID", "Error"])
    if not good_rows:
        return ScoringResult(empty_scores, empty_features, errors)

    model_rows = pd.DataFrame(good_rows).set_index("_row_number")
    features = build_features(model_rows, artifact["county_rates"],
                              artifact["school_categories"], artifact["faculty_schools"])
    features = features.loc[:, artifact["feature_columns"]]
    labels = list(artifact["model"].classes_)
    if 1 not in labels or 0 not in labels:
        raise ValueError("Model must have class labels 0 and 1 for approval outcomes")
    proba = artifact["model"].predict_proba(features)[:, labels.index(1)]
    threshold = float(artifact.get("threshold", 0.5))
    if not 0 < threshold < 1:
        raise ValueError("Artifact threshold must be between 0 and 1")
    scores = pd.DataFrame(metadata)
    scores["Approval probability"] = np.asarray(proba, dtype=float)
    scores["Not approved probability"] = 1.0 - scores["Approval probability"]
    scores["Predicted status"] = np.where(scores["Approval probability"] >= threshold,
                                           "Approved", "Not Approved")
    return ScoringResult(scores, features, errors)


def get_feature_groups(preprocessor, original_columns: list[str]) -> list[str]:
    """Map transformed (including one-hot) columns back to 10 original model features."""
    total = len(preprocessor.get_feature_names_out())
    groups: list[str | None] = [None] * total
    for branch, transformer, columns in preprocessor.transformers_:
        if branch not in ("num", "cat"):
            continue
        start = preprocessor.output_indices_[branch].start
        if branch == "num":
            for j, col in enumerate(columns):
                groups[start + j] = str(col)
        else:
            encoder = transformer.named_steps["encoder"]
            pos = start
            for col, categories in zip(columns, encoder.categories_):
                for _ in categories:
                    groups[pos] = str(col)
                    pos += 1
            if pos - start != len(encoder.get_feature_names_out(columns)):
                raise ValueError("Unexpected one-hot layout")
    if any(x is None for x in groups) or not set(groups).issubset(original_columns):
        raise ValueError("Could not map transformed columns to model features")
    return [str(g) for g in groups]


def build_shap_explainer(artifact: dict[str, Any], training_path: Path = TRAINING_PATH,
                         background_size: int = 100):
    """Construct a probability-space TreeExplainer; requires transformed training features."""
    import shap
    if not Path(training_path).is_file():
        raise FileNotFoundError(f"For SHAP, place training_data.csv at {training_path}")
    model = artifact["model"]
    if "preprocessor" not in model.named_steps or "classifier" not in model.named_steps:
        raise ValueError("SHAP integration expects a model with preprocessor and classifier steps")
    preprocessor = model.named_steps["preprocessor"]
    original_columns = list(artifact["feature_columns"])
    raw = pd.read_csv(training_path)
    missing = set(original_columns).difference(raw.columns)
    if missing:
        raise ValueError(f"Training features required for SHAP are missing: {sorted(missing)}")
    raw = raw.loc[:, original_columns]
    if raw.empty:
        raise ValueError("SHAP background training dataset is empty")
    bg = raw.sample(n=min(background_size, len(raw)), random_state=42)
    Z_bg = np.asarray(preprocessor.transform(bg), dtype=float)
    explainer = shap.TreeExplainer(
        model.named_steps["classifier"], data=Z_bg,
        feature_perturbation="interventional", model_output="probability"
    )
    return explainer, get_feature_groups(preprocessor, original_columns)


def explain_applicant(features: pd.DataFrame, artifact: dict[str, Any], explainer,
                      groups: list[str]) -> tuple[float, pd.DataFrame]:
    """Return SHAP baseline and per-original-feature contributions in percentage points."""
    if len(features) != 1:
        raise ValueError("Explain exactly one selected student")
    model = artifact["model"]
    Z = np.asarray(model.named_steps["preprocessor"].transform(
        features.loc[:, artifact["feature_columns"]]), dtype=float)
    explanation = explainer(Z, check_additivity=False)
    values = np.asarray(explanation.values, dtype=float)
    if values.ndim == 3:  # defensive handling for multioutput SHAP versions
        values = values[:, :, 1]
    baseline = float(np.asarray(explanation.base_values).reshape(-1)[0])
    labels = list(model.classes_)
    probability = float(model.predict_proba(features)[:, labels.index(1)][0])
    if abs((baseline + values[0].sum()) - probability) > 1e-4:
        raise RuntimeError("SHAP contributions do not reconstruct the approval probability")
    cols = list(artifact["feature_columns"])
    grouped = []
    for col in cols:
        effect = values[0, np.asarray(groups) == col].sum()
        grouped.append({"Feature": col,
                        "Applicant value": str(features.iloc[0][col]),
                        "Contribution (pp)": round(float(effect * 100), 5)})
    contributions = pd.DataFrame(grouped)
    contributions = contributions.iloc[
        np.argsort(-contributions["Contribution (pp)"].abs().to_numpy())
    ].reset_index(drop=True)
    return baseline, contributions
