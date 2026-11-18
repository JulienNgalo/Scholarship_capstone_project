"""Run: python -m pytest tests/test_v3_integration.py -v"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.data_pipeline import (FEATURE_COLUMNS, build_features, classify_programme,
                               lookup_map, load_training_data)
from src.train import train_model
from src.dashboard_service import score_applicants, explain_applicant, build_shap_explainer


@pytest.fixture(scope="module")
def fixture_artifact(tmp_path_factory):
    base = tmp_path_factory.mktemp("fao")
    df = []
    for i in range(90):
        sid = f"STUDENT-{i:04d}"
        for j in range(2):
            df.append({
                "ID": sid, "Course": ["Bachelor of Commerce", "Diploma in Technology"][j],
                "Status": "Approved" if (i % 3 != 0 or j == 1 and i % 5 == 0) else "Not Approved",
                "Country": "Kenya", "County": "Nairobi" if i % 2 else "Kiambu",
                "High School": "Alliance High School" if i % 2 else "Example County High School",
                "Mean Grade": ["B+", "B", "C+"][i % 3],
                "O LEVEL YEAR": 2022 + i % 3,
                "Parent Status": "Orphan" if i % 7 == 0 else "Both Parents",
                "Date Applied": f"2026-02-{(i % 27)+1:02d}",
            })
    for i in range(10):
        df.append({"ID": f"MASTER-{i}", "Course": "Master of Business Administration",
                   "Status": "Approved" if i % 2 else "Not Approved", "Country": "Kenya",
                   "County": "Nairobi", "High School": "Alliance High School", "Mean Grade": "B+",
                   "O LEVEL YEAR": 2021, "Parent Status": "Both Parents", "Date Applied": "2026-01-11"})
    # Reference duplicate key should not create extra applicant rows
    df.append(df[0].copy())
    input_csv = base / "applicants.csv"
    pd.DataFrame(df).to_csv(input_csv, index=False)
    county_csv = base / "counties.csv"
    pd.DataFrame({"County": ["Nairobi", "Kiambu", "Nairobi"],
                  "County Rate": [1, 2, 1]}).to_csv(county_csv, index=False)
    high_csv = base / "schools.csv"
    pd.DataFrame({"High School": ["Alliance High School", "Example County High School"],
                  "High School Category": ["NATIONAL", "COUNTY"]}).to_csv(high_csv, index=False)
    faculty_csv = base / "faculty.csv"
    pd.DataFrame({"Course": ["Bachelor of Commerce", "Diploma in Technology", "Master of Business Administration", "CPA"],
                  "School": ["SBS", "SIMT", "SBS-PG", "SBS"]}).to_csv(faculty_csv, index=False)
    output = base / "artifacts" / "scholarship_model.joblib"
    model = train_model(output, applicants_path=input_csv, counties_path=county_csv,
                        schools_path=high_csv, faculties_path=faculty_csv)
    return base, model, output


def test_notebook_features_and_policy():
    assert classify_programme("Bachelor of Commerce") == ("Undergraduate Degree", 1)
    assert classify_programme("Diploma in IT") == ("Diploma", 1)
    assert classify_programme("CPA") == ("CPA", 1)
    assert classify_programme("Master of Business Administration") == ("Masters", 0)
    assert classify_programme("Chartered Financial Analyst") == ("Other Professional Qualification", 0)
    assert classify_programme("Postgraduate Diploma") == ("Unclassified", None)
    X = build_features(pd.DataFrame([{"Mean Grade": "A", "O LEVEL YEAR": 2025,
              "Apply_Year": 2026, "County": "Nairobi", "High School": "British International School",
              "Parent Status": "Orphan", "Course": "Bachelor of Commerce"}]),
              {"nairobi": 2}, {"british international school": "INTERNATIONAL"})
    assert X.columns.tolist() == FEATURE_COLUMNS
    assert X["high_school_category_score"].iloc[0] == 0
    assert X["o_level_apply_score"].iloc[0] == 5
    assert "School" not in X.columns
    assert "programme_eligible" not in X.columns


def test_group_training_and_artifact(fixture_artifact):
    base, model, output = fixture_artifact
    assert output.exists()
    assert model["schema_version"] == "3.0"
    assert "School" not in model["feature_columns"]
    assert model["test_student_overlap"] == 0
    assert model["holdout_rows"] > 0
    assert (base / "artifacts" / "group_cv_folds.csv").exists()
    assert (base / "artifacts" / "training_data.csv").exists()
    audit = json.loads((base / "artifacts" / "data_audit.json").read_text())
    assert audit["exact_duplicates_removed"] >= 1
    assert audit["ineligible_rows"] == 10


def test_dashboard_retain_university_school_but_not_shap(fixture_artifact):
    base, artifact, _ = fixture_artifact
    input_data = pd.DataFrame([
        {"student_id": "E1", "mean_grade": "B+", "o_level_year": 2023,
         "county": "Nairobi", "date_applied": "2026-11-18", "high_school": "Alliance High School",
         "parent_status": "Both Parents", "course": "Bachelor of Commerce"},
        {"student_id": "E2", "mean_grade": "B", "o_level_year": 2023,
         "county": "Nairobi", "date_applied": "2026-11-18", "high_school": "Alliance High School",
         "parent_status": "Both Parents", "course": "Master of Business Administration"},
        {"student_id": "E3", "mean_grade": "B", "o_level_year": 2023,
         "county": "Nairobi", "date_applied": "2026-11-18", "high_school": "Alliance High School",
         "parent_status": "Both Parents", "course": "Unknown Professional Training"},
    ])
    scored = score_applicants(input_data, artifact)
    assert scored.errors.empty
    assert scored.scores["University School"].tolist() == ["SBS", "SBS-PG", "UNKNOWN"]
    assert scored.scores["Predicted status"].iloc[1:].tolist() == ["Not Eligible", "Manual Review"]
    assert scored.scores["Approval probability"].iloc[1:].isna().all()
    assert len(scored.features) == 1
    explainer, groups = build_shap_explainer(artifact, base / "artifacts" / "training_data.csv", 25)
    baseline, effects = explain_applicant(scored.features.iloc[[0]], artifact, explainer, groups)
    assert "School" not in effects["Feature"].tolist()
    assert len(effects) == 9
    approx = baseline + effects["Contribution (pp)"].sum() / 100
    assert np.isclose(approx, scored.scores["Approval probability"].iloc[0], atol=1e-4)


def test_reference_mapping_conflicts(tmp_path):
    p = tmp_path / "duplicates.csv"
    pd.DataFrame({"County": ["Nairobi", "Nairobi"], "County Rate": [1, 3]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="Conflicting"):
        lookup_map(p, "County", "County Rate", numeric=True)


def test_api_endpoints(fixture_artifact, monkeypatch):
    import src.api as api
    base, artifact, file = fixture_artifact
    api._load_artifact.cache_clear()
    monkeypatch.setattr(api, "ARTIFACT_PATH", file)
    client = TestClient(api.app)
    assert client.get("/health").json()["model_ready"] is True
    info = client.get("/model-info")
    assert info.status_code == 200
    assert info.json()["university_school_is_predictive"] is False
    request = {"mean_grade": "B+", "o_level_year": 2023, "county": "Nairobi",
               "date_applied": "2026-11-18T10:00:00", "high_school": "Alliance High School",
               "parent_status": "Both Parents", "course": "Bachelor of Commerce"}
    r = client.post("/predict", json=request)
    assert r.status_code == 200, r.text
    assert r.json()["university_school"] == "SBS"
    assert r.json()["programme_eligible"] == 1
    assert 0 < r.json()["approved_probability"] < 1
    request["course"] = "Master of Business Administration"
    r2 = client.post("/predict", json=request)
    assert r2.status_code == 200
    assert r2.json()["prediction"] == "Not Eligible"
    assert r2.json()["approved_probability"] is None
    api._load_artifact.cache_clear()
