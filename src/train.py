"""Group-aware training of the School-free Gradient Boosting scholarship model.

No data leakage: split by Student ID, compare on development folds only, and
evaluate once on a reserved student-level test set. All transforms are fit
inside each CV fold. School retained in audit CSV, never a deployed model feature.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.data_pipeline import (ROOT, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS,
                               FEATURE_COLUMNS, load_training_data)

ARTIFACT_PATH = ROOT / "artifacts" / "scholarship_model.joblib"


def make_pipeline(include_school: bool = False) -> Pipeline:
    cats = CATEGORICAL_COLUMNS + (["School"] if include_school else [])
    preprocess = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", RobustScaler())]), NUMERIC_COLUMNS),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", OneHotEncoder(handle_unknown="ignore",
                                                    sparse_output=False))]), cats),
    ])
    return Pipeline([("preprocessor", preprocess),
                     ("classifier", GradientBoostingClassifier(random_state=42))])


def evaluate(y_true, probabilities, k=50):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(np.unique(y_true)) != 2:
        raise ValueError("Each validation partition must contain both approval classes")
    pred = (probabilities >= .5).astype(int)
    top = np.argsort(-probabilities)[:min(k, len(y_true))]
    return {"accuracy": float(accuracy_score(y_true, pred)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, probabilities)),
            "precision_at_50": float(y_true[top].mean()),
            "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist()}


def _json_dump(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def train_model(artifact_path: Path = ARTIFACT_PATH, *, applicants_path=None,
                counties_path=None, schools_path=None, faculties_path=None,
                include_comparator: bool = True):
    paths = {key: value for key, value in {
        "applicants_path": applicants_path,
        "counties_path": counties_path,
        "schools_path": schools_path,
        "faculties_path": faculties_path}.items() if value is not None}
    eligible, X, y, references, audit = load_training_data(**paths)
    if min(y.value_counts()) < 5:
        raise ValueError("Too few records in one approval class for stratified 5-fold CV")

    # Every applicant ID must be present only on one side of the split.
    groups = eligible["ID"].astype(str).reset_index(drop=True)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_ids, test_ids = next(outer.split(X, y, groups=groups))
    assert set(groups.iloc[train_ids]).isdisjoint(set(groups.iloc[test_ids]))
    X_dev, X_test = X.iloc[train_ids], X.iloc[test_ids]
    y_dev, y_test = y.iloc[train_ids], y.iloc[test_ids]
    development_groups = groups.iloc[train_ids]

    # Only for comparison. 'School' is never an input for the deployed model.
    if include_comparator:
        course_to_school = references[2]
        school = eligible["Course"].map(
            lambda course: course_to_school.get(" ".join(str(course).casefold().split()), "UNKNOWN"))
        X_compare = X.copy()
        X_compare["School"] = school.astype(str)
    else:
        X_compare = None

    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=123)
    cv_records = []
    for fold, (tr, val) in enumerate(inner.split(X_dev, y_dev, groups=development_groups), start=1):
        true_train_ids = train_ids[tr]
        true_val_ids = train_ids[val]
        assert set(groups.iloc[true_train_ids]).isdisjoint(set(groups.iloc[true_val_ids]))
        models = [("Without School", False, X)]
        if include_comparator:
            models.append(("With School", True, X_compare))
        for label, has_school, df in models:
            model = make_pipeline(include_school=has_school)
            model.fit(df.iloc[true_train_ids], y.iloc[true_train_ids])
            p = model.predict_proba(df.iloc[true_val_ids])[:, list(model.classes_).index(1)]
            scores = evaluate(y.iloc[true_val_ids], p)
            cv_records.append({"fold": fold, "model": label,
                               "train_rows": len(tr), "validation_rows": len(val),
                               **scores})

    # Fit ONLY on development subset. Holdout is not used for model selection or SHAP background.
    deployed_model = make_pipeline(include_school=False)
    deployed_model.fit(X_dev, y_dev)
    probabilities_test = deployed_model.predict_proba(X_test)[:, list(deployed_model.classes_).index(1)]
    holdout = evaluate(y_test, probabilities_test)
    assert not any("School" == col for col in FEATURE_COLUMNS)

    artifact = {
        "schema_version": "3.0",
        "model": deployed_model,
        "model_name": "Gradient Boosting — School-free",
        "feature_columns": list(FEATURE_COLUMNS),
        "numeric_columns": list(NUMERIC_COLUMNS),
        "categorical_columns": list(CATEGORICAL_COLUMNS),
        "county_rates": references[0],
        "school_categories": references[1],
        "faculty_schools": references[2],  # For dashboard metadata only.
        "threshold": 0.5,
        "positive_label": "Approved",
        "eligible_programmes": ["Undergraduate Degree", "Diploma", "CPA"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "holdout_metrics": holdout,
        "development_rows": len(train_ids),
        "holdout_rows": len(test_ids),
        "development_unique_students": len(set(groups.iloc[train_ids])),
        "holdout_unique_students": len(set(groups.iloc[test_ids])),
        "test_student_overlap": 0,
        "policy_note": "programme_eligible is a separate eligibility flag, not an ML predictor",
    }
    output = Path(artifact_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output, compress=3)
    cv_df = pd.DataFrame(cv_records)
    cv_df.to_csv(output.parent / "group_cv_folds.csv", index=False)
    summary = (cv_df.drop(columns=["confusion_matrix"])
               .groupby("model").agg({
                   m: ["mean", "std"] for m in
                   ["accuracy", "precision", "recall", "f1", "roc_auc", "precision_at_50"]}))
    summary.to_csv(output.parent / "group_cv_summary.csv")
    # SHAP background must come only from training records for this evaluated model.
    X_dev.to_csv(output.parent / "training_data.csv", index=False)
    # A non-sensitive split manifest for reproducibility: counts, not student IDs.
    _json_dump(output.parent / "data_audit.json", audit)
    _json_dump(output.parent / "holdout_metrics.json", holdout)
    _json_dump(output.parent / "split_summary.json", {
        "development_rows": len(train_ids), "holdout_rows": len(test_ids),
        "development_unique_students": len(set(groups.iloc[train_ids])),
        "holdout_unique_students": len(set(groups.iloc[test_ids])),
        "student_overlap": 0,
        "development_approval_rate": float(y_dev.mean()),
        "holdout_approval_rate": float(y_test.mean()),
        "note": "StratifiedGroupKFold ~80/20 with student-ID grouping; 5-fold group CV on development only",
    })
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ARTIFACT_PATH)
    parser.add_argument("--applicants", type=Path)
    parser.add_argument("--counties", type=Path)
    parser.add_argument("--schools", type=Path)
    parser.add_argument("--faculties", type=Path)
    args = parser.parse_args()
    result = train_model(args.output, applicants_path=args.applicants,
                         counties_path=args.counties, schools_path=args.schools,
                         faculties_path=args.faculties)
    print("Saved:", args.output)
    print("Features:", result["feature_columns"])
    print("Holdout metrics:\n", json.dumps(result["holdout_metrics"], indent=2))


if __name__ == "__main__":
    main()
