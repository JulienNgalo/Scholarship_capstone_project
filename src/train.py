"""Train, evaluate, and serialize the scholarship outcome model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.data_pipeline import FEATURE_COLUMNS, load_training_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = PROJECT_ROOT / "artifacts" / "scholarship_model.joblib"


def train_model(
    artifact_path: Path = DEFAULT_ARTIFACT,
    applicants_path: Path | None = None,
    counties_path: Path | None = None,
    schools_path: Path | None = None,
) -> dict[str, Any]:
    """Train a reproducible logistic-regression model and save its metadata."""
    data_args: dict[str, Path] = {}
    if applicants_path is not None:
        data_args["applicants_path"] = applicants_path
    if counties_path is not None:
        data_args["counties_path"] = counties_path
    if schools_path is not None:
        data_args["schools_path"] = schools_path
    features, labels, county_rates, school_categories = load_training_data(**data_args)
    if labels.nunique() != 2:
        raise ValueError("Training data must contain both Approved and Not Approved labels.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "test_rows": int(len(y_test)),
        "training_rows": int(len(y_train)),
    }
    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "county_rates": county_rates,
        "school_categories": school_categories,
        "metrics": metrics,
        "positive_label": "Approved",
        "model_name": "LogisticRegression",
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path, compress=3)
    metrics_path = artifact_path.with_name("metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARTIFACT,
        help="Path for the serialized model artifact.",
    )
    args = parser.parse_args()
    artifact = train_model(artifact_path=args.output)
    print(f"Saved model: {args.output}")
    print(json.dumps(artifact["metrics"], indent=2))


if __name__ == "__main__":
    main()
