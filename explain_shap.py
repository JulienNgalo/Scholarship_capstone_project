"""Explain the EXISTING FAO Gradient Boosting model with SHAP.

Run from the Scholarship Project root:
    python explain_shap.py
    python explain_shap.py --applicants-json examples/shap_applicants.json

This script does not train or replace the deployed model or modify the API.
SHAP values are probability contributions for historical *approval* (class 1),
not causal effects or determinations of eligibility.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.data_pipeline import build_features

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "artifacts" / "scholarship_model.joblib"
TRAINING = ROOT / "artifacts" / "training_data.csv"
EXAMPLES = ROOT / "examples" / "shap_applicants.json"
OUT = ROOT / "artifacts" / "shap"
SEED = 42


def feature_groups(preprocessor, all_columns: list[str]) -> list[str]:
    """Map every transformed column back to its original engineered feature.

    The notebook pipeline has 'num' and 'cat' branches, with a OneHotEncoder
    inside cat. Unlike substring matching, this uses the fitted encoder's
    category counts and the fitted ColumnTransformer's output slices.
    """
    groups: list[str | None] = [None] * len(preprocessor.get_feature_names_out())
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" or name not in ("num", "cat"):
            continue
        start = preprocessor.output_indices_[name].start
        if name == "num":
            for offset, col in enumerate(columns):
                groups[start + offset] = str(col)
        else:
            encoder = transformer.named_steps["encoder"]
            output = encoder.get_feature_names_out(columns)
            pos = start
            for col, categories in zip(columns, encoder.categories_):
                # Our fitted OneHotEncoder uses handle_unknown='ignore'
                # without grouped infrequent categories or dropped classes.
                for _ in categories:
                    groups[pos] = str(col)
                    pos += 1
            if pos - start != len(output):
                raise ValueError("Unexpected one-hot encoding layout; review feature_groups().")
    if any(g is None for g in groups) or not set(groups).issubset(all_columns):
        raise ValueError("Could not map the model's transformed columns to original features.")
    return [str(g) for g in groups]


def group_attributions(values: np.ndarray, groups: list[str], columns: list[str]) -> np.ndarray:
    """Sum SHAP contributions of all one-hot columns in the same original feature."""
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D SHAP values, got {values.shape}")
    return np.column_stack([values[:, np.array(groups) == feature].sum(axis=1)
                            for feature in columns])


def load_applicants(path: Path) -> tuple[pd.DataFrame, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not payload or any(not isinstance(x, dict) for x in payload):
        raise ValueError("Applicants JSON must be one object or a nonempty list of objects")
    rows, labels = [], []
    required = ["mean_grade", "o_level_year", "county", "date_applied", "high_school"]
    for i, person in enumerate(payload):
        missing = [x for x in required if x not in person]
        if missing:
            raise ValueError(f"Applicant {i+1} missing required API fields: {missing}")
        date = pd.to_datetime(person["date_applied"], errors="raise")
        rows.append({
            "Mean Grade": person["mean_grade"],
            "O LEVEL YEAR": person["o_level_year"],
            "County": person["county"],
            "High School": person["high_school"],
            "Parent Status": person.get("parent_status", "Both Parents"),
            "Course": person.get("course"),
            "Apply_Year": date.year,
        })
        labels.append(str(person.get("test_id", f"Applicant {i+1}")))
    return pd.DataFrame(rows), labels


def save_global_bar(grouped: np.ndarray, columns: list[str], path: Path) -> pd.DataFrame:
    imp = pd.DataFrame({"feature": columns,
                        "mean_abs_shap_probability": np.abs(grouped).mean(axis=0)})
    imp["mean_abs_percentage_points"] = imp["mean_abs_shap_probability"] * 100
    imp = imp.sort_values("mean_abs_percentage_points", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ranked = imp.iloc[::-1]
    ax.barh(ranked["feature"], ranked["mean_abs_percentage_points"])
    ax.set_xlabel("Mean absolute SHAP contribution (percentage points)")
    ax.set_title("FAO scholarship model — global SHAP importance")
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return imp


def save_beeswarm(values, background, feature_names, outpath):
    e = shap.Explanation(values=np.asarray(values.values),
                         base_values=np.asarray(values.base_values),
                         data=np.asarray(background, dtype=float),
                         feature_names=feature_names)
    shap.plots.beeswarm(e, max_display=16, show=False)
    fig = plt.gcf()
    fig.set_size_inches(11, 7)
    plt.title("SHAP on transformed model features (probability units)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--training-csv", type=Path, default=TRAINING)
    parser.add_argument("--applicants-json", type=Path, default=EXAMPLES,
                        help="One applicant JSON object or a list of applicant objects")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--background-size", type=int, default=100)
    args = parser.parse_args()
    if args.sample_size < 1 or args.background_size < 1:
        parser.error("--sample-size and --background-size must be positive")

    for item in [args.artifact, args.training_csv, args.applicants_json]:
        if not item.is_file():
            raise FileNotFoundError(f"Required file does not exist: {item}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = joblib.load(args.artifact)  # Load only your own trusted model artifact.
    model = artifact["model"]
    if artifact.get("model_name") != "Gradient Boosting":
        raise ValueError("This script expects the deployed Gradient Boosting model")
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    columns = list(artifact["feature_columns"])
    training = pd.read_csv(args.training_csv)
    missing = set(columns).difference(training.columns)
    if missing:
        raise ValueError(f"training_data.csv missing expected feature columns: {sorted(missing)}")
    X = training.loc[:, columns].copy()
    background = X.sample(n=min(len(X), args.background_size), random_state=SEED)
    explained = X.sample(n=min(len(X), args.sample_size), random_state=SEED + 1)
    Zbackground = np.asarray(preprocessor.transform(background), dtype=float)
    Zexplained = np.asarray(preprocessor.transform(explained), dtype=float)
    transformed_names = list(preprocessor.get_feature_names_out())
    mapping = feature_groups(preprocessor, columns)
    if len(mapping) != Zexplained.shape[1]:
        raise ValueError("SHAP grouping does not match the transformed feature count")
    print(f"Model: {artifact['model_name']} | original features: {len(columns)} | transformed: {len(mapping)}")
    print(f"Background rows: {len(background)} | explained rows: {len(explained)}")

    # Interventional probability-space SHAP for a binary GradientBoostingClassifier.
    # The attributions sum to the *approval probability*, not raw log-odds.
    explainer = shap.TreeExplainer(classifier, data=Zbackground,
                                  feature_perturbation="interventional",
                                  model_output="probability")
    global_exp = explainer(Zexplained, check_additivity=False)
    class_pos = list(classifier.classes_).index(1)
    predicted_global = classifier.predict_proba(Zexplained)[:, class_pos]
    reconstructed = np.asarray(global_exp.base_values) + np.asarray(global_exp.values).sum(axis=1)
    error = np.max(np.abs(predicted_global - reconstructed))
    print(f"SHAP global reconstruction max error: {error:.9f}")
    if error > 1e-4:
        raise RuntimeError("SHAP contribution totals differ from predict_proba; refusing to report explanations")

    grouped = group_attributions(global_exp.values, mapping, columns)
    global_table = save_global_bar(grouped, columns, args.output_dir / "global_importance.png")
    global_table.to_csv(args.output_dir / "global_importance.csv", index=False)
    save_beeswarm(global_exp, Zexplained, transformed_names, args.output_dir / "transformed_beeswarm.png")

    input_rows, labels = load_applicants(args.applicants_json)
    applicant_features = build_features(input_rows, artifact["county_rates"],
                                       artifact["school_categories"], artifact["faculty_schools"])
    applicant_features = applicant_features[columns]
    Zlocal = np.asarray(preprocessor.transform(applicant_features), dtype=float)
    local_exp = explainer(Zlocal, check_additivity=False)
    local_grouped = group_attributions(local_exp.values, mapping, columns)
    model_probs = classifier.predict_proba(Zlocal)[:, class_pos]
    baselines = np.asarray(local_exp.base_values, dtype=float).reshape(-1)
    reconstructed_local = baselines + local_grouped.sum(axis=1)
    if not np.allclose(model_probs, reconstructed_local, atol=1e-4):
        raise RuntimeError("Local SHAP reconstruction check failed")
    threshold = float(artifact.get("threshold", .5))
    local_output_rows = []
    for i, (label, predicted) in enumerate(zip(labels, model_probs)):
        row = {
            "test_id": label,
            "prediction": "Approved" if predicted >= threshold else "Not Approved",
            "approval_probability": float(predicted),
            "approval_probability_pct": round(float(predicted)*100, 3),
            "baseline_probability": float(baselines[i]),
            "shap_reconstruction": float(reconstructed_local[i]),
            "threshold": threshold,
        }
        row.update({f"SHAP_pp | {col}": round(float(v)*100, 5)
                    for col, v in zip(columns, local_grouped[i])})
        local_output_rows.append(row)
    local_result = pd.DataFrame(local_output_rows)
    local_result.to_csv(args.output_dir / "local_explanations.csv", index=False)

    # Provide individual charts for selected illustrative cases. Can be edited
    # without changing the model or API; all supplied cases remain in the CSV.
    selected = [0] + [i for i, label in enumerate(labels) if label.lower() in
                      {"test 7", "test 8", "test 9"} and i != 0]
    selected = list(dict.fromkeys(selected))
    for i in selected:
        display = applicant_features.iloc[i].to_numpy(dtype=object)
        individual = shap.Explanation(values=local_grouped[i], base_values=float(baselines[i]),
                                      data=display, feature_names=columns)
        shap.plots.waterfall(individual, max_display=len(columns), show=False)
        fig = plt.gcf()
        fig.set_size_inches(10, 7)
        fig.suptitle(f"{labels[i]}: model approval probability {model_probs[i]:.2%}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        safe_label = "".join(ch if ch.isalnum() else "_" for ch in labels[i]).strip("_")
        fig.savefig(args.output_dir / f"waterfall_{safe_label}.png", dpi=180,
                    bbox_inches="tight")
        plt.close(fig)

    summary = {
        "model": artifact.get("model_name"), "artifact_version": artifact.get("version"),
        "model_threshold": threshold,
        "shap_method": "TreeExplainer / interventional / probability output",
        "shap_units": "probability change; 0.01 equals one percentage point",
        "background_rows": len(background), "global_explained_rows": len(explained),
        "input_features": columns, "transformed_feature_count": len(mapping),
        "max_probability_reconstruction_error": float(error),
        "top_5_features_mean_abs_percentage_points": global_table.head(5)[
            ["feature", "mean_abs_percentage_points"]].to_dict(orient="records"),
        "caveat": "Explains historical approval predictions, not causality, deservingness, or eligibility."
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                    encoding="utf-8")
    print("\nExplained applicants:")
    print(local_result[["test_id", "prediction", "approval_probability_pct"]].to_string(index=False))
    print("\nTop 5 global features (mean absolute percentage point contribution):")
    print(global_table[["feature", "mean_abs_percentage_points"]].head(5).to_string(index=False))
    print(f"\nOutputs saved to: {args.output_dir.resolve()}")

if __name__ == "__main__":
    main()
