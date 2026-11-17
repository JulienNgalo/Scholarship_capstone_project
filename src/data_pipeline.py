"""Prepare applicant records and construct the model's training features."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APPLICANTS = PROJECT_ROOT / "data" / "fao_2021_2026_applicants_cleaned.csv"
DEFAULT_COUNTIES = PROJECT_ROOT / "data" / "County_GVA_Contribution_Ranked_2019_2023.csv"
DEFAULT_SCHOOLS = PROJECT_ROOT / "data" / "school_classifications.csv"

FEATURE_COLUMNS = [
    "Apply_Year",
    "mean_grade_score",
    "County Rate",
    "high_school_category_score",
    "o_level_apply_score",
    "parent_status_score",
]

GRADE_SCORES = {
    "A": 9,
    "A-": 9,
    "B+": 8,
    "B": 7,
    "B-": 6,
    "C+": 5,
    "A LEVELS": 4,
    "DIPLOMA": 3,
    "C": 2,
    "C-": 1,
    "D+": 0,
    "D": 0,
    "D-": 0,
    "E": 0,
}
SCHOOL_CATEGORY_SCORES = {
    "INTERNATIONAL": 1,
    "NATIONAL": 2,
    "PRIVATE": 3,
    "UNKNOWN": 4,
    "COUNTY": 5,
    "SUB_COUNTY": 6,
}
PARENT_STATUS_SCORES = {
    "BOTH PARENTS": 1,
    "FATHER DECEASED": 3,
    "MOTHER DECEASED": 2,
    "ORPHAN": 4,
}
STATUS_LABELS = {"Approved": 1, "Not Approved": 0}


def _normalized_key(value: object) -> str:
    return str(value).strip().casefold()


def _require_columns(frame: pd.DataFrame, columns: set[str], source: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(sorted(missing))}")


def load_reference_data(
    counties_path: Path = DEFAULT_COUNTIES,
    schools_path: Path = DEFAULT_SCHOOLS,
) -> tuple[dict[str, int], dict[str, str]]:
    """Load county rates and school categories as normalized lookup maps."""
    counties = pd.read_csv(counties_path)
    schools = pd.read_csv(schools_path)
    _require_columns(counties, {"County", "County Rate"}, str(counties_path))
    _require_columns(schools, {"High School", "High School Category"}, str(schools_path))

    county_rows = counties[["County", "County Rate"]].dropna(
        subset=["County", "County Rate"]
    )
    county_rates = {
        _normalized_key(county): int(rate)
        for county, rate in county_rows.itertuples(index=False, name=None)
    }
    school_categories = {
        _normalized_key(school): str(category).strip().upper()
        for school, category in schools[["High School", "High School Category"]]
        .dropna(subset=["High School", "High School Category"])
        .itertuples(index=False, name=None)
    }
    return county_rates, school_categories


def load_training_frame(
    applicants_path: Path = DEFAULT_APPLICANTS,
    counties_path: Path = DEFAULT_COUNTIES,
    schools_path: Path = DEFAULT_SCHOOLS,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, str]]:
    """Read, validate, deduplicate, and filter records for Kenyan applicants."""
    applicants = pd.read_csv(applicants_path)
    required = {
        "ID",
        "Gender",
        "High School",
        "Mean Grade",
        "O LEVEL YEAR",
        "County",
        "Country",
        "Parent Status",
        "Date Applied",
        "Status",
        "Course",
    }
    _require_columns(applicants, required, str(applicants_path))

    applicants["Status"] = applicants["Status"].astype("string").str.strip()
    unknown_statuses = sorted(set(applicants["Status"].dropna()) - STATUS_LABELS.keys())
    if unknown_statuses:
        raise ValueError(f"Unexpected Status values: {', '.join(unknown_statuses)}")
    if applicants["Status"].isna().any():
        raise ValueError("Applicant data contains rows without a Status label.")

    # Keep one outcome per applicant/course, preferring Approved when labels conflict.
    applicants["_status_priority"] = applicants["Status"].eq("Not Approved")
    applicants = applicants.drop_duplicates(
        subset=["ID", "Course", "Status"], keep="first"
    )
    applicants = applicants.sort_values(
        ["_status_priority"], ascending=True, kind="stable"
    ).drop_duplicates(subset=["ID", "Course"], keep="first")
    applicants = applicants.drop(columns="_status_priority")

    applicants["Date Applied"] = pd.to_datetime(
        applicants["Date Applied"], format="ISO8601", errors="coerce"
    )
    if applicants["Date Applied"].isna().any():
        raise ValueError("Applicant data contains missing or invalid Date Applied values.")
    applicants = applicants[
        applicants["Country"].astype("string").str.strip().str.casefold() == "kenya"
    ].copy()
    applicants = applicants[
        applicants["County"].astype("string").str.strip().str.casefold()
        != "international student"
    ].copy()
    applicants["Parent Status"] = applicants["Parent Status"].fillna("Both Parents")
    applicants["Apply_Year"] = applicants["Date Applied"].dt.year

    county_rates, school_categories = load_reference_data(counties_path, schools_path)
    return applicants, county_rates, school_categories


def build_features(
    applicants: pd.DataFrame,
    county_rates: Mapping[str, int],
    school_categories: Mapping[str, str],
) -> pd.DataFrame:
    """Convert applicant fields into the numeric features used for training."""
    _require_columns(
        applicants,
        {"Mean Grade", "O LEVEL YEAR", "County", "Parent Status", "High School", "Apply_Year"},
        "Applicant frame",
    )
    grade = applicants["Mean Grade"].astype("string").str.strip().str.upper()
    category = applicants["High School"].map(
        lambda value: school_categories.get(_normalized_key(value), "UNKNOWN")
    )
    o_level_year = pd.to_numeric(applicants["O LEVEL YEAR"], errors="coerce")
    gap = applicants["Apply_Year"] - o_level_year

    features = pd.DataFrame(index=applicants.index)
    features["Apply_Year"] = pd.to_numeric(applicants["Apply_Year"], errors="coerce")
    features["mean_grade_score"] = grade.map(GRADE_SCORES).astype(float)
    features["County Rate"] = applicants["County"].map(
        lambda value: county_rates.get(_normalized_key(value))
    )
    features["high_school_category_score"] = category.map(SCHOOL_CATEGORY_SCORES)
    features["o_level_apply_score"] = np.select(
        [
            gap <= 1,
            gap == 2,
            gap == 3,
            gap.between(4, 5),
            gap.between(6, 7),
            gap >= 8,
        ],
        [5, 4, 3, 2, 1, 0],
        default=np.nan,
    )
    features["parent_status_score"] = (
        applicants["Parent Status"]
        .astype("string")
        .str.strip()
        .str.upper()
        .map(PARENT_STATUS_SCORES)
    )
    return features[FEATURE_COLUMNS]


def load_training_data(
    applicants_path: Path = DEFAULT_APPLICANTS,
    counties_path: Path = DEFAULT_COUNTIES,
    schools_path: Path = DEFAULT_SCHOOLS,
) -> tuple[pd.DataFrame, pd.Series, dict[str, int], dict[str, str]]:
    """Load model-ready features and binary labels."""
    applicants, county_rates, school_categories = load_training_frame(
        applicants_path, counties_path, schools_path
    )
    features = build_features(applicants, county_rates, school_categories)
    labels = applicants["Status"].map(STATUS_LABELS).astype(int)
    return features, labels, county_rates, school_categories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "training_data.csv",
        help="Path for the prepared feature and target CSV.",
    )
    args = parser.parse_args()
    features, labels, _, _ = load_training_data()
    prepared = features.copy()
    prepared["Status"] = labels
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(args.output, index=False)
    print(f"Saved {len(prepared)} prepared records to {args.output}")


if __name__ == "__main__":
    main()
