"""Notebook-aligned scholarship features and programme eligibility (November 2026).

Model inputs deliberately exclude University School and programme_eligible.
The latter is an independent eligibility rule; School is retained for reporting.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APPLICANTS = ROOT / "data" / "fa_final.csv"
DEFAULT_COUNTIES = ROOT / "data" / "County_GVA_Contribution_Ranked_2019_2023.csv"
DEFAULT_SCHOOLS = ROOT / "data" / "school_classifications.csv"
DEFAULT_FACULTIES = ROOT / "data" / "su_faculty.csv"

NUMERIC_COLUMNS = ["O LEVEL YEAR", "Apply_Year", "County Rate", "mean_grade_score",
                   "high_school_category_score", "o_level_apply_score", "parent_status_score"]
CATEGORICAL_COLUMNS = ["Parent Status", "High School Category"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS  # No University School

GRADE_SCORES = {"A": 10, "A-": 9, "B+": 8, "B": 7, "B-": 6, "C+": 5,
                "A LEVELS": 4, "DIPLOMA": 3, "C": 2, "C-": 1,
                "D+": 0, "D": 0, "D-": 0, "E": 0}
# This is the *actual* updated map in the supplied November 18 notebook.
SCHOOL_CATEGORY_SCORES = {"INTERNATIONAL": 0, "NATIONAL": 2, "PRIVATE": 3,
                          "UNKNOWN": 4, "COUNTY": 5, "SUB_COUNTY": 6}
PARENT_STATUS_SCORES = {"BOTH PARENTS": 1, "FATHER DECEASED": 3,
                        "MOTHER DECEASED": 2, "ORPHAN": 4}
STATUS_LABELS = {"approved": 1, "not approved": 0}


def normalized_key(value: object) -> str:
    return "" if pd.isna(value) else " ".join(str(value).strip().casefold().split())


def require_columns(frame: pd.DataFrame, names: set[str], context: str) -> None:
    missing = names - set(frame.columns)
    if missing:
        raise ValueError(f"{context} missing columns: {sorted(missing)}")


def classify_programme(course: object) -> tuple[str, int | None]:
    """Apply notebook CPA/degree/diploma policy; unknown programmes require review."""
    if pd.isna(course) or not str(course).strip():
        return "Unclassified", None
    name = re.sub(r"[.\-']", "", str(course).upper())
    name = " ".join(name.split())
    if re.search(r"\bCPA\b|CERTIFIED PUBLIC ACCOUNTANT", name):
        return "CPA", 1
    if re.search(r"\b(MASTER|MASTERS|MSC|MBA|LLM|MPHIL)\b", name):
        return "Masters", 0
    if "POSTGRADUATE DIPLOMA" in name:
        return "Unclassified", None
    if name in {"ASSOCIATION OF CHARTERED CERTIFIED ACCOUNTANTS",
                "CHARTERED FINANCIAL ANALYST", "CHARTERED INSTITUTE OF MARKETING"}:
        return "Other Professional Qualification", 0
    if re.search(r"\bCERTIFICATE\b", name):
        return "Certificate", 0
    if re.search(r"\b(BACHELOR|BACHELORS|BSC|BCOM|BBIT|LLB|BBA)\b", name):
        return "Undergraduate Degree", 1
    if re.search(r"\bDIPLOMA\b", name):
        return "Diploma", 1
    return "Unclassified", None


def lookup_map(path: Path, key: str, value: str, numeric: bool = False) -> dict:
    df = pd.read_csv(path)
    require_columns(df, {key, value}, str(path))
    rows = df[[key, value]].dropna().copy()
    rows["_key"] = rows[key].map(normalized_key)
    rows = rows[rows["_key"].ne("")].copy()
    if numeric:
        rows[value] = pd.to_numeric(rows[value], errors="raise")
    else:
        rows[value] = rows[value].astype(str).str.strip()
    # Duplicate reference keys with distinct values are ambiguous; reject rather than
    # multiplying applicant rows through left merges (53 duplicates in old notebook).
    conflicts = rows.groupby("_key")[value].nunique(dropna=False)
    if (conflicts > 1).any():
        raise ValueError(f"Conflicting {value} lookup entries in {path}: "
                         f"{list(conflicts[conflicts > 1].index)[:10]}")
    rows = rows.drop_duplicates("_key")
    return dict(zip(rows["_key"], rows[value]))


def load_reference_data(counties_path: Path = DEFAULT_COUNTIES,
                        schools_path: Path = DEFAULT_SCHOOLS,
                        faculties_path: Path = DEFAULT_FACULTIES):
    county_rates = lookup_map(Path(counties_path), "County", "County Rate", numeric=True)
    high_schools = {key: str(cat).strip().upper() for key, cat in
                    lookup_map(Path(schools_path), "High School", "High School Category").items()}
    faculties = lookup_map(Path(faculties_path), "Course", "School")
    return county_rates, high_schools, faculties


def prepare_applicants(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Filter Kenya, audit exact duplicates and retain legitimate reapplications."""
    required = {"ID", "Course", "Status", "Country", "County", "High School",
                "Mean Grade", "O LEVEL YEAR", "Parent Status", "Date Applied"}
    require_columns(frame, required, "Application data")
    data = frame.copy()
    initial = len(data)
    data["Status"] = data["Status"].astype("string").str.strip().str.casefold()
    if data["Status"].isna().any() or not data["Status"].isin(STATUS_LABELS).all():
        raise ValueError("Statuses must be Approved or Not Approved")
    data["ID"] = data["ID"].astype("string").str.strip()
    if data["ID"].isna().any() or data["ID"].eq("").any():
        raise ValueError("Student ID cannot be missing for group-aware validation")
    data = data[data["Country"].map(normalized_key).eq("kenya") &
                ~data["County"].map(normalized_key).eq("international student")].copy()
    # Convert ISO and mixed dates in the original notebook export.
    data["Date Applied"] = pd.to_datetime(data["Date Applied"], errors="coerce", format="mixed")
    if data["Date Applied"].isna().any():
        raise ValueError("Invalid Date Applied in source dataset")
    data["Apply_Year"] = data["Date Applied"].dt.year
    # Only remove exact repeated records. Reapplications on different dates are
    # legitimate observations, protected by ID grouping in every split.
    exact_dups = int(data.duplicated().sum())
    data = data.drop_duplicates().copy()
    # Conflicting labels on same application/date cannot be resolved by a model.
    contradictions = (data.groupby(["ID", "Course", "Date Applied"], dropna=False)
                      ["Status"].nunique().gt(1).sum())
    if contradictions:
        raise ValueError(f"{int(contradictions)} repeated ID+Course+Date groups "
                         "have conflicting statuses; resolve these before training")
    data["Parent Status"] = data["Parent Status"].fillna("Both Parents")
    levels = data["Course"].apply(classify_programme)
    data["programme_level"] = levels.map(lambda pair: pair[0])
    data["programme_eligible"] = levels.map(lambda pair: pair[1])
    data["target"] = data["Status"].map(STATUS_LABELS).astype(int)
    audit = {"source_rows": initial, "kenya_rows_before_exact_dedup": len(data) + exact_dups,
             "exact_duplicates_removed": exact_dups,
             "remaining_rows": len(data), "unique_student_ids": int(data["ID"].nunique()),
             "repeated_student_ids": int(data["ID"].value_counts().gt(1).sum()),
             "repeated_id_course_groups": int(data.groupby(["ID", "Course"], dropna=False).size().gt(1).sum()),
             "eligible_rows": int(data["programme_eligible"].eq(1).sum()),
             "ineligible_rows": int(data["programme_eligible"].eq(0).sum()),
             "unclassified_rows": int(data["programme_eligible"].isna().sum())}
    return data.reset_index(drop=True), audit


def build_features(applicants: pd.DataFrame, county_rates: Mapping,
                   school_categories: Mapping,
                   faculty_schools: Mapping | None = None) -> pd.DataFrame:
    """Exactly nine predictive features. School/Course are NEVER model columns."""
    require_columns(applicants, {"Mean Grade", "O LEVEL YEAR", "County", "High School",
                                 "Parent Status", "Apply_Year"}, "Applicant feature input")
    grade = applicants["Mean Grade"].astype("string").str.strip().str.upper()
    high_category = applicants["High School"].map(
        lambda v: school_categories.get(normalized_key(v), "UNKNOWN"))
    high_category = high_category.astype("string").str.upper().fillna("UNKNOWN")
    parent = applicants["Parent Status"].fillna("Both Parents").astype("string").str.strip().str.upper()
    apply_year = pd.to_numeric(applicants["Apply_Year"], errors="coerce")
    o_year = pd.to_numeric(applicants["O LEVEL YEAR"], errors="coerce")
    gap = apply_year - o_year
    X = pd.DataFrame(index=applicants.index)
    X["O LEVEL YEAR"] = o_year.astype(float)
    X["Apply_Year"] = apply_year.astype(float)
    X["County Rate"] = applicants["County"].map(
        lambda v: county_rates.get(normalized_key(v))).astype(float)
    X["mean_grade_score"] = grade.map(GRADE_SCORES).astype(float)
    X["high_school_category_score"] = high_category.map(SCHOOL_CATEGORY_SCORES).astype(float)
    X["o_level_apply_score"] = np.select(
        [gap.le(1).fillna(False), gap.eq(2).fillna(False), gap.eq(3).fillna(False),
         gap.between(4, 5).fillna(False), gap.between(6, 7).fillna(False), gap.ge(8).fillna(False)],
        [5, 4, 3, 2, 1, 0], default=np.nan)
    X["parent_status_score"] = parent.map(PARENT_STATUS_SCORES).astype(float)
    X["Parent Status"] = parent.replace("", "BOTH PARENTS").fillna("BOTH PARENTS").astype(str)
    X["High School Category"] = high_category.astype(str)
    return X.loc[:, FEATURE_COLUMNS]


def load_training_data(applicants_path: Path = DEFAULT_APPLICANTS,
                       counties_path: Path = DEFAULT_COUNTIES,
                       schools_path: Path = DEFAULT_SCHOOLS,
                       faculties_path: Path = DEFAULT_FACULTIES):
    data, audit = prepare_applicants(pd.read_csv(applicants_path, dtype={"ID": str}))
    references = load_reference_data(counties_path, schools_path, faculties_path)
    eligible = data[data["programme_eligible"].eq(1)].copy().reset_index(drop=True)
    if eligible.empty:
        raise ValueError("No eligible degree, diploma or CPA applicants in training data")
    features = build_features(eligible, *references)
    return eligible, features, eligible["target"].astype(int), references, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--applicants", type=Path, default=DEFAULT_APPLICANTS)
    parser.add_argument("--counties", type=Path, default=DEFAULT_COUNTIES)
    parser.add_argument("--schools", type=Path, default=DEFAULT_SCHOOLS)
    parser.add_argument("--faculties", type=Path, default=DEFAULT_FACULTIES)
    args = parser.parse_args()
    eligible, X, y, refs, audit = load_training_data(args.applicants, args.counties,
                                                      args.schools, args.faculties)
    print(json.dumps(audit, indent=2))
    print(f"Predictive features: {FEATURE_COLUMNS}")
    print(f"Eligible model-ready rows: {len(X)}")


if __name__ == "__main__":
    main()
