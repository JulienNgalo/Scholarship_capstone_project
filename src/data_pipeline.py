"""Data preparation and feature engineering matching fao_project.ipynb.

The historical labels describe past decisions, NOT objective applicant eligibility.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Mapping
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APPLICANTS = PROJECT_ROOT / 'data' / 'fao_2021_2026_applicants_cleaned.csv'
DEFAULT_COUNTIES = PROJECT_ROOT / 'data' / 'County_GVA_Contribution_Ranked_2019_2023.csv'
DEFAULT_SCHOOLS = PROJECT_ROOT / 'data' / 'school_classifications.csv'
DEFAULT_FACULTIES = PROJECT_ROOT / 'data' / 'su_faculty.csv'
NUMERIC_COLUMNS = ['O LEVEL YEAR', 'Apply_Year', 'County Rate', 'mean_grade_score',
                   'high_school_category_score', 'o_level_apply_score', 'parent_status_score']
CATEGORICAL_COLUMNS = ['Parent Status', 'High School Category', 'School']
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
GRADE_SCORES = {'A':10,'A-':9,'B+':8,'B':7,'B-':6,'C+':5,'A LEVELS':4,'DIPLOMA':3,
                'C':2,'C-':1,'D+':0,'D':0,'D-':0,'E':0}
SCHOOL_CATEGORY_SCORES = {'INTERNATIONAL':-1,'NATIONAL':2,'PRIVATE':3,'UNKNOWN':4,
                          'COUNTY':5,'SUB_COUNTY':6}
PARENT_STATUS_SCORES = {'BOTH PARENTS':1,'FATHER DECEASED':3,'MOTHER DECEASED':2,'ORPHAN':4}
STATUS_LABELS = {'Approved':1, 'Not Approved':0}

def normalized_key(value: object) -> str:
    return '' if pd.isna(value) else ' '.join(str(value).strip().casefold().split())

def require_columns(frame: pd.DataFrame, names: set[str], description: str) -> None:
    missing = names.difference(frame.columns)
    if missing:
        raise ValueError(f'{description} missing columns: {sorted(missing)}')

def _read_lookup(path: Path, key: str, value: str, numeric: bool = False) -> dict:
    frame = pd.read_csv(path)
    require_columns(frame, {key, value}, str(path))
    frame = frame[[key,value]].dropna(subset=[key,value]).copy()
    frame['_key'] = frame[key].map(normalized_key)
    frame = frame.loc[frame['_key'].ne('')]
    # Prevent an ambiguous mapping silently varying with row order.
    conflicts = frame.groupby('_key')[value].nunique(dropna=True)
    if (conflicts > 1).any():
        raise ValueError(f'Conflicting {value} values in {path}: {list(conflicts[conflicts>1].index)[:5]}')
    frame = frame.drop_duplicates('_key')
    if numeric:
        frame[value] = pd.to_numeric(frame[value], errors='raise')
    else:
        frame[value] = frame[value].astype(str).str.strip()
    return dict(zip(frame['_key'], frame[value]))

def load_reference_data(counties_path: Path = DEFAULT_COUNTIES,
                        schools_path: Path = DEFAULT_SCHOOLS,
                        faculties_path: Path = DEFAULT_FACULTIES) -> tuple[dict,dict,dict]:
    county_rates = _read_lookup(counties_path, 'County', 'County Rate', numeric=True)
    school_categories = {k:v.upper() for k,v in _read_lookup(
        schools_path, 'High School', 'High School Category').items()}
    faculty_schools = _read_lookup(faculties_path, 'Course', 'School')
    return county_rates, school_categories, faculty_schools

def load_training_frame(applicants_path: Path = DEFAULT_APPLICANTS) -> pd.DataFrame:
    applicants = pd.read_csv(applicants_path)
    require_columns(applicants, {'ID','Course','Status','Country','County','High School',
        'Mean Grade','O LEVEL YEAR','Parent Status','Date Applied'}, str(applicants_path))
    applicants['Status'] = applicants['Status'].astype('string').str.strip().str.title()
    if applicants['Status'].isna().any() or not applicants['Status'].isin(STATUS_LABELS).all():
        raise ValueError('Invalid or missing Status labels; expected Approved and Not Approved')
    # Notebook convention: retain Approved when ID + Course have contradictory statuses.
    applicants['_status_priority'] = applicants['Status'].ne('Approved').astype(int)
    applicants = (applicants.drop_duplicates(['ID','Course','Status'])
                  .sort_values('_status_priority', kind='stable')
                  .drop_duplicates(['ID','Course']).drop(columns='_status_priority').copy())
    applicants = applicants.loc[applicants['Country'].map(normalized_key).eq('kenya') &
        ~applicants['County'].map(normalized_key).eq('international student')].copy()
    applicants['Date Applied'] = pd.to_datetime(applicants['Date Applied'], errors='coerce', format='mixed')
    if applicants['Date Applied'].isna().any():
        raise ValueError('Invalid Date Applied values in training data')
    applicants['Apply_Year'] = applicants['Date Applied'].dt.year
    applicants['Parent Status'] = applicants['Parent Status'].fillna('Both Parents')
    return applicants

def build_features(applicants: pd.DataFrame, county_rates: Mapping,
                   school_categories: Mapping, faculty_schools: Mapping | None = None) -> pd.DataFrame:
    require_columns(applicants, {'Mean Grade','O LEVEL YEAR','County','Parent Status',
        'High School','Apply_Year'}, 'Applicant input')
    faculty_schools = faculty_schools or {}
    grade = applicants['Mean Grade'].astype('string').str.strip().str.upper()
    high_school_cat = applicants['High School'].map(
        lambda value: school_categories.get(normalized_key(value), 'UNKNOWN'))
    parent = applicants['Parent Status'].fillna('Both Parents').astype(str).str.strip().str.upper()
    apply_year = pd.to_numeric(applicants['Apply_Year'], errors='coerce')
    o_year = pd.to_numeric(applicants['O LEVEL YEAR'], errors='coerce')
    gap = apply_year - o_year
    features = pd.DataFrame(index=applicants.index)
    features['O LEVEL YEAR'] = o_year.astype(float)
    features['Apply_Year'] = apply_year.astype(float)
    features['County Rate'] = applicants['County'].map(
        lambda x: county_rates.get(normalized_key(x))).astype(float)
    features['mean_grade_score'] = grade.map(GRADE_SCORES).astype(float)
    features['high_school_category_score'] = high_school_cat.map(SCHOOL_CATEGORY_SCORES).astype(float)
    features['o_level_apply_score'] = np.select(
        [gap.le(1).fillna(False),gap.eq(2).fillna(False),gap.eq(3).fillna(False),
         gap.between(4,5).fillna(False),gap.between(6,7).fillna(False),gap.ge(8).fillna(False)],
        [5,4,3,2,1,0], default=np.nan)
    features['parent_status_score'] = parent.map(PARENT_STATUS_SCORES).astype(float)
    features['Parent Status'] = parent.replace('', 'BOTH PARENTS')
    features['High School Category'] = high_school_cat.astype(str).str.upper()
    if 'Course' in applicants:
        features['School'] = applicants['Course'].map(lambda x: faculty_schools.get(normalized_key(x), 'UNKNOWN'))
    else:
        features['School'] = 'UNKNOWN'
    for name in CATEGORICAL_COLUMNS:
        features[name] = features[name].fillna('UNKNOWN').astype(str)
    return features[FEATURE_COLUMNS]

def load_training_data(applicants_path: Path = DEFAULT_APPLICANTS,
                       counties_path: Path = DEFAULT_COUNTIES,
                       schools_path: Path = DEFAULT_SCHOOLS,
                       faculties_path: Path = DEFAULT_FACULTIES):
    applicants = load_training_frame(applicants_path)
    references = load_reference_data(counties_path, schools_path, faculties_path)
    X = build_features(applicants, *references)
    y = applicants['Status'].map(STATUS_LABELS).astype(int)
    return X,y,*references

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT/'artifacts'/'training_data.csv')
    parser.add_argument('--applicants', type=Path, default=DEFAULT_APPLICANTS)
    parser.add_argument('--counties', type=Path, default=DEFAULT_COUNTIES)
    parser.add_argument('--schools', type=Path, default=DEFAULT_SCHOOLS)
    parser.add_argument('--faculties', type=Path, default=DEFAULT_FACULTIES)
    args = parser.parse_args()
    X,y,*_ = load_training_data(args.applicants,args.counties,args.schools,args.faculties)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    X.assign(Status=y).to_csv(args.output, index=False)
    print(f'Saved {len(X)} records to {args.output}')

if __name__ == '__main__':
    main()
