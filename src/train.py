"""Train notebook-aligned benchmark models and save a deployable Gradient Boosting pipeline."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from src.data_pipeline import (PROJECT_ROOT, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS,
                               FEATURE_COLUMNS, load_training_data)

DEFAULT_ARTIFACT = PROJECT_ROOT/'artifacts'/'scholarship_model.joblib'

def make_pipeline(estimator):
    transformer = ColumnTransformer([
        ('num', Pipeline([('num_imputer',SimpleImputer(strategy='median')),
                          ('num_scaler',RobustScaler())]), NUMERIC_COLUMNS),
        ('cat',Pipeline([('cat_imputer',SimpleImputer(strategy='most_frequent')),
                         ('encoder',OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL_COLUMNS),
    ])
    return Pipeline([('preprocessor', transformer), ('classifier', estimator)])

def get_estimators(include_xgboost=False):
    estimators = {
        'Logistic Regression': LogisticRegression(max_iter=2000,random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    }
    if include_xgboost:
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError('Install xgboost to use --include-xgboost') from error
        estimators['XGBoost'] = XGBClassifier(random_state=42, eval_metric='logloss')
    return estimators

def train_model(artifact_path=DEFAULT_ARTIFACT, applicants_path=None,
                counties_path=None, schools_path=None, faculties_path=None,
                include_xgboost=False):
    paths = {key:value for key,value in {
        'applicants_path':applicants_path,'counties_path':counties_path,
        'schools_path':schools_path,'faculties_path':faculties_path}.items() if value is not None}
    X,y,county_rates,school_categories,faculty_schools = load_training_data(**paths)
    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError('At least two examples of each Status class are needed')
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    if y_test.nunique() < 2:
        raise ValueError('Test split needs both status classes; provide more records')
    results = []
    # The final GB model is intentionally chosen to match the notebook, not based on the test scores.
    selected_model = None
    for name, estimator in get_estimators(include_xgboost).items():
        pipeline = make_pipeline(estimator)
        pipeline.fit(X_train,y_train)
        pred = pipeline.predict(X_test)
        prob = pipeline.predict_proba(X_test)[:,list(pipeline.classes_).index(1)]
        results.append({'model':name,'accuracy':accuracy_score(y_test,pred),
            'precision':precision_score(y_test,pred,zero_division=0),
            'recall':recall_score(y_test,pred,zero_division=0),
            'f1':f1_score(y_test,pred,zero_division=0),
            'roc_auc':roc_auc_score(y_test,prob),
            'confusion_matrix':confusion_matrix(y_test,pred,labels=[0,1]).tolist()})
        if name == 'Gradient Boosting':
            selected_model = pipeline
    # Fit final deployment model on all available labeled rows after held-out evaluation.
    deployed_model = make_pipeline(GradientBoostingClassifier(random_state=42)).fit(X,y)
    metrics = next(row for row in results if row['model']=='Gradient Boosting')
    artifact = {'model':deployed_model,'model_name':'Gradient Boosting',
                'feature_columns':FEATURE_COLUMNS,'county_rates':county_rates,
                'school_categories':school_categories,'faculty_schools':faculty_schools,
                'metrics':metrics,'positive_label':'Approved',
                'test_rows':len(y_test),'train_rows':len(y_train),
                'total_rows':len(y),'threshold':0.5,
                'version':'2.0','trained_at':datetime.now(timezone.utc).isoformat(),
                'note':'Historical approval prediction for human review, not scholarship eligibility.'}
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact,artifact_path,compress=3)
    (artifact_path.parent/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
    (artifact_path.parent/'model_comparison.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    pd.DataFrame([{k:v for k,v in row.items() if k!='confusion_matrix'} for row in results]).to_csv(
        artifact_path.parent/'model_comparison.csv',index=False)
    return artifact

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_ARTIFACT)
    parser.add_argument('--applicants',type=Path)
    parser.add_argument('--counties',type=Path)
    parser.add_argument('--schools',type=Path)
    parser.add_argument('--faculties',type=Path)
    parser.add_argument('--include-xgboost',action='store_true')
    args=parser.parse_args()
    artifact=train_model(args.output,args.applicants,args.counties,args.schools,args.faculties,args.include_xgboost)
    print(f'Saved model: {args.output}')
    print(json.dumps(artifact['metrics'],indent=2))
if __name__=='__main__':
    main()
