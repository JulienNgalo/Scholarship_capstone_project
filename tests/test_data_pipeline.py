import pandas as pd
from src.data_pipeline import build_features

def test_notebook_scores():
    df=pd.DataFrame([{'Mean Grade':'A','O LEVEL YEAR':2025,'County':'Nairobi',
                      'Parent Status':'Orphan','High School':'School A', 'Apply_Year':2026,
                      'Course':'Course A'},
                     {'Mean Grade':'B+','O LEVEL YEAR':2018,'County':'Other',
                      'Parent Status':'Both Parents','High School':'missing','Apply_Year':2026,
                      'Course':'Course B'}])
    features=build_features(df,{'nairobi':10},{'school a':'COUNTY'},{'course a':'SBS'})
    assert features['mean_grade_score'].tolist()==[10,8]
    assert features['o_level_apply_score'].tolist()==[5,0]
    assert features['high_school_category_score'].tolist()==[5,4]
    assert features['parent_status_score'].tolist()==[4,1]
    #assert features['School'].tolist()==['SBS','UNKNOWN']
    # Confirm that School is excluded from model inputs
    assert "School" not in features.columns

    # Confirm that only the intended predictive features are returned
    from src.data_pipeline import FEATURE_COLUMNS
    assert features.columns.tolist() == FEATURE_COLUMNS 
