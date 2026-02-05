import joblib
import pandas as pd

from . import data_pipeline
process_features = data_pipeline.process_features

MODEL_PATH = 'models/catboost_credit.pkl'
THRESHOLD = 0.42

def load_model():
    return joblib.load(MODEL_PATH)

def apply_policy(pd):
    if pd <0.3:
        return 'approve'
    elif pd < 0.5:
        return "manual_review"
    else:
        return 'reject'

def predict_application(df_raw):
    model = load_model()
    df = process_features(df_raw)
    X = df.drop(columns = ["TARGET"], errors='ignore')

    proba = model.predict_proba(X)[:, 1]
    result = df_raw.copy()
    result["PD"] = proba
    result["DECISION"] = result['PD'].apply(apply_policy)

    return result