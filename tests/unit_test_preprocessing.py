import pandas as pd
from churn_mlops_pipeline.scripts.preprocessing_and_feature_engineering import clean_events, featurize

def test_clean_events_simple():
    df = pd.DataFrame([
        {"user_id": "1", "event_type": "a", "event_time": "2023-01-01", "session_id": 1, "paying": None, "country": None, "device_type": None},
        {"user_id": None, "event_type": "b", "event_time": "2023-01-01", "session_id": 2, "paying": None, "country": None, "device_type": None},
    ])
    cleaned = clean_events(df)
    assert cleaned.shape[0] == 1
    assert cleaned.iloc[0]["device_type"] == "unknown"

def test_featurize_simple():
    df = pd.DataFrame([
        {"user_id": "1", "event_type": "a", "event_time": "2023-01-01", "session_id": 1, "paying": False, "country": "US", "device_type": "Android"},
        {"user_id": "1", "event_type": "b", "event_time": "2023-01-02", "session_id": 2, "paying": False, "country": "US", "device_type": "Android"},
    ])
    df["event_time"] = pd.to_datetime(df["event_time"])
    installs = df.groupby("user_id")["event_time"].min().reset_index(name="install_time")
    feats = featurize(df, installs)
    assert feats.shape[0] == 1