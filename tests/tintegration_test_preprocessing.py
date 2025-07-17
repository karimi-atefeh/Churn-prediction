# churn_mlops_pipeline/tests/test_preprocessing_s3.py

import pytest
import pandas as pd
from churn_mlops_pipeline.scripts.preprocessing_and_feature_engineering import clean_events, featurize
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


PARQUET_PATH = "churn_mlops_pipeline/prepration_data_test.parquet"

@pytest.mark.slow
def test_clean_events():
    df = pd.read_parquet(PARQUET_PATH)
    cleaned = clean_events(df)
    assert not cleaned.isnull().any().any()
    assert "user_id" in cleaned.columns
    assert cleaned.shape[0] > 0

@pytest.mark.slow
def test_featurize():
    df = pd.read_parquet(PARQUET_PATH)
    installs = df.groupby("user_id")["event_time"].min().reset_index(name="install_time")
    feats = featurize(df, installs)
    assert "user_id" in feats.columns
    assert feats.shape[0] > 0