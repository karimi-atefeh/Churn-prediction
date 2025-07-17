#!/usr/bin/env python

import os
import argparse
import pandas as pd
import numpy as np
import joblib
import s3fs
import shap

def load_data(path):
    """Load dataframe from S3 or local."""
    fs = s3fs.S3FileSystem(anon=False)
    if path.startswith("s3://"):
        df = pd.read_parquet(path, storage_options={'anon': False})
    else:
        df = pd.read_parquet(path)
    return df

def load_artifact(path):
    """Load artifact (model/scaler/encoder) from S3 or local."""
    fs = s3fs.S3FileSystem(anon=False)
    if path.startswith("s3://"):
        with fs.open(path, "rb") as f:
            obj = joblib.load(f)
    else:
        obj = joblib.load(path)
    return obj

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", type=str, required=True, help="Input features Parquet path")
    parser.add_argument("--model-path", type=str, required=True, help="Trained model .pkl")
    parser.add_argument("--scaler-path", type=str, required=True, help="StandardScaler .pkl")
    parser.add_argument("--encoder-path", type=str, required=True, help="OneHotEncoder .pkl (only used if categoricals exist)")
    parser.add_argument("--output-path", type=str, required=True, help="Output parquet path")
    args = parser.parse_args()

    # 1. Load data and model artifacts
    df = load_data(args.features_path)
    model = load_artifact(args.model_path)
    scaler = load_artifact(args.scaler_path)
    encoder = load_artifact(args.encoder_path)

    # 2. Prepare columns
    X = df.drop(columns=["user_id"])
    categorical_cols = [c for c in X.columns if c.startswith("country") or c.startswith("device_type")]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    use_encoder = len(categorical_cols) > 0

    # 3. Prepare X_final and feature_names
    if use_encoder:
        # Handle categorical and numeric features
        X_cat = encoder.transform(X[categorical_cols])
        X_num = scaler.transform(X[numeric_cols])
        X_final = np.hstack([X_num, X_cat])
        # Feature names = numeric + encoded categoricals
        feature_names = list(numeric_cols) + list(encoder.get_feature_names_out(categorical_cols))
    else:
        # Only numeric features
        X_final = scaler.transform(X[numeric_cols])
        feature_names = list(numeric_cols)

    # 4. Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_final)

    # 5. Churn probability and label
    churn_proba = model.predict_proba(X_final)[:, 1]
    churn_label = (churn_proba >= 0.5).astype(int)

    # 6. Get top 3 important features per user
    top_feats = []
    for i in range(X_final.shape[0]):
        abs_shap = np.abs(shap_values[i])
        top_idx = abs_shap.argsort()[-3:][::-1]
        feats = [(feature_names[j], shap_values[i][j]) for j in top_idx]
        top_feats.append(feats)

    # 7. Add output columns to df
    df["churn_probability"] = churn_proba
    df["churn_label"] = churn_label
    for k in range(3):
        df[f"top{k+1}_feature"] = [x[k][0] for x in top_feats]
        df[f"top{k+1}_shap_value"] = [x[k][1] for x in top_feats]

    # 8. Save result
    save_cols = ["user_id", "churn_probability", "churn_label"] + \
        [f"top{i}_feature" for i in range(1, 4)] + [f"top{i}_shap_value" for i in range(1, 4)]

    if args.output_path.startswith("s3://"):
        df[save_cols].to_parquet(args.output_path, index=False, storage_options={'anon': False})
    else:
        df[save_cols].to_parquet(args.output_path, index=False)

    print("Saved per-user SHAP explanations to:", args.output_path)