import os
import argparse
import pandas as pd
import numpy as np
import joblib
import s3fs
from datetime import datetime
import re
import glob


def find_latest_parquet(input_dir):
    """
    Find the latest date-versioned .parquet file in an S3 or local directory.
    For S3, expects folders named as YYYY-MM-DD or YYYY-MM-DD_HH-MM-SS.
    """
    if input_dir.startswith("s3://"):
        fs = s3fs.S3FileSystem(anon=False)
        items = fs.ls(input_dir, detail=True)
        # Find timestamped folders 
        date_re = re.compile(r".*/(\d{4}-\d{2}-\d{2}(?:_\d{2}-\d{2}-\d{2})?)$")
        subfolders = [
            (f["Key"] if "Key" in f else f["name"])
            for f in items
            if f["type"] == "directory" and date_re.match(f["Key"] if "Key" in f else f["name"])
        ]
        if not subfolders:
            raise ValueError(f"No date-like folders found in {input_dir}")
        latest_folder = sorted(subfolders)[-1]
        # List parquet files in the latest folder
        parquet_files = fs.ls(latest_folder)
        parquet_files = [
            "s3://" + f if not f.startswith("s3://") else f
            for f in parquet_files if f.endswith(".parquet")
        ]
        if not parquet_files:
            raise ValueError(f"No parquet files found in {latest_folder}")
        return parquet_files[0]
    else:
        # Local mode
        subfolders = [os.path.join(input_dir, d) for d in os.listdir(input_dir)
                      if os.path.isdir(os.path.join(input_dir, d)) and re.match(r"\d{4}-\d{2}-\d{2}(_\d{2}-\d{2}-\d{2})?$", d)]
        if not subfolders:
            raise ValueError(f"No data subfolders found in {input_dir}")
        latest_folder = sorted(subfolders)[-1]
        parquet_files = glob.glob(os.path.join(latest_folder, "*.parquet"))
        if not parquet_files:
            raise ValueError(f"No parquet files found in {latest_folder}")
        return parquet_files[0]

def load_artifact(path):
    """
    Loads a model/artifact from S3 or local path using joblib.
    """
    fs = s3fs.S3FileSystem(anon=False)
    if path.startswith("s3://"):
        with fs.open(path, "rb") as f:
            obj = joblib.load(f)
    else:
        obj = joblib.load(path)
    return obj

def find_latest_artifact(s3_dir, pattern):
    """
    Finds the latest artifact in an S3 directory matching the regex pattern.
    Returns the S3 path of the latest artifact.
    """
    fs = s3fs.S3FileSystem(anon=False)
    files = fs.ls(s3_dir)
    # Only keep files matching the pattern
    matched = [f for f in files if re.match(pattern, os.path.basename(f))]
    if not matched:
        raise ValueError(f"No files matching {pattern} in {s3_dir}")
    latest = sorted(matched)[-1]
    return "s3://" + latest if not latest.startswith("s3://") else latest

def find_latest_artifact_local(local_dir, pattern):
    """Find the latest file matching pattern in a local directory."""
    files = glob.glob(os.path.join(local_dir, "*"))
    matched = [f for f in files if re.match(pattern, os.path.basename(f))]
    if not matched:
        raise ValueError(f"No files matching {pattern} in {local_dir}")
    latest = sorted(matched)[-1]
    return latest

def auto_resolve_path(input_path, file_pattern):
    if input_path.startswith("s3://") and not input_path.endswith(".pkl"):
        return find_latest_artifact(input_path, file_pattern)
    elif os.path.isdir(input_path):  # Local directory (like /opt/ml/processing/model/)
        return find_latest_artifact_local(input_path, file_pattern)
    else:
        return input_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=str, required=True, help="Directory (S3 or local) containing features.parquet files, or a direct features.parquet file path")
    parser.add_argument("--model-path", type=str, required=True, help="S3 folder or file path for model(s)")
    parser.add_argument("--scaler-path", type=str, required=True, help="S3 folder or file path for scaler(s)")
    parser.add_argument("--encoder-path", type=str, required=True, help="S3 folder or file path for encoder(s)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output S3 or local directory to save predictions (timestamped)")
    args = parser.parse_args()

    # Find the latest features file
    features_path = find_latest_parquet(args.features_dir)
    print("Using features file:", features_path)

    # Automatically select the latest model/scaler/encoder if a directory is given
    model_path = auto_resolve_path(args.model_path, r"xgboost_model\d*\.pkl")
    scaler_path = auto_resolve_path(args.scaler_path, r"standard_scaler\d*\.pkl")
    encoder_path = auto_resolve_path(args.encoder_path, r"onehot_encoder.*\.pkl")
    print("Using model:", model_path)
    print("Using scaler:", scaler_path)
    print("Using encoder:", encoder_path)

    # Read features
    if features_path.startswith("s3://"):
        df = pd.read_parquet(features_path, storage_options={'anon': False})
    else:
        df = pd.read_parquet(features_path)

    # Load model, scaler, encoder
    model = load_artifact(model_path)
    scaler = load_artifact(scaler_path)
    try:
        encoder = load_artifact(encoder_path)
    except Exception:
        encoder = None

    # Prepare features for prediction
    X = df.drop(columns=["user_id"], errors="ignore")
    # Detect categorical columns
    categorical_cols = [c for c in X.columns if c.startswith("country") or c.startswith("device_type")]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    # Encode categorical features
    if encoder is not None and categorical_cols:
        X_cat = encoder.transform(X[categorical_cols])
    else:
        X_cat = np.zeros((X.shape[0], 0))
    X_num = scaler.transform(X[numeric_cols])
    X_final = np.hstack([X_num, X_cat])

    # Predict churn probability
    churn_proba = model.predict_proba(X_final)[:, 1]
    df["churn_probability"] = churn_proba
    df["churn_label"] = (df["churn_probability"] >= 0.7).astype(int)

    # Save output in a timestamped subfolder (S3 or local)
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = f"{args.output_dir.rstrip('/')}/{now_str}"
    out_path = f"{out_dir}/predictions.parquet"

    if out_path.startswith("s3://"):
        df[["user_id", "churn_probability", "churn_label"]].to_parquet(
         out_path, index=False, storage_options={'anon': False}
        )
    else:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df[["user_id", "churn_probability", "churn_label"]].to_parquet(out_path, index=False)

    print("Saved churn predictions to:", out_path)

    print("DEBUG: Writing to out_path:", out_path)
    print("DEBUG: output-dir exists:", os.path.exists(args.output_dir))
    print("DEBUG: All files in output-dir:", os.listdir(args.output_dir))