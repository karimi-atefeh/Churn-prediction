import os
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, precision_score

import xgboost as xgb
import s3fs
import tempfile
import shutil
import tarfile
import json

from datetime import datetime
from sagemaker.session import Session
from sagemaker.experiments.run import Run

# Ensure AWS region is set for SageMaker
if "AWS_DEFAULT_REGION" not in os.environ:
    os.environ["AWS_DEFAULT_REGION"] = "eu-central-1"

def load_data(input_path):
    # Load parquet from S3 or local
    if input_path.startswith("s3://"):
        return pd.read_parquet(input_path, storage_options={'anon': False})
    else:
        return pd.read_parquet(input_path)

def save_model(model, out_path):
    # Save model (or any picklable object) to S3 or local
    if out_path.startswith("s3://"):
        with s3fs.S3FileSystem(anon=False).open(out_path, "wb") as f:
            joblib.dump(model, f)
    else:
        joblib.dump(model, out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", type=str, required=True)
    parser.add_argument("--output-model", type=str, required=True)
    parser.add_argument("--output-scaler", type=str, required=True)
    parser.add_argument("--output-encoder", type=str, required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    args = parser.parse_args()

    # Start a SageMaker Experiment Run
    sm_sess = Session()
    with Run(
        experiment_name="ChurnTrainingExperiment",
        run_name=f"train-{datetime.utcnow():%Y%m%d-%H%M%S}",
        sagemaker_session=sm_sess,
    ) as run:

        # Load features
        df = load_data(args.features_path)
        y = df["churn"].astype(int).values
        X = df.drop(columns=["churn", "user_id"])

        # Remove categorical columns (for XGBoost/scaler)
        categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
        X = X.drop(columns=categorical_cols)

        # Split data
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y)
        val_ratio = args.val_size / (1 - args.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_ratio, random_state=42, stratify=y_trainval)

        # Scale
        numeric_cols = X_train.columns
        scaler = StandardScaler()
        X_train_final = scaler.fit_transform(X_train)
        X_val_final   = scaler.transform(X_val)
        X_test_final  = scaler.transform(X_test)

        # Train model
        model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="hist",
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric="logloss"
        )
        model.fit(
            X_train_final, y_train,
            eval_set=[(X_val_final, y_val)],
            early_stopping_rounds=15,
            verbose=True
        )

        # Evaluate
        y_pred = model.predict(X_test_final)
        y_proba = model.predict_proba(X_test_final)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, pos_label=1)

        # Log metrics to SageMaker Experiments
        run.log_metric("test_accuracy", acc)
        run.log_metric("test_auc", auc)
        run.log_metric("test_precision", prec)

        # Save artifacts (model, scaler, etc.)
        save_model(model, args.output_model)
        save_model(scaler, args.output_scaler)
        save_model(None, args.output_encoder)

        # Save feature names
        feature_names_path = args.output_model.replace('.pkl', '_feature_names.txt')
        with open(feature_names_path, "w") as f:
            for col in numeric_cols:
                f.write(f"{col}\n")
        run.log_artifact("feature_names_txt", feature_names_path)

        # Save feature importances
        importance_df = pd.DataFrame({
            "feature": numeric_cols,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        importance_path = args.output_model.replace('.pkl', '_feature_importance.csv')
        importance_df.to_csv(importance_path, index=False)
        run.log_artifact("feature_importance_csv", importance_path)

        # Save evaluation JSON
        archive_dir  = os.path.dirname(args.output_model)
        eval_path = os.path.join(archive_dir, "evaluation.json")
        eval_json = {
            "binary_classification_metrics": {
                "accuracy":  {"value": float(acc)},
                "precision": {"value": float(prec)},
                "auc":       {"value": float(auc)}
            }
        }
        with open(eval_path, "w") as f:
            json.dump(eval_json, f, indent=2)
        run.log_artifact("evaluation_json", eval_path)


        # Bundle all for SageMaker Model Registry (optional)
        archive_path = os.path.join(archive_dir, "model.tar.gz")
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(args.output_model,     arcname="xgboost_model.pkl")
            tar.add(args.output_scaler,    arcname="standard_scaler.pkl")
            tar.add(args.output_encoder,   arcname="onehot_encoder.pkl")
            tar.add(feature_names_path,    arcname="feature_names.txt")
            tar.add(importance_path,       arcname="feature_importance.csv")
            tar.add(eval_path,             arcname="evaluation.json")
        run.log_artifact("model_bundle", archive_path)

        print("Model, scaler, features, importances, and evaluation saved & logged to SageMaker Experiments")