# Churn Prediction MLOps Pipeline (Impala Studios)

## Overview

This project implements a full MLOps workflow for user churn prediction at Impala Studios, a mobile app company with millions of users worldwide. All data processing, training, and inference pipelines are built using AWS SageMaker, with S3 for data storage, versioning, and all model artifacts.  
All pipeline scripts are centrally managed in S3.

---

## Project Structure

The system uses **two main SageMaker pipelines**:

### 1. Training Pipeline (`pipeline_with_train.py`)

- **Purpose:** Prepare and process raw event data, engineer features, train a model, and register it in the SageMaker Model Registry.
- **Steps:**
    1. **Data Ingestion (Training)**  
        - Script: `data_ingestion_training.py`  
        - Loads event data & install times, outputs to `train_raw/`
    2. **Feature Engineering (Training)**  
        - Script: `preprocessing_and_feature_engineering.py` (mode = train)  
        - Processes features, outputs to `train_features/`
    3. **Model Training**  
        - Script: `train.py`  
        - Trains an XGBoost model, saves model, scaler, and feature importances to `models/`
    4. **Model Registration**  
        - Registers the model in SageMaker Model Registry.

---

### 2. Inference Pipeline (`pipeline_without_train.py`)

- **Purpose:** Use the latest trained model to predict churn for new users based on new event data.
- **Steps:**
    1. **Data Ingestion (Inference)**  
        - Script: `data_ingestion_inference.py`  
        - Loads and prepares new event data, outputs to `inference_raw/`
    2. **Feature Engineering (Inference)**  
        - Script: `preprocessing_and_feature_engineering.py` (mode = inference)  
        - Processes features, outputs to `inference_features/`
    3. **Inference**  
        - Script: `inference.py`  
        - Uses the trained model and scaler to predict churn, outputs results to `predictions/`

---

## S3 Structure & Result Versioning

**All pipeline outputs and artifacts are versioned in S3 with timestamped folders** for traceability and reproducibility.

| Folder S3 Path   | Description | Example Path |
|------------------|-------------|--------------|
| `train_raw/` | Raw training data (`data_ingestion_training.py`) | `s3://analytics-v0-501209598921-churn-prediction-amplitude/train_raw/` |
| `train_features/` | Engineered features for training (`preprocessing_and_feature_engineering.py`, train mode) | `s3://analytics-v0-501209598921-churn-prediction-amplitude/train_features/` |
| `models/` | Trained models, scalers, and feature importances (`train.py`) | `s3://analytics-v0-501209598921-churn-prediction-amplitude/models/` |
| `inference_raw/` | Raw data for inference (`data_ingestion_inference.py`) | `s3://analytics-v0-501209598921-churn-prediction-amplitude/inference_raw/` |
| `inference_features/` | Features for inference (`preprocessing_and_feature_engineering.py`, inference mode) | `s3://analytics-v0-501209598921-churn-prediction-amplitude/inference_features/` |
| `predictions/` | Churn prediction results (`inference.py`) | `s3://analytics-v0-501209598921-churn-prediction-amplitude/predictions/` |

- **Each folder contains timestamped subfolders/files for every run** (e.g. `2025-07-15_13-45-22/`).
- **All scripts are stored in:**  
  `s3://analytics-v0-501209598921-churn-prediction-amplitude/scripts/`

---

## Install Time Tracking

The pipeline requires the **install time** (estimated installation date) for each user.  
Because this data was not directly available in Amplitude or any other database, we **calculate it ourselves** by taking the minimum event time for each user.  
This install time file is **updated daily** with new users and stored as a CSV in S3:

- **Install time S3 path:**  
  `s3://analytics-v0-501209598921-churn-prediction-amplitude/athena-install-time-NORD-results/NORD-install_time.csv`

Every time the inference pipeline runs, it uses the latest install times from this file.

---

## Script Management

> **Important:**  
> All pipeline scripts are versioned and stored in:
>
> ```
> s3://analytics-v0-501209598921-churn-prediction-amplitude/scripts/
> ```
>
> Whenever you update any script, always upload the latest version to this S3 path so the pipeline will use your new code.

---

## Quick Summary

- **Training pipeline:** prepares, processes, and trains models on historical data.
- **Inference pipeline:** predicts churn for new users as new data arrives.
- **All results and models are versioned in S3** for full traceability.
- **Scripts are centrally managed in S3** for easy updating.

---

## Requirements

- **AWS SageMaker**
- **AWS S3**
- **Python** (see `requirements.txt`)
- **Docker** (for dockerize the requirements.txt)

---

## How to Run

1. **Update scripts in S3** as needed.
2. **Start the desired pipeline** (training or inference) in SageMaker.
3. **Monitor results** in the relevant S3 folders (all versioned by timestamp).

Example of how to Run Preprocessing & Feature Engineering:

#### Train mode example:
python -m <your_package>.preprocessing_and_feature_engineering \
  --input-dir s3://<your-bucket>/train_raw/ \
  --output-dir s3://<your-bucket>/train_features/ \
  --install-file s3://<your-bucket>/athena-install-time-NORD-results/NORD-install_time.csv \
  --mode train

#### Inference mode example
  python -m <your_package>.preprocessing_and_feature_engineering \
  --input-dir s3://<your-bucket>/inference_raw/ \
  --output-dir s3://<your-bucket>/inference_features/ \
  --install-file s3://<your-bucket>/athena-install-time-NORD-results/NORD-install_time.csv \
  --mode inference

---

## Notes

- **Everything runs directly from and to S3—no local storage is required.**
- **Each run creates a new timestamped folder, so you always know which version created which result.**
- **Processing is fully containerized and reproducible.**

---