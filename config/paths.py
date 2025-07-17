# config/paths.py

# All global paths and constants are centralized here

S3_INSTALL_FILE_PATH = "s3://analytics-v0-501209598921-churn-prediction-amplitude/athena-install-time-NORD-results/NORD-install_time.csv"
# S3_INSTALL_FILE_PATH = "s3://analytics-v0-740804867872-eu-central-1-amplitude/machine_learning/churn/NORD-install_time.csv"

S3_EVENT_BASE_PATH = "s3://analytics-v0-501209598921-eu-central-1-amplitude/bronze/user_events/application_id=NORD"
# S3_EVENT_BASE_PATH = "s3://analytics-v0-740804867872-eu-central-1-amplitude/bronze/user_events/application_id=NORD"

# S3_MODEL_PATH = "s3://analytics-v0-501209598921-churn-prediction-amplitude/models/xgboost_model.pkl"

LOCAL_TMP_PATH = "/tmp/churn_pipeline/"

FEATURE_COLUMNS = [
    "user_id", "event_time", "event_type", "session_id",
    "country", "device_type", "paying"
]