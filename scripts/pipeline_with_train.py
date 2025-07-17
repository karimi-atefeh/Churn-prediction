import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.model import Model
from sagemaker.workflow.functions import Join
from sagemaker.model_metrics import MetricsSource, ModelMetrics

# --- S3 PATHS ---
RAW_EVENT_S3_PATH      = "s3://analytics-v0-501209598921-eu-central-1-amplitude/bronze/user_events/application_id=NORD"
TRAIN_RAW_S3           = "s3://analytics-v0-501209598921-churn-prediction-amplitude/train_raw/"
TRAIN_FEATURES_S3      = "s3://analytics-v0-501209598921-churn-prediction-amplitude/train_features/"
INSTALL_FILE_S3        = "s3://analytics-v0-501209598921-churn-prediction-amplitude/athena-install-time-NORD-results/NORD-install_time.csv"
MODEL_S3_DIR           = "s3://analytics-v0-501209598921-churn-prediction-amplitude/models/"
PREDICTIONS_S3         = "s3://analytics-v0-501209598921-churn-prediction-amplitude/predictions/"

role      = "arn:aws:iam::501209598921:role/service-role/AmazonSageMaker-ExecutionRole-20250616T101078"
image_uri = "598094125568.dkr.ecr.eu-central-1.amazonaws.com/machine_learning/behaviour/churn/mlops:v1.1.3"
session   = sagemaker.Session()

processor = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    role=role,
    instance_count=1,
    instance_type="ml.m5.large"
)

# Step 1: Data Ingestion for Training
data_ingestion_train = ProcessingStep(
    name="DataIngestionTraining",
    processor=processor,
    code="s3://analytics-v0-501209598921-churn-prediction-amplitude/scripts/data_ingestion_training.py",
    inputs=[
        ProcessingInput(source=INSTALL_FILE_S3, destination="/opt/ml/processing/install"),
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=TRAIN_RAW_S3,
            output_name="TrainRaw"
        )
    ],
    job_arguments=[
        "--event-base-path", RAW_EVENT_S3_PATH,
        "--install-file", INSTALL_FILE_S3,
        "--feature-columns", "user_id,event_time,event_type,session_id,country,device_type,paying",
        "--output-dir", "/opt/ml/processing/output"
    ]
)

# Step 2: Preprocessing & Feature Engineering (Train)
preprocessing_train = ProcessingStep(
    name="PreprocessingAndFeatureEngineeringTrain",
    processor=processor,
    code="s3://analytics-v0-501209598921-churn-prediction-amplitude/scripts/preprocessing_and_feature_engineering.py",
    inputs=[
        ProcessingInput(
            source=data_ingestion_train.properties.ProcessingOutputConfig.Outputs["TrainRaw"].S3Output.S3Uri,
            destination="/opt/ml/processing/input"
        ),
        ProcessingInput(source=INSTALL_FILE_S3, destination="/opt/ml/processing/install"),
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=TRAIN_FEATURES_S3,
            output_name="TrainFeatures"
        )
    ],
    job_arguments=[
        "--input-dir", "/opt/ml/processing/input",
        "--output-dir", "/opt/ml/processing/output",
        "--install-file", "/opt/ml/processing/install/NORD-install_time.csv",
        "--mode", "train"
    ]
)

# Step 3: Model Training
training = ProcessingStep(
    name="ModelTraining",
    processor=processor,
    code="s3://analytics-v0-501209598921-churn-prediction-amplitude/scripts/train.py",
    inputs=[
        ProcessingInput(
            source=preprocessing_train.properties.ProcessingOutputConfig.Outputs["TrainFeatures"].S3Output.S3Uri,
            destination="/opt/ml/processing/features"
        )
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=MODEL_S3_DIR,
            output_name="ModelArtifacts"
        )
    ],
    job_arguments=[
        "--features-path", "/opt/ml/processing/features",
        "--output-model", "/opt/ml/processing/output/xgboost_model.pkl",
        "--output-scaler", "/opt/ml/processing/output/standard_scaler.pkl",
        "--output-encoder", "/opt/ml/processing/output/onehot_encoder.pkl"
    ]
)

raw_uri = training.properties.ProcessingOutputConfig.\
            Outputs["ModelArtifacts"].S3Output.S3Uri      

model_tar_uri = Join(         
    on="",                   
    values=[raw_uri, "model.tar.gz"]
)

evaluation_uri = Join(        
    on="",
    values=[raw_uri, "evaluation.json"]
)

metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=evaluation_uri,
        content_type="application/json",
    )
)

churn_model = Model(
    image_uri=image_uri,
    model_data=model_tar_uri,  
    role=role,
    sagemaker_session=session,
)

register_step = RegisterModel(
    name="RegisterChurnModel",
    model_package_group_name="churn-xgboost-models",
    approval_status="Approved",
    model=churn_model,
    model_metrics=metrics,
    content_types=["application/x-parquet"],
    response_types=["application/json"],
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],
)

pipeline = Pipeline(
    name="ChurnPredictionWithTraining",
    steps=[data_ingestion_train, preprocessing_train, training, register_step],
    sagemaker_session=session
)

pipeline.upsert(role_arn=role)
print("Pipeline definition submitted.")

execution = pipeline.start()
print("Pipeline execution started:", execution.arn)