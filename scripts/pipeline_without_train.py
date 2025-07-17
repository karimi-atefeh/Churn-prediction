import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep

# ----------  S3 PATHS  ----------
RAW_EVENT_S3_PATH      = "s3://analytics-v0-501209598921-eu-central-1-amplitude/bronze/user_events/application_id=NORD"
INFERENCE_INPUT_S3     = "s3://analytics-v0-501209598921-churn-prediction-amplitude/inference_raw/"
INFERENCE_FEATURES_S3  = "s3://analytics-v0-501209598921-churn-prediction-amplitude/inference_features/"
INSTALL_FILE_S3        = "s3://analytics-v0-501209598921-churn-prediction-amplitude/athena-install-time-NORD-results/NORD-install_time.csv"
MODEL_S3_DIR           = "s3://analytics-v0-501209598921-churn-prediction-amplitude/models/"
PREDICTIONS_S3         = "s3://analytics-v0-501209598921-churn-prediction-amplitude/predictions/"

# ----------  SageMaker basics ----------
role        = "arn:aws:iam::501209598921:role/service-role/AmazonSageMaker-ExecutionRole-20250616T101078"
image_uri = "598094125568.dkr.ecr.eu-central-1.amazonaws.com/machine_learning/behaviour/churn/mlops:v1.1.3"
session     = sagemaker.Session()

processor = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    role=role,
    instance_count=1,
    instance_type="ml.m5.large"        
)

# ---------- 1) Data-Ingestion ----------
data_ingestion = ProcessingStep(
    name="DataIngestionInference",
    processor=processor,
    code="s3://analytics-v0-501209598921-churn-prediction-amplitude/scripts/data_ingestion_inference.py",
    inputs=[
        ProcessingInput(source=INSTALL_FILE_S3, destination="/opt/ml/processing/install_in"),
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=INFERENCE_INPUT_S3,
            output_name="InferenceRaw"
        )
    ],
    job_arguments=[ 
        "--events-dir",      RAW_EVENT_S3_PATH,
        "--install-file",    INSTALL_FILE_S3,
        "--output-dir",      "/opt/ml/processing/output",
        "--feature-columns", "user_id,event_time,event_type,session_id,country,device_type,paying",
        "--lookback-days",   "20",
        "--target-days-ago", "8"
    ]
)

# ---------- 2) Pre-processing ----------
preprocessing = ProcessingStep(
    name="PreprocessingAndFeatureEngineering",
    processor=processor,
    code="s3://analytics-v0-501209598921-churn-prediction-amplitude/scripts/preprocessing_and_feature_engineering.py",
    inputs=[
        ProcessingInput(
            source=data_ingestion.properties.ProcessingOutputConfig
                  .Outputs["InferenceRaw"].S3Output.S3Uri,
            destination="/opt/ml/processing/input"
        ),

        ProcessingInput(
            source=INSTALL_FILE_S3,
            destination="/opt/ml/processing/install"
        ),
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=INFERENCE_FEATURES_S3,
            output_name="Features"
        )
    ],
    job_arguments=[
        "--input-dir",    "/opt/ml/processing/input",
        "--output-dir",   "/opt/ml/processing/output",
        #2Awlii
        "--install-file", "/opt/ml/processing/install/NORD-install_time.csv",
        # "--install-file", INSTALL_FILE_S3,
        "--mode",         "inference"
    ]
)

# ---------- 3) Batch-Inference ----------
inference = ProcessingStep(
    name="BatchInference",
    processor=processor,
    code="s3://analytics-v0-501209598921-churn-prediction-amplitude/scripts/inference.py",
    inputs=[
        ProcessingInput(
            source=preprocessing.properties.ProcessingOutputConfig
                  .Outputs["Features"].S3Output.S3Uri,
            destination="/opt/ml/processing/features"
        ),
        ProcessingInput(
            source=MODEL_S3_DIR,                 
            destination="/opt/ml/processing/artifacts"
        ),
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=PREDICTIONS_S3
        )
    ],
    job_arguments=[
        "--features-dir", "/opt/ml/processing/features",
        "--model-path",   "/opt/ml/processing/artifacts/",  
        "--scaler-path",  "/opt/ml/processing/artifacts/",
        "--encoder-path", "/opt/ml/processing/artifacts/",
        "--output-dir",   "/opt/ml/processing/output"
    ]
)

# ---------- Build & Run ----------
pipeline = Pipeline(
    name="ChurnPredictionBatchInferenceOnly",
    steps=[data_ingestion, preprocessing, inference],
    sagemaker_session=session
)

pipeline.upsert(role_arn=role)   
print("Pipeline definition submitted.")

execution = pipeline.start()
print("Pipeline execution started:", execution.arn)
