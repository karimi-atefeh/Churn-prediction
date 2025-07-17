import pandas as pd

full_path = "s3://analytics-v0-501209598921-churn-prediction-amplitude/inference_raw/2025-07-03_16-07-35/inference_raw.parquet"

df = pd.read_parquet(full_path, storage_options={"anon": False})

sample = df.head(100)

sample_path = "sample_inference_raw.parquet"
sample.to_parquet(sample_path, index=False)

# sample.to_parquet("s3://YOUR_BUCKET/test_samples/sample_inference_raw.parquet", index=False, storage_options={"anon": False})