import boto3

def download_model_from_s3(s3_uri, local_path):
    s3 = boto3.client("s3")
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    s3.download_file(bucket, key, local_path)