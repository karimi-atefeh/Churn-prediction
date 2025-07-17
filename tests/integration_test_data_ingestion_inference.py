# tests/test_data_ingestion_inference.py

import subprocess
import sys
import os

def test_data_ingestion_inference_s3_runs():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "churn_mlops_pipeline.scripts.data_ingestion_inference"
        ],
        capture_output=True,
        text=True,
        cwd=os.getcwd()
    )
    print(result.stdout)

    assert result.returncode == 0, result.stderr
    assert "Saved inference user data to:" in result.stdout
