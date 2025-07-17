FROM python:3.10-slim

WORKDIR /opt/ml/code

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip show xgboost && python -c "import xgboost; print(xgboost.__version__)"