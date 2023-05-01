FROM python:3.11.3-slim

LABEL authors="Blazoned"
LABEL owners="Lectorship Data Intelligence Zuyd, Bryan Kroon"
LABEL nl.blazoned.emotionrecognition.service="mlflow server"

# Set environment variables:
# - Python create binary files (python -b)
# - Python output buffer stream (python -u)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir --upgrade mlflow==2.3.0 boto3==1.26.122 pymysql==1.0.3 cryptography==40.0.2
