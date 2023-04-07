# Using a python 3.10 image instead of 3.9 to not build with a high vulnerability.
FROM python:3.9-slim

LABEL authors="Blazoned"
LABEL owners="Lectorship Data Intelligence Zuyd, Bryan Kroon"
LABEL nl.blazoned.emotionrecognition.service="experimentation"
LABEL nl.blazoned.pythonversion=3.9

# Set environment variables:
# - Base directory,
# - Python create binary files (yes or no) (python -b)
# - Python output buffer stream (yes or no) (python -u)
# - Allow deprecated sklearn is important for Talos automl. Leave it enabled!
ENV CONTAINER_HOME=/usr/src/app
ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Copy project to container, then install requirements
ADD . $CONTAINER_HOME
WORKDIR $CONTAINER_HOME
RUN pip install -r $CONTAINER_HOME/requirements.txt

# Execute service
ENTRYPOINT ["python3", "main.py"]
