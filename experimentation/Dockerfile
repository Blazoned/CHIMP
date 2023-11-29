FROM tensorflow/tensorflow:2.12.0-gpu

LABEL authors="Blazoned"
LABEL owners="Lectorship Data Intelligence Zuyd, Bryan Kroon"
LABEL nl.blazoned.emotionrecognition.service="experimentation"

# Set environment variables:
# - Base directory,
# - Python create binary files (python -b)
# - Python output buffer stream (python -u)
# - Allow deprecated sklearn is important for Talos automl. Leave it enabled!
ENV CONTAINER_HOME=/usr/src/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Add dependencies required for opencv (ffmpeg, libsm6, libxext6), mlflow (git) and tensorflow (cuda's cudnn)
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-get update && apt-get upgrade -y && apt-get install -y -q git ffmpeg libsm6 libxext6

# Copy project to container, then install requirements
ADD . $CONTAINER_HOME
WORKDIR $CONTAINER_HOME
RUN pip install --upgrade pip
RUN pip install -r $CONTAINER_HOME/requirements.txt

# Execute service
ENTRYPOINT ["gunicorn", "--worker-class=gunicorn.workers.ggevent.GeventWorker", "--bind=0.0.0.0:8000", "--workers=8", "--threads=128", "--timeout=0", "main:get_app()"]
