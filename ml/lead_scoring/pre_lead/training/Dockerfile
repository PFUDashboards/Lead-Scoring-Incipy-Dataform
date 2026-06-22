# Base image for the KFP pipeline components. Each component runs this image and
# `import leadscoring`. Build context = repo root:
#   gcloud builds submit --config deploy/cloudbuild.training.yaml  (or see 01_build_images.sh)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# OpenMP runtime for xgboost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY training/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Install the shared library so components can `import leadscoring`
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
RUN pip install --no-deps -e /app
