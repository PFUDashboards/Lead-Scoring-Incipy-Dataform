#!/usr/bin/env bash
# One-time GCP setup: enable APIs, create the bucket and the Artifact Registry repo.
set -euo pipefail
cd "$(dirname "$0")/.."
source deploy/config.sh

echo ">> Enabling APIs on ${PROJECT_ID}"
gcloud services enable \
  aiplatform.googleapis.com \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  bigquery.googleapis.com \
  storage.googleapis.com \
  --project "${PROJECT_ID}"

echo ">> Creating GCS bucket gs://${BUCKET} (region ${REGION})"
gcloud storage buckets create "gs://${BUCKET}" \
  --project "${PROJECT_ID}" --location "${REGION}" --uniform-bucket-level-access \
  || echo "   (bucket already exists, skipping)"

echo ">> Creating Artifact Registry repo ${AR_REPO} (region ${REGION})"
gcloud artifacts repositories create "${AR_REPO}" \
  --project "${PROJECT_ID}" --location "${REGION}" --repository-format=docker \
  --description="Lead scoring images" \
  || echo "   (repo already exists, skipping)"

echo ">> Done. Next: deploy/01_build_images.sh"
