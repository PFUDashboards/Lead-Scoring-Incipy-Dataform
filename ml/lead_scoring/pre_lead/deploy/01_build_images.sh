#!/usr/bin/env bash
# Build + push the training-base and serving images to Artifact Registry (Cloud Build).
set -euo pipefail
cd "$(dirname "$0")/.."
source deploy/config.sh

echo ">> Building images via Cloud Build (context = repo root)"
echo "   training: ${TRAINING_IMAGE}"
echo "   serving : ${SERVING_IMAGE}"

# Run the build AS our deploy SA, not Cloud Build's default Compute SA (which has no
# roles here and can't even read the uploaded source). Resolve it from the gcloud
# impersonation config (local) or the active account (CI, where the GCP_SA_KEY identity
# is already the SA); override with BUILD_SA. (A custom build SA needs non-default
# logging — already set as options.logging=CLOUD_LOGGING_ONLY in cloudbuild.yaml.)
BUILD_SA="${BUILD_SA:-$(gcloud config get-value auth/impersonate_service_account 2>/dev/null || true)}"
[ -z "${BUILD_SA}" ] && BUILD_SA="$(gcloud config get-value account 2>/dev/null || true)"

EXTRA=()
if [ -n "${BUILD_SA}" ]; then
  echo "   build SA: ${BUILD_SA}"
  EXTRA+=(--service-account="projects/${PROJECT_ID}/serviceAccounts/${BUILD_SA}")
fi

gcloud builds submit . \
  --project "${PROJECT_ID}" \
  --config deploy/cloudbuild.yaml \
  --substitutions "_TRAINING_IMAGE=${TRAINING_IMAGE},_SERVING_IMAGE=${SERVING_IMAGE}" \
  "${EXTRA[@]}"

echo ">> Done. Next: deploy/02_run_pipeline.sh (train) then deploy/03_deploy_serving.sh"
