#!/usr/bin/env bash
# Tear down the lead-scoring GCP resources, in the correct order. Terraform here
# is infra-only (bucket + Artifact Registry repo + APIs), so destroying it alone
# leaves the Cloud Run services and the CI service-account key behind — this
# script handles those too.
#
# Order: Cloud Run services -> bucket + AR repo (terraform OR gcloud) -> SA key.
#
# Does NOT touch: the BigQuery source table (dataset.lead_scoring_train, pre-existing
# data), the project's default compute service account, or the APIs
# (terraform keeps them enabled via disable_on_destroy=false).
#
# Usage:
#   ./deploy/99_teardown.sh                 # interactive confirm, keeps SA keys
#   ./deploy/99_teardown.sh --yes           # skip the confirmation prompt
#   ./deploy/99_teardown.sh --delete-sa-keys# also delete user-managed keys on the
#                                           # compute SA (BREAKS GitHub Actions deploy)
set -euo pipefail
cd "$(dirname "$0")/.."
source deploy/config.sh

ASSUME_YES=0
DELETE_SA_KEYS=0
for arg in "$@"; do
  case "$arg" in
    --yes|-y) ASSUME_YES=1 ;;
    --delete-sa-keys) DELETE_SA_KEYS=1 ;;
    *) echo "Unknown flag: $arg" >&2; exit 2 ;;
  esac
done

cat <<EOF
================================================================================
  TEARDOWN  —  project ${PROJECT_ID}  (region ${REGION})
--------------------------------------------------------------------------------
  Will delete (if present):
    - Cloud Run services : lead-scoring-dev, lead-scoring-prod
    - GCS bucket         : gs://${BUCKET}   (force_destroy: models + artifacts go too)
    - Artifact Registry  : ${AR_REPO}       (with all images)
$( [ "${DELETE_SA_KEYS}" = 1 ] && echo "    - SA keys            : ALL user-managed keys on the compute SA (breaks CI!)" )
  Will NOT touch:
    - BigQuery table dataset.lead_scoring_train  (your source data)
    - the default compute service account / enabled APIs
================================================================================
EOF

if [ "${ASSUME_YES}" != 1 ]; then
  read -r -p "Type the project id to confirm destruction: " reply
  [ "${reply}" = "${PROJECT_ID}" ] || { echo "Aborted (got '${reply}')."; exit 1; }
fi

# 1) Cloud Run services (created by gcloud, not terraform) — both env namespaces.
for svc in "lead-scoring-dev" "lead-scoring-prod"; do
  echo ">> Deleting Cloud Run service ${svc} (if it exists)..."
  gcloud run services delete "${svc}" \
    --project "${PROJECT_ID}" --region "${REGION}" --quiet 2>/dev/null \
    && echo "   deleted ${svc}" || echo "   ${svc} not found, skipping"
done

# 2) Bucket + Artifact Registry repo. Prefer terraform if it has state; otherwise
#    delete directly with gcloud (infra may have been created via 00_setup_gcp.sh).
if [ -f terraform/terraform.tfstate ] && \
   terraform -chdir=terraform state list >/dev/null 2>&1 && \
   [ -n "$(terraform -chdir=terraform state list 2>/dev/null)" ]; then
  echo ">> terraform state found — running terraform destroy"
  if [ "${ASSUME_YES}" = 1 ]; then
    terraform -chdir=terraform destroy -auto-approve
  else
    terraform -chdir=terraform destroy   # terraform prompts for confirmation itself
  fi
else
  echo ">> No terraform state — deleting bucket + AR repo directly with gcloud"
  echo "   Deleting gs://${BUCKET} (recursive)..."
  gcloud storage rm -r "gs://${BUCKET}" --project "${PROJECT_ID}" --quiet 2>/dev/null \
    && echo "   bucket deleted" || echo "   bucket not found, skipping"
  echo "   Deleting Artifact Registry repo ${AR_REPO}..."
  gcloud artifacts repositories delete "${AR_REPO}" \
    --project "${PROJECT_ID}" --location "${REGION}" --quiet 2>/dev/null \
    && echo "   AR repo deleted" || echo "   AR repo not found, skipping"
fi

# 3) Service-account keys on the compute SA (the CI credential). Opt-in only.
SA_NUM="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
SA_EMAIL="${SA_NUM}-compute@developer.gserviceaccount.com"
if [ "${DELETE_SA_KEYS}" = 1 ]; then
  echo ">> Deleting user-managed keys on ${SA_EMAIL} (this breaks GitHub Actions deploy)"
  gcloud iam service-accounts keys list --iam-account "${SA_EMAIL}" \
    --project "${PROJECT_ID}" --managed-by=user --format="value(name)" \
    | while read -r key; do
        [ -n "${key}" ] || continue
        gcloud iam service-accounts keys delete "${key}" \
          --iam-account "${SA_EMAIL}" --project "${PROJECT_ID}" --quiet \
          && echo "   deleted key ${key}"
      done
  echo "   Remember to also remove the GCP_SA_KEY secret from the GitHub repo."
else
  echo ">> Leaving the CI service-account key in place (used by GitHub Actions)."
  echo "   To remove it manually: gcloud iam service-accounts keys list \\"
  echo "        --iam-account ${SA_EMAIL} --managed-by=user   # find the key id, then"
  echo "      gcloud iam service-accounts keys delete <KEY_ID> --iam-account ${SA_EMAIL}"
fi

echo ">> Teardown complete."
