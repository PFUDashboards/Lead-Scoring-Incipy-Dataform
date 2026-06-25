#!/usr/bin/env bash
# Deploy the scoring API to Cloud Run. prod: private + scale-to-zero. dev: public + warm.
set -euo pipefail
cd "$(dirname "$0")/.."
source deploy/config.sh

# Run the service AS our deploy SA, not Cloud Run's default Compute SA (which can't read
# the model from GCS -> the API would load no models and return 503). Resolve from the
# gcloud impersonation config (local) or the active account (CI); override with RUNTIME_SA.
RUNTIME_SA="${RUNTIME_SA:-$(gcloud config get-value auth/impersonate_service_account 2>/dev/null || true)}"
[ -z "${RUNTIME_SA}" ] && RUNTIME_SA="$(gcloud config get-value account 2>/dev/null || true)"

SA_ARGS=()
if [ -n "${RUNTIME_SA}" ]; then
  echo "   runtime SA: ${RUNTIME_SA}"
  SA_ARGS+=(--service-account "${RUNTIME_SA}")
fi

# dev is a sandbox for integrators without a backend: public (so they can curl /score with
# no Google identity) and kept warm (min-instances 1, no cold-start wait). prod stays private
# and scale-to-zero. Both knobs are env-overridable for one-offs.
if [ "${ENV}" = "dev" ]; then
  AUTH_ARG="--allow-unauthenticated"
  MIN_INSTANCES="${MIN_INSTANCES:-1}"
else
  AUTH_ARG="--no-allow-unauthenticated"
  MIN_INSTANCES="${MIN_INSTANCES:-0}"
fi

echo ">> Deploying ${SERVICE} to Cloud Run (${REGION}, env ${ENV}, serving the LIVE model)"
gcloud run deploy "${SERVICE}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --image "${SERVING_IMAGE}" \
  --set-env-vars "ENV=${ENV},GCS_MODEL_PREFIX=${GCS_MODEL_PREFIX},PROJECT_ID=${PROJECT_ID},REGION=${REGION}" \
  --memory 1Gi --cpu 1 --min-instances "${MIN_INSTANCES}" --max-instances 5 \
  "${AUTH_ARG}" \
  "${SA_ARGS[@]}"

URL=$(gcloud run services describe "${SERVICE}" --project "${PROJECT_ID}" --region "${REGION}" --format='value(status.url)')
echo ">> Deployed: ${URL}"

# The service is private (--no-allow-unauthenticated); callers need roles/run.invoker.
# Grant it to whoever should call /score (idempotent, so it survives a teardown+recreate).
# Default: the deploy SA (ops/smoke tests). Add real client identities via INVOKERS
# (space-separated, e.g. INVOKERS="user:foo@x.com serviceAccount:client@proj.iam...").
INVOKERS="${INVOKERS:-serviceAccount:${RUNTIME_SA}}"
for M in ${INVOKERS}; do
  [ -z "${M#serviceAccount:}" ] && continue
  echo ">> Granting run.invoker to ${M}"
  gcloud run services add-iam-policy-binding "${SERVICE}" \
    --project "${PROJECT_ID}" --region "${REGION}" \
    --member="${M}" --role="roles/run.invoker" --quiet >/dev/null
done
