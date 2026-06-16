#!/usr/bin/env bash
# Run Terraform with project/region/bucket pulled from the single source of truth
# (src/leadscoring/config.py, via deploy/config.sh -> TF_VAR_*). Usage:
#   ./deploy/tf.sh init
#   ./deploy/tf.sh plan
#   ./deploy/tf.sh apply
# Any args are forwarded to terraform verbatim.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${HERE}/config.sh"   # exports TF_VAR_project_id / _region / _bucket / _ar_repo

echo ">> terraform $* (project ${PROJECT_ID}, region ${REGION}, bucket ${BUCKET})"
terraform -chdir="${HERE}/../terraform" "$@"
