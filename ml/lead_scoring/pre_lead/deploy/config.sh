#!/usr/bin/env bash
# Shared settings for the deploy scripts (sourced by the others).
#
# Single source of truth = src/leadscoring/config.py. We DO NOT duplicate the
# project/region/bucket here any more — we read them straight from that file via a
# tiny python call, so bash, Terraform and the Vertex components can never drift.
# Every value is still env-overridable (handled inside config.py): exporting e.g.
# PROJECT_ID=other before sourcing this wins, because config.py reads the env first.
set -euo pipefail

# Resolve a config.py attribute (no heavy deps: config.py only imports os, and the
# package __init__ is a stub, so this works with just PYTHONPATH=src — no numpy etc.).
_PKG_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src"
_cfg() { PYTHONPATH="${_PKG_SRC}:${PYTHONPATH:-}" python3 -c "from leadscoring import config as c; print(c.$1)"; }

# ENV (dev|prod) namespaces the GCS model paths + Cloud Run service. Default dev so
# an un-set ENV can never overwrite prod. Run e.g.  ENV=prod ./deploy/03_...sh
export ENV="${ENV:-dev}"

# These mirror config.py exactly (read from it, not hardcoded).
export PROJECT_ID="${PROJECT_ID:-$(_cfg PROJECT_ID)}"
export REGION="${REGION:-$(_cfg REGION)}"
export BUCKET="${BUCKET:-$(_cfg BUCKET)}"
export AR_REPO="${AR_REPO:-$(_cfg AR_REPO)}"
export BQ_DATASET="${BQ_DATASET:-$(_cfg BQ_DATASET)}"
export BQ_LOCATION="${BQ_LOCATION:-$(_cfg BQ_LOCATION)}"
export SERVICE="${SERVICE:-lead-scoring-${ENV}}"

export AR_HOST="${REGION}-docker.pkg.dev"
export TRAINING_IMAGE="${AR_HOST}/${PROJECT_ID}/${AR_REPO}/training-base:latest"
export SERVING_IMAGE="${AR_HOST}/${PROJECT_ID}/${AR_REPO}/lead-scoring-serve:latest"
# Base (env-namespaced) prefix; serving appends /live, the pipeline writes /candidate.
export GCS_MODEL_PREFIX="gs://${BUCKET}/models/${ENV}"

# Terraform reads infra config from these env vars (TF_VAR_<name>) so the .tfvars
# doesn't re-declare project/region/bucket. Source this file, then run terraform
# (or use deploy/tf.sh, which does it for you).
export TF_VAR_project_id="${PROJECT_ID}"
export TF_VAR_region="${REGION}"
export TF_VAR_bucket="${BUCKET}"
export TF_VAR_ar_repo="${AR_REPO}"
