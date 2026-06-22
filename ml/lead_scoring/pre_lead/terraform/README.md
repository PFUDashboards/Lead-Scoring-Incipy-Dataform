# Terraform — lead-scoring infra (infra only)

Creates the durable infra: enables APIs, the GCS bucket (models + pipeline-root),
the Artifact Registry repo and (if `alert_emails` is set) the pipeline-failure email
alert. No service accounts or IAM are managed here.

Workloads (Cloud Build, the Vertex pipeline, Cloud Run) run as the deploy SA
`incipy-lead-scoring@bq-pfu-ga4.iam.gserviceaccount.com`, which the `deploy/01–03`
scripts pass explicitly via `--service-account`. The project's default compute SA
has no roles, so it is never relied on. See [DEPLOY.md](../DEPLOY.md) for the auth setup.

Alternative without Terraform: `deploy/00_setup_gcp.sh` (gcloud, but without the alert).

```bash
# from the model root (ml/lead_scoring/pre_lead):
./deploy/tf.sh init
./deploy/tf.sh plan      # review
./deploy/tf.sh apply     # create bucket + AR repo + enable APIs

# then:  ./deploy/01_build_images.sh -> 02 -> 03
```

State is **local** (`terraform.tfstate`, gitignored). To share/lock later, add a
GCS backend in `versions.tf` and `terraform init -migrate-state`.

`project_id` / `region` / `bucket` / `ar_repo` are NOT in `terraform.tfvars` — they
come from the single source of truth (`src/leadscoring/config.py`) via the `TF_VAR_*`
env vars that `deploy/config.sh` exports (`deploy/tf.sh` sources it for you). Only
`alert_emails` lives in `terraform.tfvars`. Region must equal the BigQuery data
location (EU → `europe-west1`). Running bare `terraform` without sourcing
`config.sh` first will just prompt for the missing vars (safe, not wrong values).
