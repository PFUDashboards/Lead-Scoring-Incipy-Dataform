# CI/CD — GitHub Actions

GitHub Actions runs the quality gate, builds images, and deploys serving. It does
**not** replace the manual deploy scripts — it *calls* them (`deploy/01_*.sh`,
`deploy/03_*.sh`, `deploy/02_run_pipeline.sh`), so local and CI deploys stay identical.

## Workflows (`.github/workflows/`)

| File         | Trigger                          | Does                                                              |
|--------------|----------------------------------|------------------------------------------------------------------|
| `ci.yml`     | every PR + push to `main`        | `ruff check` + `pytest` (no GCP creds)                           |
| `deploy.yml` | push to `develop`/`main` (or manual) | quality gate → build images → train + deploy **dev** (`develop`) or **prod** (`main`, approval) |
| `train.yml`  | manual (`workflow_dispatch`) + monthly cron | compile + submit the Vertex training pipeline for `dev`/`prod`    |

**The dev→prod gate.** `deploy.yml`'s prod job targets the **`prod` Environment**, which
has a required reviewer — so prod deploys **pause until approved** in the Actions UI. A
prod `train.yml` run hits the same environment. The monthly cron in `train.yml` runs from
the default branch and goes straight to prod (governed by the SOFT promotion gate + alert
emails, no human approval). The candidate→live *model* gate is separate and lives inside
the Vertex pipeline.

Retraining is intentionally **not** wired into merge — a Vertex run is slow/costly. Run
`train.yml` by hand when you want a new model, then re-run `deploy.yml` (or
`deploy/03_deploy_serving.sh`) so serving picks up the freshly-promoted `live` model.

## One-time setup (requires GCP + GitHub console)

These can't be automated from this repo (no account here can set IAM policy, so Workload
Identity Federation isn't an option yet — CI uses a SA key instead).

1. **Create a service-account key** for the deploy SA
   `incipy-lead-scoring@bq-pfu-ga4.iam.gserviceaccount.com`:
   ```bash
   gcloud iam service-accounts keys create key.json \
     --iam-account incipy-lead-scoring@bq-pfu-ga4.iam.gserviceaccount.com --project bq-pfu-ga4
   ```
   The SA needs: Cloud Build (`roles/cloudbuild.builds.editor`), Artifact Registry write,
   Storage object admin (models bucket), Vertex AI user, Cloud Run admin, and
   `roles/iam.serviceAccountUser`. (Locally you impersonate this same SA instead of using
   the key — see [DEPLOY.md](../DEPLOY.md).)

2. **Store it as a GitHub secret.** Repo → Settings → Secrets and variables → Actions →
   New repository secret, name **`GCP_SA_KEY`**, value = full contents of `key.json`.

3. **Environments.** `dev` auto-creates with no protection on first push. The **`prod`**
   Environment has a **required reviewer** so `deploy-prod` pauses for approval (repo →
   Settings → Environments → `prod`); creating or editing it needs repo-admin rights.

4. **Delete the local key** — `rm key.json`. It's gitignored (`**/key.json`) but never
   commit one. Rotate the key periodically.

5. *(Optional)* Override `PROJECT_ID` / `REGION` / `BUCKET` as repo **Variables** if they
   ever differ from the `deploy/config.sh` defaults.

The `GCP_SA_KEY` secret and the `prod` Environment are configured for this repo; the steps
above are documented for re-setup and key rotation.

## Teardown

`terraform destroy` alone is **not** enough — Terraform here is infra-only (bucket + AR
repo + APIs), so it leaves the Cloud Run services and the CI key behind. Use the script,
which does it in the right order (Cloud Run → bucket + AR repo → optional SA key):

```bash
./deploy/99_teardown.sh                  # interactive: type the project id to confirm
./deploy/99_teardown.sh --yes            # non-interactive
./deploy/99_teardown.sh --delete-sa-keys # also delete the CI key (BREAKS GitHub deploy)
```

It prefers `terraform destroy` when TF state exists, else deletes the bucket + AR repo
directly with `gcloud` (infra may have been created via `00_setup_gcp.sh`). It never
touches the BigQuery source table, the deploy SA, or the enabled APIs. If you pass
`--delete-sa-keys`, also remove the `GCP_SA_KEY` secret from the GitHub repo.

## Migrating to keyless later

Once an account with `setIamPolicy` is available, switch to **Workload Identity
Federation**: create a pool + provider, grant the SA `roles/iam.workloadIdentityUser`
for the repo, and replace `credentials_json` with `workload_identity_provider` +
`service_account` in each `google-github-actions/auth@v2` step. Then delete `GCP_SA_KEY`.
