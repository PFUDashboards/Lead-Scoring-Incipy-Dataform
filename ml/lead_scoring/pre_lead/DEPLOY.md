# Deployment guide — Lead Scoring (GCP)

Everything you need to **deploy, update and destroy**, step by step.
Target project: **`bq-pfu-ga4`**, region **`europe-west1`**, BigQuery data in **`EU`**
(`BQ_LOCATION="EU"`).

This guide covers the **three scenarios**:
- **[A) From scratch](#a-deploy-from-scratch)** — nothing in GCP yet.
- **[B) Update](#b-update-when-something-is-already-deployed)** — something is already deployed.
- **[C) Destroy](#c-destroy-everything)** — remove the resources.

All commands run **from `ml/lead_scoring/pre_lead/`**.

| Piece | What it is | Created/updated with |
|---|---|---|
| Infra | GCS bucket + Artifact Registry + APIs + failure alert | Terraform (`terraform/`) |
| Images | `training-base` (pipeline) + `lead-scoring-serve` (API) | `deploy/01_build_images.sh` |
| Models | 2 joblibs (`landing`+`main`) in `gs://…/models/<env>/live/` | `deploy/02_run_pipeline.sh` (Vertex) |
| API | Cloud Run service `lead-scoring-<env>` | `deploy/03_deploy_serving.sh` |

> **`ENV`** (`dev` by default \| `prod`) separates models and service per environment **without
> needing a second GCP project**. For testing, `dev` is enough; `prod` is for real
> production. Step 2 trains a *candidate* and, if it doesn't regress against the current
> *live*, promotes it to *live* (what the API serves). Details in the README →
> **Environments & model promotion**.

## TL;DR (from scratch, with prerequisites already done)

> With impersonation already configured (see [Prerequisites](#0-prerequisites-one-time-only)),
> the scripts run as the SA with nothing extra.

```bash
./deploy/tf.sh apply                                          # 0) infra (init+apply)
./deploy/01_build_images.sh                                   # 1) images
ENV=dev ./deploy/02_run_pipeline.sh                           # 2) train (Vertex)
ENV=dev ./deploy/03_deploy_serving.sh                         # 3) API (Cloud Run)
```
Order is **mandatory** `0 → 1 → 2 → 3`. Step 3 needs step 2 to have left the
models in `live/`, or the API starts but answers `503` (no models).

---

## 0. Prerequisites (one time only)

### Tools
```bash
gcloud --version        # Google Cloud CLI
bq version              # BigQuery CLI
terraform version       # >= 1.5
./.venv/bin/python -V   # the repo's venv Python
```
If Terraform is missing: `brew install terraform`.

### Auth — by impersonating the service account (no key JSON)

The deployment is done by the SA **`incipy-lead-scoring@bq-pfu-ga4.iam.gserviceaccount.com`**.
You don't have its key: you use it by **impersonation** (your account has
`roles/iam.serviceAccountTokenCreator` on it). Your user account on its own does **not**
have write roles in the project, so every `gcloud`/`bq`/Terraform command must run
as the SA.

Configure it **once, all via the command line** (without exporting variables in the session):

```bash
gcloud auth login
gcloud auth application-default login --impersonate-service-account=incipy-lead-scoring@bq-pfu-ga4.iam.gserviceaccount.com
gcloud config set project bq-pfu-ga4
gcloud config set auth/impersonate_service_account incipy-lead-scoring@bq-pfu-ga4.iam.gserviceaccount.com
```
- The 2nd line leaves the **ADC impersonated** → used by Terraform (`tf.sh`) and the Vertex SDK (step 2).
- The 4th leaves **gcloud/bq** impersonating by default → used by scripts `01` and `03`.

With that, `gcloud`, `bq`, Terraform and scripts `01/02/03` already run as the SA: **you don't
need to repeat the flag or export anything**. For a one-off command without touching the config,
append `--impersonate-service-account=incipy-lead-scoring@bq-pfu-ga4.iam.gserviceaccount.com`.

> **Web console (optional).** The UI authenticates with **your** account (it can't impersonate),
> which by default sees nothing in the project. To view pipelines/Cloud Run/GCS ask the admin
> for `roles/viewer` (or `roles/artifactregistry.reader` to see images) on `bq-pfu-ga4`.

### Quick checks
```bash
# is impersonation active? (must print the SA, not be empty)
gcloud config get-value auth/impersonate_service_account

# is the training table there? (already runs as the SA, EU)
bq --location=EU show bq-pfu-ga4:BQ_PFU_INCIPY.lead_scoring_train
```

### Configuration (review before you start)
**Single source of truth: `src/leadscoring/config.py`** ("Deployment target" block).
That's where project / region / bucket / ar_repo / dataset / `BQ_LOCATION` live. No copies:

| Where | What |
|---|---|
| `src/leadscoring/config.py` | **the only place to edit** — project/region/bucket/dataset/BQ_LOCATION + features |
| `deploy/config.sh` | **reads** them from `config.py` (no duplication) and exports `TF_VAR_*` for Terraform |
| `terraform/terraform.tfvars` | only `alert_emails` (everything else arrives via `TF_VAR_*`) |

Everything is **env-var overridable** (`PROJECT_ID=other REGION=... ./deploy/...sh`).

⚠️ The **bucket name** is global and unique. To change it edit **only** `BUCKET`
in `config.py`; bash and Terraform pick it up from there.
⚠️ **`REGION` must match the BigQuery data location.** Table in `EU` →
`europe-west1` and `BQ_LOCATION="EU"` (already configured).

---

## A) Deploy from scratch

### Step 0 — Infra with Terraform
Creates the APIs, the GCS bucket, the image repo (Artifact Registry) and the email
alert for pipeline failures.

Use the `deploy/tf.sh` wrapper (it sources `config.sh` → injects project/region/bucket
from `config.py` as `TF_VAR_*`, so Terraform has no duplicated values):

```bash
./deploy/tf.sh init        # downloads the Google provider (first time)
./deploy/tf.sh plan        # optional: review what it will create (14 resources)
./deploy/tf.sh apply       # type 'yes' to confirm
```
It's idempotent (you can re-run it without breaking anything). If you prefer raw
`terraform`, **first** `source deploy/config.sh` and then `terraform -chdir=terraform ...`.

> With impersonation configured (see Prerequisites), Terraform uses the impersonated ADC,
> so the `apply` runs as the SA with no extra config.

> **SA permissions.** The `apply` only creates what the SA is allowed to create. The
> **3 Monitoring alerts** need `roles/monitoring.editor` on the SA; without it,
> those resources fail with `403` and the rest (bucket, AR, APIs) is created anyway. Ask the admin
> for it and repeat the `apply` (idempotent). To **build images** (step 1) the SA needs
> `roles/cloudbuild.builds.editor`.

> Alternative without Terraform: `./deploy/00_setup_gcp.sh` does the same with gcloud
> (but **without** the pipeline-failure alert, which only exists in Terraform).

### Step 1 — Build the images
Builds and pushes the two images (training + serving) to Artifact Registry with
Cloud Build. Takes ~3-6 min the first time.

```bash
./deploy/01_build_images.sh
```
Pushes `…/training-base:latest` (pipeline components) and `…/lead-scoring-serve:latest`
(the API). Verify:
```bash
gcloud artifacts docker images list europe-west1-docker.pkg.dev/bq-pfu-ga4/lead-scoring
```

### Step 2 — Train (Vertex AI Pipelines)
Compiles and launches the pipeline; trains the two models and leaves the artifacts in the bucket.

```bash
ENV=dev ./deploy/02_run_pipeline.sh
```
- Installs `kfp` + `google-cloud-aiplatform` in the venv (first time).
- Prints the job ID. **View it in the console**: Vertex AI → Pipelines (europe-west1).
  You'll see the graph, the **metrics, the ROC curve and the HTML report** per segment (with
  **PR-AUC** highlighted and the **A/B/C grades** per lead).

When it finishes (green), check that the gate promoted the models to `live/`:
```bash
gcloud storage ls gs://bq-pfu-ga4-leadscoring/models/dev/live/
# lead_scoring_landing.joblib
# lead_scoring_main.joblib
```
The `validate-and-promote-<segment>` step shows `promoted=1/0` and an HTML with the reason.
If a retrain regresses, the gate does **NOT** promote (keeps the previous `live`) and the pipeline
stays **green** (**SOFT** gate).

> Compile only without launching (validate): `ENV=dev ./deploy/02_run_pipeline.sh --compile-only`

### Step 3 — Deploy the API (Cloud Run)
```bash
ENV=dev ./deploy/03_deploy_serving.sh
```
Deploys `lead-scoring-dev` (scale-to-zero, private auth, serves the environment's `live`
model). Prints the **URL**. Verify:
The service is private: the token must be an identity token from a SA with `run.invoker`
(granted by `03`) and with the audience equal to the service URL.

```bash
URL=$(gcloud run services describe lead-scoring-dev --region europe-west1 --format='value(status.url)')
TOKEN=$(gcloud auth print-identity-token \
  --impersonate-service-account=incipy-lead-scoring@bq-pfu-ga4.iam.gserviceaccount.com \
  --audiences="$URL")

curl -s "$URL/health" -H "Authorization: Bearer $TOKEN"

curl -s -X POST "$URL/score" -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"platform":"landing","page_name":"unbounce/mba","page_location":"https://x/landing/mba?utm_campaign=brand","user_studies":"es-2","language_site":"es","ga_session_number":2}'
```
Returns something like:
```json
{"segmento":"landing","score":0.07,"grade":"A","base_rate":0.023,"lift_vs_base":3.0,"features_used":[...]}
```

> **Payload fields — routing and features:** the segment is decided by **`platform`**
> (`"landing"` → landing model; anything else, e.g. `"main_site"`, → main). Also,
> always send the fields that are a model **feature** or they'll be seen as `MISSING`:
> `main` uses `form_name`, `page_name`, `product_id`, `user_country`, `user_province`,
> `user_studies`, `ga_session_number`; `landing` uses `page_name`, `language_site`,
> `utm_campaign` (from `page_location`), `user_studies`, `ga_session_number`.

---

## B) Update (when something is already deployed)

What to rebuild depends on **what you changed**. Rule: if you touched a process's code,
you must **rebuild its image** before relaunching it.

| You changed… | 01 build | 02 train | 03 deploy |
|---|:--:|:--:|:--:|
| Only data (BigQuery) | — | yes | — (auto-reload / `/reload`) |
| `serving/app.py` or `leadscoring` (serving) | yes | — | yes |
| `pipelines/` or `leadscoring` (training) | yes | yes | — |
| Infra (`terraform/`) | — | — | — → `./deploy/tf.sh apply` |

### B1. Retrain the model (new data, same code)
```bash
ENV=dev ./deploy/02_run_pipeline.sh        # trains candidate → gate → live
```
- **Caching is OFF** by default, so a relaunch **always trains** (Vertex
  caching is based on the table *reference*, not its contents).
- **No redeploy needed:** the API re-checks GCS every ~5 min (`MODEL_RELOAD_CHECK_SECONDS`)
  and *hot-swaps* the new `live` on its own. To pick it up **instantly**:
  ```bash
  curl -s -X POST "$URL/reload" -H "Authorization: Bearer $TOKEN"
  ```

### B2. You changed **serving** code (`serving/app.py`, `src/leadscoring/`)
The serving image carries the code inside → rebuild and redeploy:
```bash
./deploy/01_build_images.sh
ENV=dev ./deploy/03_deploy_serving.sh
```

### B3. You changed **pipeline/training** code (`pipelines/`, `src/leadscoring/`)
The `training-base` image carries the code → rebuild and retrain:
```bash
./deploy/01_build_images.sh
ENV=dev ./deploy/02_run_pipeline.sh
```

### B4. Promote dev → prod
Repeat **2 and 3** with `ENV=prod` (trains/promotes the prod model and brings up the
`lead-scoring-prod` service):
```bash
ENV=prod ./deploy/02_run_pipeline.sh
ENV=prod ./deploy/03_deploy_serving.sh
```

### B5. Via CI/CD (GitHub Actions) — automatic
The integration flow trains **and** deploys on every push (`deploy.yml`):
```
push develop → quality(ruff+pytest) → build → train(dev, wait) → deploy(dev)
push main    → quality → build → [APPROVE prod] → train(prod, wait) → deploy(prod)
```
- The `train` job runs the Vertex pipeline with `--wait`: it stays **"running"** until
  Vertex finishes and turns **green only if it trained successfully** (red if it fails). The `deploy`
  afterwards serves the freshly-trained model.
- On `main`, the `train` job **pauses for approval** (`prod` Environment, required
  reviewer); a single approval governs both training and deploying prod.
- **PR** (to any branch) and push to `main`: `ci.yml` (`ruff` + `pytest`).
- **Data retrain** (not code): `train.yml` → manual (`workflow_dispatch`,
  choose `dev`/`prod`) **or** the **monthly cron → prod** (day 1, 03:00 UTC; governed by the
  SOFT gate + the emails, no human approval).
- The secret **`GCP_SA_KEY`** is configured in the repo (CI authenticates with it, not by
  impersonation). The **`dev`** Environment auto-creates with no approval on the first push; the
  **`prod`** one has a *required reviewer* (editing it needs repo-admin rights).
  Details in [`deploy/CICD.md`](deploy/CICD.md).

> The **manual** deploy (section A) and the CI/CD one do the same thing under the hood (the same
> `01/02/03` scripts); you can use either.

---

## C) Destroy everything

One script deletes in the right order (Cloud Run → bucket + Artifact Registry →
optionally the CI SA keys):

```bash
./deploy/99_teardown.sh            # asks for confirmation (type the project id)
./deploy/99_teardown.sh --yes      # no prompt
```

**Deletes:** Cloud Run services (`lead-scoring-dev` *and* `lead-scoring-prod`), bucket
`gs://bq-pfu-ga4-leadscoring` (with models + artifacts), Artifact Registry repo (with
images). Uses `terraform destroy` if there's state; otherwise deletes with `gcloud`.

> With impersonation configured (see Prerequisites), the script runs as the SA. The
> **APIs** are not disabled (`disable_on_destroy=false`): they leave the Terraform state but
> stay enabled (other things in the project use them).

**Does NOT touch:** the BigQuery table `BQ_PFU_INCIPY.lead_scoring_train` (your data), the
service accounts, or the APIs.

**CI key (optional):** by default it's kept (GitHub Actions uses it). To delete the SA
keys:
```bash
./deploy/99_teardown.sh --yes --delete-sa-keys   # ⚠️ breaks the GitHub Actions deploy
```
If you delete them, also remove the `GCP_SA_KEY` secret from the GitHub repo.

> Only bring the API down (without deleting data/images):
> `gcloud run services delete lead-scoring-dev --region europe-west1`.

---

## Common problems

| Symptom | Cause / fix |
|---|---|
| `/health` or `/score` returns **503** | No models in `models/<env>/live/` → run step 2 (and let the gate promote) before step 3. |
| The new model isn't served | Wait ~5 min or do `POST /reload`; check that `validate-and-promote` did `promoted=1` (see the HTML in Vertex). |
| I changed serving and it's not reflected | Rebuild (`01`) + redeploy (`03`): the code lives inside the image, it doesn't reload on its own. |
| `PermissionDenied` launching the pipeline | The ADC is not impersonated: repeat `gcloud auth application-default login --impersonate-service-account=incipy-lead-scoring@bq-pfu-ga4.iam.gserviceaccount.com`. |
| `403` creating Monitoring alerts | The SA lacks `roles/monitoring.editor` (ask the admin and repeat `apply`). |
| Build fails: repo doesn't exist | You didn't run Terraform / step 0 (the Artifact Registry repo is missing). |
| Pipeline fails reading BigQuery | The table or bucket aren't in `EU` / `europe-west1` (they must match, `BQ_LOCATION=EU`). |
| `Dataset not found in location US` | The BQ client didn't get `location="EU"`; check `BQ_LOCATION` in `config.py`. |
| `bucket already exists` | The name is global; pick another by editing `BUCKET` in `config.py`. |
| `curl` returns 403 | The token isn't valid for a private service. Mint it as the SA and with the audience = URL: `gcloud auth print-identity-token --impersonate-service-account=incipy-lead-scoring@bq-pfu-ga4.iam.gserviceaccount.com --audiences="$URL"`. And that identity must have `run.invoker` (granted by `03`). |

---

## Summary of what each thing does

| Step | Tool | Creates / does |
|---|---|---|
| 0 | Terraform | bucket + Artifact Registry + APIs + failure alert (infra) |
| 1 | Cloud Build | Docker images (training + serving) |
| 2 | Vertex Pipelines | trains the 2 models → joblibs in GCS + metrics/HTML in the UI |
| 3 | Cloud Run | deploys the scoring API (real time) |

CI/CD (GitHub Actions) automates **1, 2 and 3** (dev on push to `develop`, prod on push to
`main` with approval) and the monthly cron retrain — see `deploy/CICD.md`.
