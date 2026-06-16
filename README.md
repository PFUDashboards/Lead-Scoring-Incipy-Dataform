# Lead-Scoring-Incipy-Dataform

ML rankers that score sales leads along OBS's commercial **funnel**, so reps call the most
promising leads first, plus the **Dataform** project that builds the training table.

This repo is the source of truth for the lead-scoring models (migrated here from the old
`lead-scoring` repo, now archived). Target GCP project **`bq-pfu-ga4`**, region
**`europe-west1`**, BigQuery data in **`EU`**.

## Structure

```
ml/lead_scoring/<stage>/   one self-contained model per funnel stage
  pre_lead/                stage 1 — scores at form submit (the only stage today)
definitions/               Dataform models that build the training table
includes/                  Dataform shared SQL/JS
workflow_settings.yaml     Dataform project config
```

Each model under `ml/lead_scoring/` is self-contained (its own `src/`, `pipelines/`,
`serving/`, `deploy/`, Terraform and docs) and runs all commands from its own root.

## The `pre_lead` model

Stage-1 ranker: BigQuery → Vertex AI Pipeline (KFP v2) trains two segment models
(`landing`/unbounce, `main`/web) → Cloud Run (FastAPI) serves real-time scores. Scores are
for **ranking** (grade-band lift), not calibrated probabilities.

- Overview + scoring API: [`ml/lead_scoring/pre_lead/README.md`](ml/lead_scoring/pre_lead/README.md)
- Deploy / update / destroy: [`ml/lead_scoring/pre_lead/DEPLOY.md`](ml/lead_scoring/pre_lead/DEPLOY.md)
- CI/CD (GitHub Actions): [`ml/lead_scoring/pre_lead/deploy/CICD.md`](ml/lead_scoring/pre_lead/deploy/CICD.md)
