# Infra-only: APIs + GCS bucket + Artifact Registry repo.
# Workloads (Vertex pipeline, Cloud Build, Cloud Run) run as the project DEFAULT
# service accounts, which already carry the access they need — so no custom SA /
# IAM bindings here (and none required: project Editor can't set IAM policy).

locals {
  apis = [
    "serviceusage.googleapis.com",
    "aiplatform.googleapis.com",
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "bigquery.googleapis.com",
    "storage.googleapis.com",
    "compute.googleapis.com",
    "monitoring.googleapis.com",
  ]
}

resource "google_project_service" "enabled" {
  for_each = toset(local.apis)
  project  = var.project_id
  service  = each.value

  disable_on_destroy = false # don't tear down APIs (other things may use them)
}

resource "google_storage_bucket" "models" {
  name                        = var.bucket
  project                     = var.project_id
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = var.force_destroy_bucket

  depends_on = [google_project_service.enabled]
}

resource "google_artifact_registry_repository" "images" {
  repository_id = var.ar_repo
  project       = var.project_id
  location      = var.region
  format        = "DOCKER"
  description   = "Lead scoring training + serving images"

  # Keep the registry from growing unbounded. Every deploy pushes :latest, which
  # leaves the previous digest UNTAGGED but still stored — those pile up. Policy:
  # keep the most recent few of each image for rollback, prune old untagged ones.
  # KEEP rules take precedence over DELETE, so recent versions are always safe.
  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"
    most_recent_versions {
      keep_count            = var.ar_keep_recent
      package_name_prefixes = ["lead-scoring-serve", "training-base"]
    }
  }

  cleanup_policies {
    id     = "delete-old-untagged"
    action = "DELETE"
    condition {
      tag_state  = "UNTAGGED"
      older_than = "${var.ar_untagged_ttl_days * 24 * 60 * 60}s"
    }
  }

  # Set true to preview deletions in the AR logs without actually deleting.
  cleanup_policy_dry_run = var.ar_cleanup_dry_run

  depends_on = [google_project_service.enabled]
}

# --- Alerting: email someone when a Vertex training pipeline FAILS -------------
# Decoupled from the pipeline code: Cloud Monitoring watches Vertex's pipeline-level
# state events (logName .../pipeline_job_events, jsonPayload.state). Skipped entirely
# when var.alert_emails is empty.

resource "google_monitoring_notification_channel" "email" {
  for_each     = toset(var.alert_emails)
  project      = var.project_id
  display_name = "lead-scoring alerts → ${each.value}"
  type         = "email"
  labels       = { email_address = each.value }

  depends_on = [google_project_service.enabled]
}

resource "google_monitoring_alert_policy" "pipeline_failed" {
  count        = length(var.alert_emails) > 0 ? 1 : 0
  project      = var.project_id
  display_name = "Vertex training pipeline failed (lead-scoring)"
  combiner     = "OR"

  conditions {
    display_name = "PipelineJob state = FAILED"
    # Pipeline-level failure event for our training pipeline only (verified against
    # the real log schema). Implicit AND across the lines.
    condition_matched_log {
      filter = <<-EOT
        logName="projects/${var.project_id}/logs/aiplatform.googleapis.com%2Fpipeline_job_events"
        jsonPayload.pipelineName="lead-scoring-train"
        jsonPayload.state="PIPELINE_STATE_FAILED"
      EOT
    }
  }

  # Required for log-match policies; throttle repeated notifications.
  alert_strategy {
    notification_rate_limit { period = "300s" }
  }

  notification_channels = [for c in google_monitoring_notification_channel.email : c.id]

  documentation {
    subject = "❌ Vertex training pipeline failed — lead-scoring"
    content = "A lead-scoring training PipelineJob reached PIPELINE_STATE_FAILED. Check the Vertex AI > Pipelines console for the failed step and logs."
  }

  depends_on = [google_project_service.enabled]
}

# Success notification: a retrain finished — someone should review the new model and
# decide whether to promote dev -> prod (the monthly cron retrains DEV only). Fires on
# EVERY successful run of the pipeline (manual or scheduled), so reviewers are prompted.
resource "google_monitoring_alert_policy" "pipeline_succeeded" {
  count        = length(var.alert_emails) > 0 ? 1 : 0
  project      = var.project_id
  display_name = "Vertex training pipeline succeeded (lead-scoring)"
  combiner     = "OR"

  conditions {
    display_name = "PipelineJob state = SUCCEEDED"
    condition_matched_log {
      filter = <<-EOT
        logName="projects/${var.project_id}/logs/aiplatform.googleapis.com%2Fpipeline_job_events"
        jsonPayload.pipelineName="lead-scoring-train"
        jsonPayload.state="PIPELINE_STATE_SUCCEEDED"
      EOT
    }
  }

  alert_strategy {
    notification_rate_limit { period = "300s" }
  }

  notification_channels = [for c in google_monitoring_notification_channel.email : c.id]

  documentation {
    subject = "✅ Vertex training pipeline finished — review the model (lead-scoring)"
    content = "A lead-scoring training PipelineJob reached PIPELINE_STATE_SUCCEEDED. Open Vertex AI > Pipelines: check the evaluate-<segment> report (capacity table, grades, lift) and validate-and-promote-<segment> (candidate vs current live). If it looks good, promote dev -> prod (ENV=prod ./deploy/02_run_pipeline.sh, or approve the prod gate in GitHub Actions)."
  }

  depends_on = [google_project_service.enabled]
}
