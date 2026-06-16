variable "project_id" {
  type        = string
  description = "GCP project where the lead-scoring infra lives."
}

variable "region" {
  type        = string
  description = "Region for the bucket, Artifact Registry, Vertex and Cloud Run. Must match the BigQuery data location."
}

variable "bucket" {
  type        = string
  description = "GCS bucket name (globally unique) for models + pipeline-root."
}

variable "ar_repo" {
  type        = string
  default     = "lead-scoring"
  description = "Artifact Registry (Docker) repository id."
}

variable "force_destroy_bucket" {
  type        = bool
  default     = true
  description = "Allow `terraform destroy` to delete the bucket even if it has objects (handy for a test env)."
}

variable "ar_keep_recent" {
  type        = number
  default     = 5
  description = "Artifact Registry cleanup: keep this many most-recent versions of each image (for rollback)."
}

variable "ar_untagged_ttl_days" {
  type        = number
  default     = 7
  description = "Artifact Registry cleanup: delete UNTAGGED images older than this many days (the digests left behind by each :latest push)."
}

variable "ar_cleanup_dry_run" {
  type        = bool
  default     = false
  description = "If true, AR cleanup policies only log what they WOULD delete instead of deleting. Set true to preview first."
}

variable "alert_emails" {
  type        = list(string)
  default     = []
  description = "Emails to notify when a Vertex training pipeline FAILS. Empty list = no alerting (resources skipped)."
}
