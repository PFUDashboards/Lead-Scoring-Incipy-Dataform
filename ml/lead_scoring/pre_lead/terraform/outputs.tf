output "bucket" {
  value       = "gs://${google_storage_bucket.models.name}"
  description = "Model + pipeline-root bucket."
}

output "artifact_registry" {
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
  description = "Docker image path prefix."
}

output "enabled_apis" {
  value = sort([for s in google_project_service.enabled : s.service])
}
