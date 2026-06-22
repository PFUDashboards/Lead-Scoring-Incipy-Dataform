terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.40, < 7.0"
    }
  }
  # Local state (terraform.tfstate on disk). To share/lock later, switch to a
  # GCS backend:  backend "gcs" { bucket = "<state-bucket>" prefix = "leadscoring" }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
