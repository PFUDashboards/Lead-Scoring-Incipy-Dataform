# project_id / region / bucket / ar_repo are NOT set here on purpose: they come from
# the single source of truth (src/leadscoring/config.py) via the TF_VAR_* env vars that
# deploy/config.sh exports. Source it (or use deploy/tf.sh) before running terraform.
# Only Terraform-specific knobs live here.
alert_emails = ["acortina@knowmadmood.com"]
