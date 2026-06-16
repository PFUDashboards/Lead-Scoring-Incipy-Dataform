"""Trigger a Dataform workflow invocation and wait for it, returning a data_version.

The training pipeline reads its BigQuery source AFTER Dataform rebuilds it, so the
table reflects the latest GA4 data on every run. The returned invocation name is the
provenance ``data_version`` stamped into the model artifact.

The google-cloud-dataform SDK is imported lazily inside ``run_workflow`` so that a pure
``--compile-only`` run and the GCP-free unit tests never need the dependency.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

# Defaults for the bq-pfu-ga4 Dataform repository (override via env / args).
DEFAULT_PROJECT_NUMBER = "992321307436"
DEFAULT_LOCATION = "europe-west1"
DEFAULT_REPO = "Lead-Scoring-Incipy-Dataform"
DEFAULT_WORKFLOW = "lead_scoring_main"


class DataformError(RuntimeError):
    """Raised when the workflow invocation ends FAILED/CANCELLED or times out."""


def _repo_path(project_number: str, location: str, repo: str) -> str:
    """Build the Dataform repository resource path.

    Args:
        project_number: GCP project NUMBER (not id).
        location: Dataform region (e.g. ``europe-west1``).
        repo: Dataform repository name.

    Returns:
        The ``projects/.../locations/.../repositories/...`` resource name.
    """
    return f"projects/{project_number}/locations/{location}/repositories/{repo}"


def skipped_data_version() -> str:
    """Provenance marker when Dataform is skipped: a UTC timestamp, never empty.

    Returns:
        A string like ``skipped-20260616T130000Z``.
    """
    return "skipped-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_workflow(
    *,
    project_number: str | None = None,
    location: str | None = None,
    repo: str | None = None,
    workflow: str | None = None,
    poll_seconds: float = 15.0,
    timeout_seconds: float = 3600.0,
) -> str:
    """Create a Dataform workflow invocation, poll to SUCCEEDED, return its name.

    Args:
        project_number: GCP project number; defaults to ``DATAFORM_PROJECT_NUMBER`` env or the constant.
        location: Dataform region; defaults to ``DATAFORM_LOCATION`` env or the constant.
        repo: Dataform repository; defaults to ``DATAFORM_REPO`` env or the constant.
        workflow: workflowConfig name; defaults to ``DATAFORM_WORKFLOW`` env or the constant.
        poll_seconds: Seconds between state polls.
        timeout_seconds: Give up (raise) after this many seconds.

    Returns:
        The workflow invocation resource name, used as the provenance ``data_version``.

    Raises:
        DataformError: if the invocation ends FAILED/CANCELLED or times out.
    """
    from google.cloud import dataform_v1beta1 as dataform

    project_number = project_number or os.environ.get("DATAFORM_PROJECT_NUMBER", DEFAULT_PROJECT_NUMBER)
    location = location or os.environ.get("DATAFORM_LOCATION", DEFAULT_LOCATION)
    repo = repo or os.environ.get("DATAFORM_REPO", DEFAULT_REPO)
    workflow = workflow or os.environ.get("DATAFORM_WORKFLOW", DEFAULT_WORKFLOW)

    client = dataform.DataformClient()
    parent = _repo_path(project_number, location, repo)
    inv = client.create_workflow_invocation(
        parent=parent,
        workflow_invocation=dataform.WorkflowInvocation(
            workflow_config=f"{parent}/workflowConfigs/{workflow}",
        ),
    )
    name = inv.name
    states = dataform.WorkflowInvocation.State
    deadline = time.monotonic() + timeout_seconds
    while True:
        state = client.get_workflow_invocation(name=name).state
        if state == states.SUCCEEDED:
            return name
        if state in (states.FAILED, states.CANCELLED, states.CANCELING):
            raise DataformError(f"Dataform invocation {name} ended in state {state.name}")
        if time.monotonic() > deadline:
            raise DataformError(f"Dataform invocation {name} timed out after {timeout_seconds}s")
        time.sleep(poll_seconds)
