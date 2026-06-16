"""Unit tests for pipelines/dataform_trigger (GCP-free).

The google-cloud-dataform SDK is imported lazily inside ``run_workflow``; here we
inject a fake ``google.cloud.dataform_v1beta1`` module so the polling logic is tested
without credentials or the real dependency.
"""

import enum
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipelines"))
import dataform_trigger as dt  # noqa: E402


def test_repo_path():
    assert (
        dt._repo_path("123", "europe-west1", "Repo")
        == "projects/123/locations/europe-west1/repositories/Repo"
    )


def test_skipped_data_version_marker():
    v = dt.skipped_data_version()
    assert v.startswith("skipped-") and v.endswith("Z")


def _fake_dataform_module(final_state_name):
    """Build a fake dataform_v1beta1 module whose invocation ends in final_state_name."""
    mod = types.ModuleType("google.cloud.dataform_v1beta1")

    class _State(enum.Enum):
        SUCCEEDED = 1
        FAILED = 2
        CANCELLED = 3
        CANCELING = 4

    class WorkflowInvocation:
        State = _State

        def __init__(self, **kw):
            self.__dict__.update(kw)

    class _Inv:
        def __init__(self, name, state):
            self.name = name
            self.state = state

    class DataformClient:
        def create_workflow_invocation(self, parent, workflow_invocation):
            return _Inv(name=f"{parent}/workflowInvocations/abc", state=_State.SUCCEEDED)

        def get_workflow_invocation(self, name):
            return _Inv(name=name, state=getattr(_State, final_state_name))

    mod.WorkflowInvocation = WorkflowInvocation
    mod.DataformClient = DataformClient
    return mod


def _install(monkeypatch, mod):
    google = sys.modules.get("google") or types.ModuleType("google")
    cloud = sys.modules.get("google.cloud") or types.ModuleType("google.cloud")
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.dataform_v1beta1", mod)
    monkeypatch.setattr(cloud, "dataform_v1beta1", mod, raising=False)


def test_run_workflow_success(monkeypatch):
    _install(monkeypatch, _fake_dataform_module("SUCCEEDED"))
    name = dt.run_workflow(project_number="123", location="eu", repo="R", workflow="W", poll_seconds=0)
    assert name.endswith("/workflowInvocations/abc")


def test_run_workflow_failure_raises(monkeypatch):
    _install(monkeypatch, _fake_dataform_module("FAILED"))
    with pytest.raises(dt.DataformError):
        dt.run_workflow(project_number="123", location="eu", repo="R", workflow="W", poll_seconds=0)
