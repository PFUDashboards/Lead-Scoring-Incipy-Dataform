"""Unit tests for the GCP-free pieces of leadscoring.data.

The target rename maps the Dataform raw column (``apd_es_matricula``) to the model's
``y`` contract. These exercise the pure helper, no BigQuery needed.
"""

import pandas as pd

from leadscoring import config, data


def test_target_rename_applied():
    df = pd.DataFrame({config.RAW_TARGET: [0, 1], "x": [1, 2]})
    out = data._apply_target_rename(df)
    assert config.TARGET in out.columns
    assert config.RAW_TARGET not in out.columns
    assert list(out[config.TARGET]) == [0, 1]


def test_target_rename_idempotent_when_y_present():
    # A table that already carries `y` is left untouched and never clobbered.
    df = pd.DataFrame({config.TARGET: [0, 1], config.RAW_TARGET: [9, 9]})
    out = data._apply_target_rename(df)
    assert list(out[config.TARGET]) == [0, 1]
    assert config.RAW_TARGET in out.columns  # not renamed over the existing y


def test_target_rename_noop_when_neither_present():
    df = pd.DataFrame({"x": [1], "z": [2]})
    out = data._apply_target_rename(df)
    assert list(out.columns) == ["x", "z"]
