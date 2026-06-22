"""Train/serve parity tests for the derived columns.

`derive_columns` runs identically in training (`data.load`) and serving (`app.py`).
If it drifts, the model sees different features at serve time than it trained on — the
single most important invariant in this repo. Sources:
  - `page_path`    <- `page_name`      (the GA content label, NOT the URL path)
  - `utm_campaign` <- `page_location`  (REGEXP_EXTRACT of the utm_campaign query param)
Nulls must be preserved as NaN, never imputed.
"""

import pandas as pd

from leadscoring import preprocess


def test_derive_columns_page_path_from_page_name():
    df = pd.DataFrame(
        {
            "page_name": ["producto/detalle/masters-online/mba"],
            "page_location": ["https://obs.edu/mba?utm_campaign=spring"],
        }
    )
    out = preprocess.derive_columns(df)
    assert out.loc[0, "page_path"] == "producto/detalle/masters-online/mba"
    assert out.loc[0, "utm_campaign"] == "spring"


def test_derive_columns_campaign_among_other_params():
    df = pd.DataFrame(
        {
            "page_name": ["home/detalle/home"],
            "page_location": ["https://obs.edu/x?gclid=abc&utm_campaign=fall&foo=1"],
        }
    )
    out = preprocess.derive_columns(df)
    assert out.loc[0, "page_path"] == "home/detalle/home"
    assert out.loc[0, "utm_campaign"] == "fall"


def test_derive_columns_missing_is_nan_not_imputed():
    # None / empty / non-string must yield NaN (XGBoost-native missing), not a default.
    df = pd.DataFrame(
        {"page_name": [None, "", 123], "page_location": [None, "", 123]}
    )
    out = preprocess.derive_columns(df)
    assert out["page_path"].isna().all()
    assert out["utm_campaign"].isna().all()


def test_derive_columns_no_campaign_param_is_nan():
    df = pd.DataFrame(
        {"page_name": ["home/detalle/home"], "page_location": ["https://obs.edu/home"]}
    )
    out = preprocess.derive_columns(df)
    assert out.loc[0, "page_path"] == "home/detalle/home"
    assert out["utm_campaign"].isna().all()


def test_derive_columns_always_adds_both_when_sources_absent():
    # A schema without page_name / page_location must not crash; both columns are NaN.
    df = pd.DataFrame({"product_id": [1, 2]})
    out = preprocess.derive_columns(df)
    assert list(preprocess.DERIVED_COLUMNS) == ["page_path", "utm_campaign"]
    for col in preprocess.DERIVED_COLUMNS:
        assert col in out.columns
        assert out[col].isna().all()
    # Original frame is untouched (derive_columns copies).
    assert "page_path" not in df.columns
