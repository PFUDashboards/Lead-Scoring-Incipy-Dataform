"""Shared preprocessing — single source of truth for train AND serve.

Baked into both images and the fitted ``ColumnTransformer`` is persisted and
re-applied at serve time (never re-fitted) -> zero train/serve skew.

Design rules (do not "improve" silently):
- Nulls are information: categoricals -> ``MISSING`` category; numerics keep
  ``np.nan`` (XGBoost routes it natively). Never mean/median-impute.
- Dynamic schema: features derived from existing columns minus id/target/segment.
- High-cardinality categoricals tamed by ``OneHotEncoder(min_frequency=20)``.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from . import config

MISSING = "MISSING"

# Features engineered from raw columns (not materialized in the table).
_UTM_CAMPAIGN_RE = re.compile(r"[?&]utm_campaign=([^&]+)")
DERIVED_COLUMNS = ("page_path", "utm_campaign")


def _utm_campaign_of(u) -> object:
    if not isinstance(u, str) or not u:
        return np.nan
    m = _UTM_CAMPAIGN_RE.search(u)
    return m.group(1) if m else np.nan


def derive_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features identically in train and serve.

    ``page_path`` = ``page_name`` (GA content label, NOT parsed from the URL).
    ``utm_campaign`` = utm_campaign param extracted from ``page_location``.

    Args:
        df: Raw rows carrying ``page_name`` / ``page_location`` (either may be absent).

    Returns:
        A copy of ``df`` with ``page_path`` and ``utm_campaign`` added (NaN when the
        source is missing), so a model that lists them never hits a missing column.
    """
    out = df.copy()
    name = out["page_name"] if "page_name" in out.columns else pd.Series(
        np.nan, index=out.index
    )
    loc = out["page_location"] if "page_location" in out.columns else pd.Series(
        np.nan, index=out.index
    )
    # Empty/non-string page_name is missing (NaN), never an empty-string category.
    out["page_path"] = pd.Series(
        [v if (isinstance(v, str) and v) else np.nan for v in name],
        index=out.index, dtype=object,
    )
    out["utm_campaign"] = [_utm_campaign_of(u) for u in loc]
    return out


def resolve_features(
    df: pd.DataFrame,
    id_cols: list[str] | None = None,
    target: str = config.TARGET,
    segment_col: str = config.SEGMENT_COL,
    override: list[str] | None = None,
) -> list[str]:
    """Return the feature columns to model on.

    Args:
        df: The (segment) DataFrame whose columns are the candidate features.
        id_cols: Identifier columns to exclude; defaults to ``config.ID_COLS``.
        target: Target column name to exclude.
        segment_col: Segment column name to exclude.
        override: Explicit feature list. If given, only those present in ``df`` are
            kept (a stale override is skipped, not fatal).

    Returns:
        The ordered list of feature column names.
    """
    id_cols = list(id_cols if id_cols is not None else config.ID_COLS)
    excluded = set(id_cols) | {target, segment_col}
    if override:
        return [c for c in override if c in df.columns and c not in excluded]
    return [c for c in df.columns if c not in excluded]


def split_types(df: pd.DataFrame, feats: list[str]) -> tuple[list[str], list[str]]:
    """Split features into numeric and categorical.

    A column is numeric only if its dtype is numeric AND it is not id-like (``*_id``
    or in ``ID_COLS``); e.g. ``product_id`` is numeric but an identifier -> categorical.

    Args:
        df: The DataFrame holding the feature columns.
        feats: Feature column names to classify.

    Returns:
        A ``(numeric, categorical)`` tuple of column-name lists.
    """
    num, cat = [], []
    for c in feats:
        is_id_like = c.endswith("_id") or c in config.ID_COLS
        if pd.api.types.is_numeric_dtype(df[c]) and not is_id_like:
            num.append(c)
        else:
            cat.append(c)
    return num, cat


def prep_X(df: pd.DataFrame, num: list[str], cat: list[str]) -> pd.DataFrame:
    """Build the model-ready frame, preserving nulls.

    Numerics become float with NaN kept. Categoricals become python ``str`` with
    real ``np.nan`` for missing (NOT ``pd.NA``, which makes ``SimpleImputer`` raise).

    Args:
        df: Raw rows; missing categorical columns are filled with NaN.
        num: Numeric feature names.
        cat: Categorical feature names.

    Returns:
        A DataFrame with one column per feature, ready for the preprocessor.
    """
    out = pd.DataFrame(index=df.index)
    for c in num:
        out[c] = pd.to_numeric(df.get(c), errors="coerce").astype(float)
    for c in cat:
        s = df[c] if c in df.columns else pd.Series(np.nan, index=df.index)
        out[c] = pd.Series(
            np.where(s.isna(), np.nan, s.astype(str)), index=df.index, dtype=object
        )
    return out


def build_preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    """Build the (unfitted) ColumnTransformer for the feature matrix.

    Numerics pass through (NaN kept for XGBoost); categoricals get a ``MISSING``
    constant fill then one-hot encoding (``min_frequency=20`` to tame rare levels).

    Args:
        num: Numeric feature names.
        cat: Categorical feature names.

    Returns:
        An unfitted ``ColumnTransformer``.
    """
    cat_pipe = Pipeline(
        [
            ("miss", SimpleImputer(strategy="constant", fill_value=MISSING)),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore", min_frequency=20, sparse_output=False
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("num", "passthrough", num), ("cat", cat_pipe, cat)],
        remainder="drop",
    )


def fit_preprocessor(
    df: pd.DataFrame, override: list[str] | None = None
) -> tuple[ColumnTransformer, list[str], list[str], list[str]]:
    """Resolve features, build and fit the transformer on ``df``.

    Args:
        df: The segment DataFrame to fit on.
        override: Optional explicit feature list (see :func:`resolve_features`).

    Returns:
        A ``(fitted_preprocessor, features, num, cat)`` tuple.
    """
    feats = resolve_features(df, override=override)
    num, cat = split_types(df, feats)
    pre = build_preprocessor(num, cat)
    pre.fit(prep_X(df, num, cat))
    return pre, feats, num, cat


def transform(pre: ColumnTransformer, df: pd.DataFrame, num: list[str], cat: list[str]):
    """Apply a fitted preprocessor to raw rows.

    Args:
        pre: A preprocessor already fitted by :func:`fit_preprocessor`.
        df: Raw rows (missing columns are handled gracefully).
        num: Numeric feature names.
        cat: Categorical feature names.

    Returns:
        The transformed feature matrix (numpy array).
    """
    return pre.transform(prep_X(df, num, cat))
