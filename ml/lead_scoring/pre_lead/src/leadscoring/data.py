"""BigQuery I/O for training."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, preprocess


def _apply_target_rename(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the raw target column to the model's ``TARGET`` contract (``y``).

    The Dataform table exposes the target as ``config.RAW_TARGET``
    (e.g. ``apd_es_matricula``); the model expects ``config.TARGET``. Idempotent:
    a table that already carries ``y`` is left untouched (and never clobbered).

    Args:
        df: The raw DataFrame loaded from BigQuery.

    Returns:
        The DataFrame with the target column renamed to ``config.TARGET`` when needed.
    """
    if config.TARGET not in df.columns and config.RAW_TARGET in df.columns:
        return df.rename(columns={config.RAW_TARGET: config.TARGET})
    return df


def load(table_ref: str | None = None, limit: int | None = None) -> pd.DataFrame:
    """Load the training table from BigQuery into a DataFrame.

    Adds the ``segmento`` routing column if the table doesn't already carry it
    (``platform == "landing"`` -> 'landing', else 'main' — same rule as
    ``config.route_segment`` at serve time), then runs ``derive_columns`` — the
    identical derive step used at serve time, so there is no train/serve skew.

    The BigQuery client is pinned to ``config.BQ_LOCATION``; this is REQUIRED for
    non-US datasets, else queries fail with "Dataset not found in location US".

    Args:
        table_ref: Fully-qualified ``project.dataset.table``; defaults to
            ``config.BQ_TABLE_REF``.
        limit: Optional row cap (for quick local runs).

    Returns:
        The training DataFrame with the ``segmento`` and derived columns added, and
        the raw target (``config.RAW_TARGET``) renamed to the model contract (``y``).
    """
    from google.cloud import bigquery

    table_ref = table_ref or config.BQ_TABLE_REF
    client = bigquery.Client(project=config.PROJECT_ID, location=config.BQ_LOCATION)
    sql = f"SELECT * FROM `{table_ref}`"
    if limit:
        sql += f" LIMIT {int(limit)}"
    df = client.query(sql).to_dataframe()

    df = _apply_target_rename(df)

    if config.SEGMENT_COL not in df.columns:
        plat = df.get("platform", pd.Series("", index=df.index)).astype(str).str.strip().str.lower()
        df[config.SEGMENT_COL] = np.where(plat == "landing", "landing", "main")
    df = preprocess.derive_columns(df)
    return df


def segment_frame(df: pd.DataFrame, segment: str) -> pd.DataFrame:
    """Subset a DataFrame to a single segment.

    Args:
        df: The full training DataFrame (must carry ``config.SEGMENT_COL``).
        segment: Segment name to keep.

    Returns:
        A copy of ``df`` with only the rows for ``segment``.
    """
    return df[df[config.SEGMENT_COL] == segment].copy()
