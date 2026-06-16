"""BigQuery I/O for training."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, preprocess


def load(table_ref: str | None = None, limit: int | None = None) -> pd.DataFrame:
    """Load the training table from BigQuery into a DataFrame.

    Adds the ``segmento`` routing column if the table doesn't already carry it
    (``platform == "landing"`` -> 'landing', else 'main' — same rule as
    ``config.route_segment`` at serve time), then runs ``derive_columns``.

    Args:
        table_ref: Fully-qualified ``project.dataset.table``; defaults to
            ``config.BQ_TABLE_REF``.
        limit: Optional row cap (for quick local runs).

    Returns:
        The training DataFrame with the ``segmento`` and derived columns added.
    """
    from google.cloud import bigquery

    table_ref = table_ref or config.BQ_TABLE_REF
    # location is REQUIRED for non-US datasets, else "Dataset not found in location US".
    client = bigquery.Client(project=config.PROJECT_ID, location=config.BQ_LOCATION)
    sql = f"SELECT * FROM `{table_ref}`"
    if limit:
        sql += f" LIMIT {int(limit)}"
    df = client.query(sql).to_dataframe()

    if config.SEGMENT_COL not in df.columns:
        plat = df.get("platform", pd.Series("", index=df.index)).astype(str).str.strip().str.lower()
        df[config.SEGMENT_COL] = np.where(plat == "landing", "landing", "main")
    # Same derive step as serving (no train/serve skew).
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
