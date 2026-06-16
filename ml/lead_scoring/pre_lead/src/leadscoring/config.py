"""Central configuration for the lead-scoring train + serve pipeline.

Project/dataset/naming live here so the rest of the code is environment-agnostic.
Every value is overridable via env vars (Cloud Run / Vertex without rebuilding).
"""
from __future__ import annotations

import os

# Logical environment (dev | prod). Namespaces the GCS model paths and the Cloud
# Run service so dev never overwrites the live prod model. Defaults to 'dev'.
ENV = os.environ.get("ENV", "dev")

# Deployment target — the single place to retarget another GCP project.
PROJECT_ID = os.environ.get("PROJECT_ID", "bq-pfu-ga4")
REGION = os.environ.get("REGION", "europe-west1")  # must match the BQ data location's continent
BUCKET = os.environ.get("BUCKET", "bq-pfu-ga4-leadscoring")  # gs://<BUCKET> — globally unique
AR_REPO = os.environ.get("AR_REPO", "lead-scoring")

# BigQuery source.
BQ_DATASET = os.environ.get("BQ_DATASET", "BQ_PFU_INCIPY")
BQ_TABLE = os.environ.get("BQ_TABLE", "lead_scoring_train")
BQ_TABLE_REF = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
BQ_LOCATION = os.environ.get("BQ_LOCATION", "EU")  # must be set for non-US datasets; keep in step with REGION

# Schema contract. Features are derived dynamically as
# (all columns) - ID_COLS - {TARGET, SEGMENT_COL}, so a BQ schema change can't break training.
TARGET = "y"
SEGMENT_COL = "segmento"
ID_COLS = [
    "event_timestamp",
    "user_pseudo_id",
    "ga_session_id",
    "transaction_id",
]

SEGMENTS = ["landing", "main"]

# Frozen feature list per segment; a segment without an entry falls back to fully
# dynamic. Only columns present after derive_columns are kept (resolve_features
# intersects). NOTE: page_path and utm_campaign are DERIVED in preprocess.derive_columns.
FEATURE_OVERRIDES: dict[str, list[str]] = {
    "landing": ["ga_session_number", "user_studies", "language_site", "utm_campaign", "page_path"],
    "main": ["ga_session_number", "product_id", "user_country", "user_province", "user_studies", "form_name", "page_name"],
}

# GCS layout: {MODELS_PREFIX}/{candidate,live}/lead_scoring_<segment>.joblib.
# The retrain writes 'candidate'; the in-pipeline gate promotes it to 'live'.
MODELS_PREFIX = os.environ.get("GCS_MODEL_PREFIX", f"gs://{BUCKET}/models/{ENV}")
PIPELINE_ROOT = os.environ.get("PIPELINE_ROOT", f"gs://{BUCKET}/pipeline-root")

# Promotion gate (candidate -> live), on the multi-seed grade-A lift mean (top 25%):
#   sanity: candidate >= min_abs; no-regression: candidate >= live - max_regression.
# Soft gate: failing keeps the live model and never fails the pipeline.
PROMOTION = {
    "metric": "lift_A",
    "min_abs": 1.0,
    "max_regression": 0.15,
}


def model_uri(segment: str, stage: str = "live") -> str:
    """Build the GCS URI of a segment artifact at a given stage.

    Args:
        segment: Segment name (e.g. ``"landing"`` or ``"main"``).
        stage: Lifecycle stage, ``"live"`` or ``"candidate"``.

    Returns:
        The full ``gs://`` URI of the segment's joblib at that stage.
    """
    return f"{MODELS_PREFIX}/{stage}/lead_scoring_{segment}.joblib"


# Score grades are percentile bands of the model's own (uncalibrated) score
# distribution: A = top 25%, B = 25–50%, C = bottom 50%. Cutoffs are fitted per
# model at train time (evaluate.grade_thresholds) and stored in the artifact.
GRADE_BANDS = [("A", 75), ("B", 50)]  # (grade, lower percentile); below B -> GRADE_FALLBACK
GRADE_FALLBACK = "C"


def grade_of(score, thresholds):
    """Map a raw score to its letter grade using fitted per-model thresholds.

    Args:
        score: The model's raw score for one lead.
        thresholds: Per-model score cutoffs, e.g. ``{"A": q75, "B": q50}``.

    Returns:
        The letter grade (``"A"``/``"B"``/``GRADE_FALLBACK``), or ``None`` when the
        artifact carries no thresholds (older model), so the caller degrades gracefully.
    """
    if not thresholds:
        return None
    for g, _ in GRADE_BANDS:
        if score >= thresholds[g]:
            return g
    return GRADE_FALLBACK


def route_segment(payload: dict) -> str:
    """Decide which segment model scores this lead, by ``platform``.

    ``platform == "landing"`` -> the landing (unbounce) model; anything else
    (``main_site``, missing, ...) -> the main (web) model. Matches how the BigQuery
    ``segmento`` column is built upstream.

    Args:
        payload: The incoming lead/form data.

    Returns:
        The segment name: ``"landing"`` or ``"main"``.
    """
    plat = str(payload.get("platform", "") or "").strip().lower()
    return "landing" if plat == "landing" else "main"
