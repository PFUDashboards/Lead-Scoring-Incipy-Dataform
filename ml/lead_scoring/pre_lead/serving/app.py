"""Real-time lead-scoring API (Cloud Run).

Loads the per-segment ``live`` artifacts from GCS at startup, routes each lead to
its segment model, applies the same ``leadscoring.preprocess`` + saved
``ColumnTransformer`` (no train/serve skew), and returns a score in [0, 1].

The live model is hot-swapped without a redeploy: requests trigger a throttled
GCS re-check (MODEL_RELOAD_CHECK_SECONDS); POST /reload forces it immediately.

Env:
  GCS_MODEL_PREFIX            gs://<bucket>/models (where the pipeline wrote the joblibs)
  MODEL_RELOAD_CHECK_SECONDS  min seconds between live-model re-checks (default 300; 0 disables)
  PORT                        provided by Cloud Run (default 8080)
"""
from __future__ import annotations

import os
import tempfile
import threading
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from leadscoring import config, preprocess

app = FastAPI(title="OBS lead scoring", version="1.0")

# In-memory artifacts + the GCS generation each was loaded from, so we can skip
# re-downloading an unchanged model.
MODELS: dict[str, dict] = {}
_GENERATIONS: dict[str, int] = {}

# Request-driven self-refresh (not a background timer, so it fires under Cloud
# Run's idle-CPU throttling): re-check GCS at most once per CHECK_INTERVAL.
CHECK_INTERVAL = int(os.environ.get("MODEL_RELOAD_CHECK_SECONDS", "300"))
_reload_lock = threading.Lock()
_last_check = 0.0


def _live_blob(uri: str):
    """Fetch the GCS blob (with metadata) behind a live-model URI.

    Args:
        uri: A ``gs://bucket/path`` URI.

    Returns:
        The ``storage.Blob`` for that object, or ``None`` if it doesn't exist.
    """
    from google.cloud import storage

    bkt, _, name = uri[len("gs://"):].partition("/")
    return storage.Client(project=config.PROJECT_ID).bucket(bkt).get_blob(name)


def _load_segment(segment: str, *, force: bool) -> bool:
    """(Re)load one segment's live artifact if its GCS generation changed.

    Always the promoted 'live' artifact, never 'candidate'.

    Args:
        segment: Segment name to load.
        force: Reload even if the GCS generation is unchanged.

    Returns:
        ``True`` if a (new) model was loaded, else ``False``.
    """
    uri = config.model_uri(segment, stage="live")
    try:
        if not uri.startswith("gs://"):  # local path (tests / dev)
            if not force and segment in MODELS:
                return False
            MODELS[segment] = joblib.load(uri)
            return True

        blob = _live_blob(uri)
        if blob is None:
            if force:
                print(f"WARNING: {segment} model not found at {uri}")
            return False
        if not force and _GENERATIONS.get(segment) == blob.generation:
            return False  # already serving this exact object

        fd, local = tempfile.mkstemp(suffix=".joblib")
        os.close(fd)
        blob.download_to_filename(local)
        MODELS[segment] = joblib.load(local)
        _GENERATIONS[segment] = blob.generation
        os.remove(local)
        print(f"loaded {segment} model from {uri} (generation {blob.generation})")
        return True
    except Exception as e:  # one bad segment must not take down the others
        print(f"WARNING: could not load {segment} model from {uri}: {e}")
        return False


def reload_models(*, force: bool) -> list[str]:
    """Check every segment and (re)load those whose model changed.

    Args:
        force: Reload every segment even if its GCS generation is unchanged.

    Returns:
        The list of segment names that were (re)loaded.
    """
    return [s for s in config.SEGMENTS if _load_segment(s, force=force)]


def maybe_reload() -> None:
    """Refresh live models on a throttle (request-driven).

    At most one GCS check per ``CHECK_INTERVAL``, hot-swapping any segment whose
    live model changed. No-op when ``CHECK_INTERVAL <= 0`` or another check is
    already in flight.
    """
    global _last_check
    if CHECK_INTERVAL <= 0:
        return
    now = time.monotonic()
    if now - _last_check < CHECK_INTERVAL:
        return
    if not _reload_lock.acquire(blocking=False):
        return  # another request is already checking
    try:
        _last_check = now
        changed = reload_models(force=False)
        if changed:
            print(f"auto-reload: refreshed {changed}")
    finally:
        _reload_lock.release()


@app.on_event("startup")
def _startup() -> None:
    reload_models(force=True)


class ScoreRequest(BaseModel):
    # Free-form payload; only the model's features are used.
    model_config = {"extra": "allow"}


@app.get("/")
def root():
    """Report which segment models are loaded and their headline info.

    Returns:
        A dict with the service name and, per loaded segment, its features,
        base rate and metrics.
    """
    return {
        "service": "lead-scoring",
        "segments_loaded": list(MODELS.keys()),
        "models": {
            s: {
                "features": m["features"],
                "base_rate": m.get("base_rate"),
                "metrics": m.get("metrics"),
            }
            for s, m in MODELS.items()
        },
    }


@app.get("/health")
def health():
    """Liveness/readiness probe (also triggers a throttled model refresh).

    Returns:
        ``{"status": "ok", "segments": [...]}`` when at least one model is loaded.

    Raises:
        HTTPException: 503 if no models are loaded.
    """
    maybe_reload()
    if not MODELS:
        raise HTTPException(503, "no models loaded")
    return {"status": "ok", "segments": list(MODELS.keys())}


@app.post("/reload")
def reload_endpoint():
    """Force an immediate reload of all live models (e.g. right after a retrain).

    Returns:
        A dict with the ``reloaded`` segments and all currently loaded ``segments``.
    """
    return {"reloaded": reload_models(force=True), "segments": list(MODELS.keys())}


@app.post("/score")
def score(payload: dict):
    """Score one lead and return its score, grade and routing info.

    Routes the payload to its segment model, derives features the same way as
    training, and applies the saved preprocessor + model.

    Args:
        payload: Free-form lead/form data; only the model's features are used.

    Returns:
        A dict with ``segmento``, ``score``, ``grade``, ``base_rate``,
        ``lift_vs_base``, ``features_used`` and ``schema_version``.

    Raises:
        HTTPException: 503 if no models are loaded.
    """
    maybe_reload()
    if not MODELS:
        raise HTTPException(503, "no models loaded")
    segment = config.route_segment(payload)
    art = MODELS.get(segment) or next(iter(MODELS.values()))

    row = preprocess.derive_columns(pd.DataFrame([payload]))  # same derive step as training
    X = preprocess.transform(art["preprocessor"], row, art["num"], art["cat"])
    proba = float(art["model"].predict_proba(X)[0, 1])

    return {
        "segmento": segment,
        "score": proba,
        "grade": config.grade_of(proba, art.get("grade_thresholds")),
        "base_rate": art.get("base_rate"),
        "lift_vs_base": (proba / art["base_rate"]) if art.get("base_rate") else None,
        "features_used": art["features"],
        "schema_version": art.get("schema_version"),
    }
