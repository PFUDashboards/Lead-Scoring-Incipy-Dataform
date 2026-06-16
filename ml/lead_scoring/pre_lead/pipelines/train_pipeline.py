"""Vertex AI Pipeline (KFP v2) — train both segment models.

DAG, built once per segment so each emits its own metrics/HTML in the Vertex UI:

    ingest ─► [ prepare_segment ─► fit_preprocess ─► train_model ─► evaluate_model ─► package_artifact ]

All components run the custom ``training-base`` image (with ``leadscoring``
installed). ``fit_preprocess`` is a separate component from ``train_model``; the
fitted ``ColumnTransformer`` it outputs is the exact object reused at serve time.

Compile/submit with ``pipelines/compile_and_run.py``.

NOTE: do NOT add ``from __future__ import annotations`` here — KFP v2 component
introspection needs real (non-stringized) annotations.
"""
import os

from kfp import dsl
from kfp.dsl import (
    HTML,
    Artifact,
    ClassificationMetrics,
    Dataset,
    Input,
    Metrics,
    Model,
    Output,
)

from leadscoring import config as _config

# compile_and_run injects the built Artifact Registry path via TRAINING_IMAGE;
# the fallback is derived from config (no hardcoded project/region).
BASE_IMAGE = os.environ.get(
    "TRAINING_IMAGE",
    f"{_config.REGION}-docker.pkg.dev/{_config.PROJECT_ID}/{_config.AR_REPO}/training-base:latest",
)


@dsl.component(base_image=BASE_IMAGE)
def ingest(table: str, project: str, data: Output[Dataset]):
    """Load the BigQuery training table and write it to parquet.

    Args:
        table: Fully-qualified ``project.dataset.table`` to read.
        project: GCP project for the BigQuery client.
        data: Output dataset; the parquet is written to ``data.path`` and the row
            count / column list are recorded in its metadata.
    """
    from leadscoring import config
    from leadscoring import data as dataio

    config.PROJECT_ID = project
    df = dataio.load(table_ref=table)
    df.to_parquet(data.path)
    data.metadata["rows"] = len(df)
    data.metadata["columns"] = list(df.columns)


@dsl.component(base_image=BASE_IMAGE)
def prepare_segment(data: Input[Dataset], segment: str, seg_out: Output[Dataset]):
    """Subset the dataset to a single segment and write it to parquet.

    Args:
        data: Input dataset (the full training table from :func:`ingest`).
        segment: Segment name to keep.
        seg_out: Output dataset; the segment parquet is written to ``seg_out.path``.
    """
    import pandas as pd

    from leadscoring import data as dataio

    df = pd.read_parquet(data.path)
    seg = dataio.segment_frame(df, segment)
    seg.to_parquet(seg_out.path)
    seg_out.metadata["segment"] = segment
    seg_out.metadata["rows"] = len(seg)


@dsl.component(base_image=BASE_IMAGE)
def fit_preprocess(
    seg: Input[Dataset], override_json: str, preprocessor: Output[Artifact]
):
    """Fit the ColumnTransformer on 100% of the segment (the serving transformer).

    Args:
        seg: Input segment dataset.
        override_json: JSON-encoded explicit feature list (empty to use the dynamic
            schema).
        preprocessor: Output artifact; the fitted transformer + feature lists are
            dumped to ``preprocessor.path``.
    """
    import json

    import joblib
    import pandas as pd

    from leadscoring import preprocess

    df = pd.read_parquet(seg.path)
    override = json.loads(override_json) if override_json else None
    pre, feats, num, cat = preprocess.fit_preprocessor(df, override=override)
    joblib.dump(
        {"preprocessor": pre, "features": feats, "num": num, "cat": cat},
        preprocessor.path,
    )
    preprocessor.metadata.update({"n_features": len(feats), "num": num, "cat": cat})


@dsl.component(base_image=BASE_IMAGE)
def train_model(
    seg: Input[Dataset],
    preprocessor: Input[Artifact],
    n_iter: int,
    n_seeds: int,
    model: Output[Model],
):
    """Tune (leak-free) + multi-seed stability + refit deployable model on 100%.

    Uses the ``fit_preprocess`` preprocessor for the final fit. Metrics + params
    are stored alongside the model for the evaluate/package steps.

    Args:
        seg: Input segment dataset.
        preprocessor: The fitted transformer bundle from :func:`fit_preprocess`.
        n_iter: ``RandomizedSearchCV`` iterations for tuning.
        n_seeds: Seeds for the multi-seed stability evaluation.
        model: Output model; the native XGBoost json is written to ``model.path``
            and the full meta (params/stability/...) to ``model.path + ".meta.joblib"``.
    """
    import joblib
    import pandas as pd

    from leadscoring import evaluate, train

    df = pd.read_parquet(seg.path)
    bundle = joblib.load(preprocessor.path)
    feats = bundle["features"]

    tuned = train.tune_segment(df, override=feats, n_iter=n_iter)
    stability = evaluate.holdout_stability(df, tuned["params"], override=feats, n_seeds=n_seeds)

    xgb_model = train.fit_with_preprocessor(
        df, bundle["preprocessor"], bundle["num"], bundle["cat"],
        tuned["params"], stability["median_best_iter"],
    )
    xgb_model.save_model(model.path)  # native xgb json

    meta = {
        "params": tuned["params"],
        "n_estimators": stability["median_best_iter"],
        "test_metrics": tuned["test_metrics"],
        "stability": stability,
        "base_rate": float(df["y"].mean()),
        "n_train": int(len(df)),
        "features": feats,
        "num": bundle["num"],
        "cat": bundle["cat"],
    }
    model.metadata.update({k: meta[k] for k in ("n_estimators", "base_rate", "n_train")})
    joblib.dump(meta, model.path + ".meta.joblib")


@dsl.component(base_image=BASE_IMAGE)
def evaluate_model(
    seg: Input[Dataset],
    data: Input[Dataset],
    model: Input[Model],
    segment: str,
    daily_volume: int,
    metrics: Output[Metrics],
    cls_metrics: Output[ClassificationMetrics],
    report: Output[HTML],
):
    """Emit scalar Metrics + ROC curve + an HTML lift report to the Vertex UI.

    Args:
        seg: Input segment dataset.
        data: The full dataset (to scale ``daily_volume`` by the segment's row share).
        model: The trained model artifact (its ``.meta.joblib`` is read).
        segment: Segment name, shown in the report.
        daily_volume: Avg total leads/day across all segments.
        metrics: Output scalar metrics.
        cls_metrics: Output classification metrics (ROC curve).
        report: Output HTML report.
    """
    import joblib
    import pandas as pd

    from leadscoring import evaluate

    df = pd.read_parquet(seg.path)
    # Segment's slice of the global daily volume, by its share of the training rows.
    n_total = len(pd.read_parquet(data.path))
    seg_daily = daily_volume * (len(df) / max(n_total, 1))
    meta = joblib.load(model.path + ".meta.joblib")
    params, feats, stab = meta["params"], meta["features"], meta["stability"]

    metrics.log_metric("pr_auc", stab["pr_auc"]["mean"])
    metrics.log_metric("pr_auc_std", stab["pr_auc"]["std"])
    metrics.log_metric("roc_auc", stab["roc"]["mean"])
    metrics.log_metric("lift_A_top25pct", stab["lift_A"]["mean"])
    metrics.log_metric("lift_B_25_50pct", stab["lift_B"]["mean"])
    metrics.log_metric("lift_C_bottom50pct", stab["lift_C"]["mean"])
    metrics.log_metric("base_rate", meta["base_rate"])
    metrics.log_metric("n_train", meta["n_train"])

    # Decile lift + test metrics on a held-out split.
    lift_tab, base, (y_true, scores) = evaluate.lift_by_decile(df, params, override=feats)
    test = evaluate.test_block(y_true, scores)
    metrics.log_metric("test_pr_auc", test["pr_auc"])
    metrics.log_metric("test_precision_top10", test["precision"])
    metrics.log_metric("test_recall_top10", test["recall"])
    metrics.log_metric("seg_daily_volume", seg_daily)

    # A/B/C grade legend on the SAME held-out scores -> honest rates.
    grade_tab = evaluate.grade_table(y_true, scores, base, seg_daily)
    import math

    fpr, tpr, thr = evaluate.roc_points(y_true, scores)
    # Clamp non-finite thresholds (sklearn>=1.3 thr[0]=inf -> "Infinity", which
    # Vertex rejects). Repeated here since the component source is embedded in the
    # compiled pipeline, so the fix applies without rebuilding the image.
    thr = [t if math.isfinite(t) else 1.0 for t in thr]
    cls_metrics.log_roc_curve(fpr, tpr, thr)

    with open(report.path, "w") as f:
        f.write(evaluate.html_report(segment, lift_tab, base, stab, grade_tab=grade_tab))


@dsl.component(base_image=BASE_IMAGE)
def package_artifact(
    seg: Input[Dataset],
    preprocessor: Input[Artifact],
    model: Input[Model],
    segment: str,
    candidate_uri: str,
):
    """Bundle {preprocessor, model, features, metrics} to a joblib and upload it.

    Writes to the ``candidate`` stage only; ``validate_and_promote`` decides whether
    this becomes the ``live`` model that serving loads.

    Args:
        seg: Input segment dataset (used to fit the grade thresholds).
        preprocessor: The fitted transformer bundle from :func:`fit_preprocess`.
        model: The trained model artifact (its ``.meta.joblib`` is read).
        segment: Segment name.
        candidate_uri: ``gs://`` URI to upload the candidate joblib to.
    """
    import joblib
    import pandas as pd
    import xgboost as xgb

    from leadscoring import config, evaluate, preprocess

    bundle = joblib.load(preprocessor.path)
    meta = joblib.load(model.path + ".meta.joblib")
    clf = xgb.XGBClassifier()
    clf.load_model(model.path)

    # Grade cutoffs from the production model's own score distribution, so a live
    # score grades consistently with the deployed model.
    df = pd.read_parquet(seg.path)
    X = preprocess.transform(bundle["preprocessor"], df, bundle["num"], bundle["cat"])
    grade_thr = evaluate.grade_thresholds(clf.predict_proba(X)[:, 1])

    artifact = {
        "preprocessor": bundle["preprocessor"],
        "model": clf,
        "features": bundle["features"],
        "num": bundle["num"],
        "cat": bundle["cat"],
        "segmento": segment,
        "params": meta["params"],
        "n_estimators": meta["n_estimators"],
        "base_rate": meta["base_rate"],
        "n_train": meta["n_train"],
        "metrics": meta["stability"],
        "grade_thresholds": grade_thr,
        "schema_version": 2,
    }
    local = f"/tmp/lead_scoring_{segment}.joblib"
    joblib.dump(artifact, local)

    from google.cloud import storage

    assert candidate_uri.startswith("gs://"), candidate_uri
    bkt, _, blob = candidate_uri[len("gs://"):].partition("/")
    storage.Client(project=config.PROJECT_ID).bucket(bkt).blob(blob).upload_from_filename(local)
    print(f"uploaded candidate {candidate_uri}")


@dsl.component(base_image=BASE_IMAGE)
def validate_and_promote(
    model: Input[Model],
    segment: str,
    candidate_uri: str,
    live_uri: str,
    metric: str,
    min_abs: float,
    max_regression: float,
    decision: Output[Metrics],
    report: Output[HTML],
):
    """Promote the candidate to 'live' only if it doesn't regress vs current live.

    Soft gate: on failure it keeps the live model and records the reason, but never
    raises. Compares the multi-seed ``metric`` (e.g. ``lift_A``) mean:
      * sanity:        candidate >= min_abs
      * no-regression: candidate >= live - max_regression
    First run (no live model yet) bootstraps: promote if sanity passes.

    Args:
        model: The trained model artifact (its ``.meta.joblib`` holds the candidate
            metric).
        segment: Segment name.
        candidate_uri: ``gs://`` URI of the candidate joblib.
        live_uri: ``gs://`` URI of the live joblib (read for the current metric,
            overwritten on promotion).
        metric: Gate metric key (e.g. ``"lift_top"``).
        min_abs: Minimum absolute metric for the sanity check.
        max_regression: Allowed drop vs the live metric before it counts as a regression.
        decision: Output metrics (``promoted``, ``candidate_lift``, ``live_lift``).
        report: Output HTML decision report.
    """
    import tempfile

    import joblib
    from google.cloud import storage

    from leadscoring import config

    client = storage.Client(project=config.PROJECT_ID)

    def _read_metric(uri):
        """Return the gate metric mean from a joblib at `uri`, or None if absent.

        Also None if the artifact predates this metric key (e.g. an older model
        scored on a different metric) -> the gate bootstraps instead of crashing.
        """
        bkt, _, blob = uri[len("gs://"):].partition("/")
        b = client.bucket(bkt).blob(blob)
        if not b.exists():
            return None
        fd, local = tempfile.mkstemp(suffix=".joblib")
        import os as _os

        _os.close(fd)
        b.download_to_filename(local)
        art = joblib.load(local)
        m = art.get("metrics", {}).get(metric)
        return float(m["mean"]) if m else None

    meta = joblib.load(model.path + ".meta.joblib")
    cand = float(meta["stability"][metric]["mean"])
    live = _read_metric(live_uri)

    sane = cand >= min_abs
    if not sane:
        promote, reason = False, f"candidate {metric}={cand:.3f} < min {min_abs} (no better than random)"
    elif live is None:
        promote, reason = True, f"no live model yet — bootstrap promote (candidate {metric}={cand:.3f})"
    elif cand >= live - max_regression:
        promote, reason = True, f"candidate {metric}={cand:.3f} >= live {live:.3f} - {max_regression} (ok)"
    else:
        promote, reason = False, f"REGRESSION: candidate {metric}={cand:.3f} < live {live:.3f} - {max_regression}"

    if promote:
        # copy candidate -> live within GCS (no re-upload).
        cb, _, cblob = candidate_uri[len("gs://"):].partition("/")
        lb, _, lblob = live_uri[len("gs://"):].partition("/")
        src_bucket = client.bucket(cb)
        src_bucket.copy_blob(src_bucket.blob(cblob), client.bucket(lb), lblob)
        print(f"PROMOTED {segment}: {candidate_uri} -> {live_uri} ({reason})")
    else:
        print(f"NOT promoted {segment}: live model kept. {reason}")

    decision.log_metric("promoted", 1.0 if promote else 0.0)
    decision.log_metric("candidate_lift", cand)
    decision.log_metric("live_lift", live if live is not None else -1.0)

    color = "#d9ead3" if promote else "#f4cccc"
    verdict = "PROMOTED ✅" if promote else "NOT promoted — live kept ⛔"
    live_txt = f"{live:.3f}" if live is not None else "(none — first run)"
    with open(report.path, "w") as f:
        f.write(f"""
        <html><body style="font-family:system-ui,sans-serif">
        <h2>Promotion decision — segment: {segment}</h2>
        <div style="background:{color};padding:10px;border-radius:6px"><b>{verdict}</b></div>
        <ul>
          <li>gate metric: <b>{metric}</b> (multi-seed mean)</li>
          <li>candidate: <b>{cand:.3f}</b></li>
          <li>current live: <b>{live_txt}</b></li>
          <li>rule: candidate &ge; {min_abs} (sanity) and &ge; live - {max_regression}</li>
          <li>reason: {reason}</li>
        </ul>
        <p><small>SOFT gate: a failure keeps the live model and never fails the pipeline.</small></p>
        </body></html>
        """)


@dsl.pipeline(name="lead-scoring-train", description="Segmented lead-scoring train + package")
def lead_scoring_pipeline(
    table: str,
    project: str,
    models_prefix: str,
    n_iter: int = 60,
    n_seeds: int = 5,
    daily_volume: int = 250,
    gate_metric: str = "lift_A",
    gate_min_abs: float = 1.0,
    gate_max_regression: float = 0.15,
):
    """Define the segmented train + package + promote DAG.

    Args:
        table: Fully-qualified BigQuery training table.
        project: GCP project for the BigQuery/GCS clients.
        models_prefix: Base ``gs://`` prefix; ``candidate/`` and ``live/`` are appended.
        n_iter: ``RandomizedSearchCV`` iterations per segment.
        n_seeds: Seeds for the multi-seed stability evaluation.
        daily_volume: Avg total leads/day, split per segment in the report.
        gate_metric: Promotion gate metric key.
        gate_min_abs: Minimum absolute metric for the sanity check.
        gate_max_regression: Allowed drop vs the live metric.
    """
    from leadscoring import config as cfg

    SEGMENTS = cfg.SEGMENTS
    OVERRIDES = cfg.FEATURE_OVERRIDES

    raw = ingest(table=table, project=project)
    for segment in SEGMENTS:
        seg = prepare_segment(data=raw.outputs["data"], segment=segment)
        seg.set_display_name(f"prepare-{segment}")

        import json

        pre = fit_preprocess(
            seg=seg.outputs["seg_out"],
            override_json=json.dumps(OVERRIDES.get(segment, [])),
        )
        pre.set_display_name(f"preprocess-{segment}")

        trained = train_model(
            seg=seg.outputs["seg_out"],
            preprocessor=pre.outputs["preprocessor"],
            n_iter=n_iter,
            n_seeds=n_seeds,
        )
        trained.set_display_name(f"train-{segment}")

        ev = evaluate_model(
            seg=seg.outputs["seg_out"],
            data=raw.outputs["data"],
            model=trained.outputs["model"],
            segment=segment,
            daily_volume=daily_volume,
        )
        ev.set_display_name(f"evaluate-{segment}")

        candidate_uri = f"{models_prefix}/candidate/lead_scoring_{segment}.joblib"
        live_uri = f"{models_prefix}/live/lead_scoring_{segment}.joblib"

        pkg = package_artifact(
            seg=seg.outputs["seg_out"],
            preprocessor=pre.outputs["preprocessor"],
            model=trained.outputs["model"],
            segment=segment,
            candidate_uri=candidate_uri,
        )
        pkg.set_display_name(f"package-{segment}")

        promote = validate_and_promote(
            model=trained.outputs["model"],
            segment=segment,
            candidate_uri=candidate_uri,
            live_uri=live_uri,
            metric=gate_metric,
            min_abs=gate_min_abs,
            max_regression=gate_max_regression,
        )
        promote.after(pkg)  # candidate joblib must exist before promotion
        promote.set_display_name(f"validate-and-promote-{segment}")
