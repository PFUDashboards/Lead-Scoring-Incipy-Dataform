"""Compile the KFP pipeline and (optionally) submit it to Vertex AI Pipelines.

    python pipelines/compile_and_run.py --compile-only        # just produce pipeline.json
    python pipelines/compile_and_run.py                       # compile + submit to Vertex

The training image tag is injected via the TRAINING_IMAGE env var BEFORE importing
the pipeline module, because @dsl.component captures base_image at decoration time.
"""
from __future__ import annotations

import argparse
import os
import sys

# Make `import leadscoring` work from the repo without installing.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from leadscoring import config  # noqa: E402


def main() -> None:
    """Parse CLI args, compile the KFP pipeline and optionally submit it to Vertex."""
    p = argparse.ArgumentParser()
    p.add_argument("--compile-only", action="store_true")
    p.add_argument("--output", default="pipeline.json")
    p.add_argument("--n-iter", type=int, default=60)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--daily-volume", type=int, default=250,
                   help="avg total leads/day; scales the capacity table's per-day columns")
    p.add_argument("--wait", action="store_true",
                   help="block until the Vertex pipeline finishes; exit non-zero if it failed")
    p.add_argument("--training-image", default=None)
    p.add_argument("--service-account", default=os.environ.get("PIPELINE_SA"),
                   help="run the PipelineJob as this SA (else Vertex uses the default "
                        "Compute SA, which usually lacks the metadata-store/BQ/GCS access)")
    p.add_argument("--env", default=os.environ.get("ENV", config.ENV),
                   help="logical environment (dev|prod) — namespaces the GCS model paths")
    p.add_argument("--cache", dest="cache", action="store_true",
                   help="enable Vertex step caching (OFF by default). KFP keys on the table "
                        "reference + the ':latest' image tag, NOT the code/data contents, so a "
                        "cache hit can serve stale results after a rebuild. Only opt in locally "
                        "when iterating on changes that don't touch upstream components")
    p.add_argument("--dataform-repo", default=os.environ.get("DATAFORM_REPO"))
    p.add_argument("--dataform-workflow", default=os.environ.get("DATAFORM_WORKFLOW"))
    p.add_argument("--dataform-location", default=os.environ.get("DATAFORM_LOCATION"))
    p.add_argument("--dataform-project-number", default=os.environ.get("DATAFORM_PROJECT_NUMBER"))
    args = p.parse_args()

    env = args.env
    # Env-namespaced model prefix; the pipeline appends candidate/ and live/.
    models_prefix = f"gs://{config.BUCKET}/models/{env}"

    image = args.training_image or os.environ.get("TRAINING_IMAGE") or (
        f"{config.REGION}-docker.pkg.dev/{config.PROJECT_ID}/{config.AR_REPO}/training-base:latest"
    )
    os.environ["TRAINING_IMAGE"] = image  # @dsl.component captures base_image at import time

    import train_pipeline  # noqa: E402  (imported after env is set)
    from kfp import compiler

    compiler.Compiler().compile(train_pipeline.lead_scoring_pipeline, args.output)
    print(f"compiled -> {args.output}  (base image: {image})")

    if args.compile_only:
        return

    # Refresh the training table via Dataform BEFORE submitting, so we train on the
    # latest GA4 data. The invocation name becomes the provenance `data_version`. A
    # Dataform failure raises here -> we exit non-zero before spending on Vertex.
    import dataform_trigger  # noqa: E402  (same pipelines/ dir; lazy-imports the SDK)

    print("triggering Dataform refresh of the training table ...")
    data_version = dataform_trigger.run_workflow(
        project_number=args.dataform_project_number,
        location=args.dataform_location,
        repo=args.dataform_repo,
        workflow=args.dataform_workflow,
    )
    print(f"Dataform refresh SUCCEEDED; data_version={data_version}")

    from google.cloud import aiplatform

    aiplatform.init(project=config.PROJECT_ID, location=config.REGION,
                    staging_bucket=f"gs://{config.BUCKET}")
    job = aiplatform.PipelineJob(
        display_name=f"lead-scoring-train-{env}",
        template_path=args.output,
        pipeline_root=config.PIPELINE_ROOT,
        parameter_values={
            "table": config.BQ_TABLE_REF,
            "project": config.PROJECT_ID,
            "models_prefix": models_prefix,
            "data_version": data_version,
            "n_iter": args.n_iter,
            "n_seeds": args.n_seeds,
            "daily_volume": args.daily_volume,
            "gate_metric": config.PROMOTION["metric"],
            "gate_min_abs": config.PROMOTION["min_abs"],
            "gate_max_regression": config.PROMOTION["max_regression"],
        },
        enable_caching=args.cache,
    )
    job.submit(service_account=args.service_account)
    print(f"submitted Vertex pipeline job ({env}):", job.resource_name)
    if args.service_account:
        print(f"  running as service account: {args.service_account}")

    if args.wait:
        # wait() raises if the pipeline ends in error -> non-zero exit -> CI goes red.
        print("waiting for the pipeline to finish (this blocks until Vertex is done)...")
        job.wait()
        print(f"pipeline SUCCEEDED ({env}):", job.resource_name)


if __name__ == "__main__":
    main()
