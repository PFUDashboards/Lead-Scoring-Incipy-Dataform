"""Lead-scoring shared library — preprocessing, training and evaluation used by
both the Vertex training pipeline and the Cloud Run serving app.

Submodules are imported EXPLICITLY by callers (``from leadscoring import config`` /
``preprocess`` / ...). We deliberately do NOT eager-import them here: the pipeline
*submit* step (``compile_and_run``) only needs ``config`` (stdlib only) and runs in a
lightweight env (kfp + aiplatform, see ``requirements-pipeline.txt``) WITHOUT
numpy/pandas/scikit-learn/xgboost. Eager-importing ``data``/``evaluate``/``train`` here
would drag that whole stack in and break the submit job.
"""

__all__ = ["config", "data", "preprocess", "train", "evaluate"]
