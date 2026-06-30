"""Training: leak-free tuning + production refit for one segment.

- Stratified 70/10/20 train/eval/test split (test untouched).
- Preprocessor fitted on TRAIN ONLY (no leakage).
- ``RandomizedSearchCV`` (PR-AUC) for hyper-params, early stopping on eval.
- Production refit on 100% of the segment, ``n_estimators`` = median
  ``best_iteration`` (avoids degenerate early-stop draws).
"""
from __future__ import annotations

import pandas as pd
import scipy.stats as st
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)

from . import config, preprocess

# Hyper-parameter search space.
PARAM_DIST = {
    "max_depth": st.randint(2, 8),
    "learning_rate": st.loguniform(1e-2, 3e-1),
    "subsample": st.uniform(0.6, 0.4),
    "colsample_bytree": st.uniform(0.6, 0.4),
    "min_child_weight": st.randint(1, 40),
    "gamma": st.uniform(0, 5),
    "reg_lambda": st.loguniform(0.1, 10),
    "reg_alpha": st.loguniform(1e-2, 10),
}
TUNED_KEYS = list(PARAM_DIST.keys())


def _xgb(spw: float, seed: int, n_estimators: int, early: bool, **params):
    kw = dict(
        n_estimators=n_estimators,
        tree_method="hist",
        eval_metric="aucpr",
        scale_pos_weight=spw,
        n_jobs=-1,
        random_state=seed,
        **params,
    )
    if early:
        kw["early_stopping_rounds"] = 50
    return xgb.XGBClassifier(**kw)


def tune_segment(df: pd.DataFrame, override=None, n_iter: int = 60, seed: int = 42) -> dict:
    """Search hyper-params on a leak-free split and report a held-out test score.

    Does not return a deployable model (call :func:`fit_production` for that). The
    preprocessor is fitted on the train split only (never on eval/test), so the
    reported test score carries no leakage.

    Args:
        df: The segment DataFrame to tune on.
        override: Optional explicit feature list (see ``preprocess.resolve_features``).
        n_iter: Number of ``RandomizedSearchCV`` sampling iterations.
        seed: Random seed for the splits, CV and estimators.

    Returns:
        A dict with ``params``, ``best_iteration``, ``features``, ``num``, ``cat``
        and ``test_metrics``.
    """
    feats = preprocess.resolve_features(df, override=override)
    num, cat = preprocess.split_types(df, feats)
    X = preprocess.prep_X(df, num, cat)
    y = df[config.TARGET].astype(int)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    X_ev, X_te, y_ev, y_te = train_test_split(
        X_tmp, y_tmp, test_size=2 / 3, stratify=y_tmp, random_state=seed
    )

    pre = preprocess.build_preprocessor(num, cat).fit(X_tr)
    A_tr, A_ev, A_te = pre.transform(X_tr), pre.transform(X_ev), pre.transform(X_te)
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    base = _xgb(spw, seed, n_estimators=400, early=False)
    search = RandomizedSearchCV(
        base,
        PARAM_DIST,
        n_iter=n_iter,
        scoring="average_precision",
        cv=StratifiedKFold(4, shuffle=True, random_state=seed),
        n_jobs=1,
        random_state=seed,
    )
    search.fit(A_tr, y_tr)

    final = _xgb(spw, seed, n_estimators=2000, early=True, **search.best_params_)
    final.fit(A_tr, y_tr, eval_set=[(A_ev, y_ev)], verbose=False)
    p = final.predict_proba(A_te)[:, 1]

    dec = pd.qcut(p, 10, labels=False, duplicates="drop")
    lift = pd.DataFrame({"y": y_te.values, "d": dec}).groupby("d")["y"].mean()
    metrics = {
        "cv_best_pr_auc": float(search.best_score_),
        "test_pr_auc": float(average_precision_score(y_te, p)),
        "test_roc_auc": float(roc_auc_score(y_te, p)),
        "test_lift_top": float(lift.iloc[-1] / y_te.mean()),
        "base_rate": float(y_te.mean()),
        "best_iteration": int(final.best_iteration or 1),
    }
    return {
        "params": {k: search.best_params_[k] for k in TUNED_KEYS},
        "best_iteration": metrics["best_iteration"],
        "features": feats,
        "num": num,
        "cat": cat,
        "test_metrics": metrics,
    }


def fit_with_preprocessor(df: pd.DataFrame, pre, num, cat, params: dict,
                          n_estimators: int, seed: int = 42):
    """Fit the deployable XGBoost on 100% of the segment.

    Uses an already-fitted preprocessor (from the ``fit_preprocess`` step).

    Args:
        df: The segment DataFrame to fit on.
        pre: A fitted ``ColumnTransformer``.
        num: Numeric feature names.
        cat: Categorical feature names.
        params: Tuned XGBoost hyper-params.
        n_estimators: Number of boosting rounds (floored at 10).
        seed: Random seed for the estimator.

    Returns:
        The fitted ``XGBClassifier``.
    """
    y = df[config.TARGET].astype(int)
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    model = _xgb(spw, seed, n_estimators=max(int(n_estimators), 10), early=False, **params)
    model.fit(preprocess.transform(pre, df, num, cat), y)
    return model


def fit_production(df: pd.DataFrame, params: dict, n_estimators: int, override=None,
                   seed: int = 42) -> dict:
    """Standalone refit: fit preprocessor + model on 100% of the segment.

    Args:
        df: The segment DataFrame to fit on.
        params: Tuned XGBoost hyper-params.
        n_estimators: Number of boosting rounds (floored at 10).
        override: Optional explicit feature list.
        seed: Random seed.

    Returns:
        A dict bundling ``preprocessor``, ``model``, ``features``, ``num``, ``cat``,
        ``params``, ``n_estimators``, ``base_rate`` and ``n_train``.
    """
    pre, feats, num, cat = preprocess.fit_preprocessor(df, override=override)
    model = fit_with_preprocessor(df, pre, num, cat, params, n_estimators, seed)
    y = df[config.TARGET].astype(int)
    return {
        "preprocessor": pre, "model": model, "features": feats, "num": num, "cat": cat,
        "params": params, "n_estimators": int(max(int(n_estimators), 10)),
        "base_rate": float(y.mean()), "n_train": int(len(y)),
    }
