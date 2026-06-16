"""Multi-seed evaluation — single-split lift is unreliable, so report mean +/- std.

Also produces the artifacts the KFP ``evaluate`` component renders in the Vertex
Pipelines UI (ROC points + an HTML lift-by-decile report).
"""
from __future__ import annotations

import base64
import io

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from . import config, preprocess, train


def holdout_stability(df: pd.DataFrame, params: dict, override=None, n_seeds: int = 5) -> dict:
    """Refit on N seeds and report robust metrics.

    Args:
        df: The segment DataFrame.
        params: Tuned XGBoost hyper-params.
        override: Optional explicit feature list.
        n_seeds: Number of seeded train/eval/test splits to average over.

    Returns:
        A dict with mean/std of ``pr_auc``, ``roc`` and the per-grade-band lift
        (``lift_A``, ``lift_B``, ``lift_C`` — same 0-25/25-50/50-100% bands as the
        A/B/C grades), plus ``best_iters``, ``median_best_iter`` and ``n_seeds``.
    """
    feats = preprocess.resolve_features(df, override=override)
    num, cat = preprocess.split_types(df, feats)
    X = preprocess.prep_X(df, num, cat)
    y = df[config.TARGET].astype(int)
    # Rank bands, same as the A/B/C grades (config.GRADE_BANDS): A=top 0-25%, B=25-50%,
    # C=50-100% (non-cumulative). Per-band lift by RANK (sort by score, slice by position)
    # — robust to ties, unlike qcut deciles which collapse when many scores are equal.
    grades = [g for g, _ in config.GRADE_BANDS] + [config.GRADE_FALLBACK]
    uppers = [100 - q for _, q in config.GRADE_BANDS] + [100]
    rows, best_iters = [], []
    for s in range(n_seeds):
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=s
        )
        X_ev, X_te, y_ev, y_te = train_test_split(
            X_tmp, y_tmp, test_size=2 / 3, stratify=y_tmp, random_state=s
        )
        pre = preprocess.build_preprocessor(num, cat).fit(X_tr)
        A_tr, A_ev, A_te = pre.transform(X_tr), pre.transform(X_ev), pre.transform(X_te)
        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        m = train._xgb(spw, s, n_estimators=2000, early=True, **params)
        m.fit(A_tr, y_tr, eval_set=[(A_ev, y_ev)], verbose=False)
        p = m.predict_proba(A_te)[:, 1]
        base = y_te.mean()
        ys = y_te.values[np.argsort(-p, kind="stable")]
        n = len(ys)
        band, prev = [], 0
        for up in uppers:
            lo, hi = int(round(prev / 100 * n)), int(round(up / 100 * n))
            seg = ys[lo:hi]
            band.append(float(seg.mean() / base) if (len(seg) and base) else float("nan"))
            prev = up
        rows.append([average_precision_score(y_te, p), roc_auc_score(y_te, p), *band])
        best_iters.append(int(m.best_iteration or 1))
    r = np.array(rows)
    labels = ["pr_auc", "roc"] + [f"lift_{g}" for g in grades]
    summary = {
        lab: {"mean": float(r[:, i].mean()), "std": float(r[:, i].std())}
        for i, lab in enumerate(labels)
    }
    summary["best_iters"] = best_iters
    summary["median_best_iter"] = int(np.median(best_iters))
    summary["n_seeds"] = n_seeds
    return summary


def lift_by_decile(df: pd.DataFrame, params: dict, override=None, seed: int = 42) -> pd.DataFrame:
    """Build a decile lift table on one held-out split (for display).

    Args:
        df: The segment DataFrame.
        params: Tuned XGBoost hyper-params.
        override: Optional explicit feature list.
        seed: Random seed for the split and estimator.

    Returns:
        A ``(lift_table, base_rate, (y_true, scores))`` tuple, where ``lift_table``
        has columns ``decile``/``conv``/``n``/``lift`` sorted best-first.
    """
    feats = preprocess.resolve_features(df, override=override)
    num, cat = preprocess.split_types(df, feats)
    X = preprocess.prep_X(df, num, cat)
    y = df[config.TARGET].astype(int)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    X_ev, X_te, y_ev, y_te = train_test_split(X_tmp, y_tmp, test_size=2 / 3, stratify=y_tmp, random_state=seed)
    pre = preprocess.build_preprocessor(num, cat).fit(X_tr)
    m = train._xgb((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), seed, 2000, True, **params)
    m.fit(pre.transform(X_tr), y_tr, eval_set=[(pre.transform(X_ev), y_ev)], verbose=False)
    p = m.predict_proba(pre.transform(X_te))[:, 1]
    base = y_te.mean()
    d = pd.DataFrame({"y": y_te.values, "decile": pd.qcut(p, 10, labels=False, duplicates="drop")})
    tab = d.groupby("decile")["y"].agg(["mean", "size"]).rename(columns={"mean": "conv", "size": "n"})
    tab["lift"] = tab["conv"] / base
    tab = tab.sort_index(ascending=False).reset_index()
    tab["decile"] = tab["decile"].astype(int) + 1
    return tab, float(base), (y_te.values, p)


def test_block(y_true, scores, frac: float = 0.10) -> dict:
    """Held-out test metrics: PR-AUC + confusion matrix at the operating point.

    The confusion matrix is at the top-``frac`` cutoff, not 0.5 (meaningless under
    scale_pos_weight).

    Args:
        y_true: Ground-truth binary labels.
        scores: Model scores for the same rows.
        frac: Top fraction flagged positive for the confusion matrix (e.g. 0.10).

    Returns:
        A dict with ``pr_auc``, ``roc_auc``, ``frac``, ``threshold``, ``n_test``,
        the confusion matrix ``cm`` and ``precision``/``recall`` at the cutoff.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    thr = float(np.quantile(scores, 1 - frac))
    y_pred = (scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "frac": frac,
        "threshold": thr,
        "n_test": int(len(y_true)),
        "cm": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
    }


def capacity_table(
    y_true,
    scores,
    base: float,
    daily_volume: float,
    cuts=(2, 5, 10, 15, 20, 30, 50, 60, 70, 80, 90),
) -> pd.DataFrame:
    """Build the cumulative-gains ("capacity") table — the client-facing view.

    For each ``Top P%`` of leads (ranked by score), reports leads/day, conversion
    rate, share of conversions captured, and lift vs random. Rank-based (first
    ``k = round(P% · n)``) so score ties don't distort the cut.

    Args:
        y_true: Ground-truth binary labels.
        scores: Model scores for the same rows.
        base: Base conversion rate (for the lift column).
        daily_volume: Avg total leads/day; scales the per-day columns only.
        cuts: Top-percentage cut points to tabulate.

    Returns:
        A DataFrame with one row per cut (``top_pct``, ``leads_dia``, ``conv_dia``,
        ``tasa_exito``, ``pct_capturadas``, ``vs_azar``).
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    n = len(y)
    order = np.argsort(-s, kind="stable")  # highest score first
    y_sorted = y[order]
    total_conv = max(int(y.sum()), 1)
    rows = []
    for p in cuts:
        k = max(int(round(p / 100 * n)), 1)
        top = y_sorted[:k]
        tasa = float(top.mean())
        leads_dia = p / 100 * daily_volume
        rows.append(
            {
                "top_pct": p,
                "leads_dia": leads_dia,
                "conv_dia": leads_dia * tasa,
                "tasa_exito": tasa,
                "pct_capturadas": float(top.sum()) / total_conv,
                "vs_azar": (tasa / base) if base else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def grade_thresholds(scores) -> dict:
    """Fit the score cutoffs from a score distribution (``config.GRADE_BANDS``).

    Stored in the artifact so serving grades a live score the same way.

    Args:
        scores: The model's score distribution to derive cutoffs from.

    Returns:
        The score at each band's lower percentile, e.g. ``{"A": q75, "B": q50}``.
    """
    s = np.asarray(scores, dtype=float)
    return {g: float(np.quantile(s, q / 100)) for g, q in config.GRADE_BANDS}


def grade_table(y_true, scores, base: float, daily_volume: float) -> pd.DataFrame:
    """Build the per-grade legend (non-cumulative).

    Reports conversion rate and lift vs random for each band, so the grade returned
    by ``/score`` is readable as a number. Bands come from ``config.GRADE_BANDS``,
    sliced on the rank-sorted leads.

    Args:
        y_true: Ground-truth binary labels.
        scores: Model scores for the same rows.
        base: Base conversion rate (for the lift column).
        daily_volume: Avg total leads/day; scales the ``leads_dia`` column.

    Returns:
        A DataFrame with one row per grade (``grade``, ``banda``, ``leads_dia``,
        ``tasa_exito``, ``vs_azar``).
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    n = len(y)
    y_sorted = y[np.argsort(-s, kind="stable")]
    # Cumulative upper edge of each band as a percentile-from-top (e.g. 75 -> 25%).
    uppers = [100 - q for _, q in config.GRADE_BANDS] + [100]
    grades = [g for g, _ in config.GRADE_BANDS] + [config.GRADE_FALLBACK]
    rows = []
    prev = 0
    for g, up in zip(grades, uppers):
        lo, hi = int(round(prev / 100 * n)), int(round(up / 100 * n))
        seg = y_sorted[lo:hi]
        tasa = float(seg.mean()) if len(seg) else float("nan")
        rows.append(
            {
                "grade": g,
                "banda": f"{prev}–{up}%",
                "leads_dia": (hi - lo) / max(n, 1) * daily_volume,
                "tasa_exito": tasa,
                "vs_azar": (tasa / base) if base else float("nan"),
            }
        )
        prev = up
    return pd.DataFrame(rows)


def roc_points(y_true, scores, n: int = 200):
    """Down-sample ROC points for KFP ClassificationMetrics.

    sklearn>=1.3 sets ``thresholds[0] = np.inf``, which serializes to ``Infinity``
    and Vertex's metadata store rejects. Clamp non-finite thresholds to 1.0.

    Args:
        y_true: Ground-truth binary labels.
        scores: Model scores for the same rows.
        n: Max number of points to keep (linearly down-sampled).

    Returns:
        A ``(fpr, tpr, thresholds)`` tuple of plain python lists.
    """
    fpr, tpr, thr = roc_curve(y_true, scores)
    thr = np.where(np.isfinite(thr), thr, 1.0)
    if len(fpr) > n:
        idx = np.linspace(0, len(fpr) - 1, n).astype(int)
        fpr, tpr, thr = fpr[idx], tpr[idx], thr[idx]
    return fpr.tolist(), tpr.tolist(), thr.tolist()


_GRADE_COLORS = {"A": "#1e8e5a", "B": "#2f6fed", "C": "#9aa3ad"}

_REPORT_CSS = """
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         margin: 0; background: #f4f6fb; color: #1d2330; }
  .wrap { max-width: 760px; margin: 0 auto; padding: 28px 22px 40px; }
  .head { border-left: 6px solid #2f6fed; padding: 4px 0 4px 16px; margin-bottom: 6px; }
  .head h1 { font-size: 22px; margin: 0; letter-spacing: .2px; }
  .head .seg { color: #2f6fed; }
  .head p { margin: 6px 0 0; color: #5b6472; font-size: 13px; }
  .kpis { display: flex; flex-wrap: wrap; gap: 12px; margin: 20px 0 8px; }
  .kpi { flex: 1 1 150px; background: #fff; border: 1px solid #e7eaf1; border-radius: 12px;
         padding: 14px 16px; box-shadow: 0 1px 2px rgba(20,30,60,.04); }
  .kpi .lbl { font-size: 11px; text-transform: uppercase; letter-spacing: .6px; color: #7a8494; }
  .kpi .val { font-size: 30px; font-weight: 700; line-height: 1.1; margin-top: 4px; }
  .kpi .sub { font-size: 12px; color: #8a93a3; margin-top: 2px; }
  .kpi.hero { background: linear-gradient(135deg,#1e8e5a,#23a86b); border: none; color: #fff; }
  .kpi.hero .lbl, .kpi.hero .sub { color: rgba(255,255,255,.85); }
  .pitch { background: #eef3ff; border: 1px solid #d8e3ff; border-radius: 12px;
           padding: 14px 18px; margin: 14px 0 4px; font-size: 15px; line-height: 1.5; }
  .pitch b { color: #1e8e5a; }
  h2.sec { font-size: 15px; margin: 28px 0 10px; color: #2a3140; }
  table { width: 100%; border-collapse: separate; border-spacing: 0; background: #fff;
          border: 1px solid #e7eaf1; border-radius: 12px; overflow: hidden; font-size: 14px; }
  thead th { background: #2a3140; color: #fff; font-weight: 600; padding: 11px 14px;
             text-align: right; font-size: 12px; letter-spacing: .3px; }
  thead th:first-child, thead th:nth-child(2) { text-align: left; }
  tbody td { padding: 11px 14px; text-align: right; border-top: 1px solid #eef1f6; }
  tbody td:first-child, tbody td:nth-child(2) { text-align: left; }
  .badge { display: inline-block; width: 26px; height: 26px; line-height: 26px; text-align: center;
           border-radius: 50%; color: #fff; font-weight: 700; font-size: 13px; }
  .bar { position: relative; background: #eef1f6; border-radius: 6px; height: 9px; width: 110px;
         display: inline-block; vertical-align: middle; overflow: hidden; }
  .bar > span { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 6px; }
  .liftpill { display: inline-block; padding: 2px 9px; border-radius: 999px; font-weight: 700;
              font-size: 13px; background: #eaf6ef; color: #1e8e5a; }
  .liftpill.low { background: #f0f2f5; color: #7a8494; }
  .chart { margin: 18px 0 0; text-align: center; }
  .chart img { max-width: 100%; border: 1px solid #e7eaf1; border-radius: 12px; background:#fff; }
  .foot { margin-top: 22px; font-size: 12px; color: #8a93a3; line-height: 1.5; }
"""


def _grades_section(grade_tab: pd.DataFrame, base: float) -> str:
    """Render the A/B/C grade legend as a styled client-facing HTML table.

    Args:
        grade_tab: The table from :func:`grade_table`.
        base: Base conversion rate (drives the relative bar widths).

    Returns:
        An HTML fragment (heading + table) for the grade legend.
    """
    max_tasa = max(float(grade_tab["tasa_exito"].max()), 1e-9)
    rows = []
    for r in grade_tab.itertuples():
        color = _GRADE_COLORS.get(r.grade, "#2f6fed")
        width = max(float(r.tasa_exito) / max_tasa * 100, 3)
        pill = "liftpill" if r.vs_azar >= 1.0 else "liftpill low"
        rows.append(
            f"<tr>"
            f"<td><span class='badge' style='background:{color}'>{r.grade}</span></td>"
            f"<td>{r.banda}</td>"
            f"<td>~{r.leads_dia:.0f}</td>"
            f"<td>{r.tasa_exito*100:.1f}%"
            f" <span class='bar'><span style='width:{width:.0f}%;background:{color}'></span></span></td>"
            f"<td><span class='{pill}'>{r.vs_azar:.1f}x</span></td>"
            f"</tr>"
        )
    return f"""
    <h2 class="sec">Grados A / B / C &mdash; prioridad de llamada</h2>
    <table>
      <thead><tr>
        <th>Grado</th><th>Posici&oacute;n en el ranking</th><th>Leads/d&iacute;a</th>
        <th>Tasa de conversi&oacute;n</th><th>vs. media</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


def html_report(segment: str, lift_tab: pd.DataFrame, base: float, stability: dict,
                test: dict | None = None, capacity: pd.DataFrame | None = None,
                grade_tab: pd.DataFrame | None = None) -> str:
    """Render the self-contained HTML evaluation report.

    Layout: a KPI card row (base rate, grade-A lift, PR-AUC, ROC-AUC), a plain-language
    pitch line, the A/B/C grade legend table, and the lift-by-grade chart.

    Args:
        segment: Segment name, shown in the title.
        lift_tab: The table from :func:`lift_by_decile`.
        base: Base conversion rate.
        stability: The summary from :func:`holdout_stability`.
        test: Accepted for back-compat; no longer rendered.
        capacity: Accepted for back-compat; no longer rendered.
        grade_tab: Optional table from :func:`grade_table`; renders the grade legend.

    Returns:
        A complete HTML document as a string.
    """
    chart = ""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.2, 3))
        if grade_tab is not None:
            xs = grade_tab["grade"].astype(str)
            ys = grade_tab["vs_azar"].astype(float)
            bars = ax.bar(xs, ys, width=0.62,
                          color=[_GRADE_COLORS.get(g, "#2f6fed") for g in xs])
            for b, v in zip(bars, ys):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}x",
                        ha="center", va="bottom", fontsize=10, fontweight="bold", color="#2a3140")
            ax.set_xlabel("grado")
            ax.set_title(f"{segment} — lift por grado")
            ax.set_ylim(0, max(float(ys.max()) * 1.25, 1.3))
        else:
            ax.bar(lift_tab["decile"].astype(str), lift_tab["lift"], color="#2f6fed")
            ax.set_xlabel("decil (10 = score más alto)")
            ax.set_title(f"{segment} — lift por decil")
        ax.axhline(1.0, color="#9aa3ad", ls="--", lw=1)
        ax.set_ylabel("veces mejor que la media")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode()
        chart = f'<div class="chart"><img src="data:image/png;base64,{b64}"/></div>'
    except Exception as e:  # matplotlib optional
        chart = f"<p><i>(gráfico no disponible: {e})</i></p>"

    n_seeds = stability.get("n_seeds")
    pr = stability["pr_auc"]
    roc = stability["roc"]
    lift_a = stability.get("lift_A", {"mean": float("nan"), "std": 0.0})

    # Grade-A conversion rate for the pitch line (single-split legend, if present).
    tasa_a = None
    if grade_tab is not None and len(grade_tab):
        tasa_a = float(grade_tab.iloc[0]["tasa_exito"])

    if tasa_a is not None and base:
        pitch = (
            f'<div class="pitch">Llamando primero al <b>25% mejor</b> (grado A), el equipo '
            f'contacta leads que convierten <b>{lift_a["mean"]:.1f}x</b> más que la media: '
            f'<b>{tasa_a*100:.1f}%</b> frente al {base*100:.1f}% de tasa base.</div>'
        )
    else:
        pitch = ""

    kpis = f"""
    <div class="kpis">
      <div class="kpi hero">
        <div class="lbl">Lift grado A</div>
        <div class="val">{lift_a['mean']:.1f}x</div>
        <div class="sub">top 25% vs. media &middot; &plusmn;{lift_a['std']:.1f}</div>
      </div>
      <div class="kpi">
        <div class="lbl">Tasa base</div>
        <div class="val">{base*100:.1f}%</div>
        <div class="sub">conversión media</div>
      </div>
      <div class="kpi">
        <div class="lbl">PR-AUC</div>
        <div class="val">{pr['mean']:.3f}</div>
        <div class="sub">&plusmn;{pr['std']:.3f} &middot; {n_seeds} semillas</div>
      </div>
      <div class="kpi">
        <div class="lbl">ROC-AUC</div>
        <div class="val">{roc['mean']:.3f}</div>
        <div class="sub">&plusmn;{roc['std']:.3f} &middot; {n_seeds} semillas</div>
      </div>
    </div>
    """

    grades_section = _grades_section(grade_tab, base) if grade_tab is not None else ""

    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><style>{_REPORT_CSS}</style></head>
<body><div class="wrap">
  <div class="head">
    <h1>Lead Scoring &middot; segmento <span class="seg">{segment}</span></h1>
    <p>Prioriza qué leads llamar primero. El score ordena por probabilidad de convertir
       (ranking, no probabilidad calibrada).</p>
  </div>
  {kpis}
  {pitch}
  {grades_section}
  <h2 class="sec">Lift por grado</h2>
  {chart}
  <div class="foot">
    Métricas robustas sobre {n_seeds} semillas (media &plusmn; desviación). &ldquo;Lift&rdquo; =
    cuántas veces más convierte ese grupo frente a la conversión media. PR-AUC y ROC-AUC miden
    la capacidad de ordenar (1.0 = perfecto, 0.5 = azar).
  </div>
</div></body></html>"""
