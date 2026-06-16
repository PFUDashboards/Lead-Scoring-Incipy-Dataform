"""Unit tests for the client-facing capacity table + A/B/C/D grades.

GCP-free: pure numpy/pandas + the config grade mapper. No model training, no GCS.
"""

import numpy as np

from leadscoring import config, evaluate


def _sample(n=10_000, base=0.05, seed=0):
    """A monotone-ish signal: higher score -> higher conversion (so lift is real)."""
    rng = np.random.default_rng(seed)
    scores = rng.random(n)
    # conversion prob increases with score; mean ~= base
    p = np.clip(base * 2 * scores / scores.mean() * 0.5, 0, 1)
    y = rng.binomial(1, p)
    return y, scores


def test_capacity_table_identities():
    y, scores = _sample()
    base = y.mean()
    vol = 150
    cap = evaluate.capacity_table(y, scores, base, daily_volume=vol)

    # vs_azar is exactly tasa_exito / base.
    assert np.allclose(cap["vs_azar"], cap["tasa_exito"] / base)
    # Leads/dia is the percentile slice of the daily volume.
    assert np.allclose(cap["leads_dia"], cap["top_pct"] / 100 * vol)
    # Conversiones/dia = leads/dia * tasa.
    assert np.allclose(cap["conv_dia"], cap["leads_dia"] * cap["tasa_exito"])


def test_capacity_recall_is_monotonic():
    y, scores = _sample()
    cap = evaluate.capacity_table(y, scores, y.mean(), daily_volume=250)
    # Capturing a wider top% can only capture more (or equal) conversions.
    rec = cap["pct_capturadas"].to_numpy()
    assert np.all(np.diff(rec) >= -1e-9)
    assert (rec <= 1.0 + 1e-9).all()


def test_capacity_full_population_captures_everything():
    y, scores = _sample()
    cap = evaluate.capacity_table(y, scores, y.mean(), daily_volume=250, cuts=(50, 100))
    # Top 100% must capture all conversions and hit the base rate exactly.
    last = cap.iloc[-1]
    assert last["top_pct"] == 100
    assert np.isclose(last["pct_capturadas"], 1.0)
    assert np.isclose(last["tasa_exito"], y.mean())


def test_grade_thresholds_are_ordered():
    y, scores = _sample()
    thr = evaluate.grade_thresholds(scores)
    assert set(thr) == {"A", "B"}
    # A (top 25%) cutoff is a higher score than B (top 50%).
    assert thr["A"] >= thr["B"]


def test_grade_of_boundaries():
    thr = {"A": 0.75, "B": 0.5}
    assert config.grade_of(0.95, thr) == "A"
    assert config.grade_of(0.75, thr) == "A"  # inclusive lower edge
    assert config.grade_of(0.60, thr) == "B"
    assert config.grade_of(0.50, thr) == "B"  # inclusive lower edge
    assert config.grade_of(0.10, thr) == "C"  # below B -> fallback C


def test_grade_of_without_thresholds_is_none():
    # Old artifacts have no grade_thresholds -> serving must not crash.
    assert config.grade_of(0.5, None) is None
    assert config.grade_of(0.5, {}) is None


def test_grade_table_shape_and_lift():
    y, scores = _sample()
    base = y.mean()
    gt = evaluate.grade_table(y, scores, base, daily_volume=200)
    assert list(gt["grade"]) == ["A", "B", "C"]
    assert np.allclose(gt["vs_azar"], gt["tasa_exito"] / base, equal_nan=True)
    # The three bands' daily leads sum back to the segment volume.
    assert np.isclose(gt["leads_dia"].sum(), 200, atol=1.0)
