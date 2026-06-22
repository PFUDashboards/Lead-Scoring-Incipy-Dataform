"""Unit tests for the schema/routing/gate contract in leadscoring.config.

GCP-free: these exercise pure functions and constants only, so they run in CI
without credentials.
"""

from leadscoring import config


def test_model_uri_paths():
    # The candidate/live stage and segment name must land in the exact GCS layout
    # serving and the promotion gate both depend on.
    assert config.model_uri("landing", "candidate").endswith(
        "/candidate/lead_scoring_landing.joblib"
    )
    assert config.model_uri("main", "live").endswith("/live/lead_scoring_main.joblib")
    # Defaults to the live stage (what serving loads).
    assert config.model_uri("main") == config.model_uri("main", "live")


def test_bq_source_contract():
    # The training table is the Dataform output; lock the name + the raw-target map.
    assert config.BQ_TABLE == "model_train_GTM"
    assert config.BQ_TABLE_REF.endswith("BQ_PFU_INCIPY.model_train_GTM")
    assert config.RAW_TARGET == "apd_es_matricula"
    # ld_mcs_id (lead id) must be excluded from features (and is the data_hash key).
    assert "ld_mcs_id" in config.ID_COLS


def test_route_segment_landing_platform():
    assert config.route_segment({"platform": "landing"}) == "landing"
    assert config.route_segment({"platform": "  LANDING "}) == "landing"  # normalized


def test_route_segment_main_platform():
    assert config.route_segment({"platform": "main_site"}) == "main"
    assert config.route_segment({"platform": "anything_else"}) == "main"


def test_route_segment_ignores_form_name_and_segmento():
    # Only `platform` decides now; legacy fields must not influence routing.
    assert config.route_segment({"form_name": "unbounce_x"}) == "main"
    assert config.route_segment({"segmento": "landing"}) == "main"


def test_route_segment_missing_or_null_platform_defaults_to_main():
    assert config.route_segment({}) == "main"
    assert config.route_segment({"platform": None}) == "main"
    assert config.route_segment({"platform": "  "}) == "main"


def test_grade_letters_per_segment():
    # Display letters differ by segment: main A/B/C, landing D/E/F (same percentile
    # bands; landing is relabelled because it converts less in absolute terms).
    thr = {"A": 0.6, "B": 0.4}
    assert [config.grade_of(s, thr, "main") for s in (0.7, 0.5, 0.2)] == ["A", "B", "C"]
    assert [config.grade_of(s, thr, "landing") for s in (0.7, 0.5, 0.2)] == ["D", "E", "F"]
    # Unknown/absent segment falls back to the canonical letters.
    assert config.grade_of(0.7, thr, None) == "A"
    # No thresholds (older artifact) -> None, caller degrades gracefully.
    assert config.grade_of(0.7, None, "landing") is None


def test_promotion_gate_contract():
    # The soft promotion gate's thresholds are a non-negotiable contract; lock them in.
    assert config.PROMOTION["metric"] == "lift_A"
    assert config.PROMOTION["min_abs"] == 1.0
    assert config.PROMOTION["max_regression"] == 0.15
