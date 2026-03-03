"""
SHAP explanation stability and determinism tests.

Verifies that:
- Same input always produces the same top-5 SHAP features (deterministic)
- SHAP values are returned in descending absolute importance order
- All expected fields are present in each SHAP entry
- SHAP values are finite floats
"""

import os
import sys
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from tests.conftest import DETERMINISTIC_VARIANT


# ─── Determinism ─────────────────────────────────────────────────────────────

def test_shap_deterministic_top5(client):
    """Same variant produces identical top-5 SHAP features across 3 calls."""
    results = []
    for _ in range(3):
        resp = client.post("/predict", json=DETERMINISTIC_VARIANT)
        assert resp.status_code == 200, f"Predict failed: {resp.status_code} — {resp.text}"
        results.append(resp.json())

    top5_sets = [
        frozenset(entry["feature"] for entry in r["shap_explanation"][:5])
        for r in results
    ]
    assert top5_sets[0] == top5_sets[1] == top5_sets[2], (
        f"SHAP top-5 features differ across runs:\n"
        f"  Run 1: {top5_sets[0]}\n"
        f"  Run 2: {top5_sets[1]}\n"
        f"  Run 3: {top5_sets[2]}"
    )


def test_shap_probability_deterministic(client):
    """Same variant produces identical probability across 3 calls."""
    probs = []
    for _ in range(3):
        resp = client.post("/predict", json=DETERMINISTIC_VARIANT)
        assert resp.status_code == 200
        probs.append(resp.json()["probability"])

    assert probs[0] == probs[1] == probs[2], (
        f"Probability differs across runs: {probs}"
    )


# ─── SHAP Structure ──────────────────────────────────────────────────────────

def test_shap_explanation_present(client):
    """Response must contain a non-empty shap_explanation list."""
    resp = client.post("/predict", json=DETERMINISTIC_VARIANT)
    assert resp.status_code == 200
    data = resp.json()
    assert "shap_explanation" in data, "Missing shap_explanation key"
    assert len(data["shap_explanation"]) > 0, "shap_explanation is empty"


def test_shap_entries_have_required_fields(client):
    """Each SHAP entry must have 'feature', 'value', and 'direction'."""
    resp = client.post("/predict", json=DETERMINISTIC_VARIANT)
    assert resp.status_code == 200
    for entry in resp.json()["shap_explanation"]:
        assert "feature" in entry, f"Missing 'feature' in: {entry}"
        assert "value" in entry, f"Missing 'value' in: {entry}"
        assert "direction" in entry, f"Missing 'direction' in: {entry}"


def test_shap_values_are_finite(client):
    """All SHAP values must be finite floats (no NaN or Inf)."""
    import math
    resp = client.post("/predict", json=DETERMINISTIC_VARIANT)
    assert resp.status_code == 200
    for entry in resp.json()["shap_explanation"]:
        val = entry["value"]
        assert isinstance(val, (int, float)), f"SHAP value is not numeric: {val}"
        assert math.isfinite(val), f"SHAP value is not finite: {val}"


def test_shap_sorted_by_absolute_value(client):
    """SHAP entries should be sorted by |value| descending."""
    resp = client.post("/predict", json=DETERMINISTIC_VARIANT)
    assert resp.status_code == 200
    entries = resp.json()["shap_explanation"]
    abs_vals = [abs(e["value"]) for e in entries]
    assert abs_vals == sorted(abs_vals, reverse=True), (
        "SHAP explanation is not sorted by absolute value descending"
    )


# ─── Full Response Structure ──────────────────────────────────────────────────

def test_predict_response_has_all_fields(client):
    """Successful prediction must include all expected top-level keys."""
    resp = client.post("/predict", json=DETERMINISTIC_VARIANT)
    assert resp.status_code == 200
    data = resp.json()

    required_keys = [
        "prediction", "probability", "shap_explanation",
        "data_sources", "acmg_evidence", "confidence",
        "risk_tier", "features_used",
    ]
    for key in required_keys:
        assert key in data, f"Missing top-level key: '{key}'"


def test_predict_probability_in_range(client):
    """Predicted probability must be in [0, 1]."""
    resp = client.post("/predict", json=DETERMINISTIC_VARIANT)
    assert resp.status_code == 200
    prob = resp.json()["probability"]
    assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"


def test_predict_prediction_is_valid_label(client):
    """Prediction field must be 'Pathogenic' or 'Benign'."""
    resp = client.post("/predict", json=DETERMINISTIC_VARIANT)
    assert resp.status_code == 200
    pred = resp.json()["prediction"]
    assert pred in ("Pathogenic", "Benign"), f"Unexpected prediction value: {pred}"


def test_data_sources_has_expected_keys(client):
    """data_sources must contain expected sub-keys."""
    resp = client.post("/predict", json=DETERMINISTIC_VARIANT)
    assert resp.status_code == 200
    ds = resp.json()["data_sources"]
    assert isinstance(ds, dict), "data_sources should be a dict"
    # At least one of the core sources should be present
    core_sources = {"alphamissense", "phylop", "mave", "structure"}
    assert len(core_sources & set(ds.keys())) > 0, (
        f"None of the core data sources found in: {list(ds.keys())}"
    )


def test_confidence_has_ci_fields(client):
    """Confidence object must include ci_lower, ci_upper, std."""
    resp = client.post("/predict", json=DETERMINISTIC_VARIANT)
    assert resp.status_code == 200
    ci = resp.json()["confidence"]
    assert "ci_lower" in ci, "Missing ci_lower in confidence"
    assert "ci_upper" in ci, "Missing ci_upper in confidence"
    assert "std" in ci, "Missing std in confidence"
    assert "label" in ci, "Missing label in confidence"
    assert ci["ci_lower"] <= ci["ci_upper"], "ci_lower should be <= ci_upper"
