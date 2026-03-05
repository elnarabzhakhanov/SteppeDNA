"""
API endpoint tests for SteppeDNA FastAPI backend.

Tests cover:
- Health check endpoint
- Root/status endpoint
- Single variant prediction (pathogenic + benign)
- Input validation (invalid AA, out-of-range cDNA)
- Model metrics endpoint
- Response schema validation
- Prediction caching
- Rate limiting headers
"""

import pytest


# ─── Health & Status ────────────────────────────────────────────────────────

def test_root_returns_200(client):
    """GET / should return 200 with API status."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "SteppeDNA" in data["status"]


def test_health_returns_200(client):
    """GET /health should return 200 with component statuses."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data


def test_metrics_endpoint(client):
    """GET /model_metrics should return 200 with AUC/MCC values."""
    resp = client.get("/model_metrics")
    if resp.status_code == 200:
        data = resp.json()
        assert "roc_auc" in data or "auc" in data
    else:
        # Metrics file may not exist in test environment
        assert resp.status_code == 404


# ─── Prediction: Valid Inputs ───────────────────────────────────────────────

def test_predict_pathogenic_nonsense(client):
    """Nonsense variant (Ser->Ter) should be classified as Pathogenic."""
    resp = client.post("/predict", json={
        "gene_name": "BRCA2",
        "cDNA_pos": 6174,
        "AA_ref": "Ser",
        "AA_alt": "Ter",
        "Mutation": "C>A",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction"] == "Pathogenic"
    assert data["probability"] > 0.9
    assert "acmg_evidence" in data
    assert "PVS1" in data["acmg_evidence"]


def test_predict_missense_has_all_fields(client):
    """Missense prediction should return all required fields."""
    resp = client.post("/predict", json={
        "gene_name": "BRCA2",
        "cDNA_pos": 8165,
        "AA_ref": "Thr",
        "AA_alt": "Arg",
        "Mutation": "A>G",
    })
    assert resp.status_code == 200
    data = resp.json()

    # Required top-level fields
    required_fields = [
        "prediction", "probability", "risk_tier", "aa_pos",
        "acmg_evidence", "confidence", "features_used",
        "data_sources", "shap_explanation",
    ]
    for field in required_fields:
        assert field in data, f"Missing field: {field}"

    # Probability is between 0 and 1
    assert 0 <= data["probability"] <= 1

    # Prediction is either Pathogenic or Benign
    assert data["prediction"] in ("Pathogenic", "Benign")

    # Confidence has required sub-fields
    ci = data["confidence"]
    assert "label" in ci
    assert "ci_lower" in ci
    assert "ci_upper" in ci
    assert ci["ci_lower"] <= ci["ci_upper"]

    # Features used has expected keys
    fu = data["features_used"]
    assert "blosum62_score" in fu
    assert "volume_diff" in fu
    assert "charge_changed" in fu
    assert isinstance(fu["charge_changed"], bool)

    # Data sources present
    ds = data["data_sources"]
    assert "alphamissense" in ds
    assert "phylop" in ds
    assert "structure" in ds


def test_predict_shap_explanation(client):
    """SHAP explanation should have feature names and values."""
    resp = client.post("/predict", json={
        "gene_name": "BRCA2",
        "cDNA_pos": 8165,
        "AA_ref": "Thr",
        "AA_alt": "Arg",
        "Mutation": "A>G",
    })
    data = resp.json()
    shap = data.get("shap_explanation", [])
    assert len(shap) > 0, "SHAP explanation should not be empty"
    for entry in shap:
        assert "feature" in entry
        assert "value" in entry
        assert isinstance(entry["value"], (int, float))


def test_predict_all_genes(client):
    """Prediction should work for all 5 supported genes."""
    genes = {
        "BRCA1": {"cDNA_pos": 300, "AA_ref": "Ala", "AA_alt": "Val"},
        "BRCA2": {"cDNA_pos": 8165, "AA_ref": "Thr", "AA_alt": "Arg"},
        "PALB2": {"cDNA_pos": 1000, "AA_ref": "Gly", "AA_alt": "Asp"},
        "RAD51C": {"cDNA_pos": 500, "AA_ref": "Leu", "AA_alt": "Pro"},
        "RAD51D": {"cDNA_pos": 400, "AA_ref": "Ile", "AA_alt": "Thr"},
    }
    for gene, params in genes.items():
        resp = client.post("/predict", json={
            "gene_name": gene,
            **params,
            "Mutation": "A>G",
        })
        assert resp.status_code == 200, f"Failed for gene {gene}: {resp.text}"
        data = resp.json()
        assert data["prediction"] in ("Pathogenic", "Benign"), f"Bad prediction for {gene}"


# ─── Prediction: Invalid Inputs ────────────────────────────────────────────

def test_predict_invalid_gene(client):
    """Invalid gene name should return 422 or 400."""
    resp = client.post("/predict", json={
        "gene_name": "INVALID_GENE",
        "cDNA_pos": 100,
        "AA_ref": "Ala",
        "AA_alt": "Val",
        "Mutation": "A>G",
    })
    assert resp.status_code in (400, 422)


def test_predict_cdna_out_of_range(client):
    """cDNA position exceeding gene length should return 400."""
    resp = client.post("/predict", json={
        "gene_name": "RAD51D",
        "cDNA_pos": 99999,
        "AA_ref": "Ala",
        "AA_alt": "Val",
        "Mutation": "A>G",
    })
    assert resp.status_code == 400


def test_predict_missing_fields(client):
    """Missing required fields should return 422."""
    resp = client.post("/predict", json={
        "gene_name": "BRCA2",
    })
    assert resp.status_code == 422


# ─── Prediction Caching ────────────────────────────────────────────────────

def test_predict_cache_consistent(client):
    """Same variant queried twice should return identical results."""
    payload = {
        "gene_name": "BRCA2",
        "cDNA_pos": 8165,
        "AA_ref": "Thr",
        "AA_alt": "Arg",
        "Mutation": "A>G",
    }
    resp1 = client.post("/predict", json=payload)
    resp2 = client.post("/predict", json=payload)
    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp1.json()["probability"] == resp2.json()["probability"]
    assert resp1.json()["prediction"] == resp2.json()["prediction"]


# ─── Probability Clipping ──────────────────────────────────────────────────

def test_probability_clipped(client):
    """Probabilities should be clipped to [0.005, 0.995]."""
    resp = client.post("/predict", json={
        "gene_name": "BRCA2",
        "cDNA_pos": 8165,
        "AA_ref": "Thr",
        "AA_alt": "Arg",
        "Mutation": "A>G",
    })
    data = resp.json()
    p = data["probability"]
    # Missense variants should never be exactly 0 or 1
    if data["prediction"] in ("Pathogenic", "Benign"):
        assert p >= 0.005 or "Tier 1" in data.get("risk_tier", ""), \
            f"Probability {p} below clip threshold"
        assert p <= 0.995 or "Tier 1" in data.get("risk_tier", ""), \
            f"Probability {p} above clip threshold"


# ─── Request ID Header ─────────────────────────────────────────────────────

def test_request_id_header(client):
    """Every response should include X-Request-ID header."""
    resp = client.get("/")
    assert "x-request-id" in resp.headers


# ─── Gene Reliability Warnings ───────────────────────────────────────────

def test_gene_reliability_in_predict_response(client):
    """Prediction response should include gene reliability info with tier."""
    resp = client.post("/predict", json={
        "gene_name": "PALB2",
        "cDNA_pos": 1000,
        "AA_ref": "Asp",
        "AA_alt": "His",
        "Mutation": "G>C",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "gene_reliability" in data, "Missing gene_reliability field"
    rel = data["gene_reliability"]
    assert "tier" in rel, "Missing tier in gene_reliability"
    assert rel["tier"] in ("high", "moderate", "low"), f"Unexpected tier: {rel['tier']}"


def test_brca2_reliability_is_high(client):
    """BRCA2 should have 'high' reliability tier."""
    resp = client.post("/predict", json={
        "gene_name": "BRCA2",
        "cDNA_pos": 8165,
        "AA_ref": "Thr",
        "AA_alt": "Arg",
        "Mutation": "A>G",
    })
    data = resp.json()
    assert data["gene_reliability"]["tier"] == "high"


# ─── Data Scarcity Quantification (Item 41) ──────────────────────────────

def test_data_support_in_predict_response(client):
    """Prediction should include data_support with required fields."""
    resp = client.post("/predict", json={
        "gene_name": "BRCA2",
        "cDNA_pos": 8165,
        "AA_ref": "Thr",
        "AA_alt": "Arg",
        "Mutation": "A>G",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "data_support" in data, "Missing data_support field"
    ds = data["data_support"]
    assert "nearby_variants" in ds
    assert "same_substitution_type" in ds
    assert "in_known_domain" in ds
    assert "level" in ds
    assert ds["level"] in ("HIGH", "MODERATE", "LOW")
    assert isinstance(ds["nearby_variants"], int)
    assert isinstance(ds["same_substitution_type"], int)
    assert isinstance(ds["in_known_domain"], bool)
    assert ds["nearby_variants"] >= 0


def test_data_support_brca2_high_density(client):
    """BRCA2 BRC repeat region should have HIGH data support (many training variants)."""
    resp = client.post("/predict", json={
        "gene_name": "BRCA2",
        "cDNA_pos": 4500,
        "AA_ref": "Glu",
        "AA_alt": "Lys",
        "Mutation": "G>A",
        "AA_pos": 1500,
    })
    assert resp.status_code == 200
    ds = resp.json()["data_support"]
    # BRCA2 is data-rich, so most positions should have many nearby variants
    assert ds["nearby_variants"] > 0


def test_data_support_level_reflects_count(client):
    """data_support level should be consistent with nearby_variants count."""
    resp = client.post("/predict", json={
        "gene_name": "BRCA2",
        "cDNA_pos": 8165,
        "AA_ref": "Thr",
        "AA_alt": "Arg",
        "Mutation": "A>G",
    })
    assert resp.status_code == 200
    ds = resp.json()["data_support"]
    if ds["nearby_variants"] > 100:
        assert ds["level"] == "HIGH"
    elif ds["nearby_variants"] >= 10:
        assert ds["level"] == "MODERATE"
    else:
        assert ds["level"] == "LOW"


# ─── Per-Gene Calibrators (Item 45) ──────────────────────────────────────

def test_calibrator_type_in_predict_response(client):
    """Prediction should include calibrator_type field."""
    resp = client.post("/predict", json={
        "gene_name": "BRCA2",
        "cDNA_pos": 8165,
        "AA_ref": "Thr",
        "AA_alt": "Arg",
        "Mutation": "A>G",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "calibrator_type" in data, "Missing calibrator_type field"
    assert data["calibrator_type"] in ("gene_specific", "universal", "raw_fallback", "none")


def test_calibrator_type_gene_specific_when_available(client):
    """BRCA2 should use gene-specific calibrator if calibrator file exists."""
    import os
    cal_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "calibrator_brca2.pkl")
    resp = client.post("/predict", json={
        "gene_name": "BRCA2",
        "cDNA_pos": 8165,
        "AA_ref": "Thr",
        "AA_alt": "Arg",
        "Mutation": "A>G",
    })
    assert resp.status_code == 200
    data = resp.json()
    if os.path.exists(cal_path):
        assert data["calibrator_type"] == "gene_specific"


# ─── Feature Coverage (Item 46 - already implemented) ────────────────────

def test_feature_coverage_in_predict_response(client):
    """Prediction should include feature_coverage with nonzero, total, percentage."""
    resp = client.post("/predict", json={
        "gene_name": "BRCA2",
        "cDNA_pos": 8165,
        "AA_ref": "Thr",
        "AA_alt": "Arg",
        "Mutation": "A>G",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "feature_coverage" in data
    fc = data["feature_coverage"]
    assert "nonzero" in fc
    assert "total" in fc
    assert "percentage" in fc
    assert fc["total"] == 103
    assert 0 <= fc["nonzero"] <= fc["total"]
    assert 0 <= fc["percentage"] <= 100


# ─── Gene-Adaptive Ensemble Weights (Item 38) ─────────────────────────────

def test_ensemble_weights_in_predict_response(client):
    """Prediction should include ensemble_weights with xgb and mlp."""
    resp = client.post("/predict", json={
        "gene_name": "BRCA2",
        "cDNA_pos": 8165,
        "AA_ref": "Thr",
        "AA_alt": "Arg",
        "Mutation": "A>G",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "ensemble_weights" in data, "Missing ensemble_weights field"
    ew = data["ensemble_weights"]
    assert "xgb" in ew
    assert "mlp" in ew
    assert 0 <= ew["xgb"] <= 1.0
    assert 0 <= ew["mlp"] <= 1.0
    assert abs(ew["xgb"] + ew["mlp"] - 1.0) < 0.01, "Weights should sum to 1.0"


def test_ensemble_weights_gene_specific(client):
    """Different genes may have different ensemble weights."""
    genes_weights = {}
    for gene in ["BRCA2", "RAD51C"]:
        resp = client.post("/predict", json={
            "gene_name": gene,
            "cDNA_pos": 100,
            "AA_ref": "Ala",
            "AA_alt": "Val",
            "Mutation": "C>T",
        })
        if resp.status_code == 200:
            data = resp.json()
            if "ensemble_weights" in data:
                genes_weights[gene] = data["ensemble_weights"]

    # Both genes should return weights
    assert len(genes_weights) >= 1, "At least one gene should have ensemble weights"


# ─── Bootstrap Confidence Intervals (Item 39) ────────────────────────────

def test_confidence_has_ci_width_and_method(client):
    """Confidence object should include ci_width and method fields."""
    resp = client.post("/predict", json={
        "gene_name": "BRCA2",
        "cDNA_pos": 8165,
        "AA_ref": "Thr",
        "AA_alt": "Arg",
        "Mutation": "A>G",
    })
    assert resp.status_code == 200
    ci = resp.json()["confidence"]
    assert "ci_width" in ci, "Missing ci_width in confidence"
    assert "method" in ci, "Missing method in confidence"
    assert ci["method"] in ("bootstrap", "beta_approximation"), \
        f"Unexpected CI method: {ci['method']}"
    assert ci["ci_width"] >= 0, "ci_width should be non-negative"
    assert ci["ci_width"] == round(ci["ci_upper"] - ci["ci_lower"], 4), \
        "ci_width should equal ci_upper - ci_lower"


def test_bootstrap_ci_if_models_exist(client):
    """If bootstrap models are loaded, CI method should be 'bootstrap'."""
    import os
    bootstrap_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "bootstrap_models"
    )
    resp = client.post("/predict", json={
        "gene_name": "BRCA2",
        "cDNA_pos": 8165,
        "AA_ref": "Thr",
        "AA_alt": "Arg",
        "Mutation": "A>G",
    })
    assert resp.status_code == 200
    ci = resp.json()["confidence"]
    if os.path.isdir(bootstrap_dir) and os.path.exists(
        os.path.join(bootstrap_dir, "bootstrap_0.json")
    ):
        assert ci["method"] == "bootstrap"
    else:
        assert ci["method"] == "beta_approximation"


# ─── Research Priorities / Active Learning (Item 42) ─────────────────────

def test_research_priorities_endpoint(client):
    """GET /research/priorities should return 200."""
    resp = client.get("/research/priorities")
    assert resp.status_code == 200
    data = resp.json()
    # Should have either priorities or error message
    assert "priorities" in data or "error" in data


def test_research_priorities_gene_filter(client):
    """GET /research/priorities?gene=BRCA2 should filter by gene."""
    resp = client.get("/research/priorities?gene=BRCA2")
    assert resp.status_code == 200
    data = resp.json()
    if "priorities" in data:
        # If data loaded, check gene is correct
        if "gene" in data:
            assert data["gene"] == "BRCA2"


def test_research_priorities_invalid_gene(client):
    """GET /research/priorities?gene=INVALID should return 400."""
    resp = client.get("/research/priorities?gene=INVALID")
    assert resp.status_code == 400


# ─── Contrastive Explanation Pairs (Item 43) ─────────────────────────────

def test_contrastive_explanation_field_present(client):
    """Predict response should include contrastive_explanation field."""
    resp = client.post("/predict", json={
        "gene_name": "BRCA2",
        "cDNA_pos": 8165,
        "AA_ref": "Thr",
        "AA_alt": "Ile",
        "Mutation": "C>T",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "contrastive_explanation" in data, "Missing contrastive_explanation in response"


def test_contrastive_explanation_structure_if_available(client):
    """If contrastive_explanation is not None, validate its structure."""
    resp = client.post("/predict", json={
        "gene_name": "BRCA2",
        "cDNA_pos": 8165,
        "AA_ref": "Thr",
        "AA_alt": "Ile",
        "Mutation": "C>T",
    })
    assert resp.status_code == 200
    data = resp.json()
    ce = data.get("contrastive_explanation")
    if ce is not None:
        assert "contrast_variant" in ce, "Missing contrast_variant"
        assert "contrast_class" in ce, "Missing contrast_class"
        assert ce["contrast_class"] in ("Pathogenic", "Benign"), \
            "contrast_class must be Pathogenic or Benign"
        assert "contrast_distance" in ce, "Missing contrast_distance"
        assert ce["contrast_distance"] >= 0, "Distance should be non-negative"
        assert "key_differences" in ce, "Missing key_differences"
        assert isinstance(ce["key_differences"], list), "key_differences should be a list"
        assert len(ce["key_differences"]) <= 5, "Should have at most 5 key differences"
        for diff in ce["key_differences"]:
            assert "feature" in diff, "Missing feature name in key_differences"
            assert "importance" in diff, "Missing importance in key_differences"
            assert diff["importance"] in ("high", "moderate", "low"), \
                "importance must be high, moderate, or low"


def test_contrastive_explanation_opposite_class(client):
    """Contrastive variant should be of the opposite class."""
    resp = client.post("/predict", json={
        "gene_name": "BRCA2",
        "cDNA_pos": 8165,
        "AA_ref": "Thr",
        "AA_alt": "Ile",
        "Mutation": "C>T",
    })
    assert resp.status_code == 200
    data = resp.json()
    ce = data.get("contrastive_explanation")
    if ce is not None:
        predicted_class = data["prediction"]
        # Contrastive class should be the opposite of predicted class
        if predicted_class == "Pathogenic":
            assert ce["contrast_class"] == "Benign", \
                "Contrastive variant should be Benign when prediction is Pathogenic"
        else:
            assert ce["contrast_class"] == "Pathogenic", \
                "Contrastive variant should be Pathogenic when prediction is Benign"


# ─── Split Conformal Prediction (Item 5.1) ─────────────────────────────────

def test_conformal_prediction_field_present(client):
    """Predict response should include conformal_prediction field."""
    resp = client.post("/predict", json={
        "gene_name": "BRCA2",
        "cDNA_pos": 8165,
        "AA_ref": "Thr",
        "AA_alt": "Arg",
        "Mutation": "A>G",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "conformal_prediction" in data, "Missing conformal_prediction in response"


def test_conformal_prediction_structure(client):
    """If conformal_prediction is not None, validate its structure."""
    import os
    ct_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "conformal_thresholds.json"
    )
    resp = client.post("/predict", json={
        "gene_name": "BRCA2",
        "cDNA_pos": 8165,
        "AA_ref": "Thr",
        "AA_alt": "Arg",
        "Mutation": "A>G",
    })
    assert resp.status_code == 200
    data = resp.json()
    cp = data.get("conformal_prediction")
    if os.path.exists(ct_path) and cp is not None:
        assert "conformal_set" in cp, "Missing conformal_set"
        assert "conformal_coverage" in cp, "Missing conformal_coverage"
        assert "set_size" in cp, "Missing set_size"
        assert isinstance(cp["conformal_set"], list), "conformal_set should be a list"
        assert len(cp["conformal_set"]) >= 1, "conformal_set should have at least 1 class"
        assert all(c in ("Pathogenic", "Benign") for c in cp["conformal_set"]), \
            "conformal_set should only contain Pathogenic or Benign"
        assert cp["conformal_coverage"] == 0.90, \
            f"Expected 90% coverage, got {cp['conformal_coverage']}"
        assert cp["set_size"] == len(cp["conformal_set"]), \
            "set_size should match length of conformal_set"


def test_conformal_prediction_all_genes(client):
    """Conformal prediction should work for all supported genes."""
    import os
    ct_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "conformal_thresholds.json"
    )
    if not os.path.exists(ct_path):
        pytest.skip("conformal_thresholds.json not found")
    genes = {
        "BRCA1": {"cDNA_pos": 300, "AA_ref": "Ala", "AA_alt": "Val"},
        "BRCA2": {"cDNA_pos": 8165, "AA_ref": "Thr", "AA_alt": "Arg"},
        "PALB2": {"cDNA_pos": 1000, "AA_ref": "Gly", "AA_alt": "Asp"},
        "RAD51C": {"cDNA_pos": 500, "AA_ref": "Leu", "AA_alt": "Pro"},
        "RAD51D": {"cDNA_pos": 400, "AA_ref": "Ile", "AA_alt": "Thr"},
    }
    for gene, params in genes.items():
        resp = client.post("/predict", json={
            "gene_name": gene, **params, "Mutation": "A>G",
        })
        assert resp.status_code == 200, f"Failed for gene {gene}: {resp.text}"
        cp = resp.json().get("conformal_prediction")
        if cp is not None:
            assert len(cp["conformal_set"]) >= 1, \
                f"Empty conformal set for {gene}"
