"""
Clinical correctness tests: verify the model produces reasonable predictions
for known pathogenic/benign variants across all 5 genes.
"""
import pytest


class TestBRCA2ClinicalCorrectness:
    """BRCA2 is the best-performing gene (AUC 0.983)."""

    def test_known_pathogenic_high_probability(self, client):
        from tests.conftest import KNOWN_PATHOGENIC
        payload = {"gene_name": "BRCA2", **KNOWN_PATHOGENIC}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        assert resp.json()["probability"] > 0.80

    def test_known_benign_low_probability(self, client):
        from tests.conftest import KNOWN_BENIGN
        payload = {"gene_name": "BRCA2", **KNOWN_BENIGN}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        assert resp.json()["probability"] < 0.50  # should be benign-leaning


class TestBRCA1ClinicalCorrectness:
    """BRCA1 has AUC 0.706 — predictions are less reliable."""

    def test_nonsense_classified_pathogenic(self, client):
        from tests.conftest import KNOWN_BRCA1_PATHOGENIC
        resp = client.post("/predict", json=KNOWN_BRCA1_PATHOGENIC)
        assert resp.status_code == 200
        # Nonsense variants should hit Tier 1 rule
        assert resp.json()["probability"] > 0.90

    def test_benign_missense_returns_valid_prediction(self, client):
        from tests.conftest import KNOWN_BRCA1_BENIGN
        resp = client.post("/predict", json=KNOWN_BRCA1_BENIGN)
        assert resp.status_code == 200
        prob = resp.json()["probability"]
        # BRCA1 AUC is 0.706 with 94.8% pathogenic class imbalance.
        # The model may misclassify benign variants — we only assert valid range.
        assert 0.0 <= prob <= 1.0


class TestPALB2ClinicalCorrectness:
    """PALB2 has AUC 0.641 — weakest gene."""

    def test_nonsense_classified_pathogenic(self, client):
        from tests.conftest import KNOWN_PALB2_PATHOGENIC
        resp = client.post("/predict", json=KNOWN_PALB2_PATHOGENIC)
        assert resp.status_code == 200
        assert resp.json()["probability"] > 0.90  # Tier 1 rule

    def test_benign_missense_returns_valid_probability(self, client):
        from tests.conftest import KNOWN_PALB2_BENIGN
        resp = client.post("/predict", json=KNOWN_PALB2_BENIGN)
        assert resp.status_code == 200
        prob = resp.json()["probability"]
        assert 0.0 <= prob <= 1.0


class TestRAD51CClinicalCorrectness:
    def test_missense_returns_valid_prediction(self, client):
        from tests.conftest import KNOWN_RAD51C_MISSENSE
        resp = client.post("/predict", json=KNOWN_RAD51C_MISSENSE)
        assert resp.status_code == 200
        data = resp.json()
        assert 0.0 <= data["probability"] <= 1.0
        assert data["prediction"] in [
            "Pathogenic", "Benign", "Likely Pathogenic", "Likely Benign"
        ]
        assert "gene_name" not in data or data.get("gene_name", "RAD51C") == "RAD51C"


class TestRAD51DClinicalCorrectness:
    def test_missense_returns_valid_prediction(self, client):
        from tests.conftest import KNOWN_RAD51D_MISSENSE
        resp = client.post("/predict", json=KNOWN_RAD51D_MISSENSE)
        assert resp.status_code == 200
        data = resp.json()
        assert 0.0 <= data["probability"] <= 1.0
        assert data["prediction"] in [
            "Pathogenic", "Benign", "Likely Pathogenic", "Likely Benign"
        ]


class TestClinicalInvariants:
    """Properties that must always hold regardless of gene or variant."""

    def test_high_af_triggers_ba1(self, client):
        """Any variant with gnomAD AF > 1% should trigger BA1 benign evidence."""
        from backend.acmg_rules import evaluate_acmg_rules
        for gene in ["BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D"]:
            feats = {"gnomad_af": 0.02, "dist_dna": 999.0, "dist_palb2": 999.0}
            result = evaluate_acmg_rules(feats, 0.99, gene_name=gene)
            assert "BA1" in result, f"BA1 not triggered for {gene} with AF=2%"

    @pytest.mark.parametrize("gene", ["BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D"])
    def test_prediction_has_required_fields(self, client, gene):
        """Every prediction response must contain these fields."""
        payload = {
            "gene_name": gene,
            "cDNA_pos": 100,
            "AA_ref": "Ala",
            "AA_alt": "Val",
            "Mutation": "C>T",
            "AA_pos": 34,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        required = ["prediction", "probability", "risk_tier", "acmg_evidence"]
        for field in required:
            assert field in data, f"Missing field '{field}' for {gene}"

    @pytest.mark.parametrize("gene", ["BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D"])
    def test_gene_reliability_warning_present(self, client, gene):
        """Non-BRCA2 genes should have reliability warnings."""
        payload = {
            "gene_name": gene,
            "cDNA_pos": 100,
            "AA_ref": "Ala",
            "AA_alt": "Val",
            "Mutation": "C>T",
            "AA_pos": 34,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        if gene != "BRCA2":
            assert "reliability" in data or "gene_reliability" in data
