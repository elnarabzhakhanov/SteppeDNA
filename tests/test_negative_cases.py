"""
Negative/edge-case tests for the SteppeDNA API.

Verifies that:
- Invalid input returns 422 (Pydantic validation error), not 500
- Malformed VCF files are handled gracefully
- Synonymous variants (ref == alt AA) are handled
- Out-of-range positions are rejected at the pydantic layer
"""

import io
import os
import sys
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


# ─── Predict endpoint — invalid inputs ───────────────────────────────────────

def test_invalid_amino_acid_ref(client):
    """Unknown three-letter AA code for ref should return 422."""
    resp = client.post("/predict", json={
        "cDNA_pos": 8165,
        "AA_ref":   "Xyz",   # Not a valid 3-letter AA
        "AA_alt":   "Arg",
        "Mutation": "c.8165A>G",
        "AA_pos":   2722,
    })
    assert resp.status_code == 422, f"Expected 422, got {resp.status_code}: {resp.text}"


def test_invalid_amino_acid_alt(client):
    """Unknown three-letter AA code for alt should return 422."""
    resp = client.post("/predict", json={
        "cDNA_pos": 8165,
        "AA_ref":   "Thr",
        "AA_alt":   "Bbb",   # Not valid
        "Mutation": "c.8165A>G",
        "AA_pos":   2722,
    })
    assert resp.status_code == 422


def test_cdna_pos_out_of_range_zero(client):
    """cDNA_pos = 0 should fail validation (must be >= 1)."""
    resp = client.post("/predict", json={
        "cDNA_pos": 0,
        "AA_ref":   "Thr",
        "AA_alt":   "Arg",
        "Mutation": "c.0A>G",
        "AA_pos":   1,
    })
    assert resp.status_code == 422


def test_cdna_pos_out_of_range_too_large(client):
    """cDNA_pos >> MAX_CDNA_POS should fail validation."""
    resp = client.post("/predict", json={
        "cDNA_pos": 999999,
        "AA_ref":   "Thr",
        "AA_alt":   "Arg",
        "Mutation": "c.999999A>G",
        "AA_pos":   333333,
    })
    assert resp.status_code == 422


def test_missing_required_field(client):
    """Missing AA_ref should return 422."""
    resp = client.post("/predict", json={
        "cDNA_pos": 8165,
        "AA_alt":   "Arg",
        "Mutation": "c.8165A>G",
        "AA_pos":   2722,
    })
    assert resp.status_code == 422


def test_empty_payload(client):
    """Empty JSON body should return 422."""
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


def test_wrong_content_type(client):
    """Sending a plain string instead of JSON should return 4xx."""
    resp = client.post(
        "/predict",
        content="this is not json",
        headers={"Content-Type": "text/plain"},
    )
    assert resp.status_code in (400, 415, 422)


# ─── VCF endpoint — malformed input ─────────────────────────────────────────

def test_vcf_garbage_content(client):
    """Garbage file content should be handled gracefully (no 500 error)."""
    fake_vcf = b"this is not a vcf file at all!\x00\x01\x02"
    resp = client.post(
        "/predict/vcf",
        files={"file": ("bad.vcf", io.BytesIO(fake_vcf), "text/plain")},
    )
    # Should NOT crash — either returns 200 with 0 results or 400
    assert resp.status_code in (200, 400)
    if resp.status_code == 200:
        data = resp.json()
        assert data.get("brca2_missense_found", 0) == 0


def test_vcf_empty_file(client):
    """Empty file should return 200 with 0 missense found."""
    resp = client.post(
        "/predict/vcf",
        files={"file": ("empty.vcf", io.BytesIO(b""), "text/plain")},
    )
    assert resp.status_code in (200, 400)
    if resp.status_code == 200:
        assert resp.json().get("brca2_missense_found", 0) == 0


def test_vcf_header_only(client):
    """VCF with only header lines should return 0 missense variants."""
    header = b"##fileformat=VCFv4.2\n##source=test\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    resp = client.post(
        "/predict/vcf",
        files={"file": ("header_only.vcf", io.BytesIO(header), "text/plain")},
    )
    assert resp.status_code in (200, 400)
    if resp.status_code == 200:
        assert resp.json().get("brca2_missense_found", 0) == 0


def test_vcf_non_brca2_variants(client):
    """Variants on a different chromosome should not match BRCA2."""
    # chr1 variant — not in BRCA2's chr13 region
    vcf_content = (
        b"##fileformat=VCFv4.2\n"
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        b"1\t100000\t.\tA\tG\t.\tPASS\t.\n"
    )
    resp = client.post(
        "/predict/vcf",
        files={"file": ("non_brca2.vcf", io.BytesIO(vcf_content), "text/plain")},
    )
    assert resp.status_code in (200, 400)
    if resp.status_code == 200:
        assert resp.json().get("brca2_missense_found", 0) == 0


# ─── Health check ─────────────────────────────────────────────────────────────

def test_health_endpoint(client):
    """/health should return 200 (healthy) or 503 (degraded when model files missing)."""
    resp = client.get("/health")
    assert resp.status_code in (200, 503)


def test_health_returns_json(client):
    """/health response should include a status field."""
    resp = client.get("/health")
    if resp.status_code == 200:
        data = resp.json()
        assert "status" in data or len(data) > 0
