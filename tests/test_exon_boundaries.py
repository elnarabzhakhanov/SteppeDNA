"""Tests for exon boundary derivation and variant type classification."""

import sys
import os
import pickle
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ─── Exon Boundary Derivation Tests ──────────────────────────────────────────

def _load_cdna_to_genomic(gene):
    """Load the cDNA-to-genomic mapping for a gene."""
    path = os.path.join(DATA_DIR, f"{gene.lower()}_cdna_to_genomic.pkl")
    if not os.path.exists(path):
        pytest.skip(f"No mapping file found for {gene}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _get_strand(gene):
    """Get strand from gene config."""
    import json
    config_path = os.path.join(os.path.dirname(__file__), "..", "backend", "gene_configs", f"{gene.lower()}.json")
    if not os.path.exists(config_path):
        pytest.skip(f"No gene config for {gene}")
    with open(config_path, "r") as f:
        return json.load(f).get("strand", "+")


def _derive():
    """Import the derivation function."""
    from backend.main import _derive_exon_boundaries
    return _derive_exon_boundaries


def test_brca2_exon_boundaries_nonempty():
    """BRCA2 (forward strand) should produce ~26 exon-exon boundaries."""
    derive = _derive()
    mapping = _load_cdna_to_genomic("BRCA2")
    strand = _get_strand("BRCA2")
    assert strand == "+"
    result = derive(mapping, strand)
    assert len(result["exon_boundaries"]) > 20, f"Expected ~26 boundaries, got {len(result['exon_boundaries'])}"
    assert len(result["exon_boundaries"]) < 35, f"Too many boundaries: {len(result['exon_boundaries'])}"


def test_brca1_exon_boundaries_nonempty():
    """BRCA1 (reverse strand) should produce ~22 exon-exon boundaries."""
    derive = _derive()
    mapping = _load_cdna_to_genomic("BRCA1")
    strand = _get_strand("BRCA1")
    assert strand == "-"
    result = derive(mapping, strand)
    assert len(result["exon_boundaries"]) > 15, f"Expected ~22 boundaries, got {len(result['exon_boundaries'])}"
    assert len(result["exon_boundaries"]) < 30, f"Too many boundaries: {len(result['exon_boundaries'])}"


def test_canonical_near_splice_mutual_exclusion():
    """Canonical and near-splice sets should not overlap."""
    derive = _derive()
    mapping = _load_cdna_to_genomic("BRCA2")
    strand = _get_strand("BRCA2")
    result = derive(mapping, strand)
    overlap = result["canonical_splice"] & result["near_splice"]
    assert len(overlap) == 0, f"Overlap found: {len(overlap)} positions"


def test_splice_positions_not_in_cds():
    """Splice zone positions should not overlap with CDS positions."""
    derive = _derive()
    mapping = _load_cdna_to_genomic("BRCA2")
    strand = _get_strand("BRCA2")
    result = derive(mapping, strand)
    cds_positions = set(mapping.values())
    canonical_in_cds = result["canonical_splice"] & cds_positions
    near_in_cds = result["near_splice"] & cds_positions
    assert len(canonical_in_cds) == 0, f"Canonical positions in CDS: {len(canonical_in_cds)}"
    assert len(near_in_cds) == 0, f"Near-splice positions in CDS: {len(near_in_cds)}"


def test_splice_info_consistency():
    """Every position in canonical/near sets should have a splice_info entry."""
    derive = _derive()
    mapping = _load_cdna_to_genomic("BRCA2")
    strand = _get_strand("BRCA2")
    result = derive(mapping, strand)
    for pos in list(result["canonical_splice"])[:50]:
        assert pos in result["splice_info"], f"Missing splice_info for canonical pos {pos}"
        assert result["splice_info"][pos]["zone"] == "canonical"
    for pos in list(result["near_splice"])[:50]:
        assert pos in result["splice_info"], f"Missing splice_info for near-splice pos {pos}"
        assert result["splice_info"][pos]["zone"] == "near"


def test_empty_mapping():
    """Empty mapping should produce empty results."""
    derive = _derive()
    result = derive({}, "+")
    assert result["exon_boundaries"] == []
    assert len(result["canonical_splice"]) == 0
    assert len(result["near_splice"]) == 0


# ─── parse_vcf_line Tests ────────────────────────────────────────────────────

def test_parse_vcf_line_basic():
    """Basic 5-column VCF line should parse correctly."""
    from backend.main import parse_vcf_line
    result = parse_vcf_line("13\t32316462\t.\tA\tG")
    assert result is not None
    chrom, pos, ref, alt, gt, vcf_filter, warnings, multi_sample = result
    assert chrom == "13"
    assert pos == 32316462
    assert ref == "A"
    assert alt == "G"
    assert gt is None


def test_parse_vcf_line_with_genotype():
    """VCF line with FORMAT + SAMPLE columns should extract genotype."""
    from backend.main import parse_vcf_line
    line = "13\t32316462\t.\tA\tG\t30\tPASS\t.\tGT:DP:GQ\t0/1:30:99"
    result = parse_vcf_line(line)
    assert result is not None
    chrom, pos, ref, alt, gt, vcf_filter, warnings, multi_sample = result
    assert gt == "0/1"


def test_parse_vcf_line_homozygous():
    """Homozygous genotype should be parsed."""
    from backend.main import parse_vcf_line
    line = "13\t32316462\t.\tA\tG\t30\tPASS\t.\tGT:DP\t1/1:30"
    result = parse_vcf_line(line)
    chrom, pos, ref, alt, gt, vcf_filter, warnings, multi_sample = result
    assert gt == "1/1"


def test_parse_vcf_line_chr_prefix():
    """chr prefix should be stripped."""
    from backend.main import parse_vcf_line
    result = parse_vcf_line("chr13\t32316462\t.\tA\tG")
    assert result[0] == "13"


# ─── Variant Type Classification Tests ───────────────────────────────────────

def test_vcf_variant_frameshift_has_type():
    """Frameshift variants should have variant_type='frameshift'."""
    from backend.main import vcf_variant_to_prediction, get_gene_data
    gd = get_gene_data("BRCA2")
    # Use a known CDS position with a 2bp deletion (frameshift)
    cds_positions = list(gd["genomic_to_cdna"].keys())[:5]
    if not cds_positions:
        pytest.skip("No CDS positions available")
    pos = cds_positions[0]
    result, reason = vcf_variant_to_prediction("13", pos, "AT", "A", gene_name="BRCA2", gene_data=gd)
    assert result is not None, f"Expected result, got reason: {reason}"
    assert result["variant_type"] == "frameshift"
    assert result["probability"] == 0.9999
