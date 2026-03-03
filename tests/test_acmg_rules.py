"""
Unit tests for the ACMG rule engine (backend/acmg_rules.py).

Verifies that each ACMG criterion fires correctly under the right conditions
and does NOT fire under conditions where it shouldn't.
"""

import os
import sys
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from backend.acmg_rules import evaluate_acmg_rules


# ─── PM1: Structural proximity to functional sites ────────────────────────────

def test_pm1_fires_for_dna_contact():
    """Variant within 5Å of DNA binding interface should trigger PM1."""
    feats = {"domain": "DNA_binding", "dist_dna": 3.0, "dist_palb2": 50.0, "gnomad_af": 0.0}
    result = evaluate_acmg_rules(feats, 0.5)
    assert "PM1" in result, f"Expected PM1, got: {result}"


def test_pm1_fires_for_palb2_contact():
    """Variant within 5Å of PALB2 interaction domain should trigger PM1."""
    feats = {"domain": "uncharacterized", "dist_dna": 100.0, "dist_palb2": 2.5, "gnomad_af": 0.0}
    result = evaluate_acmg_rules(feats, 0.5)
    assert "PM1" in result


def test_pm1_fires_for_brc_repeat():
    """Variant inside a BRC repeat should trigger PM1."""
    feats = {"in_critical_domain": True, "dist_dna": 50.0, "dist_palb2": 50.0, "gnomad_af": 0.0}
    result = evaluate_acmg_rules(feats, 0.5)
    assert "PM1" in result


def test_pm1_not_fired_far_from_sites():
    """Variant far from all functional sites should NOT trigger PM1."""
    feats = {"domain": "uncharacterized", "dist_dna": 999.0, "dist_palb2": 999.0, "gnomad_af": 0.0}
    result = evaluate_acmg_rules(feats, 0.5)
    assert "PM1" not in result


def test_pm1_boundary_at_5A():
    """dist_dna exactly at 5.0Å should trigger PM1 (≤ 5.0)."""
    feats = {"domain": "uncharacterized", "dist_dna": 5.0, "dist_palb2": 999.0, "gnomad_af": 0.0}
    result = evaluate_acmg_rules(feats, 0.5)
    assert "PM1" in result


def test_pm1_not_fired_just_outside_5A():
    """dist_dna of 5.01Å should NOT trigger PM1."""
    feats = {"domain": "uncharacterized", "dist_dna": 5.01, "dist_palb2": 999.0, "gnomad_af": 0.0}
    result = evaluate_acmg_rules(feats, 0.5)
    assert "PM1" not in result


# ─── BS1: Common population frequency ────────────────────────────────────────

def test_bs1_fires_for_common_variant():
    """gnomAD AF > gene BS1 threshold but below BA1 should trigger BS1 (not BA1)."""
    feats = {"domain": "uncharacterized", "dist_dna": 999.0, "dist_palb2": 999.0, "gnomad_af": 0.005}
    result = evaluate_acmg_rules(feats, 0.3, gene_name="BRCA2")
    assert "BS1" in result
    assert "BA1" not in result


def test_ba1_fires_for_very_common_variant():
    """gnomAD AF > BA1 threshold should trigger BA1 and suppress BS1."""
    feats = {"domain": "uncharacterized", "dist_dna": 999.0, "dist_palb2": 999.0, "gnomad_af": 0.08}
    result = evaluate_acmg_rules(feats, 0.3, gene_name="BRCA2")
    assert "BA1" in result
    assert "BS1" not in result


def test_bs1_not_fired_for_rare_variant():
    """gnomAD AF below gene threshold should NOT trigger BS1."""
    feats = {"domain": "uncharacterized", "dist_dna": 999.0, "dist_palb2": 999.0, "gnomad_af": 0.0005}
    result = evaluate_acmg_rules(feats, 0.3, gene_name="BRCA2")
    assert "BS1" not in result


def test_bs1_boundary_at_gene_threshold():
    """gnomAD AF exactly at gene threshold (0.001 for BRCA2) should NOT fire (needs to exceed)."""
    feats = {"domain": "uncharacterized", "dist_dna": 999.0, "dist_palb2": 999.0, "gnomad_af": 0.001}
    result = evaluate_acmg_rules(feats, 0.3, gene_name="BRCA2")
    assert "BS1" not in result


def test_bs1_fires_above_gene_threshold():
    """gnomAD AF above BRCA2 threshold (0.001) should trigger BS1."""
    feats = {"domain": "uncharacterized", "dist_dna": 999.0, "dist_palb2": 999.0, "gnomad_af": 0.002}
    result = evaluate_acmg_rules(feats, 0.3, gene_name="BRCA2")
    assert "BS1" in result


def test_bs1_gene_specific_rad51d():
    """RAD51D with AF=0.003 should NOT trigger BS1 (threshold is 0.005)."""
    feats = {"domain": "uncharacterized", "dist_dna": 999.0, "dist_palb2": 999.0, "gnomad_af": 0.003}
    result = evaluate_acmg_rules(feats, 0.3, gene_name="RAD51D")
    assert "BS1" not in result


def test_bs1_gene_specific_rad51d_above():
    """RAD51D with AF=0.006 should trigger BS1 (threshold is 0.005)."""
    feats = {"domain": "uncharacterized", "dist_dna": 999.0, "dist_palb2": 999.0, "gnomad_af": 0.006}
    result = evaluate_acmg_rules(feats, 0.3, gene_name="RAD51D")
    assert "BS1" in result


# ─── PP3: Strong computational prediction of pathogenicity ────────────────────

def test_pp3_fires_at_high_probability():
    """Probability >= 0.90 should trigger PP3."""
    feats = {"domain": "uncharacterized", "dist_dna": 999.0, "dist_palb2": 999.0, "gnomad_af": 0.0}
    result = evaluate_acmg_rules(feats, 0.95)
    assert "PP3" in result


def test_pp3_not_fired_at_moderate_probability():
    """Probability of 0.75 (below 0.90 threshold) should NOT trigger PP3."""
    feats = {"domain": "uncharacterized", "dist_dna": 999.0, "dist_palb2": 999.0, "gnomad_af": 0.0}
    result = evaluate_acmg_rules(feats, 0.75)
    assert "PP3" not in result


# ─── BP4: Strong computational prediction of benign ──────────────────────────

def test_bp4_fires_at_low_probability():
    """Probability <= 0.10 should trigger BP4."""
    feats = {"domain": "uncharacterized", "dist_dna": 999.0, "dist_palb2": 999.0, "gnomad_af": 0.0}
    result = evaluate_acmg_rules(feats, 0.05)
    assert "BP4" in result


def test_bp4_not_fired_at_moderate_probability():
    """Probability of 0.50 should NOT trigger BP4."""
    feats = {"domain": "uncharacterized", "dist_dna": 999.0, "dist_palb2": 999.0, "gnomad_af": 0.0}
    result = evaluate_acmg_rules(feats, 0.50)
    assert "BP4" not in result


# ─── PP3 and BP4 mutual exclusion ────────────────────────────────────────────

def test_pp3_and_bp4_not_both_active():
    """PP3 and BP4 should never both be active simultaneously."""
    feats = {"domain": "uncharacterized", "dist_dna": 999.0, "dist_palb2": 999.0, "gnomad_af": 0.0}
    for prob in [0.0, 0.05, 0.10, 0.50, 0.85, 0.90, 0.95, 1.0]:
        result = evaluate_acmg_rules(feats, prob)
        assert not ("PP3" in result and "BP4" in result), (
            f"Both PP3 and BP4 active at prob={prob}"
        )


# ─── Missing/default features ────────────────────────────────────────────────

def test_empty_features_no_crash():
    """Empty features dict should not raise an exception."""
    result = evaluate_acmg_rules({}, 0.5)
    assert isinstance(result, dict)


def test_partial_features_no_crash():
    """Partial features dict (missing some keys) should not raise."""
    result = evaluate_acmg_rules({"gnomad_af": 0.08}, 0.95)
    assert isinstance(result, dict)
    assert "BA1" in result  # AF 8% exceeds BA1 threshold (1%)
    assert "PP3" in result


def test_pm5_fires_for_known_pathogenic_position():
    """PM5 should fire when known_pathogenic_at_pos is True."""
    feats = {"domain": "uncharacterized", "dist_dna": 999.0, "dist_palb2": 999.0,
             "gnomad_af": 0.0, "known_pathogenic_at_pos": True}
    result = evaluate_acmg_rules(feats, 0.85, gene_name="BRCA2")
    assert "PM5" in result


def test_pm5_not_fired_without_flag():
    """PM5 should not fire when known_pathogenic_at_pos is absent or False."""
    feats = {"domain": "uncharacterized", "dist_dna": 999.0, "dist_palb2": 999.0, "gnomad_af": 0.0}
    result = evaluate_acmg_rules(feats, 0.85, gene_name="BRCA2")
    assert "PM5" not in result


# ─── PVS1: Null variant (nonsense/frameshift) ────────────────────────────────

def test_pvs1_fires_for_nonsense():
    """Nonsense variant should trigger PVS1."""
    feats = {"is_nonsense": True, "gnomad_af": 0.0, "dist_dna": 999.0, "dist_palb2": 999.0}
    result = evaluate_acmg_rules(feats, 0.95, gene_name="BRCA2")
    assert "PVS1" in result
    assert "Nonsense" in result["PVS1"]


def test_pvs1_fires_for_frameshift():
    """Frameshift variant should trigger PVS1."""
    feats = {"is_frameshift": True, "gnomad_af": 0.0, "dist_dna": 999.0, "dist_palb2": 999.0}
    result = evaluate_acmg_rules(feats, 0.95, gene_name="BRCA1")
    assert "PVS1" in result
    assert "Frameshift" in result["PVS1"]


def test_pvs1_not_fired_for_missense():
    """Missense variant should NOT trigger PVS1."""
    feats = {"is_nonsense": False, "is_frameshift": False, "gnomad_af": 0.0,
             "dist_dna": 999.0, "dist_palb2": 999.0}
    result = evaluate_acmg_rules(feats, 0.95, gene_name="BRCA2")
    assert "PVS1" not in result


# ─── PM4: In-frame indel ─────────────────────────────────────────────────────

def test_pm4_fires_for_inframe_indel_in_domain():
    """In-frame indel in a functional domain should trigger PM4."""
    feats = {"is_inframe_indel": True, "in_critical_domain": True,
             "gnomad_af": 0.0, "dist_dna": 999.0, "dist_palb2": 999.0}
    result = evaluate_acmg_rules(feats, 0.70, gene_name="BRCA2")
    assert "PM4" in result
    assert "functional domain" in result["PM4"]


def test_pm4_fires_for_inframe_indel_outside_domain():
    """In-frame indel outside functional domain should still trigger PM4 (reduced weight)."""
    feats = {"is_inframe_indel": True, "in_critical_domain": False,
             "gnomad_af": 0.0, "dist_dna": 999.0, "dist_palb2": 999.0}
    result = evaluate_acmg_rules(feats, 0.70, gene_name="BRCA2")
    assert "PM4" in result
    assert "reduced weight" in result["PM4"]


def test_pm4_not_fired_without_indel():
    """Non-indel variant should NOT trigger PM4."""
    feats = {"is_inframe_indel": False, "gnomad_af": 0.0,
             "dist_dna": 999.0, "dist_palb2": 999.0}
    result = evaluate_acmg_rules(feats, 0.70, gene_name="BRCA2")
    assert "PM4" not in result


# ─── BP7: Synonymous variant far from splice ─────────────────────────────────

def test_bp7_fires_for_synonymous_far_from_splice():
    """Synonymous variant far from splice site should trigger BP7."""
    feats = {"is_synonymous": True, "near_splice": False, "canonical_splice": False,
             "gnomad_af": 0.0, "dist_dna": 999.0, "dist_palb2": 999.0}
    result = evaluate_acmg_rules(feats, 0.10, gene_name="BRCA2")
    assert "BP7" in result


def test_bp7_not_fired_for_synonymous_near_splice():
    """Synonymous variant near splice site should NOT trigger BP7."""
    feats = {"is_synonymous": True, "near_splice": True, "canonical_splice": False,
             "gnomad_af": 0.0, "dist_dna": 999.0, "dist_palb2": 999.0}
    result = evaluate_acmg_rules(feats, 0.10, gene_name="BRCA2")
    assert "BP7" not in result


def test_bp7_not_fired_for_non_synonymous():
    """Non-synonymous variant should NOT trigger BP7."""
    feats = {"is_synonymous": False, "near_splice": False, "canonical_splice": False,
             "gnomad_af": 0.0, "dist_dna": 999.0, "dist_palb2": 999.0}
    result = evaluate_acmg_rules(feats, 0.10, gene_name="BRCA2")
    assert "BP7" not in result


# ─── PP3_splice: Near splice site ────────────────────────────────────────────

def test_pp3_splice_fires_for_near_splice():
    """Variant near splice site should trigger PP3_splice."""
    feats = {"near_splice": True, "canonical_splice": False,
             "gnomad_af": 0.0, "dist_dna": 999.0, "dist_palb2": 999.0}
    result = evaluate_acmg_rules(feats, 0.70, gene_name="BRCA2")
    assert "PP3_splice" in result


def test_pvs1_fires_for_canonical_splice():
    """Variant at canonical splice site should trigger PVS1 (not PP3_splice)."""
    feats = {"canonical_splice": True, "near_splice": False,
             "gnomad_af": 0.0, "dist_dna": 999.0, "dist_palb2": 999.0}
    result = evaluate_acmg_rules(feats, 0.95, gene_name="BRCA2")
    assert "PVS1" in result


def test_pp3_splice_not_fired_far_from_splice():
    """Variant far from splice site should NOT trigger PP3_splice."""
    feats = {"near_splice": False, "canonical_splice": False,
             "gnomad_af": 0.0, "dist_dna": 999.0, "dist_palb2": 999.0}
    result = evaluate_acmg_rules(feats, 0.70, gene_name="BRCA2")
    assert "PP3_splice" not in result
