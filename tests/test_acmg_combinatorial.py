"""
Combinatorial, boundary-value, and regression tests for the ACMG rule engine.

Tests focus on:
- PVS1 priority regression (nonsense/frameshift must not be overwritten by canonical splice)
- BA1 supersedes BS1 across all five genes
- PP3/BP4 boundary values (exact thresholds and one-unit steps away)
- BS1 gene-specific thresholds (above and below per gene)
- PM4 domain-aware rationale strings
- BP7 synonymous suppression near splice sites
- PM5 known-pathogenic-at-position flag

Run from project root:
  pytest tests/test_acmg_combinatorial.py -v
"""

import os
import sys
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from backend.acmg_rules import evaluate_acmg_rules


# ─── Minimal default features used as a base for most tests ──────────────────
# Keys that are not relevant to the criterion under test are given neutral values
# so that no other rule accidentally fires and obscures the assertion.

_BASE = {
    "gnomad_af": 0.0,
    "dist_dna": 999.0,
    "dist_palb2": 999.0,
    "in_critical_domain": False,
    "known_pathogenic_at_pos": False,
    "is_nonsense": False,
    "is_frameshift": False,
    "is_inframe_indel": False,
    "is_synonymous": False,
    "near_splice": False,
    "canonical_splice": False,
}


def _feats(**overrides):
    """Return a copy of _BASE with the supplied keys overridden."""
    f = dict(_BASE)
    f.update(overrides)
    return f


# ─── PVS1 overwrite regression tests ─────────────────────────────────────────


def test_pvs1_nonsense_not_overwritten_by_splice():
    """Regression: a nonsense variant with canonical_splice must keep the
    'Nonsense' PVS1 label — the splice branch must not replace it."""
    feats = _feats(is_nonsense=True, canonical_splice=True)
    result = evaluate_acmg_rules(feats, 0.95, gene_name="BRCA2")
    assert "PVS1" in result, f"PVS1 missing from result: {result}"
    assert "Nonsense" in result["PVS1"], (
        f"Expected 'Nonsense' in PVS1 rationale, got: {result['PVS1']!r}"
    )


def test_pvs1_frameshift_not_overwritten_by_splice():
    """Regression: a frameshift variant with canonical_splice must keep the
    'Frameshift' PVS1 label — the splice branch must not replace it."""
    feats = _feats(is_frameshift=True, canonical_splice=True)
    result = evaluate_acmg_rules(feats, 0.95, gene_name="BRCA1")
    assert "PVS1" in result, f"PVS1 missing from result: {result}"
    assert "Frameshift" in result["PVS1"], (
        f"Expected 'Frameshift' in PVS1 rationale, got: {result['PVS1']!r}"
    )


def test_pvs1_nonsense_takes_priority_over_splice_no_pp3_splice():
    """When nonsense sets PVS1, the canonical splice branch is skipped entirely,
    so PP3_splice must also be absent (near_splice=False confirms this)."""
    feats = _feats(is_nonsense=True, canonical_splice=True, near_splice=False)
    result = evaluate_acmg_rules(feats, 0.95, gene_name="BRCA2")
    assert "PP3_splice" not in result, (
        f"PP3_splice should not fire when PVS1 already set: {result}"
    )


# ─── Canonical splice PVS1 when no prior PVS1 exists ─────────────────────────


def test_pvs1_splice_when_no_nonsense_or_frameshift():
    """A pure canonical splice variant (no nonsense/frameshift) should set PVS1
    with a splice-related rationale string."""
    feats = _feats(canonical_splice=True)
    result = evaluate_acmg_rules(feats, 0.95, gene_name="BRCA2")
    assert "PVS1" in result, f"PVS1 missing from result: {result}"
    assert "splice" in result["PVS1"].lower(), (
        f"Expected 'splice' in PVS1 rationale, got: {result['PVS1']!r}"
    )


def test_pvs1_splice_not_pp3_splice():
    """Canonical splice must fire PVS1, not PP3_splice."""
    feats = _feats(canonical_splice=True)
    result = evaluate_acmg_rules(feats, 0.70, gene_name="BRCA2")
    assert "PVS1" in result
    assert "PP3_splice" not in result


# ─── BA1 supersedes BS1 for all five genes ───────────────────────────────────


@pytest.mark.parametrize("gene", ["BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D"])
def test_ba1_supersedes_bs1_all_genes(gene):
    """gnomAD AF of 5 % is above the BA1 threshold (1 %) for every gene.
    BA1 must be present and BS1 must be absent (BA1 supersedes BS1)."""
    feats = _feats(gnomad_af=0.05)
    result = evaluate_acmg_rules(feats, 0.5, gene_name=gene)
    assert "BA1" in result, f"BA1 missing for {gene}: {result}"
    assert "BS1" not in result, f"BS1 should be suppressed by BA1 for {gene}: {result}"


@pytest.mark.parametrize("gene", ["BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D"])
def test_ba1_not_triggered_at_exact_threshold(gene):
    """AF exactly equal to the BA1 threshold (0.01) must NOT trigger BA1
    because the rule requires strictly greater-than."""
    feats = _feats(gnomad_af=0.01)
    result = evaluate_acmg_rules(feats, 0.5, gene_name=gene)
    assert "BA1" not in result, (
        f"BA1 fired at exact threshold for {gene}: {result}"
    )


# ─── PP3 / BP4 boundary values ───────────────────────────────────────────────


@pytest.mark.parametrize("pred, code_present, code_absent", [
    (0.70,  "PP3", "BP4"),   # at PP3 threshold — PP3 fires
    (0.699, None,  "PP3"),   # one step below PP3 threshold — PP3 must not fire
    (0.20,  "BP4", "PP3"),   # at BP4 threshold — BP4 fires
    (0.201, None,  "BP4"),   # one step above BP4 threshold — BP4 must not fire
    (0.50,  None,  "PP3"),   # middle value — neither fires
])
def test_pp3_bp4_boundaries(pred, code_present, code_absent):
    """Check PP3/BP4 fire exactly at their thresholds and not one step outside."""
    feats = _feats()
    result = evaluate_acmg_rules(feats, pred)
    if code_present is not None:
        assert code_present in result, (
            f"Expected {code_present} at pred={pred}, got: {result}"
        )
    assert code_absent not in result, (
        f"Unexpected {code_absent} at pred={pred}, got: {result}"
    )


def test_pp3_and_bp4_never_both_active_combinatorial():
    """PP3 and BP4 must never both appear in the same result across a range of
    probability values — including at and around both thresholds."""
    feats = _feats()
    for prob in [0.00, 0.05, 0.20, 0.201, 0.50, 0.699, 0.70, 0.95, 1.00]:
        result = evaluate_acmg_rules(feats, prob)
        assert not ("PP3" in result and "BP4" in result), (
            f"PP3 and BP4 both active at prob={prob}: {result}"
        )


def test_middle_probability_fires_neither_pp3_nor_bp4():
    """A probability of 0.50 should fire neither PP3 nor BP4."""
    feats = _feats()
    result = evaluate_acmg_rules(feats, 0.50)
    assert "PP3" not in result
    assert "BP4" not in result


# ─── BS1 gene-specific threshold tests ───────────────────────────────────────


@pytest.mark.parametrize("gene, af, should_trigger", [
    # BRCA1 threshold = 0.001
    ("BRCA1", 0.0011, True),   # just above
    ("BRCA1", 0.0009, False),  # just below
    ("BRCA1", 0.001,  False),  # exactly at threshold (strict >)
    # PALB2 threshold = 0.002
    ("PALB2", 0.0021, True),   # just above
    ("PALB2", 0.0019, False),  # just below
    # RAD51C threshold = 0.005
    ("RAD51C", 0.0051, True),  # just above
    ("RAD51C", 0.0049, False), # just below
    # RAD51D threshold = 0.005 (same family as RAD51C)
    ("RAD51D", 0.0051, True),  # just above
    ("RAD51D", 0.0049, False), # just below
])
def test_bs1_gene_specific_thresholds(gene, af, should_trigger):
    """BS1 must use the gene-specific AF threshold and not fire below it."""
    feats = _feats(gnomad_af=af)
    result = evaluate_acmg_rules(feats, 0.5, gene_name=gene)
    if should_trigger:
        assert "BS1" in result, (
            f"Expected BS1 for {gene} at AF={af}, got: {result}"
        )
    else:
        assert "BS1" not in result, (
            f"Unexpected BS1 for {gene} at AF={af}, got: {result}"
        )


# ─── PM4 domain-aware rationale strings ──────────────────────────────────────


def test_pm4_inframe_in_domain_rationale():
    """In-frame indel inside a critical domain must trigger PM4 with a rationale
    that includes 'within' (i.e. 'within a known functional domain')."""
    feats = _feats(is_inframe_indel=True, in_critical_domain=True)
    result = evaluate_acmg_rules(feats, 0.5)
    assert "PM4" in result, f"PM4 missing: {result}"
    assert "within" in result["PM4"].lower(), (
        f"Expected 'within' in PM4 rationale, got: {result['PM4']!r}"
    )


def test_pm4_inframe_outside_domain_rationale():
    """In-frame indel outside a critical domain must trigger PM4 with 'reduced
    weight' in the rationale string."""
    feats = _feats(is_inframe_indel=True, in_critical_domain=False)
    result = evaluate_acmg_rules(feats, 0.5)
    assert "PM4" in result, f"PM4 missing: {result}"
    assert "reduced weight" in result["PM4"].lower(), (
        f"Expected 'reduced weight' in PM4 rationale, got: {result['PM4']!r}"
    )


def test_pm4_not_fired_for_missense():
    """A plain missense (is_inframe_indel=False) must NOT trigger PM4."""
    feats = _feats(is_inframe_indel=False)
    result = evaluate_acmg_rules(feats, 0.5)
    assert "PM4" not in result, f"Unexpected PM4: {result}"


# ─── BP7 synonymous suppression ───────────────────────────────────────────────


def test_bp7_synonymous_no_splice():
    """Synonymous variant far from any splice site should trigger BP7."""
    feats = _feats(is_synonymous=True, near_splice=False, canonical_splice=False)
    result = evaluate_acmg_rules(feats, 0.5)
    assert "BP7" in result, f"BP7 missing: {result}"


def test_bp7_not_triggered_near_splice():
    """Synonymous variant near a splice site must NOT trigger BP7 (potential
    splicing impact cannot be ruled out)."""
    feats = _feats(is_synonymous=True, near_splice=True, canonical_splice=False)
    result = evaluate_acmg_rules(feats, 0.5)
    assert "BP7" not in result, f"Unexpected BP7: {result}"


def test_bp7_not_triggered_at_canonical_splice():
    """Synonymous variant at a canonical splice site must NOT trigger BP7."""
    feats = _feats(is_synonymous=True, near_splice=False, canonical_splice=True)
    result = evaluate_acmg_rules(feats, 0.5)
    assert "BP7" not in result, f"Unexpected BP7: {result}"


def test_bp7_not_triggered_for_non_synonymous():
    """A non-synonymous variant (is_synonymous=False) must NOT trigger BP7."""
    feats = _feats(is_synonymous=False, near_splice=False, canonical_splice=False)
    result = evaluate_acmg_rules(feats, 0.5)
    assert "BP7" not in result, f"Unexpected BP7: {result}"


# ─── PM5 known pathogenic at position ────────────────────────────────────────


def test_pm5_known_pathogenic_at_pos():
    """PM5 must fire when known_pathogenic_at_pos is True."""
    feats = _feats(known_pathogenic_at_pos=True)
    result = evaluate_acmg_rules(feats, 0.5)
    assert "PM5" in result, f"PM5 missing: {result}"


def test_pm5_not_fired_when_flag_false():
    """PM5 must not fire when known_pathogenic_at_pos is False."""
    feats = _feats(known_pathogenic_at_pos=False)
    result = evaluate_acmg_rules(feats, 0.5)
    assert "PM5" not in result, f"Unexpected PM5: {result}"


def test_pm5_not_fired_when_flag_absent():
    """PM5 must not fire when known_pathogenic_at_pos key is absent entirely."""
    feats = {k: v for k, v in _BASE.items() if k != "known_pathogenic_at_pos"}
    result = evaluate_acmg_rules(feats, 0.5)
    assert "PM5" not in result, f"Unexpected PM5: {result}"


# ─── Combinatorial multi-code interactions ────────────────────────────────────


def test_pvs1_and_pp3_both_fire_for_nonsense_high_prob():
    """A high-confidence nonsense variant should fire both PVS1 (variant type)
    and PP3 (computational) simultaneously."""
    feats = _feats(is_nonsense=True)
    result = evaluate_acmg_rules(feats, 0.95, gene_name="BRCA2")
    assert "PVS1" in result, f"PVS1 missing: {result}"
    assert "PP3" in result, f"PP3 missing: {result}"


def test_ba1_and_pvs1_can_coexist():
    """A very common nonsense variant should trigger both BA1 (population frequency)
    and PVS1 (variant type) — the rules are independent."""
    feats = _feats(is_nonsense=True, gnomad_af=0.05)
    result = evaluate_acmg_rules(feats, 0.5, gene_name="BRCA2")
    assert "BA1" in result, f"BA1 missing: {result}"
    assert "PVS1" in result, f"PVS1 missing: {result}"
    assert "BS1" not in result, f"BS1 should be suppressed by BA1: {result}"


def test_empty_features_returns_dict_without_crash():
    """Passing an empty dict must not raise an exception and must return a dict."""
    result = evaluate_acmg_rules({}, 0.5)
    assert isinstance(result, dict)


def test_all_flags_neutral_returns_only_pm2():
    """With all neutral feature values (including gnomad_af=0) and a mid-range
    probability, only PM2 should fire (since AF=0 means absent from population DB)."""
    feats = _feats()
    result = evaluate_acmg_rules(feats, 0.5, gene_name="BRCA2")
    assert result == {"PM2": "Variant absent from gnomAD population database (AF = 0)"}, (
        f"Expected only PM2, got: {result}"
    )


def test_no_pm2_when_gnomad_af_nonzero():
    """PM2 should not fire when gnomAD AF is > 0."""
    feats = _feats(gnomad_af=0.0001)
    result = evaluate_acmg_rules(feats, 0.5, gene_name="BRCA2")
    assert "PM2" not in result, f"PM2 should not fire when AF > 0, got: {result}"
