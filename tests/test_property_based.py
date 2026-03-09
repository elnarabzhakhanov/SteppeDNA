"""
Property-based tests using Hypothesis.
Tests invariants that must always hold regardless of input.

Each test uses @given to generate hundreds of random inputs and verifies
that structural guarantees in the ACMG rule engine never break.  The
approach finds edge cases that hand-crafted unit tests routinely miss,
such as floating-point boundary conditions and rare feature combinations.
"""
import os
import sys
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

try:
    from hypothesis import given, strategies as st, settings, HealthCheck
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

pytestmark = pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_GENES = ["BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D"]

# Every code the engine is allowed to emit.
# BP7 and PM4 are valid in addition to the core set because the engine emits
# them for synonymous and in-frame indel variants respectively.
VALID_ACMG_CODES = {
    "PVS1",
    "PVS1_moderate",
    "PM1",
    "PM2",
    "PM4",
    "PM5",
    "PS1",
    "PP3",
    "PP3_splice",
    "BA1",
    "BS1",
    "BP4",
    "BP7",
}

# Minimal "neutral" feature dict: rare variant, far from all structural sites.
_NEUTRAL = {"gnomad_af": 0.0, "dist_dna": 999.0, "dist_palb2": 999.0}


def _neutral(**overrides):
    """Return a copy of _NEUTRAL with any overrides applied."""
    d = dict(_NEUTRAL)
    d.update(overrides)
    return d


# ---------------------------------------------------------------------------
# Guard: only define @given-decorated tests when Hypothesis is importable.
# The decorators themselves (@given, @settings) are evaluated at module load
# time by Python, so placing them inside an `if HAS_HYPOTHESIS` block is the
# only way to avoid a NameError during collection when Hypothesis is absent.
# The pytestmark skip at the top of the file will still cause pytest to skip
# any test that reaches the call phase without Hypothesis.
# ---------------------------------------------------------------------------

if HAS_HYPOTHESIS:

    # -----------------------------------------------------------------------
    # 1. BA1 and BS1 never co-occur
    # -----------------------------------------------------------------------

    @given(
        gene=st.sampled_from(ALL_GENES),
        prob=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        af=st.floats(min_value=0.0, max_value=0.1, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_ba1_bs1_never_cooccur(gene, prob, af):
        """
        BA1 (stand-alone benign) and BS1 (strong benign) are mutually exclusive.

        BA1 supersedes BS1: the engine only evaluates BS1 when BA1 has not
        already been triggered.  This invariant must hold for every gene,
        probability and allele frequency combination.
        """
        from backend.acmg_rules import evaluate_acmg_rules
        feats = _neutral(gnomad_af=af)
        result = evaluate_acmg_rules(feats, prob, gene_name=gene)
        assert not ("BA1" in result and "BS1" in result), (
            f"BA1 and BS1 both triggered: gene={gene}, AF={af:.6f}, prob={prob:.4f}"
        )

    # -----------------------------------------------------------------------
    # 2. PP3 and BP4 never co-occur
    # -----------------------------------------------------------------------

    @given(prob=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    @settings(max_examples=200)
    def test_pp3_bp4_never_cooccur(prob):
        """
        PP3 (computational evidence of pathogenicity) and BP4 (computational
        evidence of benign effect) represent opposite conclusions and must never
        both appear in the same result.

        PP3 requires prob >= 0.70; BP4 requires prob <= 0.20.  No single float
        can satisfy both conditions simultaneously.
        """
        from backend.acmg_rules import evaluate_acmg_rules
        feats = _neutral()
        result = evaluate_acmg_rules(feats, prob)
        assert not ("PP3" in result and "BP4" in result), (
            f"PP3 and BP4 both triggered at prob={prob:.6f}"
        )

    # -----------------------------------------------------------------------
    # 3. PP3 and BP4 never co-occur — across all genes and AF values
    # -----------------------------------------------------------------------

    @given(
        prob=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        gene=st.sampled_from(ALL_GENES),
        af=st.floats(min_value=0.0, max_value=0.1, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_pp3_bp4_never_cooccur_any_gene(prob, gene, af):
        """
        The PP3/BP4 mutual-exclusion invariant must hold regardless of which
        gene is selected or what allele frequency is supplied.  Gene-specific
        thresholds only affect BA1/BS1, not the computational prediction codes,
        but this test confirms the isolation holds end-to-end.
        """
        from backend.acmg_rules import evaluate_acmg_rules
        feats = _neutral(gnomad_af=af)
        result = evaluate_acmg_rules(feats, prob, gene_name=gene)
        assert not ("PP3" in result and "BP4" in result), (
            f"PP3 and BP4 both triggered: gene={gene}, prob={prob:.6f}, AF={af:.6f}"
        )

    # -----------------------------------------------------------------------
    # 4. All emitted ACMG codes belong to the known valid set
    # -----------------------------------------------------------------------

    @given(
        prob=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        af=st.floats(min_value=0.0, max_value=0.1, allow_nan=False),
        gene=st.sampled_from(ALL_GENES),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_acmg_codes_from_known_set(prob, af, gene):
        """
        The engine must only emit codes that are part of the approved ACMG/AMP
        vocabulary.  Any novel or misspelled code key is a programming error
        that would silently corrupt downstream classification logic.
        """
        from backend.acmg_rules import evaluate_acmg_rules
        feats = _neutral(gnomad_af=af)
        result = evaluate_acmg_rules(feats, prob, gene_name=gene)
        for code in result:
            assert code in VALID_ACMG_CODES, (
                f"Unknown ACMG code '{code}' emitted for gene={gene}, "
                f"prob={prob:.4f}, AF={af:.6f}"
            )

    # -----------------------------------------------------------------------
    # 5. All emitted codes are valid — full boolean feature space
    # -----------------------------------------------------------------------

    @given(
        prob=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        af=st.floats(min_value=0.0, max_value=0.1, allow_nan=False),
        gene=st.sampled_from(ALL_GENES),
        is_nonsense=st.booleans(),
        is_frameshift=st.booleans(),
        is_inframe_indel=st.booleans(),
        is_synonymous=st.booleans(),
        near_splice=st.booleans(),
        canonical_splice=st.booleans(),
        in_critical_domain=st.booleans(),
        known_pathogenic_at_pos=st.booleans(),
    )
    @settings(max_examples=300)
    def test_acmg_codes_from_known_set_full_features(
        prob, af, gene, is_nonsense, is_frameshift, is_inframe_indel,
        is_synonymous, near_splice, canonical_splice, in_critical_domain,
        known_pathogenic_at_pos,
    ):
        """
        With every boolean feature flag randomised, the engine must still only
        produce codes from the approved vocabulary.  This exercises all code
        paths including PVS1, PM4, BP7 and PP3_splice which are absent from
        neutral-only tests.
        """
        from backend.acmg_rules import evaluate_acmg_rules
        feats = _neutral(
            gnomad_af=af,
            is_nonsense=is_nonsense,
            is_frameshift=is_frameshift,
            is_inframe_indel=is_inframe_indel,
            is_synonymous=is_synonymous,
            near_splice=near_splice,
            canonical_splice=canonical_splice,
            in_critical_domain=in_critical_domain,
            known_pathogenic_at_pos=known_pathogenic_at_pos,
        )
        result = evaluate_acmg_rules(feats, prob, gene_name=gene)
        for code in result:
            assert code in VALID_ACMG_CODES, (
                f"Unknown ACMG code '{code}' emitted with full feature set"
            )

    # -----------------------------------------------------------------------
    # 6. All rationale strings are non-empty
    # -----------------------------------------------------------------------

    @given(
        prob=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        af=st.floats(min_value=0.0, max_value=0.1, allow_nan=False),
        in_critical_domain=st.booleans(),
        dist_dna=st.floats(min_value=0.0, max_value=999.0, allow_nan=False),
    )
    @settings(max_examples=150)
    def test_acmg_rationale_non_empty(prob, af, in_critical_domain, dist_dna):
        """
        Every emitted ACMG code must map to a non-empty string rationale.
        Empty strings or non-string values would break the frontend rendering
        and any downstream PDF/CSV export logic.
        """
        from backend.acmg_rules import evaluate_acmg_rules
        feats = _neutral(
            gnomad_af=af,
            dist_dna=dist_dna,
            in_critical_domain=in_critical_domain,
        )
        result = evaluate_acmg_rules(feats, prob)
        for code, rationale in result.items():
            assert isinstance(rationale, str), (
                f"Rationale for {code} is not a string: {type(rationale)}"
            )
            assert len(rationale.strip()) > 0, (
                f"Empty rationale for code '{code}' "
                f"(prob={prob:.4f}, AF={af:.6f})"
            )

    # -----------------------------------------------------------------------
    # 7. PVS1 rationale always names the triggering mechanism
    # -----------------------------------------------------------------------

    @given(
        is_nonsense=st.booleans(),
        is_frameshift=st.booleans(),
        canonical_splice=st.booleans(),
    )
    @settings(max_examples=100)
    def test_pvs1_rationale_matches_mechanism(is_nonsense, is_frameshift, canonical_splice):
        """
        When PVS1 fires, its rationale string must mention the specific
        molecular mechanism that caused it (Nonsense, Frameshift, or splice).
        A generic or mismatched rationale would mislead clinical reviewers.

        Priority in the engine: nonsense/frameshift are evaluated first;
        canonical_splice only sets PVS1 if neither null-variant flag is set.
        """
        from backend.acmg_rules import evaluate_acmg_rules
        feats = _neutral(
            is_nonsense=is_nonsense,
            is_frameshift=is_frameshift,
            canonical_splice=canonical_splice,
        )
        result = evaluate_acmg_rules(feats, 0.95)
        if "PVS1" not in result:
            return  # Nothing to check if PVS1 did not fire
        rationale = result["PVS1"].lower()
        if is_nonsense or is_frameshift:
            # Null-variant block fired first
            assert "nonsense" in rationale or "frameshift" in rationale, (
                f"PVS1 rationale does not name null variant: '{result['PVS1']}'"
            )
        elif canonical_splice:
            # Only the splice path remains
            assert "splice" in rationale, (
                f"PVS1 rationale does not mention splice: '{result['PVS1']}'"
            )

    # -----------------------------------------------------------------------
    # 8. BA1 always fires when AF strictly exceeds the gene-specific threshold
    # -----------------------------------------------------------------------

    @given(gene=st.sampled_from(ALL_GENES))
    @settings(max_examples=50)
    def test_ba1_fires_above_threshold(gene):
        """
        For every supported gene, an allele frequency that strictly exceeds the
        published BA1 threshold must trigger BA1.  This validates that the
        gene-specific lookup in GENE_BA1_THRESHOLDS is wired correctly and does
        not accidentally fall back to the wrong gene's threshold.
        """
        from backend.acmg_rules import evaluate_acmg_rules, GENE_BA1_THRESHOLDS
        threshold = GENE_BA1_THRESHOLDS[gene]
        af_above = threshold + 0.001  # safely above the threshold
        feats = _neutral(gnomad_af=af_above)
        result = evaluate_acmg_rules(feats, 0.5, gene_name=gene)
        assert "BA1" in result, (
            f"BA1 did not fire for {gene} at AF={af_above:.4f} "
            f"(threshold={threshold})"
        )
        assert "BS1" not in result, (
            f"BS1 also fired alongside BA1 for {gene}"
        )

    # -----------------------------------------------------------------------
    # 9. BS1 absent at or below the gene-specific BS1 threshold
    # -----------------------------------------------------------------------

    @given(gene=st.sampled_from(ALL_GENES))
    @settings(max_examples=50)
    def test_bs1_absent_at_or_below_threshold(gene):
        """
        The BS1 criterion requires AF to *strictly exceed* the threshold.  At
        exactly the threshold value BS1 must not fire.  This boundary condition
        is clinically significant: erroneously flagging a variant as BS1 could
        suppress a correct pathogenic classification.
        """
        from backend.acmg_rules import evaluate_acmg_rules, GENE_BS1_THRESHOLDS
        threshold = GENE_BS1_THRESHOLDS[gene]
        feats_at = _neutral(gnomad_af=threshold)
        result_at = evaluate_acmg_rules(feats_at, 0.5, gene_name=gene)
        assert "BS1" not in result_at, (
            f"BS1 fired at the threshold boundary for {gene}: AF={threshold}"
        )

        feats_below = _neutral(gnomad_af=max(0.0, threshold - 0.0001))
        result_below = evaluate_acmg_rules(feats_below, 0.5, gene_name=gene)
        assert "BS1" not in result_below, (
            f"BS1 fired below the threshold for {gene}: "
            f"AF={max(0.0, threshold - 0.0001):.6f}"
        )

    # -----------------------------------------------------------------------
    # 10. evaluate_acmg_rules always returns a dict and never raises
    # -----------------------------------------------------------------------

    @given(
        prob=st.floats(allow_nan=False, allow_infinity=False),
        af=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        gene=st.sampled_from(ALL_GENES),
        is_nonsense=st.booleans(),
        is_frameshift=st.booleans(),
    )
    @settings(max_examples=200)
    def test_evaluate_never_raises(prob, af, gene, is_nonsense, is_frameshift):
        """
        The engine must be robust to any valid (non-NaN, non-inf) float
        probability and must always return a dict rather than raising an
        exception.  Callers such as the /predict endpoint do not wrap this call
        in a try/except, so an unhandled exception here would propagate
        directly to the user as an HTTP 500 error.
        """
        from backend.acmg_rules import evaluate_acmg_rules
        feats = _neutral(
            gnomad_af=af,
            is_nonsense=is_nonsense,
            is_frameshift=is_frameshift,
        )
        try:
            result = evaluate_acmg_rules(feats, prob, gene_name=gene)
        except Exception as exc:
            pytest.fail(
                f"evaluate_acmg_rules raised {type(exc).__name__}: {exc} "
                f"(prob={prob}, gene={gene}, AF={af})"
            )
        assert isinstance(result, dict), "Return value must be a dict"

    # -----------------------------------------------------------------------
    # 11. PP3 fires if and only if prob >= 0.70 (neutral feature set)
    # -----------------------------------------------------------------------

    @given(prob=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    @settings(max_examples=300)
    def test_pp3_threshold_invariant(prob):
        """
        With a neutral feature set (no structural proximity, no population
        frequency signal), PP3 must fire if and only if the probability is at
        or above 0.70.  This confirms the threshold is correctly implemented
        and that no incidental feature interaction silently suppresses or
        triggers PP3.
        """
        from backend.acmg_rules import evaluate_acmg_rules
        feats = _neutral()
        result = evaluate_acmg_rules(feats, prob)
        if prob >= 0.70:
            assert "PP3" in result, (
                f"PP3 absent at prob={prob:.6f} "
                f"(expected >= 0.70 to trigger PP3)"
            )
        else:
            assert "PP3" not in result, (
                f"PP3 present at prob={prob:.6f} "
                f"(expected < 0.70 to suppress PP3)"
            )

    # -----------------------------------------------------------------------
    # 12. BP4 fires if and only if prob <= 0.20 (neutral feature set)
    # -----------------------------------------------------------------------

    @given(prob=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    @settings(max_examples=300)
    def test_bp4_threshold_invariant(prob):
        """
        With a neutral feature set, BP4 must fire if and only if the
        probability is at or below 0.20.  This is the mirror invariant of
        test_pp3_threshold_invariant and together they guarantee complete and
        non-overlapping coverage of the probability axis for computational
        prediction codes.
        """
        from backend.acmg_rules import evaluate_acmg_rules
        feats = _neutral()
        result = evaluate_acmg_rules(feats, prob)
        if prob <= 0.20:
            assert "BP4" in result, (
                f"BP4 absent at prob={prob:.6f} "
                f"(expected <= 0.20 to trigger BP4)"
            )
        else:
            assert "BP4" not in result, (
                f"BP4 present at prob={prob:.6f} "
                f"(expected > 0.20 to suppress BP4)"
            )
