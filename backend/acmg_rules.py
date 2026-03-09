"""
ACMG/AMP Clinical Rule Engine

Evaluates missense variants against ACMG criteria using
computed structural features and model predictions.

Evidence Codes Evaluated:
- PVS1 (Very Strong): Null variant in a gene with known LOF mechanism (with last-exon modifier)
- PS1 (Strong): Same amino acid change as an established pathogenic variant
- PM1 (Moderate): Located in a critical functional domain
- PM2 (Moderate): Absent from population databases (gnomAD AF = 0)
- PM4 (Moderate): In-frame indel in a non-repetitive region
- PM5 (Moderate): Novel missense at a position with a known pathogenic missense
- PP3 (Supporting): Computational evidence supports deleterious effect
- BP4 (Supporting): Computational evidence suggests no impact
- BP7 (Supporting): Synonymous with no splice impact
- BS1 (Strong): Allele frequency too high for disorder
- BA1 (Stand-alone): Allele frequency above population threshold (benign)

Note: PS2/PM6 (de novo), PP1/BS4 (co-segregation), PS3/BS3 (functional studies),
PS4 (prevalence), PP2/BP1 (gene mechanism) require data not available to this tool.
"""

# Gene-specific BS1 AF thresholds based on ClinGen SVI recommendations
# High-penetrance genes (BRCA1/2) use stricter thresholds
GENE_BS1_THRESHOLDS = {
    "BRCA1": 0.001,
    "BRCA2": 0.001,
    "PALB2": 0.002,
    "RAD51C": 0.005,
    "RAD51D": 0.005,
}

# BA1 stand-alone benign threshold (ClinGen SVI: AF > 0.05 for dominant disorders)
# For AR cancer predisposition genes, a lower threshold is appropriate
GENE_BA1_THRESHOLDS = {
    "BRCA1": 0.01,
    "BRCA2": 0.01,
    "PALB2": 0.01,
    "RAD51C": 0.01,
    "RAD51D": 0.01,
}


def evaluate_acmg_rules(features, model_prediction, gene_name="BRCA2"):
    """
    Evaluate ACMG criteria for a variant.

    Args:
        features: dict with domain, dist_dna, dist_palb2, gnomad_af,
                  in_critical_domain, known_pathogenic_at_pos,
                  exon_number, total_exons (optional for PVS1 modifier) keys
        model_prediction: calibrated probability of pathogenicity (0.0 to 1.0)
        gene_name: gene name for gene-specific thresholds

    Returns:
        dict mapping ACMG codes to human-readable rationale strings
    """
    met_codes = {}

    # PM1: structural proximity to functional sites (5A threshold)
    domain = features.get('domain', 'uncharacterized')
    dist_dna = features.get('dist_dna', 999.0)
    dist_palb2 = features.get('dist_palb2', 999.0)

    pm1_reasons = []
    if dist_dna <= 5.0 and dist_dna != 999.0:
        pm1_reasons.append(f"Located within {dist_dna}A of DNA binding interface")
    if dist_palb2 <= 5.0 and dist_palb2 != 999.0:
        pm1_reasons.append(f"Located within {dist_palb2}A of Primary Interaction interface")
    if features.get('in_critical_domain', False):
        pm1_reasons.append("Located inside a Critical Functional Repeat or Domain")

    if pm1_reasons:
        met_codes['PM1'] = "Structural Disruption: " + " & ".join(pm1_reasons)

    # PS1: Same amino acid change as an established pathogenic variant
    # Distinct from PM5 — PS1 requires the exact same AA substitution, PM5 requires
    # a different pathogenic missense at the same position
    if features.get('same_aa_change_pathogenic', False):
        met_codes['PS1'] = "Same amino acid change as a previously established pathogenic variant"

    # PM5: Novel missense at a position where a different pathogenic missense is known
    # Only fire PM5 if PS1 was not triggered (PS1 supersedes PM5)
    if 'PS1' not in met_codes and features.get('known_pathogenic_at_pos', False):
        met_codes['PM5'] = "Different pathogenic missense variant previously established at this amino acid position"

    # PM2: Absent from population databases
    af = features.get('gnomad_af', 0.0)
    if af == 0.0:
        met_codes['PM2'] = "Variant absent from gnomAD population database (AF = 0)"

    # BA1: Stand-alone benign — AF above population threshold
    ba1_threshold = GENE_BA1_THRESHOLDS.get(gene_name.upper(), 0.01)
    if af > ba1_threshold:
        met_codes['BA1'] = f"Allele frequency ({af*100:.3f}%) exceeds stand-alone benign threshold ({ba1_threshold*100:.1f}%) for {gene_name}"

    # BS1: common variant in gnomAD (gene-specific AF threshold)
    # Only apply BS1 if BA1 was not triggered (BA1 supersedes BS1)
    if 'BA1' not in met_codes:
        bs1_threshold = GENE_BS1_THRESHOLDS.get(gene_name.upper(), 0.001)
        if af > bs1_threshold:
            met_codes['BS1'] = f"Population frequency ({af*100:.4f}%) exceeds threshold ({bs1_threshold*100:.3f}%) for {gene_name}"

    # PVS1: Null variant (nonsense, frameshift) in a gene where LOF is a known mechanism
    # With last-exon/NMD modifier per ACMG guidelines
    is_nonsense = features.get('is_nonsense', False)
    is_frameshift = features.get('is_frameshift', False)
    if is_nonsense or is_frameshift:
        variant_type = "Nonsense" if is_nonsense else "Frameshift"
        exon_number = features.get('exon_number')
        total_exons = features.get('total_exons')

        # Check last-exon rule: nonsense/frameshift in the last exon or
        # last 50 nucleotides of the penultimate exon may escape NMD
        if exon_number is not None and total_exons is not None and total_exons > 1:
            if exon_number == total_exons:
                met_codes['PVS1_moderate'] = (
                    f"{variant_type} variant in last exon (exon {exon_number}/{total_exons}) — "
                    f"may escape nonsense-mediated decay. PVS1 downgraded to moderate."
                )
            elif exon_number == total_exons - 1 and features.get('near_last_exon_boundary', False):
                met_codes['PVS1_moderate'] = (
                    f"{variant_type} variant in last 50nt of penultimate exon (exon {exon_number}/{total_exons}) — "
                    f"may escape nonsense-mediated decay. PVS1 downgraded to moderate."
                )
            else:
                met_codes['PVS1'] = f"Pathogenic truncating null variant ({variant_type}) in a recognized tumor suppressor"
        else:
            # No exon info available — apply PVS1 at full strength with caveat
            met_codes['PVS1'] = f"Pathogenic truncating null variant ({variant_type}) in a recognized tumor suppressor"

    # PM4: In-frame indel — stronger if within a known functional domain
    is_inframe_indel = features.get('is_inframe_indel', False)
    if is_inframe_indel:
        if features.get('in_critical_domain', False):
            met_codes['PM4'] = "In-frame deletion/insertion within a known functional domain"
        else:
            met_codes['PM4'] = "In-frame deletion/insertion outside known functional domain (reduced weight)"

    # PP3_splice: Near splice site variant with potential splicing impact
    near_splice = features.get('near_splice', False)
    canonical_splice = features.get('canonical_splice', False)
    if canonical_splice and 'PVS1' not in met_codes and 'PVS1_moderate' not in met_codes:
        met_codes['PVS1'] = "Canonical splice site variant disrupting mRNA splicing"
    elif near_splice:
        met_codes['PP3_splice'] = "Near splice site variant with potential splicing impact"

    # BP7: Synonymous variant with no predicted splice impact
    is_synonymous = features.get('is_synonymous', False)
    if is_synonymous and not near_splice and not canonical_splice:
        met_codes['BP7'] = "Synonymous variant with no predicted splice impact"

    # PP3/BP4: computational prediction evidence
    # Thresholds aligned with ClinGen SVI recommendations (lowered from 0.90/0.10)
    if model_prediction >= 0.70:
        met_codes['PP3'] = f"Computational model predicts pathogenicity (p={model_prediction:.3f})"
    elif model_prediction <= 0.20:
        met_codes['BP4'] = f"Computational model predicts benign (p={model_prediction:.3f})"

    return met_codes


# ─── ACMG Evidence Combining Rules (Richards et al. 2015, Table 5) ──────────

# Evidence strength hierarchy
_PATHOGENIC_VERY_STRONG = {"PVS1"}
_PATHOGENIC_STRONG = {"PS1", "PS2", "PS3", "PS4"}
_PATHOGENIC_MODERATE = {"PM1", "PM2", "PM3", "PM4", "PM5", "PM6", "PVS1_moderate"}
_PATHOGENIC_SUPPORTING = {"PP1", "PP2", "PP3", "PP3_splice", "PP4", "PP5"}
_BENIGN_STANDALONE = {"BA1"}
_BENIGN_STRONG = {"BS1", "BS2", "BS3", "BS4"}
_BENIGN_SUPPORTING = {"BP1", "BP2", "BP3", "BP4", "BP5", "BP6", "BP7"}


def combine_acmg_evidence(met_codes: dict) -> str:
    """
    Combine ACMG evidence codes into a 5-tier classification per Richards et al. 2015 Table 5.

    Args:
        met_codes: dict mapping ACMG codes to rationale strings (output of evaluate_acmg_rules)

    Returns:
        One of: "Pathogenic", "Likely Pathogenic", "Uncertain Significance",
                "Likely Benign", "Benign"
    """
    codes = set(met_codes.keys())

    # Count evidence by strength
    pvs = len(codes & _PATHOGENIC_VERY_STRONG)
    ps = len(codes & _PATHOGENIC_STRONG)
    pm = len(codes & _PATHOGENIC_MODERATE)
    pp = len(codes & _PATHOGENIC_SUPPORTING)
    ba = len(codes & _BENIGN_STANDALONE)
    bs = len(codes & _BENIGN_STRONG)
    bp = len(codes & _BENIGN_SUPPORTING)

    # ─── BENIGN (BA1 is stand-alone) ─────────────────────────────────────────
    if ba >= 1:
        return "Benign"

    # ─── BENIGN (2 strong benign) ────────────────────────────────────────────
    if bs >= 2:
        return "Benign"

    # ─── LIKELY BENIGN ───────────────────────────────────────────────────────
    # 1 Strong + 1 Supporting
    if bs >= 1 and bp >= 1:
        return "Likely Benign"

    # ─── PATHOGENIC ──────────────────────────────────────────────────────────
    # Rule i:   1 Very Strong + ≥1 Strong
    if pvs >= 1 and ps >= 1:
        return "Pathogenic"
    # Rule ii:  1 Very Strong + ≥2 Moderate
    if pvs >= 1 and pm >= 2:
        return "Pathogenic"
    # Rule iii: 1 Very Strong + 1 Moderate + 1 Supporting
    if pvs >= 1 and pm >= 1 and pp >= 1:
        return "Pathogenic"
    # Rule iv:  1 Very Strong + ≥2 Supporting
    if pvs >= 1 and pp >= 2:
        return "Pathogenic"
    # Rule v:   ≥2 Strong
    if ps >= 2:
        return "Pathogenic"
    # Rule vi:  1 Strong + ≥3 Moderate
    if ps >= 1 and pm >= 3:
        return "Pathogenic"
    # Rule vii: 1 Strong + 2 Moderate + ≥2 Supporting
    if ps >= 1 and pm >= 2 and pp >= 2:
        return "Pathogenic"
    # Rule viii: 1 Strong + 1 Moderate + ≥4 Supporting
    if ps >= 1 and pm >= 1 and pp >= 4:
        return "Pathogenic"

    # ─── LIKELY PATHOGENIC ───────────────────────────────────────────────────
    # Rule i:  1 Very Strong + 1 Moderate
    if pvs >= 1 and pm >= 1:
        return "Likely Pathogenic"
    # Rule ii: 1 Strong + 1-2 Moderate
    if ps >= 1 and 1 <= pm <= 2:
        return "Likely Pathogenic"
    # Rule iii: 1 Strong + ≥2 Supporting
    if ps >= 1 and pp >= 2:
        return "Likely Pathogenic"
    # Rule iv:  ≥3 Moderate
    if pm >= 3:
        return "Likely Pathogenic"
    # Rule v:   2 Moderate + ≥2 Supporting
    if pm >= 2 and pp >= 2:
        return "Likely Pathogenic"
    # Rule vi:  1 Moderate + ≥4 Supporting
    if pm >= 1 and pp >= 4:
        return "Likely Pathogenic"

    # ─── DEFAULT: UNCERTAIN SIGNIFICANCE ─────────────────────────────────────
    return "Uncertain Significance"
