"""
ACMG/AMP Clinical Rule Engine

Evaluates missense variants against ACMG criteria using
computed structural features and model predictions.

Evidence Codes Evaluated:
- PM1 (Moderate): Located in a critical functional domain
- PM5 (Moderate): Novel missense at a position with a known pathogenic missense
- PP3 (Supporting): Computational evidence supports deleterious effect
- BP4 (Supporting): Computational evidence suggests no impact
- BS1 (Strong): Allele frequency too high for disorder
- BA1 (Stand-alone): Allele frequency above population threshold (benign)
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
                  in_critical_domain, known_pathogenic_at_pos keys
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
        pm1_reasons.append(f"Located inside a Critical Functional Repeat or Domain")

    if pm1_reasons:
        met_codes['PM1'] = "Structural Disruption: " + " & ".join(pm1_reasons)

    # PM5: Novel missense at a position where a different pathogenic missense is known
    # This requires ClinVar data — if the caller provides known_pathogenic_at_pos flag
    if features.get('known_pathogenic_at_pos', False):
        met_codes['PM5'] = "Different pathogenic missense variant previously established at this amino acid position"

    # BA1: Stand-alone benign — AF above population threshold
    af = features.get('gnomad_af', 0.0)
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
    is_nonsense = features.get('is_nonsense', False)
    is_frameshift = features.get('is_frameshift', False)
    if is_nonsense or is_frameshift:
        variant_type = "Nonsense" if is_nonsense else "Frameshift"
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
    if canonical_splice:
        met_codes['PVS1'] = "Canonical splice site variant disrupting mRNA splicing"
    elif near_splice:
        met_codes['PP3_splice'] = "Near splice site variant with potential splicing impact"

    # BP7: Synonymous variant with no predicted splice impact
    is_synonymous = features.get('is_synonymous', False)
    if is_synonymous and not near_splice and not canonical_splice:
        met_codes['BP7'] = "Synonymous variant with no predicted splice impact"

    # PP3/BP4: computational prediction evidence
    if model_prediction >= 0.90:
        met_codes['PP3'] = f"Computational model strongly predicts pathogenicity (p={model_prediction:.3f})"
    elif model_prediction <= 0.10:
        met_codes['BP4'] = f"Computational model strongly predicts benign (p={model_prediction:.3f})"

    return met_codes
