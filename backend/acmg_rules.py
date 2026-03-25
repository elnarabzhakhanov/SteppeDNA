"""
ACMG/AMP Clinical Rule Engine
==============================

This module classifies DNA variants (mutations) as disease-causing or harmless
using the ACMG/AMP framework — the gold-standard method used by clinical
genetics labs worldwide.

WHAT IS ACMG?
    ACMG stands for the American College of Medical Genetics and Genomics.
    In 2015, Richards et al. published guidelines that define a systematic way
    to classify genetic variants into one of five categories:

        1. Pathogenic          — disease-causing (high confidence)
        2. Likely Pathogenic   — probably disease-causing (>90% certainty)
        3. Uncertain Significance (VUS) — not enough evidence either way
        4. Likely Benign       — probably harmless (>90% certainty)
        5. Benign              — harmless (high confidence)

HOW DOES IT WORK?
    The framework defines ~28 evidence codes, each representing a different
    type of evidence for or against pathogenicity. Each code has a strength
    level (very strong, strong, moderate, or supporting). Think of it like a
    scoring system:

        - Pathogenic evidence codes start with "P" (PVS1, PS1, PM1, PP3, ...)
        - Benign evidence codes start with "B" (BA1, BS1, BP4, BP7, ...)

    The second letter indicates strength:
        P_V_S = Very Strong, P_S = Strong, P_M = Moderate, P_P = suPPorting
        B_A = stAnd-alone, B_S = Strong, B_P = suPPorting

    After collecting all triggered codes for a variant, we combine them using
    specific rules (from Table 5 of Richards et al. 2015) to arrive at one
    of the five classifications above.

WHAT THIS MODULE EVALUATES:
    We can only evaluate codes where we have the necessary data. Some codes
    (like PS2 = de novo occurrence) require family studies we don't have.

    Evidence Codes Evaluated:
    - PVS1 (Very Strong): The variant truncates (cuts short) the protein.
          This is the strongest single piece of pathogenic evidence. It fires
          for nonsense mutations (which create a premature stop codon) and
          frameshift mutations (which scramble the reading frame). Has a
          "last exon" modifier — see below.
    - PS1  (Strong): The exact same amino acid change has already been
          confirmed pathogenic in clinical databases like ClinVar.
    - PM1  (Moderate): The variant sits in a critical functional domain —
          a region of the protein that does something important (e.g., binds
          DNA or interacts with a partner protein like PALB2).
    - PM2  (Moderate): The variant has never been seen in large population
          databases (gnomAD). If it were harmless, we'd expect to see it in
          healthy people.
    - PM4  (Moderate): An in-frame insertion or deletion (the protein gets
          shorter or longer by a few amino acids without scrambling the rest).
    - PM5  (Moderate): A *different* pathogenic missense variant has been
          found at the same protein position. This suggests the position is
          important, even though this specific change hasn't been seen before.
    - PP3  (Supporting): Our ML model predicts the variant is damaging.
    - BP4  (Supporting): Our ML model predicts the variant is benign.
    - BP7  (Supporting): The variant is synonymous (doesn't change the amino
          acid) and doesn't affect mRNA splicing.
    - BS1  (Strong benign): The variant is found at a frequency in the
          population that's too high for a rare disease-causing mutation.
    - BA1  (Stand-alone benign): The variant is so common in the population
          that it's almost certainly harmless — this single piece of evidence
          is enough to classify it as Benign, overriding everything else.

    NOT evaluated (require external data we don't have):
    - PS2/PM6 (de novo), PP1/BS4 (co-segregation), PS3/BS3 (functional assays),
      PP2/BP1 (gene-level missense constraint).

    Conditionally evaluated:
    - PS4 (case-control prevalence): triggered for known Kazakh founder mutations
      with published population frequency data.

KEY HIERARCHIES:
    Some codes override or supersede others:
    - BA1 overrides BS1: If a variant is common enough to trigger BA1 (the
      higher threshold), we don't also flag BS1 (the lower threshold). BA1
      alone classifies as Benign.
    - PS1 supersedes PM5: If the exact same amino acid change is known
      pathogenic (PS1), there's no point also saying "a different pathogenic
      change exists at this position" (PM5). PS1 is stronger evidence.

PVS1 LAST-EXON MODIFIER:
    Nonsense and frameshift variants normally trigger PVS1 at full (very
    strong) strength. However, if the variant is in the last exon of the
    gene, the truncated mRNA may escape "nonsense-mediated decay" (NMD) —
    the cell's quality-control system that destroys faulty mRNAs. If it
    escapes NMD, a truncated but partially functional protein might still
    be made, so the evidence is weaker. In this case, PVS1 is downgraded
    from "Very Strong" to "Moderate" (stored as PVS1_moderate).
"""

# ---------------------------------------------------------------------------
# POPULATION FREQUENCY THRESHOLDS (BS1 and BA1)
# ---------------------------------------------------------------------------
# These thresholds define how common a variant can be in the general population
# before we consider it too common to cause a rare disease.
#
# WHY gene-specific? Different genes cause diseases with different prevalence.
# BRCA1/BRCA2 mutations cause ~1 in 400 cancers, so disease-causing variants
# should be very rare (low thresholds). RAD51C/D are rarer causes, so we
# allow slightly higher frequencies before calling something benign.
#
# The key rule: BS1 threshold < BA1 threshold (always).
# BS1 = "strong benign evidence" (needs other supporting evidence to classify)
# BA1 = "stand-alone benign" (this alone is enough to call it Benign)

# Gene-specific BS1 AF thresholds based on ClinGen SVI recommendations.
# BS1 = "strong benign" � AF above this threshold provides strong evidence.
# Must be LOWER than BA1 (BA1 is stand-alone, BS1 needs corroboration).
GENE_BS1_THRESHOLDS = {
    "BRCA1": 0.0001,  # ClinGen SVI: 0.01% (BA1=0.1%)
    "BRCA2": 0.0001,  # ClinGen SVI: 0.01% (BA1=0.1%)
    "PALB2": 0.0005,  # 0.05% (BA1=0.2%)
    "RAD51C": 0.001,   # 0.1% (BA1=0.5%)
    "RAD51D": 0.001,   # 0.1% (BA1=0.5%)
}

# BA1 stand-alone benign threshold — must be HIGHER than BS1 for each gene.
# BA1 = "benign, stand-alone" (stronger evidence) vs BS1 = "strong benign" (needs corroboration).
# ClinGen SVI recommends ~0.001 for high-penetrance cancer genes (BRCA1/2).
GENE_BA1_THRESHOLDS = {
    "BRCA1": 0.001,   # ClinGen SVI high-penetrance (was 0.005)
    "BRCA2": 0.001,   # ClinGen SVI high-penetrance (was 0.005)
    "PALB2": 0.002,   # Moderate penetrance (was 0.005)
    "RAD51C": 0.005,  # Lower penetrance (was 0.01)
    "RAD51D": 0.005,  # Lower penetrance (was 0.01)
}

# ---------------------------------------------------------------------------
# COMPUTATIONAL AND STRUCTURAL THRESHOLDS
# ---------------------------------------------------------------------------
# These thresholds control when our ML model's prediction and structural
# data trigger ACMG evidence codes.

# PP3/BP4: Our ML model outputs a probability between 0.0 (benign) and
# 1.0 (pathogenic). If the probability is high enough, it triggers PP3
# (supporting pathogenic). If low enough, it triggers BP4 (supporting benign).
# The "dead zone" in between (0.20 - 0.70) means the model isn't confident
# enough to provide evidence in either direction.
# ---------------------------------------------------------------------------
# UNDERREPRESENTED POPULATION CALIBRATION
# ---------------------------------------------------------------------------
# gnomAD has ~0% Central Asian representation. For these populations:
#   1. PM2 (absent from controls) is withheld — absence means "never checked",
#      not "genuinely rare"
#   2. BA1/BS1 thresholds are relaxed (multiplied by 2x) — sparse frequency
#      data shouldn't prematurely classify variants as benign
UNDERREPRESENTED_POPULATIONS = {"kazakh", "central_asian", "kz"}
SPARSE_POP_THRESHOLD_MULTIPLIER = 2.0

PP3_PATHOGENIC_THRESHOLD = 0.70  # Model probability >= this triggers PP3
BP4_BENIGN_THRESHOLD = 0.20     # Model probability <= this triggers BP4

# PM1: If a variant's amino acid is physically close to an important part
# of the protein (like the DNA-binding site), that's moderate evidence of
# pathogenicity. We measure distance in Angstroms (1A = 0.1 nanometers)
# using 3D protein structures from AlphaFold.
PM1_DISTANCE_THRESHOLD = 5.0    # Angstroms to functional site for PM1
PM1_DISTANCE_UNKNOWN = 999.0    # Sentinel value when no structural data exists


def evaluate_acmg_rules(features, model_prediction, gene_name="BRCA2", population=None, founder_mutation=None):  # noqa: C901
    """
    Evaluate ACMG criteria for a single variant and return all triggered codes.

    This is the first half of the ACMG classification process. It examines
    each piece of evidence independently (structural proximity, population
    frequency, computational prediction, etc.) and collects whichever codes
    are triggered. The second half — combining these codes into a final
    5-tier classification — is done by combine_acmg_evidence().

    Args:
        features: dict of variant properties (structural distances, population
                  frequencies, known pathogenic status, exon info, etc.)
        model_prediction: calibrated probability of pathogenicity (0.0 to 1.0)
                          from our XGBoost + MLP ensemble
        gene_name: which gene this variant is in (affects BA1/BS1 thresholds)
        population: optional population code ('eas','afr','nfe','sas','amr')
                    for ancestry-aware allele frequency lookups

    Returns:
        dict mapping ACMG code strings (e.g. "PM1", "BA1") to human-readable
        rationale strings explaining WHY each code was triggered
    """
    # This dict accumulates all the evidence codes that apply to this variant.
    # At the end, it's returned and later passed to combine_acmg_evidence().
    met_codes = {}

    # ── STEP 1: Determine which allele frequency to use ──────────────────
    # Allele frequency (AF) = how common this variant is in the population.
    # gnomAD stores AF broken down by ancestry (African, East Asian, etc.).
    # If the user specifies their population, we use that population's AF
    # because some variants are common in one population but absent in others.
    # Otherwise, we fall back to the global (all-populations) AF.
    POP_AF_KEYS = {
        'afr': 'gnomad_af_afr', 'amr': 'gnomad_af_amr',
        'asj': 'gnomad_af_asj', 'eas': 'gnomad_af_eas',
        'fin': 'gnomad_af_fin', 'nfe': 'gnomad_af_nfe',
        'sas': 'gnomad_af_sas',
    }
    _pop_lower = population.lower() if population else None
    _is_underrepresented = _pop_lower in UNDERREPRESENTED_POPULATIONS if _pop_lower else False

    if _is_underrepresented:
        # Kazakh/Central Asian: use EAS (East Asian) as closest gnomAD proxy
        af = features.get('gnomad_af_eas', features.get('gnomad_af', 0.0))
        pop_label = "Kazakh (EAS proxy)"
    elif _pop_lower and _pop_lower in POP_AF_KEYS:
        af = features.get(POP_AF_KEYS[_pop_lower], features.get('gnomad_af', 0.0))
        pop_label = population.upper()
    else:
        af = features.get('gnomad_af', 0.0)
        pop_label = 'global'

    # ── STEP 2: PM1 — Is the variant in a critical part of the protein? ──
    # Proteins have specific regions (domains) that perform functions like
    # binding to DNA or interacting with partner proteins. A mutation in one
    # of these regions is more likely to be harmful than one in a "boring" area.
    # We check three things:
    #   1. Is it physically close to the DNA-binding site? (dist_dna)
    #   2. Is it close to where the protein touches PALB2? (dist_palb2)
    #   3. Is it inside a known critical domain? (in_critical_domain)
    # If ANY of these are true, PM1 (moderate pathogenic evidence) is triggered.
    features.get('domain', 'uncharacterized')
    dist_dna = features.get('dist_dna', 999.0)       # Distance to DNA-binding site in Angstroms
    dist_palb2 = features.get('dist_palb2', 999.0)   # Distance to PALB2 interaction site

    pm1_reasons = []  # Collect all structural reasons (there can be multiple)
    if dist_dna <= PM1_DISTANCE_THRESHOLD and dist_dna != PM1_DISTANCE_UNKNOWN:
        pm1_reasons.append(f"Located within {dist_dna}A of DNA binding interface")
    if dist_palb2 <= PM1_DISTANCE_THRESHOLD and dist_palb2 != PM1_DISTANCE_UNKNOWN:
        pm1_reasons.append(f"Located within {dist_palb2}A of Primary Interaction interface")
    if features.get('in_critical_domain', False):
        pm1_reasons.append("Located inside a Critical Functional Repeat or Domain")

    if pm1_reasons:
        met_codes['PM1'] = "Structural Disruption: " + " & ".join(pm1_reasons)

    # ── STEP 3: PS1 / PM5 — Has this position been seen before? ──────────
    # PS1 (Strong): The EXACT same amino acid substitution (e.g., Arg→Trp at
    #   position 2336) has already been classified as pathogenic in ClinVar.
    #   This is strong evidence because we already know this change is harmful.
    #
    # PM5 (Moderate): A DIFFERENT pathogenic missense exists at the same position
    #   (e.g., we see Arg→Gln, but Arg→Trp is known pathogenic). This means
    #   the position is sensitive to change, but we're less certain about this
    #   specific substitution.
    #
    # HIERARCHY: PS1 supersedes PM5. If the exact change is known pathogenic,
    # there's no need to also say "a different change at this position is
    # pathogenic" — PS1 is already stronger evidence.
    if features.get('same_aa_change_pathogenic', False):
        met_codes['PS1'] = "Same amino acid change as a previously established pathogenic variant"

    if 'PS1' not in met_codes and features.get('known_pathogenic_at_pos', False):
        met_codes['PM5'] = "Different pathogenic missense variant previously established at this amino acid position"

    # ── STEP 3b: PS4 — Founder mutation prevalence evidence ──────────────
    # PS4 fires when a variant has been shown to be significantly more common
    # in affected individuals than in controls. Known founder mutations from
    # published Kazakh cancer genetics studies meet this criterion.
    if founder_mutation and founder_mutation.get("is_founder"):
        freq = founder_mutation.get("frequency_kz", founder_mutation.get("frequency", 0))
        pop = founder_mutation.get("population", "Kazakh")
        source = founder_mutation.get("source", "published literature")
        met_codes['PS4'] = (
            f"Known {pop} founder mutation (frequency {freq*100:.1f}% in affected individuals). "
            f"Prevalence in cases significantly exceeds controls. Source: {source}"
        )

    # ── STEP 4: PM2 / BA1 / BS1 — Population frequency evidence ─────────
    # The logic here: if a variant causes a rare, serious disease, it should
    # be rare in the general population. If we see it frequently in healthy
    # people (gnomAD database), it's probably not disease-causing.
    #
    # PM2 (Moderate pathogenic): AF = 0 means this variant has NEVER been
    #   seen in gnomAD's ~140,000 people. That's consistent with it being a
    #   rare disease-causing mutation.
    if af == 0.0:
        if _is_underrepresented:
            # Kazakh/Central Asian: gnomAD has ~0% representation for this population.
            # AF=0 means "never checked", not "genuinely absent from controls".
            # Withhold PM2 to prevent ancestry-driven evidence inflation.
            met_codes['PM2_insufficient'] = (
                f"Variant absent from gnomAD ({pop_label} AF = 0), but this population has "
                f"insufficient representation in gnomAD (~0% of samples). PM2 evidence "
                f"withheld to avoid ancestry bias."
            )
        else:
            met_codes['PM2'] = f"Variant absent from gnomAD population database ({pop_label} AF = 0)"

    # BA1 (Stand-alone benign): AF is above the gene-specific "definitely too
    #   common" threshold. This SINGLE code is enough to classify as Benign,
    #   overriding all other evidence. It's the nuclear option for benign.
    ba1_threshold = GENE_BA1_THRESHOLDS.get(gene_name.upper(), 0.01)
    bs1_threshold = GENE_BS1_THRESHOLDS.get(gene_name.upper(), 0.001)

    # For underrepresented populations, relax thresholds — require higher
    # frequency before calling benign, because sparse data is unreliable.
    if _is_underrepresented:
        ba1_threshold *= SPARSE_POP_THRESHOLD_MULTIPLIER
        bs1_threshold *= SPARSE_POP_THRESHOLD_MULTIPLIER

    if af > ba1_threshold:
        _relaxed_note = " (threshold relaxed for underrepresented population)" if _is_underrepresented else ""
        met_codes['BA1'] = f"Allele frequency ({af*100:.3f}%, {pop_label}) exceeds stand-alone benign threshold ({ba1_threshold*100:.1f}%) for {gene_name}{_relaxed_note}"

    # BS1 (Strong benign): AF is elevated but not high enough for BA1.
    # HIERARCHY: BA1 supersedes BS1.
    if 'BA1' not in met_codes:
        if af > bs1_threshold:
            _relaxed_note = " (threshold relaxed for underrepresented population)" if _is_underrepresented else ""
            met_codes['BS1'] = f"Population frequency ({af*100:.4f}%, {pop_label}) exceeds threshold ({bs1_threshold*100:.3f}%) for {gene_name}{_relaxed_note}"

    # ── STEP 5: PVS1 — Does this variant destroy the protein? ────────────
    # PVS1 is the STRONGEST single pathogenic evidence code ("Very Strong").
    # It applies to "null" variants — mutations that prevent the cell from
    # making a functional protein. Two types qualify:
    #   - Nonsense: creates a premature STOP codon (e.g., Arg→Stop)
    #   - Frameshift: inserts/deletes bases not in multiples of 3, scrambling
    #     all downstream amino acids
    #
    # LAST-EXON MODIFIER (NMD escape):
    #   Cells have a quality-control system called Nonsense-Mediated Decay
    #   (NMD) that detects and destroys mRNAs with premature stop codons.
    #   BUT: NMD only works if the premature stop is NOT in the last exon.
    #   If the variant IS in the last exon (or near the end of the second-to-
    #   last exon), the truncated mRNA escapes NMD, and a shorter protein
    #   may still be produced. This shorter protein might retain partial
    #   function, so the evidence is weaker. In these cases, PVS1 is
    #   downgraded from "Very Strong" to "Moderate" (stored as PVS1_moderate).
    is_nonsense = features.get('is_nonsense', False)
    is_frameshift = features.get('is_frameshift', False)
    if is_nonsense or is_frameshift:
        variant_type = "Nonsense" if is_nonsense else "Frameshift"
        exon_number = features.get('exon_number')
        total_exons = features.get('total_exons')

        # Apply the last-exon / NMD-escape downgrade rule
        if exon_number is not None and total_exons is not None and total_exons > 1:
            if exon_number == total_exons:
                # Variant is in the last exon -> NMD escape -> downgrade to moderate
                met_codes['PVS1_moderate'] = (
                    f"{variant_type} variant in last exon (exon {exon_number}/{total_exons}) — "
                    f"may escape nonsense-mediated decay. PVS1 downgraded to moderate."
                )
            elif exon_number == total_exons - 1 and features.get('near_last_exon_boundary', False):
                # Variant near the end of the second-to-last exon -> also may escape NMD
                met_codes['PVS1_moderate'] = (
                    f"{variant_type} variant in last 50nt of penultimate exon (exon {exon_number}/{total_exons}) — "
                    f"may escape nonsense-mediated decay. PVS1 downgraded to moderate."
                )
            else:
                # Variant is in an earlier exon -> NMD will catch it -> full PVS1
                met_codes['PVS1'] = f"Pathogenic truncating null variant ({variant_type}) in a recognized tumor suppressor"
        else:
            # No exon info available — conservatively apply PVS1 at full strength
            met_codes['PVS1'] = f"Pathogenic truncating null variant ({variant_type}) in a recognized tumor suppressor"

    # ── STEP 6: PM4 — In-frame insertion/deletion ────────────────────────
    # Unlike frameshifts (which scramble everything downstream), in-frame
    # indels add or remove whole amino acids without breaking the reading
    # frame. The protein gets slightly longer or shorter but the rest stays
    # intact. This is moderate evidence of pathogenicity — especially if
    # it happens inside a critical functional domain where even small
    # structural changes can be disruptive.
    is_inframe_indel = features.get('is_inframe_indel', False)
    if is_inframe_indel:
        if features.get('in_critical_domain', False):
            met_codes['PM4'] = "In-frame deletion/insertion within a known functional domain"
        else:
            met_codes['PM4'] = "In-frame deletion/insertion outside known functional domain (reduced weight)"

    # ── STEP 7: Splice site variants ─────────────────────────────────────
    # Splice sites are the boundaries between exons and introns. The cell's
    # splicing machinery reads specific sequences at these boundaries to
    # correctly cut out introns and join exons together. Mutations here can
    # cause the mRNA to be incorrectly spliced, potentially destroying the
    # protein.
    #
    # Canonical splice sites (the GT...AG dinucleotides at exon boundaries)
    # are so critical that disrupting them is treated like a null variant
    # (PVS1). Variants merely NEAR a splice site get weaker evidence (PP3).
    near_splice = features.get('near_splice', False)
    canonical_splice = features.get('canonical_splice', False)
    if canonical_splice and 'PVS1' not in met_codes and 'PVS1_moderate' not in met_codes:
        # Canonical splice disruption is essentially a null variant -> PVS1
        met_codes['PVS1'] = "Canonical splice site variant disrupting mRNA splicing"
    elif near_splice:
        # Near (but not at) a splice site -> weaker supporting evidence
        met_codes['PP3_splice'] = "Near splice site variant with potential splicing impact"

    # ── STEP 8: BP7 — Synonymous (silent) variants ────────────────────
    # A synonymous variant changes the DNA codon but NOT the amino acid
    # (e.g., GGT->GGC both encode Glycine). These are usually harmless
    # UNLESS they affect splicing. If there's no splice concern, BP7
    # provides supporting benign evidence.
    is_synonymous = features.get('is_synonymous', False)
    if is_synonymous and not near_splice and not canonical_splice:
        met_codes['BP7'] = "Synonymous variant with no predicted splice impact"

    # ── STEP 9: PP3 / BP4 — Our ML model's prediction ─────────────────
    # This is where SteppeDNA's own prediction feeds into the ACMG framework.
    # Our calibrated ensemble model outputs a probability between 0 and 1.
    # High probability (>=0.70) -> PP3 (supporting pathogenic evidence)
    # Low probability  (<=0.20) -> BP4 (supporting benign evidence)
    # In-between       (0.20-0.70) -> no computational evidence triggered
    if model_prediction >= PP3_PATHOGENIC_THRESHOLD:
        met_codes['PP3'] = f"Computational model predicts pathogenicity (p={model_prediction:.3f})"
    elif model_prediction <= BP4_BENIGN_THRESHOLD:
        met_codes['BP4'] = f"Computational model predicts benign (p={model_prediction:.3f})"

    return met_codes


# ===========================================================================
# ACMG EVIDENCE COMBINING RULES (Richards et al. 2015, Table 5)
# ===========================================================================
# After evaluate_acmg_rules() collects all triggered evidence codes, we need
# to combine them into a final 5-tier classification. The combining rules
# work like a scoring system with thresholds:
#
# Think of it as a balance scale:
#   - Pathogenic evidence codes go on the left (trying to tip toward "disease")
#   - Benign evidence codes go on the right (trying to tip toward "harmless")
#   - Stronger codes weigh more than weaker ones
#
# The rules below define exactly how many codes of each strength level are
# needed to reach each classification tier.
#
# IMPORTANT: Benign rules are checked FIRST. If BA1 (stand-alone benign) is
# triggered, the variant is immediately classified as Benign regardless of
# any pathogenic evidence. This makes biological sense: if a variant is
# common in healthy people, it almost certainly doesn't cause rare disease,
# even if it looks structurally damaging.

# ---------------------------------------------------------------------------
# Evidence strength categories
# ---------------------------------------------------------------------------
# Each set below lists all the ACMG codes that belong to that strength level.
# Note: PVS1_moderate is our custom code for PVS1 downgraded due to last-exon
# location — it's counted as moderate-strength rather than very-strong.
# Note: PP3_splice is our custom code for near-splice-site variants — it's
# counted as supporting-strength pathogenic evidence.
_PATHOGENIC_VERY_STRONG = {"PVS1"}
_PATHOGENIC_STRONG = {"PS1", "PS2", "PS3", "PS4"}
_PATHOGENIC_MODERATE = {"PM1", "PM2", "PM3", "PM4", "PM5", "PM6", "PVS1_moderate"}
_PATHOGENIC_SUPPORTING = {"PP1", "PP2", "PP3", "PP3_splice", "PP4", "PP5"}
_BENIGN_STANDALONE = {"BA1"}
_BENIGN_STRONG = {"BS1", "BS2", "BS3", "BS4"}
_BENIGN_SUPPORTING = {"BP1", "BP2", "BP3", "BP4", "BP5", "BP6", "BP7"}


def combine_acmg_evidence(met_codes: dict) -> str:  # noqa: C901
    """
    Combine ACMG evidence codes into a 5-tier classification.

    This implements Table 5 from Richards et al. 2015 — the official rules
    for how to combine individual evidence codes into a final classification.

    The function counts how many codes were triggered at each strength level
    and then checks a series of threshold rules in order:
        1. Benign rules first (BA1 alone, or 2 strong benign, etc.)
        2. Pathogenic rules (various combinations of strong + moderate + supporting)
        3. Likely Pathogenic rules (weaker combinations)
        4. Default: Uncertain Significance (not enough evidence either way)

    Args:
        met_codes: dict mapping ACMG code strings to rationale strings,
                   as returned by evaluate_acmg_rules()

    Returns:
        One of: "Pathogenic", "Likely Pathogenic", "Uncertain Significance",
                "Likely Benign", "Benign"
    """
    codes = set(met_codes.keys())

    # Count how many triggered codes fall into each strength bucket.
    # We use set intersection (&) to find which of our triggered codes
    # appear in each strength category, then count them.
    pvs = len(codes & _PATHOGENIC_VERY_STRONG)   # Very strong pathogenic (0 or 1)
    ps = len(codes & _PATHOGENIC_STRONG)          # Strong pathogenic (0-4)
    pm = len(codes & _PATHOGENIC_MODERATE)         # Moderate pathogenic (0-7)
    pp = len(codes & _PATHOGENIC_SUPPORTING)       # Supporting pathogenic (0-6)
    ba = len(codes & _BENIGN_STANDALONE)           # Stand-alone benign (0 or 1)
    bs = len(codes & _BENIGN_STRONG)               # Strong benign (0-4)
    bp = len(codes & _BENIGN_SUPPORTING)           # Supporting benign (0-7)

    # ─── BENIGN RULES (checked first — benign evidence overrides) ────────
    # BA1 alone is sufficient for Benign. This is the strongest benign
    # evidence: if a variant is very common in healthy people, it cannot
    # be causing a rare disease, no matter how bad it looks structurally.
    if ba >= 1:
        return "Benign"

    # Two strong benign codes together also reach Benign (e.g., BS1 + BS2)
    if bs >= 2:
        return "Benign"

    # ─── LIKELY BENIGN ───────────────────────────────────────────────────
    # One strong benign + one supporting benign = Likely Benign
    # Example: BS1 (elevated population frequency) + BP4 (model says benign)
    if bs >= 1 and bp >= 1:
        return "Likely Benign"

    # ─── PATHOGENIC RULES ────────────────────────────────────────────────
    # These rules require progressively weaker combinations of evidence.
    # The first rule that matches wins (they're ordered strongest first).
    #
    # Think of it as: the more "points" of evidence you have, the more
    # confident the classification. Very Strong = 8 pts, Strong = 4 pts,
    # Moderate = 2 pts, Supporting = 1 pt (roughly).

    # Rule i:   1 Very Strong + >=1 Strong
    #   Example: PVS1 (truncating variant) + PS1 (known pathogenic change)
    if pvs >= 1 and ps >= 1:
        return "Pathogenic"
    # Rule ii:  1 Very Strong + >=2 Moderate
    #   Example: PVS1 + PM1 (critical domain) + PM2 (absent from gnomAD)
    if pvs >= 1 and pm >= 2:
        return "Pathogenic"
    # Rule iii: 1 Very Strong + 1 Moderate + 1 Supporting
    if pvs >= 1 and pm >= 1 and pp >= 1:
        return "Pathogenic"
    # Rule iv:  1 Very Strong + >=2 Supporting
    if pvs >= 1 and pp >= 2:
        return "Pathogenic"
    # Rule v:   >=2 Strong (no Very Strong needed)
    if ps >= 2:
        return "Pathogenic"
    # Rule vi:  1 Strong + >=3 Moderate
    if ps >= 1 and pm >= 3:
        return "Pathogenic"
    # Rule vii: 1 Strong + 2 Moderate + >=2 Supporting
    if ps >= 1 and pm >= 2 and pp >= 2:
        return "Pathogenic"
    # Rule viii: 1 Strong + 1 Moderate + >=4 Supporting
    if ps >= 1 and pm >= 1 and pp >= 4:
        return "Pathogenic"

    # ─── LIKELY PATHOGENIC RULES ─────────────────────────────────────────
    # These are weaker combinations — enough evidence to say "probably
    # disease-causing" but not quite enough for full Pathogenic.

    # Rule i:  1 Very Strong + 1 Moderate (not enough for full Pathogenic)
    if pvs >= 1 and pm >= 1:
        return "Likely Pathogenic"
    # Rule ii: 1 Strong + 1-2 Moderate
    if ps >= 1 and 1 <= pm <= 2:
        return "Likely Pathogenic"
    # Rule iii: 1 Strong + >=2 Supporting
    if ps >= 1 and pp >= 2:
        return "Likely Pathogenic"
    # Rule iv:  >=3 Moderate (many moderate pieces add up)
    if pm >= 3:
        return "Likely Pathogenic"
    # Rule v:   2 Moderate + >=2 Supporting
    if pm >= 2 and pp >= 2:
        return "Likely Pathogenic"
    # Rule vi:  1 Moderate + >=4 Supporting
    if pm >= 1 and pp >= 4:
        return "Likely Pathogenic"

    # ─── DEFAULT: UNCERTAIN SIGNIFICANCE (VUS) ──────────────────────────
    # If none of the above rules matched, we don't have enough evidence to
    # classify this variant in either direction. This is the most common
    # outcome — the majority of variants in clinical databases are VUS.
    return "Uncertain Significance"
