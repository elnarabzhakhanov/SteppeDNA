"""
SteppeDNA - Feature Engineering & Gene Data Cache
==================================================

This is the heart of the SteppeDNA prediction pipeline. It converts a human-readable
variant description (e.g., "BRCA2 p.Arg2520Gln") into a numeric feature vector of
120 numbers that our XGBoost + MLP ensemble model can score.

Think of it like a translator: the biologist says "this amino acid changed at position X",
and this module says "here are 120 measurements about how important that change is."

The 120 features fall into these categories:
  1. Position features        - Where in the gene/protein the variant sits
  2. Amino acid properties    - Physical/chemical differences between old and new amino acid
  3. Domain membership        - Whether the variant hits a known functional region
  4. Conservation scores      - How unchanged this position is across species (PhyloP)
  5. Functional assay scores  - Lab measurements of variant effect (MAVE/DMS, EVE)
  6. 3D structural features   - Protein shape info from AlphaFold (burial, contacts, etc.)
  7. Population frequencies   - How common the variant is in human populations (gnomAD)
  8. Protein language model   - ESM-2 neural network embeddings (learned protein patterns)
  9. Splice predictions       - Whether the variant disrupts RNA splicing (SpliceAI)
 10. One-hot encodings        - Which specific amino acids and nucleotide changes are involved

Main entry points:
  - get_gene_data(gene_name): Loads all per-gene data files into a cache (called once per gene)
  - build_feature_vector(...): THE critical function -- builds the 120-feature numeric vector
  - NICE_NAMES: Dict mapping internal feature names to human-readable labels for the UI
  - _safe_critical_domain(): Helper to check if a variant is in a critical protein domain
"""

import os
import json
import logging
import threading

import numpy as np

# Models module provides: data directory path, ESM-2 settings, pickle/dict loaders,
# the universal ML model bundle, and the ESM-2 protein language model
from backend.models import (
    DATA_DIR, ESM2_WINDOW, ESM2_PCA_COMPONENTS,
    _load_pickle, _load_variant_dict,
    _get_universal_models, esm_model,
    esm_batch_converter,
)
# Feature engineering module provides: amino acid property lookup tables (hydrophobicity,
# volume, charge), the BLOSUM62 substitution matrix, and lists of all amino acids/mutations
from backend.feature_engineering import (
    AA_HYDROPHOBICITY, AA_VOLUME, ALL_AMINO_ACIDS,
    ALL_MUTATIONS, get_blosum62,
    get_charge,
)

logger = logging.getLogger("steppedna")


# ─── Exon Boundary Derivation ────────────────────────────────────────────────
# Background: Genes are split into exons (coding parts) and introns (non-coding parts).
# The boundaries between exons and introns are called "splice sites" because the cell's
# machinery must splice (cut and join) the RNA at these exact positions. Mutations near
# splice sites can prevent proper splicing, which often causes disease.

def _derive_exon_boundaries(cdna_to_genomic: dict, strand: str) -> dict:  # noqa: C901
    """Derive exon-intron boundaries from the cDNA-to-genomic position mapping.

    How it works:
        - cDNA (coding DNA) positions map to genomic (chromosome) positions.
        - Within an exon, consecutive cDNA positions map to consecutive genomic positions.
        - When there's a jump > 1 in genomic position, that gap is an intron.
        - We find all such jumps to identify exon-intron boundaries.

    Why it matters:
        - Variants near splice sites (exon-intron boundaries) are more likely to be
          pathogenic because they can disrupt RNA processing.
        - "Canonical" splice sites (1-2 bp from boundary) are the most critical.
        - "Near-splice" sites (3-8 bp) can also affect splicing but less severely.

    Args:
        cdna_to_genomic: Dict mapping cDNA position (int) -> genomic position (int).
        strand: "+" for forward-strand genes, "-" for reverse-strand genes.
            Reverse-strand genes have decreasing genomic positions as cDNA increases.

    Returns:
        Dict with:
            exon_boundaries: list of (last_exon_gpos, first_next_exon_gpos) tuples
            canonical_splice: set of genomic positions within 1-2 bp of a boundary
            near_splice: set of genomic positions within 3-8 bp (but not canonical)
            splice_info: dict mapping gpos -> {"zone": "canonical"|"near", "distance": int}
    """
    # If no mapping data is available, return empty results
    if not cdna_to_genomic:
        return {"exon_boundaries": [], "canonical_splice": set(), "near_splice": set(), "splice_info": {}}

    # Sort cDNA positions so we can walk through them in order
    sorted_cdna = sorted(cdna_to_genomic.keys())
    boundaries = []

    # Step 1: Find exon-intron boundaries by looking for gaps in genomic positions.
    # If cDNA positions 100 and 101 map to genomic positions 5000 and 7000,
    # there's a 2000-bp intron between them -- that's a boundary.
    for i in range(len(sorted_cdna) - 1):
        gpos_curr = cdna_to_genomic[sorted_cdna[i]]
        gpos_next = cdna_to_genomic[sorted_cdna[i + 1]]

        if strand == "+":
            gap = gpos_next - gpos_curr
        else:  # reverse strand: genomic positions decrease as cDNA increases
            gap = gpos_curr - gpos_next

        if gap > 1:
            # Found an intron -- record the boundary positions
            boundaries.append((gpos_curr, gpos_next))

    # Step 2: Build sets of splice-site positions around each boundary.
    # Canonical = 1-2 bp from boundary (almost always required for splicing)
    # Near-splice = 3-8 bp from boundary (can still affect splicing)
    canonical_splice = set()
    near_splice = set()
    splice_info = {}  # maps each position to its zone type and distance

    for exon_end, next_exon_start in boundaries:
        # Generate positions in a window around each boundary endpoint
        for anchor in [exon_end, next_exon_start]:
            for offset in range(-8, 9):
                if offset == 0:
                    continue
                gpos = anchor + offset
                dist = abs(offset)

                if dist <= 2:
                    # Canonical splice site: critical for splicing machinery (GT-AG rule)
                    canonical_splice.add(gpos)
                    splice_info[gpos] = {"zone": "canonical", "distance": dist}
                elif dist <= 8:
                    # Near-splice site: can still affect branch point or enhancers
                    if gpos not in canonical_splice:
                        near_splice.add(gpos)
                        splice_info[gpos] = {"zone": "near", "distance": dist}

    # Step 3: Also mark the actual intronic positions adjacent to each boundary,
    # handling forward and reverse strand genes differently.
    for exon_end, next_exon_start in boundaries:
        if strand == "+":
            # Intron spans from exon_end+1 to next_exon_start-1
            for d in [1, 2]:
                pos = exon_end + d
                canonical_splice.add(pos)
                splice_info[pos] = {"zone": "canonical", "distance": d}
                pos = next_exon_start - d
                canonical_splice.add(pos)
                splice_info[pos] = {"zone": "canonical", "distance": d}
            for d in [3, 4, 5, 6, 7, 8]:
                pos = exon_end + d
                if pos not in canonical_splice:
                    near_splice.add(pos)
                    splice_info[pos] = {"zone": "near", "distance": d}
                pos = next_exon_start - d
                if pos not in canonical_splice:
                    near_splice.add(pos)
                    splice_info[pos] = {"zone": "near", "distance": d}
        else:
            # Reverse strand: intron spans from next_exon_start+1 to exon_end-1
            low = min(exon_end, next_exon_start)
            high = max(exon_end, next_exon_start)
            for d in [1, 2]:
                pos = low + d
                canonical_splice.add(pos)
                splice_info[pos] = {"zone": "canonical", "distance": d}
                pos = high - d
                canonical_splice.add(pos)
                splice_info[pos] = {"zone": "canonical", "distance": d}
            for d in [3, 4, 5, 6, 7, 8]:
                pos = low + d
                if pos not in canonical_splice:
                    near_splice.add(pos)
                    splice_info[pos] = {"zone": "near", "distance": d}
                pos = high - d
                if pos not in canonical_splice:
                    near_splice.add(pos)
                    splice_info[pos] = {"zone": "near", "distance": d}

    # Step 4: Remove any positions that fall inside the actual coding sequence (CDS).
    # Splice sites are by definition non-coding, so if a position is in an exon,
    # it shouldn't be labeled as a splice site.
    cds_positions = set(cdna_to_genomic.values())
    canonical_splice -= cds_positions
    near_splice -= cds_positions
    splice_info = {k: v for k, v in splice_info.items() if k not in cds_positions}

    logger.info(f"[EXON] Derived {len(boundaries)} exon boundaries, "
                f"{len(canonical_splice)} canonical splice positions, "
                f"{len(near_splice)} near-splice positions")

    return {
        "exon_boundaries": boundaries,
        "canonical_splice": canonical_splice,
        "near_splice": near_splice,
        "splice_info": splice_info,
    }


# ─── Gene Data Cache ─────────────────────────────────────────────────────────
# Loading data files from disk is slow. We cache each gene's data in memory so
# we only load once. The lock ensures thread safety -- multiple web requests
# could try to load the same gene simultaneously.
_gene_cache: dict = {}
_gene_cache_lock = threading.Lock()


def get_gene_data(gene_name: str) -> dict:
    """Load and cache all per-gene data files needed for feature construction.

    This function is the data loading backbone of SteppeDNA. For each gene (e.g., "BRCA2"),
    it loads ~10 different data files from disk:
      - PhyloP conservation scores (how conserved each position is across species)
      - MAVE/DMS functional assay scores (lab measurements of variant effects)
      - AlphaMissense scores (kept for ACMG rules, removed from ML features in v5.4)
      - AlphaFold structural features (3D protein structure info)
      - gnomAD population frequencies (how common each variant is in humans)
      - EVE evolutionary coupling scores (evolutionary constraint predictions)
      - SpliceAI splice predictions (whether variants disrupt RNA splicing)
      - ESM-2 protein language model embeddings (neural network protein representations)
      - CDS (coding DNA sequence) for on-the-fly ESM-2 embedding computation
      - Gene config (domain definitions, strand, lengths)
      - cDNA-to-genomic position mapping (for splice site detection)

    Results are cached in memory with thread-safe locking, so the second call for
    the same gene returns instantly.

    Args:
        gene_name: One of "BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D" (case-insensitive).

    Returns:
        Dict containing all loaded data plus references to the universal ML models
        (XGBoost booster, MLP, scaler, calibrator, etc.).
    """
    key = gene_name.upper()
    # Check if we already loaded this gene's data (fast path)
    with _gene_cache_lock:
        if key in _gene_cache:
            return _gene_cache[key]

    # File naming convention: files are named like "brca2_phylop_scores.pkl"
    prefix = gene_name.lower()

    def load_with_fallback(suffix, loader_func, *args, **kwargs):
        """Try loading gene-specific file first (e.g., brca2_phylop_scores.pkl),
        fall back to shared file (e.g., phylop_scores.pkl) if gene-specific is empty."""
        res = loader_func(f"{prefix}_{suffix}", *args, **kwargs)
        if hasattr(res, "__len__") and not res:
            res = loader_func(suffix, *args, **kwargs)
        elif res is None:
            res = loader_func(suffix, *args, **kwargs)
        return res

    # --- Load per-gene data files ---

    # PhyloP: conservation scores from multi-species alignment. Higher = more conserved.
    # A score > 4 means the position is highly conserved across vertebrates.
    phylop = load_with_fallback("phylop_scores.pkl", _load_pickle) or {}

    # MAVE (Multiplex Assays of Variant Effect): lab-measured functional impact scores.
    # Currently available for BRCA1 (Findlay SGE assay). Returns variant-keyed and position-keyed dicts.
    mave_v, mave_p = load_with_fallback("mave_scores.pkl", _load_variant_dict)

    # AlphaMissense: DeepMind's pathogenicity predictor. REMOVED from ML features in v5.4
    # because it caused label leakage (AM was partly trained on ClinVar labels we predict).
    # Still loaded here because ACMG rule engine may reference it for PP3/BP4 evidence.
    am_v, am_p = load_with_fallback("alphamissense_scores.pkl", _load_variant_dict)

    # AlphaFold structural features: 3D protein structure predictions (RSA, B-factor,
    # secondary structure, distances to DNA/partner binding sites)
    struct = load_with_fallback("structural_features.pkl", _load_pickle) or {}

    # gnomAD (Genome Aggregation Database): allele frequencies across human populations.
    # Common variants (high AF) are less likely to be pathogenic.
    gnomad_v, gnomad_p = load_with_fallback("gnomad_frequencies.pkl", _load_variant_dict)

    # EVE (Evolutionary model of Variant Effect): unsupervised deep learning model
    # that predicts pathogenicity from evolutionary sequence patterns alone.
    eve_v, eve_p = load_with_fallback("eve_scores.pkl", _load_variant_dict)

    # SpliceAI: deep learning model that predicts splice-site disruption.
    # Shared across genes (not gene-prefixed).
    spliceai = _load_pickle("spliceai_scores.pkl") or {}

    # ESM-2: Meta's protein language model. Pre-computed embeddings capture how much
    # a mutation shifts the learned protein representation. The PCA model reduces
    # the high-dimensional embedding difference to 20 components.
    esm2 = load_with_fallback("esm2_embeddings.pkl", _load_pickle) or {}
    esm2_dict_local = esm2.get("embeddings", {})       # Pre-computed per-variant embeddings
    esm2_pca_model_local = esm2.get("pca_model", None)  # PCA transformer for dimensionality reduction

    # CDS (Coding DNA Sequence): the raw nucleotide sequence of the gene.
    # Needed for on-the-fly ESM-2 embedding computation for novel (unseen) variants.
    cds_path = f"{DATA_DIR}/{prefix}_cds.txt"
    cds = None
    if os.path.exists(cds_path):
        with open(cds_path, "r") as f:
            cds = f.read().strip()

    # cDNA-to-genomic mapping: translates cDNA positions to chromosome positions.
    # Used for splice site detection (see _derive_exon_boundaries above).
    cdna_genomic = load_with_fallback("cdna_to_genomic.pkl", _load_pickle) or {}
    # Build the reverse mapping (genomic -> cDNA) for VCF processing
    genomic_to_cdna_local = {v: k for k, v in cdna_genomic.items()}

    # Gene config JSON: contains domain definitions (e.g., DNA binding domain spans
    # residues 2400-2800), strand direction, CDS length, protein length, etc.
    gene_config = {}
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gene_configs", f"{prefix}.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            gene_config = json.load(f)

    # Derive exon-intron boundaries from the cDNA-to-genomic mapping,
    # so we can detect splice-site variants during feature construction.
    strand = gene_config.get("strand", "+")
    exon_data = _derive_exon_boundaries(cdna_genomic, strand)

    # PM5 positions: amino acid positions where at least one pathogenic missense variant
    # has been reported in ClinVar. Used by the ACMG rule engine (PM5 = "different amino acid
    # change at same position as known pathogenic variant").
    pm5_positions = set()
    pm5_path = os.path.join(DATA_DIR, "pathogenic_positions_clinvar.json")
    if os.path.exists(pm5_path):
        with open(pm5_path, "r") as f:
            pm5_data = json.load(f)
            pm5_positions = set(pm5_data.get(prefix.upper(), []))

    # Load the universal (shared across all genes) ML models: XGBoost, MLP, scaler,
    # calibrator, threshold, and feature name list. These are the same for all genes;
    # only the per-gene calibrator and ensemble weights differ.
    uni_models = _get_universal_models()

    # Assemble everything into a single dict for easy access during feature construction
    result = {
        "phylop": phylop,
        "mave_v": mave_v, "mave_p": mave_p,
        "am_v": am_v, "am_p": am_p,
        "struct": struct,
        "gnomad_v": gnomad_v, "gnomad_p": gnomad_p,
        "eve_v": eve_v, "eve_p": eve_p,
        "spliceai": spliceai,
        "esm2_dict": esm2_dict_local,
        "esm2_pca": esm2_pca_model_local,
        "cds": cds,
        "cdna_to_genomic": cdna_genomic,
        "genomic_to_cdna": genomic_to_cdna_local,
        "gene_config": gene_config,
        "exon_data": exon_data,
        "pm5_positions": pm5_positions,
        "calibrator": uni_models.get("calibrator"),
        "scaler": uni_models.get("scaler"),
        "feature_names": uni_models.get("feature_names"),
        "threshold": uni_models.get("threshold", 0.5),
        "ensemble_model": uni_models.get("ensemble_model"),
        "booster": uni_models.get("booster")
    }
    # Store in cache so subsequent calls for this gene are instant
    with _gene_cache_lock:
        _gene_cache[key] = result
    return result


# ─── Feature Vector Construction ──────────────────────────────────────────────

def build_feature_vector(cDNA_pos, AA_ref, AA_alt, Mutation, AA_pos, gene_name="BRCA2"):  # noqa: C901
    """Build the 120-feature numeric vector that the ML model scores.

    This is THE critical function in SteppeDNA. It takes a variant description
    and produces a 1x120 numpy array of floats. Each position in the array
    corresponds to a specific feature (listed in the feature_names pickle).

    The features are computed in this order:
      1. Position features (where in the gene)
      2. Amino acid physicochemical properties (BLOSUM62, hydrophobicity, volume, charge)
      3. Domain membership (is the variant in a known functional region?)
      4. Gene-specific domain proximity features
      5. Conservation (PhyloP) and cross-features
      6. Functional assay scores (MAVE, EVE)
      7. 3D structural features (AlphaFold)
      8. Population frequencies (gnomAD)
      9. Splice predictions (SpliceAI)
     10. ESM-2 protein language model embeddings
     11. One-hot encodings (mutation type, ref AA, alt AA)

    Args:
        cDNA_pos: Position in coding DNA (e.g., 7397 for BRCA2 c.7397T>C).
        AA_ref: Original (reference) amino acid in 3-letter code (e.g., "Val").
        AA_alt: New (alternate) amino acid in 3-letter code (e.g., "Ala"), or "Ter" for stop.
        Mutation: Nucleotide change string (e.g., "T>C").
        AA_pos: Position in the protein sequence (e.g., 2466).
        gene_name: Which gene this variant is in (default "BRCA2").

    Returns:
        numpy array of shape (1, 120) containing the feature vector.
        Returns all-zeros if input positions are invalid.
    """
    # Load all pre-computed data for this gene (cached after first call)
    gene_data = get_gene_data(gene_name)
    features = {}  # Dict to accumulate features by name before ordering into final vector

    # Early guard: reject clearly invalid positions (e.g., negative or zero)
    if cDNA_pos < 1 or AA_pos < 1:
        logger.warning(f"[FEATURE] Invalid positions: cDNA={cDNA_pos}, AA={AA_pos} for {gene_name}")
        return np.zeros((1, len(gene_data.get("feature_names", []) or range(120))), dtype=np.float32)

    gene_config = gene_data.get("gene_config", {})
    if not gene_config:
        raise FileNotFoundError(f"Gene configuration not found for {gene_name}")

    # ── Section 1: Position Features ─────────────────────────────────────────
    # Where the variant sits in the gene. Relative positions (0 to 1) let the model
    # learn that certain regions of the protein are more critical than others.
    features["cDNA_pos"] = cDNA_pos
    features["AA_pos"] = AA_pos
    features["relative_cdna_pos"] = cDNA_pos / gene_config["cds_length"]    # 0.0 = start, 1.0 = end
    features["relative_aa_pos"] = AA_pos / gene_config["aa_length"]          # same but for protein

    # ── Section 2: Amino Acid Physicochemical Properties ─────────────────────
    # These capture how different the new amino acid is from the original.
    # Bigger differences = more likely to disrupt protein function.

    # BLOSUM62: a substitution matrix from protein evolution. Negative scores mean
    # the substitution is rarely seen in nature (likely damaging). Positive = common (benign).
    features["blosum62_score"] = get_blosum62(AA_ref, AA_alt)

    # Volume difference: amino acids have different physical sizes (in cubic Angstroms).
    # A big volume change can break the protein's 3D packing.
    features["volume_diff"] = abs(AA_VOLUME.get(AA_ref, 0) - AA_VOLUME.get(AA_alt, 0))

    # Hydrophobicity: measures how much an amino acid repels water.
    # Buried residues are usually hydrophobic; surface residues are hydrophilic.
    # Swapping a hydrophobic AA for a hydrophilic one in the protein core is often damaging.
    ref_hydro = AA_HYDROPHOBICITY.get(AA_ref, 0)
    alt_hydro = AA_HYDROPHOBICITY.get(AA_alt, 0)
    features["hydro_diff"] = abs(ref_hydro - alt_hydro)      # magnitude of change
    features["ref_hydro"] = ref_hydro                          # original AA's hydrophobicity
    features["alt_hydro"] = alt_hydro                          # new AA's hydrophobicity
    features["hydro_delta"] = alt_hydro - ref_hydro            # signed change (direction matters)

    # ── Section 3: Domain Membership Features ──────────────────────────────────
    # Proteins have functional domains -- specific regions that perform distinct tasks
    # (e.g., DNA binding, protein-protein interaction). Mutations in these domains
    # are more likely to be damaging because they disrupt critical functions.
    # Each gene has different domains defined in its gene_config JSON file.

    domains = gene_config.get("domains", {})

    def in_domain(pos, d_name):
        """Check if amino acid position falls within the given domain's range."""
        rng = domains.get(d_name)
        return int(rng[0] <= pos <= rng[1]) if rng else 0

    def check_domains(*d_names):
        """Check multiple domain name candidates (different genes use different names
        for similar functional concepts). Returns 1 if the position is in any of them."""
        for d in d_names:
            if domains.get(d):
                return in_domain(AA_pos, d)
        logger.debug("No matching domain found in gene config for candidates: %s (available: %s)", d_names, list(domains.keys()))
        return 0

    # These 5 domain features are binary (0 or 1). Names must match training code exactly.
    # Each feature checks several possible domain names because different genes call
    # similar regions by different names.
    features["in_critical_repeat_region"] = check_domains("BRC_repeats", "WD40_repeats", "BRCT1", "BRCT2", "SCD")
    features["in_DNA_binding"] = check_domains("DNA_binding", "ChAM_DNA_binding", "ssDNA_binding")
    features["in_OB_folds"] = check_domains("OB_folds", "RING", "Walker_A", "Walker_B")
    features["in_NLS"] = check_domains("NLS_nuclear_localization", "N_terminal_domain")
    features["in_primary_interaction"] = check_domains("PALB2_interaction", "BRCA1_interaction", "BARD1_interaction", "RAD51B_RAD51D_XRCC3_interaction", "Holliday_junction_resolution")

    # ── Section 4: Gene-Specific Domain Proximity Features ─────────────────────
    # Beyond simple "in or out of domain", these features capture HOW CLOSE a variant
    # is to functional domains. Variants just outside a domain can still be damaging.

    # Distance to nearest critical domain boundary (normalized by protein length)
    aa_len = gene_config.get("aa_length", 1000)
    min_dist_to_domain = aa_len  # start with worst case: as far as possible
    n_domains_hit = 0             # count how many domains this position falls in
    for d_name, d_range in domains.items():
        if isinstance(d_range, (list, tuple)) and len(d_range) >= 2:
            if d_range[0] <= AA_pos <= d_range[1]:
                min_dist_to_domain = 0   # inside a domain = distance 0
                n_domains_hit += 1
            else:
                # Distance to the closest edge of this domain
                dist = min(abs(AA_pos - d_range[0]), abs(AA_pos - d_range[1]))
                min_dist_to_domain = min(min_dist_to_domain, dist)
    features["dist_nearest_domain"] = min_dist_to_domain / aa_len  # normalized 0-1
    features["n_domains_hit"] = n_domains_hit           # how many domains overlap this position
    features["in_multi_domain"] = int(n_domains_hit > 1)  # binary: overlapping domain regions

    # Functional zone score: a soft measure of how "functionally important" a region is.
    # Instead of binary in/out, it sums up contributions from ALL domains, where each
    # domain contributes 1.0 if the position is inside it, or a decaying fraction
    # based on distance (halves roughly every 50 residues away).
    func_zone_score = 0.0
    for d_name, d_range in domains.items():
        if isinstance(d_range, (list, tuple)) and len(d_range) >= 2:
            if d_range[0] <= AA_pos <= d_range[1]:
                func_zone_score += 1.0  # fully inside this domain
            else:
                dist = min(abs(AA_pos - d_range[0]), abs(AA_pos - d_range[1]))
                func_zone_score += 1.0 / (1.0 + dist / 50.0)  # smooth decay with distance
    features["functional_zone_score"] = func_zone_score
    # Cross-features: multiply domain proximity with other signals for richer patterns
    features["func_zone_x_blosum"] = func_zone_score * features["blosum62_score"]
    # NOTE: func_zone_x_phylop is deferred to after phylop_score is computed (below)

    # Nonsense mutation: the new amino acid is a stop codon ("Ter" = Termination).
    # This truncates the protein, which is almost always pathogenic.
    features["is_nonsense"] = int(AA_alt == "Ter")

    # Charge features: amino acids can be positive, negative, or nonpolar (uncharged).
    # Changing the charge disrupts electrostatic interactions in the protein.
    ref_charge = get_charge(AA_ref)
    alt_charge = get_charge(AA_alt)
    features["charge_change"] = int(ref_charge != alt_charge)
    # Particularly disruptive: going from a nonpolar (greasy) residue to a charged one
    features["nonpolar_to_charged"] = int(ref_charge == "nonpolar" and alt_charge in ["positive", "negative"])

    # ── Section 5: Conservation Features (PhyloP) ──────────────────────────────
    # PhyloP measures how conserved a DNA position is across 100+ vertebrate species.
    # If a position hasn't changed in millions of years of evolution, mutations there
    # are more likely to be damaging. Scale: ~0 = neutral, >4 = highly conserved, >7 = ultra.
    phylop = gene_data["phylop"].get(int(cDNA_pos), 0.0)
    features["phylop_score"] = phylop
    features["high_conservation"] = int(phylop > 4.0)   # binary threshold: highly conserved
    features["ultra_conservation"] = int(phylop > 7.0)   # binary threshold: ultra conserved
    # Cross-features: combine conservation with substitution severity and domain proximity.
    # A bad substitution (negative BLOSUM62) at a conserved site is especially concerning.
    features["conserv_x_blosum"] = phylop * features["blosum62_score"]
    features["func_zone_x_phylop"] = features["functional_zone_score"] * phylop

    # ── Section 6: Functional Assay Scores ──────────────────────────────────────

    # MAVE (Multiplex Assays of Variant Effect): actual lab measurements of how each
    # mutation affects protein function. Currently available for BRCA1 via the Findlay
    # SGE (Saturation Genome Editing) study. HDR = Homology-Directed Repair assay.
    # Lookup by variant key (e.g., "Val2466Ala"), fall back to position-level score.
    mave_key = f"{AA_ref}{AA_pos}{AA_alt}"
    mave = gene_data["mave_v"].get(mave_key, gene_data["mave_p"].get(int(cDNA_pos), 0.0))
    features["mave_score"] = mave
    features["has_mave"] = int(mave != 0.0)        # binary: do we have lab data for this variant?
    features["mave_abnormal"] = int(0.01 <= mave <= 1.49)  # abnormal function range
    features["mave_x_blosum"] = mave * features["blosum62_score"]  # cross-feature

    # AlphaMissense features REMOVED in v5.4 (label leakage -- AM was partly trained
    # on the same ClinVar labels we predict, so including it would be "cheating")

    # EVE (Evolutionary model of Variant Effect): uses deep generative models trained
    # on protein sequence alignments to predict pathogenicity WITHOUT using clinical labels.
    # This makes EVE a valuable independent signal (unlike AlphaMissense).
    # Score range: 0 = likely benign, 1 = likely pathogenic.
    eve_key = f"{AA_ref}{AA_pos}{AA_alt}"
    eve = gene_data["eve_v"].get(eve_key, gene_data["eve_p"].get(int(AA_pos), 0.0))
    features["eve_score"] = eve
    features["eve_pathogenic"] = int(eve > 0.5)        # binary: EVE says pathogenic?
    features["eve_x_phylop"] = eve * features["phylop_score"]  # cross-feature with conservation

    # ── Section 7: 3D Structural Features (from AlphaFold) ─────────────────────
    # AlphaFold is an AI that predicts 3D protein structures. These features describe
    # the structural context of the mutated position -- is it buried inside the protein?
    # Is it near a DNA binding site? What secondary structure does it form?
    sf = gene_data["struct"].get(int(AA_pos), {})

    # RSA (Relative Solvent Accessibility): fraction of the amino acid surface exposed
    # to water. 0 = completely buried inside the protein, 1 = fully exposed on surface.
    # Buried residues are critical for protein stability; mutations there are often damaging.
    features["rsa"] = sf.get("rsa", 0.4)                    # default 0.4 = moderate exposure
    features["is_buried"] = int(features["rsa"] < 0.25)     # binary: buried if <25% exposed

    # B-factor: a measure of structural flexibility/disorder from the AlphaFold model.
    # Higher values = more flexible/uncertain region.
    features["bfactor"] = sf.get("bfactor", 50.0)

    # Distance features: how far (in Angstroms) this residue is from functional sites.
    # 999.0 = no data available (effectively "very far away").
    features["dist_dna"] = sf.get("dist_dna", 999.0)        # distance to nearest DNA-binding site
    features["dist_palb2"] = sf.get("dist_palb2", 999.0)    # distance to PALB2 interaction site
    features["is_dna_contact"] = int(sf.get("is_dna_contact", False))  # binary: directly contacts DNA?

    # Secondary structure: proteins fold into alpha helices (spirals), beta sheets (flat),
    # or coils/loops (irregular). Mutations that disrupt helices/sheets are more damaging.
    # Handle both old format (single-letter string) and new AlphaFold format (pre-computed ints)
    if "ss_helix" in sf:
        features["ss_helix"] = int(sf["ss_helix"])
        features["ss_sheet"] = int(sf.get("ss_sheet", 0))
    else:
        ss = sf.get("ss", "C")  # C = coil (default)
        # H/G/I are types of helix; E/B are types of sheet (DSSP notation)
        features["ss_helix"] = int(ss in ("H", "G", "I"))
        features["ss_sheet"] = int(ss in ("E", "B"))

    # Cross-features: structural context combined with substitution severity
    features["buried_x_blosum"] = features["is_buried"] * features["blosum62_score"]
    features["dna_contact_x_blosum"] = features["is_dna_contact"] * features["blosum62_score"]

    # ── Section 8: Population Frequency Features (gnomAD) ──────────────────────
    # gnomAD (Genome Aggregation Database) tells us how common each variant is across
    # ~140,000 human genomes. Key insight: if a variant is common in healthy people,
    # it's very unlikely to cause a rare cancer syndrome. Conversely, extremely rare
    # variants could be pathogenic.
    # AF = Allele Frequency (0 to 1, where 0.01 = 1% of the population carries it).
    gnomad_key = f"{AA_ref}{AA_pos}{AA_alt}"
    gnomad_af_val = gene_data["gnomad_v"].get(gnomad_key, gene_data["gnomad_p"].get(int(cDNA_pos), 0.0))

    # gnomAD data can come in two formats: new (dict with per-population breakdown)
    # or legacy (single float). We handle both for backward compatibility.
    if isinstance(gnomad_af_val, dict):
        gnomad_af = float(gnomad_af_val.get("af", 0.0))           # global allele frequency
        # PopMax AF: highest frequency across all populations. A variant common in ANY
        # population is less likely to be pathogenic. Handle key name variations.
        gnomad_popmax = float(gnomad_af_val.get("popmax_af", gnomad_af_val.get("popmax", gnomad_af)))
        # Per-population frequencies (important for detecting population-specific variants)
        gnomad_afr = float(gnomad_af_val.get("afr", 0.0))   # African
        gnomad_amr = float(gnomad_af_val.get("amr", 0.0))   # American (Latino)
        gnomad_eas = float(gnomad_af_val.get("eas", 0.0))   # East Asian
        gnomad_nfe = float(gnomad_af_val.get("nfe", 0.0))   # Non-Finnish European
    else:
        gnomad_af = float(gnomad_af_val) if gnomad_af_val is not None else 0.0
        gnomad_popmax = gnomad_af
        gnomad_afr = gnomad_amr = gnomad_eas = gnomad_nfe = 0.0

    features["gnomad_af"] = gnomad_af
    features["gnomad_popmax_af"] = gnomad_popmax
    features["gnomad_af_afr"] = gnomad_afr
    features["gnomad_af_amr"] = gnomad_amr
    features["gnomad_af_eas"] = gnomad_eas
    features["gnomad_af_nfe"] = gnomad_nfe
    # Legacy aliases for backward-compat with older model feature name lists
    features["gnomad_popmax"] = gnomad_popmax
    features["gnomad_afr"] = gnomad_afr
    features["gnomad_amr"] = gnomad_amr
    features["gnomad_eas"] = gnomad_eas
    features["gnomad_nfe"] = gnomad_nfe

    # Log-scale AF: the model benefits from log-transformed frequencies because
    # the raw values span many orders of magnitude (0.0001 vs 0.1).
    # We add 1e-8 to avoid log(0) = -infinity.
    features["gnomad_af_log"] = np.log10(gnomad_af + 1e-8)
    features["gnomad_popmax_log"] = np.log10(gnomad_popmax + 1e-8)
    features["is_rare"] = int(gnomad_af < 0.001)            # binary: rare variant (<0.1%)
    features["is_popmax_rare"] = int(gnomad_popmax < 0.001)  # rare in ALL populations
    features["af_x_blosum"] = gnomad_af * features["blosum62_score"]  # cross-feature

    # ── Section 9: Splice Prediction Features (SpliceAI) ────────────────────────
    # SpliceAI is a deep learning model that predicts how much a variant disrupts
    # RNA splicing. Score range: 0 = no effect on splicing, 1 = completely disrupts splicing.
    # Variants that break splicing often produce non-functional truncated proteins.
    # The dict is keyed by (ref_AA, position, alt_AA) tuples from the training pipeline.
    spliceai_score = float(gene_data["spliceai"].get((AA_ref, int(AA_pos), AA_alt), 0.0))
    features["spliceai_score"] = spliceai_score
    features["splice_pathogenic"] = int(spliceai_score > 0.5)  # binary: likely disrupts splicing

    # ── Section 10: ESM-2 Protein Language Model Embeddings ─────────────────────
    # ESM-2 is Meta's protein language model (similar to GPT but for proteins).
    # It learns patterns from millions of protein sequences and produces a numeric
    # "embedding" (vector) for each amino acid position. By comparing the embedding
    # of the wild-type (original) protein to the mutant, we measure how much the
    # mutation disrupts the protein's learned representation.
    #
    # Features extracted:
    #   - cosine_similarity: how similar wt and mut embeddings are (1 = identical, 0 = orthogonal)
    #   - l2_shift: Euclidean distance between wt and mut embeddings (larger = bigger disruption)
    #   - pca_0 through pca_19: the 20 principal components of the embedding difference,
    #     capturing the most important dimensions of variation
    #
    # Three paths:
    #   1. Pre-computed: variant was seen during training, use cached embedding
    #   2. Real-time: novel variant, compute embedding on-the-fly using ESM-2 model
    #   3. Fallback: no model available, fill with zeros

    # Lookup table to convert 3-letter amino acid codes to 1-letter (needed by ESM-2)
    AA3_TO_1 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E",
                "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F",
                "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V", "Ter": "*"}

    esm_k = f"{AA_ref}{AA_pos}{AA_alt}"

    # Path 1: Use pre-computed embedding if available (fast)
    if esm_k in gene_data["esm2_dict"]:
        esm_v = gene_data["esm2_dict"][esm_k]
        features["esm2_cosine_sim"] = esm_v.get("cosine_similarity", 0.0)
        features["esm2_l2_shift"] = esm_v.get("l2_shift", 0.0)
        pca_arr = esm_v.get("pca_components", [])
        for i in range(ESM2_PCA_COMPONENTS):
            features[f"esm2_pca_{i}"] = float(pca_arr[i]) if i < len(pca_arr) else 0.0

    # Path 2: Novel variant -- compute embedding in real-time using the ESM-2 model.
    # Requires: ESM-2 model loaded, PCA model available, CDS sequence, and not a stop codon.
    elif esm_model is not None and gene_data["esm2_pca"] is not None and gene_data["cds"] is not None and AA_alt != "Ter":
        try:
            # Translate the gene's coding DNA sequence into a protein sequence
            from Bio.Seq import Seq
            protein_seq = str(Seq(gene_data["cds"]).translate()).rstrip("*")

            # Safety check: AA_pos must be within protein bounds
            if AA_pos < 1 or AA_pos > len(protein_seq):
                logger.warning(f"[ESM-2] AA_pos {AA_pos} out of protein range (len={len(protein_seq)}) for {gene_name}")
                features["esm2_cosine_sim"] = 0.0
                features["esm2_l2_shift"] = 0.0
                for i in range(ESM2_PCA_COMPONENTS):
                    features[f"esm2_pca_{i}"] = 0.0
                raise ValueError("AA_pos out of range")  # caught by outer except

            # Extract a local window around the mutation site (not the full protein,
            # for speed). ESM2_WINDOW controls how many residues on each side to include.
            WINDOW = ESM2_WINDOW
            win_start = max(0, AA_pos - 1 - WINDOW)
            win_end = min(len(protein_seq), AA_pos - 1 + WINDOW + 1)
            local_pos = AA_pos - 1 - win_start  # position within the window

            # Create wild-type and mutant versions of the window
            wt_window = protein_seq[win_start:win_end]
            mut_window = list(wt_window)
            m_aa = AA3_TO_1.get(AA_alt, "A")  # convert 3-letter to 1-letter code
            if m_aa != "*":
                mut_window[local_pos] = m_aa  # introduce the mutation
            mut_window = "".join(mut_window)

            # Run both sequences through ESM-2 in a single batch
            data = [("wt", wt_window), ("mut", mut_window)]
            _, _, batch_tokens = esm_batch_converter(data)
            import torch
            from backend.models import DEVICE
            batch_tokens = batch_tokens.to(DEVICE)
            with torch.no_grad():
                results = esm_model(batch_tokens, repr_layers=[6], return_contacts=False)

            # Extract the embedding at the mutation position for both wt and mut
            # Note: +1 offset because ESM-2 adds a special <cls> token at position 0
            wt_emb = results["representations"][6][0, local_pos + 1].cpu().numpy()
            mut_emb = results["representations"][6][1, local_pos + 1].cpu().numpy()

            # Compute similarity metrics between wild-type and mutant embeddings
            diff = mut_emb - wt_emb
            cos_sim = float(np.dot(wt_emb, mut_emb) / (np.linalg.norm(wt_emb) * np.linalg.norm(mut_emb) + 1e-8))
            l2_shift = float(np.linalg.norm(diff))
            # Project the embedding difference onto 20 principal components
            # (same PCA model used during training for consistency)
            pca_arr = gene_data["esm2_pca"].transform([diff])[0]

            features["esm2_cosine_sim"] = cos_sim
            features["esm2_l2_shift"] = l2_shift
            for i in range(20):
                features[f"esm2_pca_{i}"] = float(pca_arr[i]) if i < len(pca_arr) else 0.0
        except (ImportError, ValueError, IndexError, KeyError, AttributeError, RuntimeError, OSError) as e:
            # If anything goes wrong (missing dependencies, GPU errors, etc.),
            # fall back to zeros rather than crashing the prediction
            logger.warning(f"ESM-2 Dynamic compilation failed: {type(e).__name__}: {e}")
            features["esm2_cosine_sim"] = 0.0
            features["esm2_l2_shift"] = 0.0
            for i in range(20):
                features[f"esm2_pca_{i}"] = 0.0
    else:
        # Path 3: No ESM-2 model available (e.g., lightweight deployment) -- fill with zeros
        features["esm2_cosine_sim"] = 0.0
        features["esm2_l2_shift"] = 0.0
        for i in range(ESM2_PCA_COMPONENTS):
            features[f"esm2_pca_{i}"] = 0.0

    # ── Section 11: One-Hot Encodings ───────────────────────────────────────────
    # One-hot encoding converts categorical variables into binary columns.
    # E.g., if Mutation is "C>T", then Mutation_C>T = 1 and all other Mutation_X>Y = 0.
    # This lets the model learn that certain nucleotide changes (e.g., C>T transitions)
    # or certain amino acids have distinct pathogenicity patterns.

    # One-hot encode the nucleotide mutation type (12 possible: A>C, A>G, A>T, C>A, ...)
    for m in ALL_MUTATIONS:
        features[f"Mutation_{m}"] = int(Mutation == m)
    # One-hot encode the reference (original) amino acid (21 possible including Ter)
    for aa in ALL_AMINO_ACIDS:
        features[f"AA_ref_{aa}"] = int(AA_ref == aa)
    # One-hot encode the alternate (new) amino acid
    for aa in ALL_AMINO_ACIDS:
        features[f"AA_alt_{aa}"] = int(AA_alt == aa)

    # ── Final Assembly ───────────────────────────────────────────────────────────
    # Convert the features dict into an ordered numpy array matching the model's
    # expected feature order (from feature_names pickle saved during training).
    # Any features not computed above default to 0.
    vector = []
    feature_names = gene_data.get("feature_names", [])
    for name in feature_names:
        vector.append(features.get(name, 0))
    if not vector:
        return np.array([[]], dtype=np.float32)
    # Return shape (1, 120) -- a single sample with 120 features, ready for model.predict()
    return np.array([vector], dtype=np.float32)


# ─── Human-readable feature labels ───────────────────────────────────────────
# Maps internal feature names (used by the model) to human-friendly labels
# (shown in the web UI's feature importance explanations). This helps users
# understand which features contributed most to a prediction without needing
# to know the technical feature naming conventions.
NICE_NAMES = {
    "blosum62_score": "BLOSUM62 Score",
    "volume_diff": "Volume Difference",
    "hydro_diff": "Hydrophobicity Diff",
    "hydro_delta": "Hydro Delta",
    "ref_hydro": "Ref Hydrophobicity",
    "alt_hydro": "Alt Hydrophobicity",
    "charge_change": "Charge Change",
    "nonpolar_to_charged": "Nonpolar to Charged",
    "is_nonsense": "Nonsense Mutation",
    "in_critical_repeat_region": "In Critical Repeat Region",
    "in_DNA_binding": "In DNA Binding Domain",
    "in_OB_folds": "In OB Folds",
    "in_NLS": "In NLS Region",
    "in_primary_interaction": "In Primary Interaction Site",
    "cDNA_pos": "cDNA Position",
    "AA_pos": "AA Position",
    "relative_cdna_pos": "Relative cDNA Pos",
    "relative_aa_pos": "Relative AA Pos",
    "phylop_score": "PhyloP Conservation",
    "high_conservation": "Highly Conserved Site",
    "ultra_conservation": "Ultra-Conserved Site",
    "conserv_x_blosum": "Conservation x BLOSUM62",
    "mave_score": "MAVE HDR Score",
    "has_mave": "Has MAVE Data",
    "mave_abnormal": "MAVE Abnormal",
    "mave_x_blosum": "MAVE x BLOSUM62",
    # AM features removed in v5.4 (label leakage)
    "rsa": "Solvent Accessibility",
    "is_buried": "Buried Residue",
    "bfactor": "Structural Confidence",
    "dist_dna": "Distance to DNA Site",
    "dist_palb2": "Distance to PALB2 Site",
    "is_dna_contact": "DNA Contact Residue",
    "ss_helix": "Alpha Helix",
    "ss_sheet": "Beta Sheet",
    "buried_x_blosum": "Buried x BLOSUM62",
    "dna_contact_x_blosum": "DNA Contact x BLOSUM62",
    "gnomad_af": "gnomAD Frequency",
    "gnomad_af_log": "gnomAD AF (log scale)",
    "is_rare": "Rare Variant",
    "af_x_blosum": "Frequency x BLOSUM62",
    "gnomad_popmax_af": "gnomAD PopMax Frequency",
    "gnomad_af_afr": "gnomAD African AF",
    "gnomad_af_amr": "gnomAD American AF",
    "gnomad_af_eas": "gnomAD East Asian AF",
    "gnomad_af_nfe": "gnomAD Non-Finnish European AF",
    # ESM-2 embedding features
    "esm2_cosine_sim": "ESM-2 Cosine Similarity",
    "esm2_l2_shift": "ESM-2 Embedding Shift",
    # SpliceAI features
    "spliceai_score": "SpliceAI Score",
    "splice_pathogenic": "Splice Pathogenic",
    **{f"esm2_pca_{i}": f"ESM-2 PCA Component {i}" for i in range(20)},
    # One-hot amino acid reference features
    "AA_ref_Ala": "Ref AA: Alanine", "AA_ref_Arg": "Ref AA: Arginine",
    "AA_ref_Asn": "Ref AA: Asparagine", "AA_ref_Asp": "Ref AA: Aspartate",
    "AA_ref_Cys": "Ref AA: Cysteine", "AA_ref_Gln": "Ref AA: Glutamine",
    "AA_ref_Glu": "Ref AA: Glutamate", "AA_ref_Gly": "Ref AA: Glycine",
    "AA_ref_His": "Ref AA: Histidine", "AA_ref_Ile": "Ref AA: Isoleucine",
    "AA_ref_Leu": "Ref AA: Leucine", "AA_ref_Lys": "Ref AA: Lysine",
    "AA_ref_Met": "Ref AA: Methionine", "AA_ref_Phe": "Ref AA: Phenylalanine",
    "AA_ref_Pro": "Ref AA: Proline", "AA_ref_Ser": "Ref AA: Serine",
    "AA_ref_Ter": "Ref AA: Stop Codon", "AA_ref_Thr": "Ref AA: Threonine",
    "AA_ref_Trp": "Ref AA: Tryptophan", "AA_ref_Tyr": "Ref AA: Tyrosine",
    "AA_ref_Val": "Ref AA: Valine",
    # One-hot amino acid alternate features
    "AA_alt_Ala": "Alt AA: Alanine", "AA_alt_Arg": "Alt AA: Arginine",
    "AA_alt_Asn": "Alt AA: Asparagine", "AA_alt_Asp": "Alt AA: Aspartate",
    "AA_alt_Cys": "Alt AA: Cysteine", "AA_alt_Gln": "Alt AA: Glutamine",
    "AA_alt_Glu": "Alt AA: Glutamate", "AA_alt_Gly": "Alt AA: Glycine",
    "AA_alt_His": "Alt AA: Histidine", "AA_alt_Ile": "Alt AA: Isoleucine",
    "AA_alt_Leu": "Alt AA: Leucine", "AA_alt_Lys": "Alt AA: Lysine",
    "AA_alt_Met": "Alt AA: Methionine", "AA_alt_Phe": "Alt AA: Phenylalanine",
    "AA_alt_Pro": "Alt AA: Proline", "AA_alt_Ser": "Alt AA: Serine",
    "AA_alt_Ter": "Alt AA: Stop Codon", "AA_alt_Thr": "Alt AA: Threonine",
    "AA_alt_Trp": "Alt AA: Tryptophan", "AA_alt_Tyr": "Alt AA: Tyrosine",
    "AA_alt_Val": "Alt AA: Valine",
    # Mutation type features
    "Mutation_A>C": "Mutation A\u2192C", "Mutation_A>G": "Mutation A\u2192G",
    "Mutation_A>T": "Mutation A\u2192T", "Mutation_C>A": "Mutation C\u2192A",
    "Mutation_C>G": "Mutation C\u2192G", "Mutation_C>T": "Mutation C\u2192T",
    "Mutation_G>A": "Mutation G\u2192A", "Mutation_G>C": "Mutation G\u2192C",
    "Mutation_G>T": "Mutation G\u2192T", "Mutation_T>A": "Mutation T\u2192A",
    "Mutation_T>C": "Mutation T\u2192C", "Mutation_T>G": "Mutation T\u2192G",
    # Missing gnomAD features
    "gnomad_popmax_log": "gnomAD PopMax AF (log)",
    "is_popmax_rare": "Rare in All Populations",
    # Gene-specific computed features
    "dist_nearest_domain": "Distance to Nearest Domain",
    "n_domains_hit": "# Domains at Position",
    "in_multi_domain": "In Multiple Domains",
    "functional_zone_score": "Functional Zone Score",
    "func_zone_x_blosum": "Func Zone x BLOSUM62",
    "func_zone_x_phylop": "Func Zone x PhyloP",
    # EVE evolutionary coupling features
    "eve_score": "EVE Score",
    "eve_pathogenic": "EVE Pathogenic",
    "eve_x_phylop": "EVE x PhyloP",
}

# ─── Population Frequency-Dependent Features ──────────────────────────────────
# These are the only features in our 120-feature vector that depend on population
# frequency databases (gnomAD). All other features use universal biological signals
# (conservation, structure, protein language models) that work equally regardless
# of patient ancestry. This is a key advantage for underrepresented populations.
FREQUENCY_DEPENDENT_FEATURES = {
    "gnomad_af", "gnomad_popmax_af",
    "gnomad_af_afr", "gnomad_af_amr", "gnomad_af_eas", "gnomad_af_nfe",
    "gnomad_af_log", "gnomad_popmax_log",
    "is_rare", "is_popmax_rare",
    "af_x_blosum",
}


def get_frequency_independence_stats() -> dict:
    """Return stats on how many ML features depend on population frequency data."""
    total = 120  # Fixed model architecture
    freq_dep = len(FREQUENCY_DEPENDENT_FEATURES)
    freq_indep = total - freq_dep
    return {
        "total_features": total,
        "frequency_dependent": freq_dep,
        "frequency_independent": freq_indep,
        "pct_independent": round(freq_indep / total * 100, 1),
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe_critical_domain(raw_vector, gene_data, aa_pos) -> bool:
    """Safely check if a variant is in a critical protein domain.

    This is used by other modules (e.g., ACMG rules, explanations) to determine
    if a variant falls in a functionally important region. It tries two approaches:

    1. Primary: Look up the domain features directly from the pre-built feature vector
       (fast, avoids re-computing domain membership).
    2. Fallback: If the feature vector lookup fails (e.g., index errors), check the
       gene config's domain definitions directly.

    Args:
        raw_vector: The 1xN numpy feature vector from build_feature_vector().
        gene_data: The gene data dict from get_gene_data().
        aa_pos: The amino acid position to check.

    Returns:
        True if the variant is in any critical domain, False otherwise.
    """
    feature_names = gene_data.get("feature_names", [])
    try:
        if "in_critical_repeat_region" in feature_names and "in_DNA_binding" in feature_names:
            idx_repeat = feature_names.index("in_critical_repeat_region")
            idx_dna = feature_names.index("in_DNA_binding")
            if idx_repeat < raw_vector.shape[1] and idx_dna < raw_vector.shape[1]:
                return bool(raw_vector[0][idx_repeat] or raw_vector[0][idx_dna])
    except (ValueError, IndexError):
        pass
    # Fallback: check gene config domains if available
    gene_config = gene_data.get("gene_config", {})
    for d_name, d_range in gene_config.get("domains", {}).items():
        if isinstance(d_range, (list, tuple)) and len(d_range) >= 2:
            if d_range[0] <= aa_pos <= d_range[1]:
                return True
    return False
