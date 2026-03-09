"""
SteppeDNA - Feature Engineering & Gene Data Cache

Handles:
- Exon boundary derivation from cDNA-to-genomic mapping
- Per-gene data cache (get_gene_data)
- Feature vector construction (build_feature_vector)
- NICE_NAMES human-readable feature labels
- _safe_critical_domain helper
"""

import os
import sys
import json
import logging
import threading

import numpy as np

from backend.models import (
    DATA_DIR, SUPPORTED_GENES, GENE_MAX_AA,
    ESM2_WINDOW, ESM2_PCA_COMPONENTS,
    _load_pickle, _load_variant_dict,
    _get_universal_models,
    esm_model, esm_batch_converter,
)
from backend.feature_engineering import (
    AA_HYDROPHOBICITY, AA_VOLUME, BLOSUM62,
    ALL_AMINO_ACIDS, ALL_MUTATIONS,
    get_blosum62, get_charge,
)

logger = logging.getLogger("steppedna")


# ─── Exon Boundary Derivation ────────────────────────────────────────────────

def _derive_exon_boundaries(cdna_to_genomic: dict, strand: str) -> dict:
    """Derive exon-intron boundaries from the cDNA-to-genomic position mapping.

    Within an exon, consecutive cDNA positions map to consecutive genomic positions.
    When there's a jump > 1 in genomic position, that's an intron (= exon boundary).

    Returns dict with:
        exon_boundaries: list of (last_exon_gpos, first_next_exon_gpos) tuples
        canonical_splice: set of genomic positions +/-1-2bp from each boundary
        near_splice: set of genomic positions +/-3-8bp from each boundary (excluding canonical)
        splice_info: dict mapping gpos -> {"zone": "canonical"|"near", "distance": int}
    """
    if not cdna_to_genomic:
        return {"exon_boundaries": [], "canonical_splice": set(), "near_splice": set(), "splice_info": {}}

    sorted_cdna = sorted(cdna_to_genomic.keys())
    boundaries = []

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

    canonical_splice = set()
    near_splice = set()
    splice_info = {}

    for exon_end, next_exon_start in boundaries:
        # Generate positions around each boundary
        for anchor in [exon_end, next_exon_start]:
            for offset in range(-8, 9):
                if offset == 0:
                    continue
                gpos = anchor + offset
                dist = abs(offset)

                if dist <= 2:
                    canonical_splice.add(gpos)
                    splice_info[gpos] = {"zone": "canonical", "distance": dist}
                elif dist <= 8:
                    if gpos not in canonical_splice:
                        near_splice.add(gpos)
                        splice_info[gpos] = {"zone": "near", "distance": dist}

    # Also add the boundary positions themselves (they are intronic positions)
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

    # Remove any splice zone positions that are actually in the CDS
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
_gene_cache: dict = {}
_gene_cache_lock = threading.Lock()

def get_gene_data(gene_name: str) -> dict:
    key = gene_name.upper()
    with _gene_cache_lock:
        if key in _gene_cache:
            return _gene_cache[key]

    prefix = gene_name.lower()

    def load_with_fallback(suffix, loader_func, *args, **kwargs):
        res = loader_func(f"{prefix}_{suffix}", *args, **kwargs)
        if hasattr(res, "__len__") and not res:
            res = loader_func(suffix, *args, **kwargs)
        elif res is None:
            res = loader_func(suffix, *args, **kwargs)
        return res

    phylop = load_with_fallback("phylop_scores.pkl", _load_pickle) or {}
    mave_v, mave_p = load_with_fallback("mave_scores.pkl", _load_variant_dict)
    am_v, am_p = load_with_fallback("alphamissense_scores.pkl", _load_variant_dict)
    struct = load_with_fallback("structural_features.pkl", _load_pickle) or {}
    gnomad_v, gnomad_p = load_with_fallback("gnomad_frequencies.pkl", _load_variant_dict)

    esm2 = load_with_fallback("esm2_embeddings.pkl", _load_pickle) or {}
    esm2_dict_local = esm2.get("embeddings", {})
    esm2_pca_model_local = esm2.get("pca_model", None)

    cds_path = f"{DATA_DIR}/{prefix}_cds.txt"
    cds = None
    if os.path.exists(cds_path):
        with open(cds_path, "r") as f:
            cds = f.read().strip()

    cdna_genomic = load_with_fallback("cdna_to_genomic.pkl", _load_pickle) or {}
    genomic_to_cdna_local = {v: k for k, v in cdna_genomic.items()}

    # Load gene config JSON (cached here to avoid redundant reads in VCF processing)
    gene_config = {}
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gene_configs", f"{prefix}.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            gene_config = json.load(f)

    # Derive exon boundaries for splice-site variant detection
    strand = gene_config.get("strand", "+")
    exon_data = _derive_exon_boundaries(cdna_genomic, strand)

    # Load PM5 pathogenic position lookup (AA positions with known pathogenic missense)
    pm5_positions = set()
    pm5_path = os.path.join(DATA_DIR, "pathogenic_positions.json")
    if os.path.exists(pm5_path):
        with open(pm5_path, "r") as f:
            pm5_data = json.load(f)
            pm5_positions = set(pm5_data.get(prefix.upper(), []))

    # Utilize the static memory singleton to pull our massive Universal ML models
    uni_models = _get_universal_models()

    result = {
        "phylop": phylop,
        "mave_v": mave_v, "mave_p": mave_p,
        "am_v": am_v, "am_p": am_p,
        "struct": struct,
        "gnomad_v": gnomad_v, "gnomad_p": gnomad_p,
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
    with _gene_cache_lock:
        _gene_cache[key] = result
    return result


# ─── Feature Vector Construction ──────────────────────────────────────────────

def build_feature_vector(cDNA_pos, AA_ref, AA_alt, Mutation, AA_pos, gene_name="BRCA2"):
    gene_data = get_gene_data(gene_name)
    features = {}

    # Early guard: reject clearly invalid positions
    if cDNA_pos < 1 or AA_pos < 1:
        logger.warning(f"[FEATURE] Invalid positions: cDNA={cDNA_pos}, AA={AA_pos} for {gene_name}")
        return np.zeros((1, len(gene_data.get("feature_names", []) or range(103))), dtype=np.float32)

    gene_config = gene_data.get("gene_config", {})
    if not gene_config:
        raise FileNotFoundError(f"Gene configuration not found for {gene_name}")

    features["cDNA_pos"] = cDNA_pos
    features["AA_pos"] = AA_pos
    features["relative_cdna_pos"] = cDNA_pos / gene_config["cds_length"]
    features["relative_aa_pos"] = AA_pos / gene_config["aa_length"]
    features["blosum62_score"] = get_blosum62(AA_ref, AA_alt)
    features["volume_diff"] = abs(AA_VOLUME.get(AA_ref, 0) - AA_VOLUME.get(AA_alt, 0))
    ref_hydro = AA_HYDROPHOBICITY.get(AA_ref, 0)
    alt_hydro = AA_HYDROPHOBICITY.get(AA_alt, 0)
    features["hydro_diff"] = abs(ref_hydro - alt_hydro)
    features["ref_hydro"] = ref_hydro
    features["alt_hydro"] = alt_hydro
    features["hydro_delta"] = alt_hydro - ref_hydro

    domains = gene_config.get("domains", {})
    def in_domain(pos, d_name):
        rng = domains.get(d_name)
        return int(rng[0] <= pos <= rng[1]) if rng else 0

    def check_domains(*d_names):
        for d in d_names:
            if domains.get(d):
                return in_domain(AA_pos, d)
        logger.debug("No matching domain found in gene config for candidates: %s (available: %s)", d_names, list(domains.keys()))
        return 0

    # Domain feature names must match feature_engineering.py (training) exactly
    features["in_critical_repeat_region"] = check_domains("BRC_repeats", "WD40_repeats", "BRCT1", "BRCT2", "SCD")
    features["in_DNA_binding"] = check_domains("DNA_binding", "ChAM_DNA_binding", "ssDNA_binding")
    features["in_OB_folds"] = check_domains("OB_folds", "RING", "Walker_A", "Walker_B")
    features["in_NLS"] = check_domains("NLS_nuclear_localization", "N_terminal_domain")
    features["in_primary_interaction"] = check_domains("PALB2_interaction", "BRCA1_interaction", "BARD1_interaction", "RAD51B_RAD51D_XRCC3_interaction", "Holliday_junction_resolution")

    features["is_nonsense"] = int(AA_alt == "Ter")
    ref_charge = get_charge(AA_ref)
    alt_charge = get_charge(AA_alt)
    features["charge_change"] = int(ref_charge != alt_charge)
    features["nonpolar_to_charged"] = int(ref_charge == "nonpolar" and alt_charge in ["positive", "negative"])

    # PhyloP conservation features
    phylop = gene_data["phylop"].get(int(cDNA_pos), 0.0)
    features["phylop_score"] = phylop
    features["high_conservation"] = int(phylop > 4.0)
    features["ultra_conservation"] = int(phylop > 7.0)
    features["conserv_x_blosum"] = phylop * features["blosum62_score"]

    # MAVE HDR functional features
    mave_key = f"{AA_ref}{AA_pos}{AA_alt}"
    mave = gene_data["mave_v"].get(mave_key, gene_data["mave_p"].get(int(cDNA_pos), 0.0))
    features["mave_score"] = mave
    features["has_mave"] = int(mave != 0.0)
    features["mave_abnormal"] = int(0.01 <= mave <= 1.49)
    features["mave_x_blosum"] = mave * features["blosum62_score"]

    # AlphaMissense features
    am_key = f"{AA_ref}{AA_pos}{AA_alt}"
    am = gene_data["am_v"].get(am_key, gene_data["am_p"].get(int(AA_pos), 0.0))
    features["am_score"] = am
    features["am_pathogenic"] = int(am > 0.564)
    features["am_x_phylop"] = am * features["phylop_score"]

    # 3D Structural Features
    sf = gene_data["struct"].get(int(AA_pos), {})
    features["rsa"] = sf.get("rsa", 0.4)
    features["is_buried"] = int(features["rsa"] < 0.25)
    features["bfactor"] = sf.get("bfactor", 50.0)
    features["dist_dna"] = sf.get("dist_dna", 999.0)
    features["dist_palb2"] = sf.get("dist_palb2", 999.0)
    features["is_dna_contact"] = int(sf.get("is_dna_contact", False))
    ss = sf.get("ss", "C")
    features["ss_helix"] = int(ss == "H")
    features["ss_sheet"] = int(ss == "E")
    features["buried_x_blosum"] = features["is_buried"] * features["blosum62_score"]
    features["dna_contact_x_blosum"] = features["is_dna_contact"] * features["blosum62_score"]

    # gnomAD Allele Frequency Features
    gnomad_key = f"{AA_ref}{AA_pos}{AA_alt}"
    gnomad_af_val = gene_data["gnomad_v"].get(gnomad_key, gene_data["gnomad_p"].get(int(cDNA_pos), 0.0))

    # Normalize gnomAD value -- it can be a dict (new format) or float (legacy)
    if isinstance(gnomad_af_val, dict):
        gnomad_af = float(gnomad_af_val.get("af", 0.0))
        gnomad_popmax = float(gnomad_af_val.get("popmax", gnomad_af))
        gnomad_afr = float(gnomad_af_val.get("afr", 0.0))
        gnomad_amr = float(gnomad_af_val.get("amr", 0.0))
        gnomad_eas = float(gnomad_af_val.get("eas", 0.0))
        gnomad_nfe = float(gnomad_af_val.get("nfe", 0.0))
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
    # Legacy aliases for backward-compat with old models
    features["gnomad_popmax"] = gnomad_popmax
    features["gnomad_afr"] = gnomad_afr
    features["gnomad_amr"] = gnomad_amr
    features["gnomad_eas"] = gnomad_eas
    features["gnomad_nfe"] = gnomad_nfe

    features["gnomad_af_log"] = np.log10(gnomad_af + 1e-8)
    features["gnomad_popmax_log"] = np.log10(gnomad_popmax + 1e-8)
    features["is_rare"] = int(gnomad_af < 0.001)
    features["is_popmax_rare"] = int(gnomad_popmax < 0.001)
    features["af_x_blosum"] = gnomad_af * features["blosum62_score"]

    # ESM-2 Structural Features
    AA3_TO_1 = {"Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E",
                "Gly":"G","His":"H","Ile":"I","Leu":"L","Lys":"K","Met":"M","Phe":"F",
                "Pro":"P","Ser":"S","Thr":"T","Trp":"W","Tyr":"Y","Val":"V","Ter":"*"}

    esm_k = f"{AA_ref}{AA_pos}{AA_alt}"
    if esm_k in gene_data["esm2_dict"]:
        esm_v = gene_data["esm2_dict"][esm_k]
        features["esm2_cosine_sim"] = esm_v.get("cosine_similarity", 0.0)
        features["esm2_l2_shift"] = esm_v.get("l2_shift", 0.0)
        pca_arr = esm_v.get("pca_components", [])
        for i in range(ESM2_PCA_COMPONENTS):
            features[f"esm2_pca_{i}"] = float(pca_arr[i]) if i < len(pca_arr) else 0.0
    elif esm_model is not None and gene_data["esm2_pca"] is not None and gene_data["cds"] is not None and AA_alt != "Ter":
        # Novel mutation: Dynamic real-time PyTorch sequence embedding
        try:
            from Bio.Seq import Seq
            protein_seq = str(Seq(gene_data["cds"]).translate()).rstrip("*")
            # Bounds check: AA_pos must be within protein length
            if AA_pos < 1 or AA_pos > len(protein_seq):
                logger.warning(f"[ESM-2] AA_pos {AA_pos} out of protein range (len={len(protein_seq)}) for {gene_name}")
                features["esm2_cosine_sim"] = 0.0
                features["esm2_l2_shift"] = 0.0
                for i in range(ESM2_PCA_COMPONENTS):
                    features[f"esm2_pca_{i}"] = 0.0
                raise ValueError("AA_pos out of range")  # caught by outer except
            WINDOW = ESM2_WINDOW
            win_start = max(0, AA_pos - 1 - WINDOW)
            win_end   = min(len(protein_seq), AA_pos - 1 + WINDOW + 1)
            local_pos = AA_pos - 1 - win_start

            wt_window  = protein_seq[win_start:win_end]
            mut_window = list(wt_window)
            m_aa = AA3_TO_1.get(AA_alt, "A")
            if m_aa != "*":
                mut_window[local_pos] = m_aa
            mut_window = "".join(mut_window)

            data = [("wt", wt_window), ("mut", mut_window)]
            _, _, batch_tokens = esm_batch_converter(data)
            import torch
            from backend.models import DEVICE
            batch_tokens = batch_tokens.to(DEVICE)
            with torch.no_grad():
                results = esm_model(batch_tokens, repr_layers=[6], return_contacts=False)

            wt_emb  = results["representations"][6][0, local_pos + 1].cpu().numpy()
            mut_emb = results["representations"][6][1, local_pos + 1].cpu().numpy()

            diff = mut_emb - wt_emb
            cos_sim = float(np.dot(wt_emb, mut_emb) / (np.linalg.norm(wt_emb) * np.linalg.norm(mut_emb) + 1e-8))
            l2_shift = float(np.linalg.norm(diff))
            pca_arr = gene_data["esm2_pca"].transform([diff])[0]

            features["esm2_cosine_sim"] = cos_sim
            features["esm2_l2_shift"] = l2_shift
            for i in range(20):
                features[f"esm2_pca_{i}"] = float(pca_arr[i]) if i < len(pca_arr) else 0.0
        except (ImportError, ValueError, IndexError, KeyError, AttributeError, RuntimeError, OSError) as e:
            logger.warning(f"ESM-2 Dynamic compilation failed: {type(e).__name__}: {e}")
            features["esm2_cosine_sim"] = 0.0
            features["esm2_l2_shift"] = 0.0
            for i in range(20):
                features[f"esm2_pca_{i}"] = 0.0
    else:
        features["esm2_cosine_sim"] = 0.0
        features["esm2_l2_shift"] = 0.0
        for i in range(ESM2_PCA_COMPONENTS):
            features[f"esm2_pca_{i}"] = 0.0

    for m in ALL_MUTATIONS:
        features[f"Mutation_{m}"] = int(Mutation == m)
    for aa in ALL_AMINO_ACIDS:
        features[f"AA_ref_{aa}"] = int(AA_ref == aa)
    for aa in ALL_AMINO_ACIDS:
        features[f"AA_alt_{aa}"] = int(AA_alt == aa)

    vector = []
    feature_names = gene_data.get("feature_names", [])
    for name in feature_names:
        vector.append(features.get(name, 0))
    if not vector:
        return np.array([[]], dtype=np.float32)
    return np.array([vector], dtype=np.float32)


# ─── Human-readable feature labels ───────────────────────────────────────────
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
    "am_score": "AlphaMissense Score",
    "am_pathogenic": "AM Pathogenic",
    "am_x_phylop": "AM x PhyloP",
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
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe_critical_domain(raw_vector, gene_data, aa_pos) -> bool:
    """Safely check if a variant is in a critical domain without risking IndexError."""
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
