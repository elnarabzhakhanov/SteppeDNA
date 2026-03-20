"""
Shared biological lookup tables and feature engineering for SteppeDNA.
Used by both training and inference. don't touch the dictionaries, they took forever to transcribe.
"""
import os
import pickle
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("steppedna")

# ============================================================
# BIOLOGICAL LOOKUP TABLES
# ============================================================
AA_HYDROPHOBICITY = {
    'Ala': 1.8,  'Arg': -4.5, 'Asn': -3.5, 'Asp': -3.5, 'Cys': 2.5,
    'Gln': -3.5, 'Glu': -3.5, 'Gly': -0.4, 'His': -3.2, 'Ile': 4.5,
    'Leu': 3.8,  'Lys': -3.9, 'Met': 1.9,  'Phe': 2.8,  'Pro': -1.6,
    'Ser': -0.8, 'Thr': -0.7, 'Trp': -0.9, 'Tyr': -1.3, 'Val': 4.2,
    'Ter': -10.0
}

AA_VOLUME = {
    'Ala': 67,  'Arg': 148, 'Asn': 96,  'Asp': 91,  'Cys': 86,
    'Gln': 114, 'Glu': 109, 'Gly': 48,  'His': 118, 'Ile': 124,
    'Leu': 124, 'Lys': 135, 'Met': 124, 'Phe': 135, 'Pro': 90,
    'Ser': 73,  'Thr': 93,  'Trp': 163, 'Tyr': 141, 'Val': 105,
    'Ter': 0
}

BLOSUM62 = {
    ('Ala', 'Ala'): 4,  ('Ala', 'Arg'): -1, ('Ala', 'Asn'): -2, ('Ala', 'Asp'): -2,
    ('Ala', 'Cys'): 0,  ('Ala', 'Gln'): -1, ('Ala', 'Glu'): -1, ('Ala', 'Gly'): 0,
    ('Ala', 'His'): -2, ('Ala', 'Ile'): -1, ('Ala', 'Leu'): -1, ('Ala', 'Lys'): -1,
    ('Ala', 'Met'): -1, ('Ala', 'Phe'): -2, ('Ala', 'Pro'): -1, ('Ala', 'Ser'): 1,
    ('Ala', 'Thr'): 0,  ('Ala', 'Trp'): -3, ('Ala', 'Tyr'): -2, ('Ala', 'Val'): 0,
    ('Arg', 'Arg'): 5,  ('Arg', 'Asn'): -1, ('Arg', 'Asp'): -2, ('Arg', 'Cys'): -3,
    ('Arg', 'Gln'): 1,  ('Arg', 'Glu'): 0,  ('Arg', 'Gly'): -2, ('Arg', 'His'): 0,
    ('Arg', 'Ile'): -3, ('Arg', 'Leu'): -2, ('Arg', 'Lys'): 2,  ('Arg', 'Met'): -1,
    ('Arg', 'Phe'): -3, ('Arg', 'Pro'): -2, ('Arg', 'Ser'): -1, ('Arg', 'Thr'): -1,
    ('Arg', 'Trp'): -3, ('Arg', 'Tyr'): -2, ('Arg', 'Val'): -3,
    ('Asn', 'Asn'): 6,  ('Asn', 'Asp'): 1,  ('Asn', 'Cys'): -3, ('Asn', 'Gln'): 0,
    ('Asn', 'Glu'): 0,  ('Asn', 'Gly'): 0,  ('Asn', 'His'): 1,  ('Asn', 'Ile'): -3,
    ('Asn', 'Leu'): -3, ('Asn', 'Lys'): 0,  ('Asn', 'Met'): -2, ('Asn', 'Phe'): -3,
    ('Asn', 'Pro'): -2, ('Asn', 'Ser'): 1,  ('Asn', 'Thr'): 0,  ('Asn', 'Trp'): -4,
    ('Asn', 'Tyr'): -2, ('Asn', 'Val'): -3,
    ('Asp', 'Asp'): 6,  ('Asp', 'Cys'): -3, ('Asp', 'Gln'): 0,  ('Asp', 'Glu'): 2,
    ('Asp', 'Gly'): -1, ('Asp', 'His'): -1, ('Asp', 'Ile'): -3, ('Asp', 'Leu'): -4,
    ('Asp', 'Lys'): -1, ('Asp', 'Met'): -3, ('Asp', 'Phe'): -3, ('Asp', 'Pro'): -1,
    ('Asp', 'Ser'): 0,  ('Asp', 'Thr'): -1, ('Asp', 'Trp'): -4, ('Asp', 'Tyr'): -3,
    ('Asp', 'Val'): -3,
    ('Cys', 'Cys'): 9,  ('Cys', 'Gln'): -3, ('Cys', 'Glu'): -4, ('Cys', 'Gly'): -3,
    ('Cys', 'His'): -3, ('Cys', 'Ile'): -1, ('Cys', 'Leu'): -1, ('Cys', 'Lys'): -3,
    ('Cys', 'Met'): -1, ('Cys', 'Phe'): -2, ('Cys', 'Pro'): -3, ('Cys', 'Ser'): -1,
    ('Cys', 'Thr'): -1, ('Cys', 'Trp'): -2, ('Cys', 'Tyr'): -2, ('Cys', 'Val'): -1,
    ('Gln', 'Gln'): 5,  ('Gln', 'Glu'): 2,  ('Gln', 'Gly'): -2, ('Gln', 'His'): 0,
    ('Gln', 'Ile'): -3, ('Gln', 'Leu'): -2, ('Gln', 'Lys'): 1,  ('Gln', 'Met'): 0,
    ('Gln', 'Phe'): -3, ('Gln', 'Pro'): -1, ('Gln', 'Ser'): 0,  ('Gln', 'Thr'): -1,
    ('Gln', 'Trp'): -2, ('Gln', 'Tyr'): -1, ('Gln', 'Val'): -2,
    ('Glu', 'Glu'): 5,  ('Glu', 'Gly'): -2, ('Glu', 'His'): 0,  ('Glu', 'Ile'): -3,
    ('Glu', 'Leu'): -3, ('Glu', 'Lys'): 1,  ('Glu', 'Met'): -2, ('Glu', 'Phe'): -3,
    ('Glu', 'Pro'): -1, ('Glu', 'Ser'): 0,  ('Glu', 'Thr'): -1, ('Glu', 'Trp'): -3,
    ('Glu', 'Tyr'): -2, ('Glu', 'Val'): -2,
    ('Gly', 'Gly'): 6,  ('Gly', 'His'): -2, ('Gly', 'Ile'): -4, ('Gly', 'Leu'): -4,
    ('Gly', 'Lys'): -2, ('Gly', 'Met'): -3, ('Gly', 'Phe'): -3, ('Gly', 'Pro'): -2,
    ('Gly', 'Ser'): 0,  ('Gly', 'Thr'): -2, ('Gly', 'Trp'): -2, ('Gly', 'Tyr'): -3,
    ('Gly', 'Val'): -3,
    ('His', 'His'): 8,  ('His', 'Ile'): -3, ('His', 'Leu'): -3, ('His', 'Lys'): -1,
    ('His', 'Met'): -2, ('His', 'Phe'): -1, ('His', 'Pro'): -2, ('His', 'Ser'): -1,
    ('His', 'Thr'): -2, ('His', 'Trp'): -2, ('His', 'Tyr'): 2,  ('His', 'Val'): -3,
    ('Ile', 'Ile'): 4,  ('Ile', 'Leu'): 2,  ('Ile', 'Lys'): -1, ('Ile', 'Met'): 1,
    ('Ile', 'Phe'): 0,  ('Ile', 'Pro'): -3, ('Ile', 'Ser'): -2, ('Ile', 'Thr'): -1,
    ('Ile', 'Trp'): -3, ('Ile', 'Tyr'): -1, ('Ile', 'Val'): 3,
    ('Leu', 'Leu'): 4,  ('Leu', 'Lys'): -2, ('Leu', 'Met'): 2,  ('Leu', 'Phe'): 0,
    ('Leu', 'Pro'): -3, ('Leu', 'Ser'): -2, ('Leu', 'Thr'): -1, ('Leu', 'Trp'): -2,
    ('Leu', 'Tyr'): -1, ('Leu', 'Val'): 1,
    ('Lys', 'Lys'): 5,  ('Lys', 'Met'): -1, ('Lys', 'Phe'): -3, ('Lys', 'Pro'): -1,
    ('Lys', 'Ser'): 0,  ('Lys', 'Thr'): -1, ('Lys', 'Trp'): -3, ('Lys', 'Tyr'): -2,
    ('Lys', 'Val'): -2,
    ('Met', 'Met'): 5,  ('Met', 'Phe'): 0,  ('Met', 'Pro'): -2, ('Met', 'Ser'): -1,
    ('Met', 'Thr'): -1, ('Met', 'Trp'): -1, ('Met', 'Tyr'): -1, ('Met', 'Val'): 1,
    ('Phe', 'Phe'): 6,  ('Phe', 'Pro'): -4, ('Phe', 'Ser'): -2, ('Phe', 'Thr'): -2,
    ('Phe', 'Trp'): 1,  ('Phe', 'Tyr'): 3,  ('Phe', 'Val'): -1,
    ('Pro', 'Pro'): 7,  ('Pro', 'Ser'): -1, ('Pro', 'Thr'): -1, ('Pro', 'Trp'): -4,
    ('Pro', 'Tyr'): -3, ('Pro', 'Val'): -2,
    ('Ser', 'Ser'): 4,  ('Ser', 'Thr'): 1,  ('Ser', 'Trp'): -3, ('Ser', 'Tyr'): -2,
    ('Ser', 'Val'): -2,
    ('Thr', 'Thr'): 5,  ('Thr', 'Trp'): -2, ('Thr', 'Tyr'): -2, ('Thr', 'Val'): 0,
    ('Trp', 'Trp'): 11, ('Trp', 'Tyr'): 2,  ('Trp', 'Val'): -3,
    ('Tyr', 'Tyr'): 7,  ('Tyr', 'Val'): -1,
    ('Val', 'Val'): 4,
}

ALL_AMINO_ACIDS = ['Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly', 'His', 'Ile',
                   'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser', 'Ter', 'Thr', 'Trp', 'Tyr', 'Val']
ALL_MUTATIONS = ['A>C', 'A>G', 'A>T', 'C>A', 'C>G', 'C>T', 'G>A', 'G>C', 'G>T', 'T>A', 'T>C', 'T>G']

# ============================================================
# HELPERS
# ============================================================


def get_blosum62(ref, alt):
    if ref == alt:
        return 4
    if alt == 'Ter' or ref == 'Ter':
        return -4
    pair = (ref, alt)
    rev = (alt, ref)
    if pair in BLOSUM62:
        return BLOSUM62[pair]
    if rev in BLOSUM62:
        return BLOSUM62[rev]
    return -1


def get_charge(aa):
    if aa in {"Arg", "Lys", "His"}:
        return "positive"
    if aa in {"Asp", "Glu"}:
        return "negative"
    if aa in {"Ala", "Val", "Ile", "Leu", "Met", "Phe", "Trp", "Pro", "Gly"}:
        return "nonpolar"
    if aa in {"Ser", "Thr", "Cys", "Tyr", "Asn", "Gln"}:
        return "polar"
    return "other"

# ============================================================
# DATA LOADING
# ============================================================


def load_phylop_scores(data_dir="data"):
    phylop_path = os.path.join(data_dir, "phylop_scores.pkl")
    if os.path.exists(phylop_path):
        with open(phylop_path, "rb") as f:
            return pickle.load(f)
    print("  [WARNING] PhyloP scores not found. Run fetch_phylop.py first.")
    return None


def load_alphamissense_scores(data_dir="data"):
    am_path = os.path.join(data_dir, "alphamissense_scores.pkl")
    if os.path.exists(am_path):
        with open(am_path, "rb") as f:
            return pickle.load(f)
    print("  [WARNING] AlphaMissense scores not found. Run fetch_alphamissense.py first.")
    return None


def load_mave_scores(data_dir="data"):
    mave_path = os.path.join(data_dir, "mave_scores.pkl")
    if os.path.exists(mave_path):
        with open(mave_path, "rb") as f:
            return pickle.load(f)
    print("  [WARNING] MAVE scores not found. Run fetch_mave.py first.")
    return None


def load_structural_features(data_dir="data"):
    path = os.path.join(data_dir, "structural_features.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    print("  [WARNING] Structural features not found. Run fetch_alphafold.py first.")
    return None


def load_gnomad_frequencies(data_dir="data"):
    path = os.path.join(data_dir, "gnomad_frequencies.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    print("  [WARNING] gnomAD frequencies not found. Run fetch_gnomad.py first.")
    return None


def load_spliceai_scores(data_dir="data"):
    path = os.path.join(data_dir, "spliceai_scores.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    print("  [WARNING] SpliceAI scores not found. Run fetch_spliceai.py first.")
    return None


def load_esm2_embeddings(data_dir="data"):
    path = os.path.join(data_dir, "esm2_embeddings.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)["embeddings"]
    print("  [WARNING] ESM-2 embeddings not found. Run generate_esm2_embeddings.py first.")
    return None

# ============================================================
# FEATURE ENGINEERING (DataFrame version — used in training)
# ============================================================


def engineer_features(mutation_df, phylop_scores=None, mave_data=None, am_data=None,  # noqa: C901
                      structural_data=None, gnomad_data=None, spliceai_data=None, esm2_data=None,
                      gene_name="BRCA2", use_mave=True, eve_data=None):
    # main feature engineering loop. warning: adding new columns here means you have to retrain the entire XGBoost model from scratch.
    X = pd.DataFrame()
    X["cDNA_pos"] = mutation_df["cDNA_pos"]
    X["AA_pos"] = mutation_df["AA_pos"]
    import json
    import os
    config_path = os.path.join(os.path.dirname(__file__), "gene_configs", f"{gene_name.lower()}.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            gene_config: dict = json.load(f)
    else:
        raise FileNotFoundError(
            f"Gene configuration not found for {gene_name}. Expected at: {config_path}. "
            f"Create a JSON config with cds_length, aa_length, and domains fields."
        )

    X["relative_cdna_pos"] = mutation_df["cDNA_pos"] / gene_config["cds_length"]
    X["relative_aa_pos"] = mutation_df["AA_pos"] / gene_config["aa_length"]
    X["blosum62_score"] = mutation_df.apply(lambda r: get_blosum62(r["AA_ref"], r["AA_alt"]), axis=1)
    X["volume_diff"] = (mutation_df["AA_ref"].map(AA_VOLUME) - mutation_df["AA_alt"].map(AA_VOLUME)).abs().fillna(0)
    ref_hydro = mutation_df["AA_ref"].map(AA_HYDROPHOBICITY).fillna(0)
    alt_hydro = mutation_df["AA_alt"].map(AA_HYDROPHOBICITY).fillna(0)
    X["hydro_diff"] = (ref_hydro - alt_hydro).abs()
    X["ref_hydro"] = ref_hydro
    X["alt_hydro"] = alt_hydro
    X["hydro_delta"] = alt_hydro - ref_hydro

    # Generic Domain Engineering
    domains = gene_config.get("domains", {})

    def in_domain(pos, d_name):
        rng = domains.get(d_name)
        if hasattr(pos, 'fillna'):  # pandas series
            return ((pos >= rng[0]) & (pos <= rng[1])).astype(int) if rng else pd.Series(0, index=pos.index)
        return int(rng[0] <= pos <= rng[1]) if rng else 0

    def check_domains(*d_names):
        for d in d_names:
            if domains.get(d):
                return in_domain(mutation_df["AA_pos"], d)
        logger.debug("No matching domain found in gene config for candidates: %s (available: %s)", d_names, list(domains.keys()))
        if hasattr(mutation_df["AA_pos"], 'fillna'):
            return pd.Series(0, index=mutation_df["AA_pos"].index)
        return 0

    # Try generic names first (for PALB2/BRCA1/RAD51), fallback to BRCA2 specifically for backward compatibility
    X["in_critical_repeat_region"] = check_domains("BRC_repeats", "WD40_repeats", "BRCT1", "BRCT2", "SCD")
    X["in_DNA_binding"] = check_domains("DNA_binding", "ChAM_DNA_binding", "ssDNA_binding")
    X["in_OB_folds"] = check_domains("OB_folds", "RING", "Walker_A", "Walker_B")
    X["in_NLS"] = check_domains("NLS_nuclear_localization", "N_terminal_domain")
    X["in_primary_interaction"] = check_domains("PALB2_interaction", "BRCA1_interaction", "BARD1_interaction", "RAD51B_RAD51D_XRCC3_interaction", "Holliday_junction_resolution")

    # Gene-specific computed features (domain distance, multi-domain, functional zone)
    aa_len = gene_config.get("aa_length", 1000)
    aa_pos_series = mutation_df["AA_pos"].astype(int)

    # Distance to nearest domain boundary (normalized)
    def _dist_nearest(pos):
        min_d = aa_len
        for d_name, d_range in domains.items():
            if isinstance(d_range, (list, tuple)) and len(d_range) >= 2:
                if d_range[0] <= pos <= d_range[1]:
                    return 0.0
                dist = min(abs(pos - d_range[0]), abs(pos - d_range[1]))
                min_d = min(min_d, dist)
        return min_d / aa_len

    # Number of domains at this position
    def _n_domains(pos):
        count = 0
        for d_name, d_range in domains.items():
            if isinstance(d_range, (list, tuple)) and len(d_range) >= 2:
                if d_range[0] <= pos <= d_range[1]:
                    count += 1
        return count

    # Functional zone score (weighted proximity to all domains)
    def _func_zone(pos):
        score = 0.0
        for d_name, d_range in domains.items():
            if isinstance(d_range, (list, tuple)) and len(d_range) >= 2:
                if d_range[0] <= pos <= d_range[1]:
                    score += 1.0
                else:
                    dist = min(abs(pos - d_range[0]), abs(pos - d_range[1]))
                    score += 1.0 / (1.0 + dist / 50.0)
        return score

    X["dist_nearest_domain"] = aa_pos_series.apply(_dist_nearest)
    X["n_domains_hit"] = aa_pos_series.apply(_n_domains)
    X["in_multi_domain"] = (X["n_domains_hit"] > 1).astype(int)
    X["functional_zone_score"] = aa_pos_series.apply(_func_zone)
    X["func_zone_x_blosum"] = X["functional_zone_score"] * X["blosum62_score"]
    # func_zone_x_phylop deferred until after phylop_score is created

    X["is_nonsense"] = (mutation_df["AA_alt"] == "Ter").astype(int)
    transitions = {"A>G", "G>A", "C>T", "T>C"}
    X["is_transition"] = mutation_df["Mutation"].isin(transitions).astype(int)
    X["is_transversion"] = (~mutation_df["Mutation"].isin(transitions)).astype(int)
    ref_charge = mutation_df["AA_ref"].apply(get_charge)
    alt_charge = mutation_df["AA_alt"].apply(get_charge)
    X["charge_change"] = (ref_charge != alt_charge).astype(int)
    X["nonpolar_to_charged"] = ((ref_charge == "nonpolar") & (alt_charge.isin(["positive", "negative"]))).astype(int)

    # PhyloP conservation
    if phylop_scores is not None:
        X["phylop_score"] = mutation_df["cDNA_pos"].astype(int).map(phylop_scores).fillna(0.0)
        X["high_conservation"] = (X["phylop_score"] > 4.0).astype(int)
        X["ultra_conservation"] = (X["phylop_score"] > 7.0).astype(int)
        X["conserv_x_blosum"] = X["phylop_score"] * X["blosum62_score"]
    else:
        X["phylop_score"] = 0.0
        X["high_conservation"] = 0
        X["ultra_conservation"] = 0
        X["conserv_x_blosum"] = 0.0

    # Deferred interaction: needs phylop_score to exist
    X["func_zone_x_phylop"] = X["functional_zone_score"] * X["phylop_score"]

    # MAVE HDR functional scores (Hu et al. 2024, BRCA2 only)
    # MAVE leakage assessment status: ASSESSED (see VALIDATION_REPORT.md Section 7).
    # Many MAVE-assayed variants overlap with ClinVar training labels, creating
    # potential indirect label leakage. Ablation shows minimal impact (deltaAUC=-0.002).
    # use_mave=False disables MAVE features to avoid this leakage. With
    # use_mave=False, MAVE feature columns are still created (zeroed out) to
    # maintain feature vector compatibility. Only 3.5% of test variants have
    # MAVE scores; the model does not rely on MAVE for its predictions.
    if mave_data is not None and use_mave:
        by_variant = mave_data.get("by_variant", {})
        by_position = mave_data.get("by_position", {})

        def get_mave_score(row):
            key = f"{row['AA_ref']}{int(row['AA_pos'])}{row['AA_alt']}"
            if key in by_variant:
                return by_variant[key]
            cdna = int(row['cDNA_pos'])
            if cdna in by_position:
                return by_position[cdna]
            return 0.0

        X["mave_score"] = mutation_df.apply(get_mave_score, axis=1)
        X["has_mave"] = (X["mave_score"] != 0.0).astype(int)
        X["mave_abnormal"] = (X["mave_score"].between(0.01, 1.49)).astype(int)
        X["mave_x_blosum"] = X["mave_score"] * X["blosum62_score"]
    else:
        X["mave_score"] = 0.0
        X["has_mave"] = 0
        X["mave_abnormal"] = 0
        X["mave_x_blosum"] = 0.0

    # AlphaMissense scores
    if am_data is not None:
        am_by_variant = am_data.get("by_variant", {})
        am_by_position = am_data.get("by_position", {})

        def get_am_score(row):
            key = f"{row['AA_ref']}{int(row['AA_pos'])}{row['AA_alt']}"
            if key in am_by_variant:
                return am_by_variant[key]
            aa_pos = int(row['AA_pos'])
            if aa_pos in am_by_position:
                return am_by_position[aa_pos]
            return 0.0

        X["am_score"] = mutation_df.apply(get_am_score, axis=1)
        X["am_pathogenic"] = (X["am_score"] > 0.564).astype(int)
        X["am_x_phylop"] = X["am_score"] * X["phylop_score"]
    else:
        X["am_score"] = 0.0
        X["am_pathogenic"] = 0
        X["am_x_phylop"] = 0.0

    # 3D Structural Features
    if structural_data is not None:
        def get_struct_feat(row, feat_name, default=0.0):
            aa_pos = int(row["AA_pos"])
            entry = structural_data.get(aa_pos, {})
            return entry.get(feat_name, default)

        X["rsa"] = mutation_df.apply(lambda r: get_struct_feat(r, "rsa", 0.4), axis=1)
        X["is_buried"] = (X["rsa"] < 0.25).astype(int)
        X["bfactor"] = mutation_df.apply(lambda r: get_struct_feat(r, "bfactor", 50.0), axis=1)
        X["dist_dna"] = mutation_df.apply(lambda r: get_struct_feat(r, "dist_dna", 999.0), axis=1)
        X["dist_palb2"] = mutation_df.apply(lambda r: get_struct_feat(r, "dist_palb2", 999.0), axis=1)
        X["is_dna_contact"] = mutation_df.apply(lambda r: int(get_struct_feat(r, "is_dna_contact", False)), axis=1)

        # Secondary structure one-hot (supports both old 'ss' string and new 'ss_helix' int format)
        def _get_ss_helix(row):
            aa_pos = int(row['AA_pos'])
            entry = structural_data.get(aa_pos, {})
            if 'ss_helix' in entry:
                return int(entry['ss_helix'])
            return 1 if entry.get('ss', 'C') in ('H', 'G', 'I') else 0

        def _get_ss_sheet(row):
            aa_pos = int(row['AA_pos'])
            entry = structural_data.get(aa_pos, {})
            if 'ss_sheet' in entry:
                return int(entry['ss_sheet'])
            return 1 if entry.get('ss', 'C') in ('E', 'B') else 0
        X["ss_helix"] = mutation_df.apply(_get_ss_helix, axis=1)
        X["ss_sheet"] = mutation_df.apply(_get_ss_sheet, axis=1)

        # Interaction features
        X["buried_x_blosum"] = X["is_buried"] * X["blosum62_score"]
        X["dna_contact_x_blosum"] = X["is_dna_contact"] * X["blosum62_score"]
    else:
        X["rsa"] = 0.4
        X["is_buried"] = 0
        X["bfactor"] = 50.0
        X["dist_dna"] = 999.0
        X["dist_palb2"] = 999.0
        X["is_dna_contact"] = 0
        X["ss_helix"] = 0
        X["ss_sheet"] = 0
        X["buried_x_blosum"] = 0.0
        X["dna_contact_x_blosum"] = 0.0

    # gnomAD Allele Frequency features
    if gnomad_data is not None:
        gnomad_by_variant = gnomad_data.get("by_variant", {})
        gnomad_by_position = gnomad_data.get("by_position", {})

        def get_gnomad_af(row, freq_type="af"):
            # Handle field name aliases (popmax -> popmax_af)
            lookup_keys = [freq_type]
            if freq_type == 'popmax':
                lookup_keys = ['popmax', 'popmax_af']
            key = f"{row['AA_ref']}{int(row['AA_pos'])}{row['AA_alt']}"
            if key in gnomad_by_variant:
                if isinstance(gnomad_by_variant[key], dict):
                    for lk in lookup_keys:
                        if lk in gnomad_by_variant[key]:
                            return gnomad_by_variant[key][lk]
                    return 0.0
                elif freq_type == "af":  # Backwards compatibility with old pickle format
                    return gnomad_by_variant[key]
                return 0.0

            if int(row["cDNA_pos"]) in gnomad_by_position:
                if isinstance(gnomad_by_position[int(row["cDNA_pos"])], dict):
                    for lk in lookup_keys:
                        if lk in gnomad_by_position[int(row["cDNA_pos"])]:
                            return gnomad_by_position[int(row["cDNA_pos"])][lk]
                    return 0.0
                elif freq_type == "af":
                    return gnomad_by_position[int(row["cDNA_pos"])]
            return 0.0

        raw_af = mutation_df.apply(lambda r: get_gnomad_af(r, "af"), axis=1)
        popmax_af = mutation_df.apply(lambda r: get_gnomad_af(r, "popmax"), axis=1)

        X["gnomad_af"] = raw_af
        X["gnomad_popmax_af"] = popmax_af
        X["gnomad_af_afr"] = mutation_df.apply(lambda r: get_gnomad_af(r, "afr"), axis=1)
        X["gnomad_af_amr"] = mutation_df.apply(lambda r: get_gnomad_af(r, "amr"), axis=1)
        X["gnomad_af_eas"] = mutation_df.apply(lambda r: get_gnomad_af(r, "eas"), axis=1)
        X["gnomad_af_nfe"] = mutation_df.apply(lambda r: get_gnomad_af(r, "nfe"), axis=1)

        X["gnomad_af_log"] = np.log10(raw_af + 1e-8)  # Log-transform
        X["gnomad_popmax_log"] = np.log10(popmax_af + 1e-8)
        X["is_rare"] = (raw_af < 0.001).astype(int)
        X["is_popmax_rare"] = (popmax_af < 0.001).astype(int)
        X["af_x_blosum"] = raw_af * X["blosum62_score"]

    else:
        X["gnomad_af"] = 0.0
        X["gnomad_popmax_af"] = 0.0
        X["gnomad_af_afr"] = 0.0
        X["gnomad_af_amr"] = 0.0
        X["gnomad_af_eas"] = 0.0
        X["gnomad_af_nfe"] = 0.0
        X["gnomad_af_log"] = np.log10(1e-8)
        X["gnomad_popmax_log"] = np.log10(1e-8)
        X["is_rare"] = 1
        X["is_popmax_rare"] = 1
        X["af_x_blosum"] = 0.0

    # SpliceAI Features
    if spliceai_data is not None:
        def get_splice_score(row):
            key = (str(row['AA_ref']).strip(), int(row['AA_pos']), str(row['AA_alt']).strip())
            return float(spliceai_data.get(key, 0.0))

        X["spliceai_score"] = mutation_df.apply(get_splice_score, axis=1)
        X["splice_pathogenic"] = (X["spliceai_score"] > 0.5).astype(int)
    else:
        X["spliceai_score"] = 0.0
        X["splice_pathogenic"] = 0

    # ESM-2 Structural Features (Meta PLM)
    if esm2_data is not None:
        def get_esm_feat(row, feat_key, default=0.0):
            k = f"{row['AA_ref']}{int(row['AA_pos'])}{row['AA_alt']}"
            if k in esm2_data:
                return float(esm2_data[k].get(feat_key, default))
            return default

        def get_esm_pca(row, comp_idx):
            k = f"{row['AA_ref']}{int(row['AA_pos'])}{row['AA_alt']}"
            if k in esm2_data and "pca_components" in esm2_data[k]:
                pca_arr = esm2_data[k]["pca_components"]
                if comp_idx < len(pca_arr):
                    return float(pca_arr[comp_idx])
            return 0.0

        X["esm2_cosine_sim"] = mutation_df.apply(lambda r: get_esm_feat(r, "cosine_similarity"), axis=1)
        X["esm2_l2_shift"] = mutation_df.apply(lambda r: get_esm_feat(r, "l2_shift"), axis=1)

        for i in range(20):
            X[f"esm2_pca_{i}"] = mutation_df.apply(lambda r: get_esm_pca(r, i), axis=1)
    else:
        X["esm2_cosine_sim"] = 0.0
        X["esm2_l2_shift"] = 0.0
        for i in range(20):
            X[f"esm2_pca_{i}"] = 0.0

    X_cat = pd.DataFrame({"Mutation": mutation_df["Mutation"], "AA_ref": mutation_df["AA_ref"], "AA_alt": mutation_df["AA_alt"]})
    X = pd.concat([X, X_cat], axis=1)

    # Enforce exact one-hot encoded columns structurally to guarantee 115 feature shape universally
    for m in ALL_MUTATIONS:
        X[f"Mutation_{m}"] = (X["Mutation"] == m).astype(int)
    for aa in ALL_AMINO_ACIDS:
        X[f"AA_ref_{aa}"] = (X["AA_ref"] == aa).astype(int)
    for aa in ALL_AMINO_ACIDS:
        X[f"AA_alt_{aa}"] = (X["AA_alt"] == aa).astype(int)

    X.drop(columns=["Mutation", "AA_ref", "AA_alt"], inplace=True)

    # EVE score (Item 44): optional evolutionary variant effect feature
    if eve_data is not None:
        by_var = eve_data.get("by_variant", {})
        by_pos = eve_data.get("by_position", {})

        def _lookup_eve(row):
            aa_ref = mutation_df.loc[row.name, "AA_ref"] if "AA_ref" in mutation_df.columns else None
            aa_alt = mutation_df.loc[row.name, "AA_alt"] if "AA_alt" in mutation_df.columns else None
            aa_pos = int(row.get("AA_pos", 0))
            if aa_ref and aa_alt and aa_pos > 0:
                vkey = f"{aa_ref}{aa_pos}{aa_alt}"
                if vkey in by_var:
                    return by_var[vkey]
                if aa_pos in by_pos:
                    return by_pos[aa_pos]
            return 0.0

        X["eve_score"] = X.apply(_lookup_eve, axis=1)
        X["eve_pathogenic"] = (X["eve_score"] > 0.5).astype(int)
        X["eve_x_phylop"] = X["eve_score"] * X["phylop_score"]
    else:
        X["eve_score"] = 0.0
        X["eve_pathogenic"] = 0
        X["eve_x_phylop"] = 0.0

    # We enforce generic structural domain names to support any gene dynamically (e.g. PALB2, RAD51C)
    # The machine learning model now receives these generic flags regardless of the exact domain name.

    # Filter only desired features
    final_cols = [
        'cDNA_pos', 'AA_pos', 'relative_cdna_pos', 'relative_aa_pos', 'blosum62_score',
        'volume_diff', 'hydro_diff', 'ref_hydro', 'alt_hydro', 'hydro_delta',
        'in_critical_repeat_region', 'in_DNA_binding', 'in_OB_folds', 'in_NLS', 'in_primary_interaction',
        'is_nonsense', 'is_transition', 'is_transversion', 'charge_change', 'nonpolar_to_charged',
        'phylop_score', 'high_conservation', 'ultra_conservation', 'conserv_x_blosum',
        'mave_score', 'has_mave', 'mave_abnormal', 'mave_x_blosum',
        # AM features removed (v5.4): ablation showed leakage hurts non-BRCA2 genes
        'dist_nearest_domain', 'n_domains_hit', 'in_multi_domain', 'functional_zone_score', 'func_zone_x_blosum', 'func_zone_x_phylop',
        'rsa', 'is_buried', 'bfactor', 'dist_dna', 'dist_palb2', 'is_dna_contact',
        'ss_helix', 'ss_sheet', 'buried_x_blosum', 'dna_contact_x_blosum',
        # Sub-pop AF integration
        'gnomad_af', 'gnomad_popmax_af', 'gnomad_af_afr', 'gnomad_af_amr', 'gnomad_af_eas', 'gnomad_af_nfe',
        'gnomad_af_log', 'gnomad_popmax_log', 'is_rare', 'is_popmax_rare', 'af_x_blosum',
        'spliceai_score', 'splice_pathogenic',
        'esm2_cosine_sim', 'esm2_l2_shift',
        'esm2_pca_0', 'esm2_pca_1', 'esm2_pca_2', 'esm2_pca_3', 'esm2_pca_4',
        'esm2_pca_5', 'esm2_pca_6', 'esm2_pca_7', 'esm2_pca_8', 'esm2_pca_9',
        'esm2_pca_10', 'esm2_pca_11', 'esm2_pca_12', 'esm2_pca_13', 'esm2_pca_14',
        'esm2_pca_15', 'esm2_pca_16', 'esm2_pca_17', 'esm2_pca_18', 'esm2_pca_19'
    ]

    # Add one-hot encodings to final cols list
    for m in ALL_MUTATIONS:
        final_cols.append(f"Mutation_{m}")
    for aa in ALL_AMINO_ACIDS:
        final_cols.append(f"AA_ref_{aa}")
    for aa in ALL_AMINO_ACIDS:
        final_cols.append(f"AA_alt_{aa}")

    # EVE evolutionary coupling features
    final_cols.extend(["eve_score", "eve_pathogenic", "eve_x_phylop"])

    for col in final_cols:
        if col not in X.columns:
            X[col] = 0.0

    return X[final_cols]
