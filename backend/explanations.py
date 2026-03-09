"""
SteppeDNA - Explanation & Uncertainty Quantification

Handles:
- Training data index for data scarcity quantification (Item 41)
- Contrastive explanation pairs via KD-trees (Item 43)
- Bootstrap confidence interval computation (Item 39)
- Split conformal prediction set computation (Item 5.1)
"""

import os
import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.spatial import KDTree

from backend.models import (
    DATA_DIR, SUPPORTED_GENES, GENE_MAX_AA,
    _get_universal_models, _load_pickle,
    _BOOTSTRAP_MODELS, _CONFORMAL_THRESHOLDS,
)

logger = logging.getLogger("steppedna")


# ─── Training Data Index for Data Scarcity Quantification (Item 41) ──────────
_TRAINING_INDEX: dict = {}  # gene -> {"aa_positions": np.array, "ref_aas": list, ...}
DATA_SCARCITY_WINDOW = 50   # +/- AA positions for neighborhood count

def _build_training_index():
    """Build per-gene training data index from master dataset for data scarcity computation."""
    csv_path = os.path.join(DATA_DIR, "master_training_dataset.csv")
    if not os.path.exists(csv_path):
        logger.warning("[DATA-SCARCITY] master_training_dataset.csv not found -- data scarcity disabled")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.warning(f"[DATA-SCARCITY] Failed to load training CSV: {e}")
        return

    if "Gene" not in df.columns or "relative_aa_pos" not in df.columns:
        logger.warning("[DATA-SCARCITY] Required columns missing from training CSV")
        return

    # Reconstruct AA position from relative_aa_pos * aa_length
    aa_ref_cols = [c for c in df.columns if c.startswith("AA_ref_")]
    aa_alt_cols = [c for c in df.columns if c.startswith("AA_alt_")]

    for gene in SUPPORTED_GENES:
        gene_mask = df["Gene"] == gene
        gene_df = df[gene_mask]
        if len(gene_df) == 0:
            continue

        aa_length = GENE_MAX_AA.get(gene, 3418)
        aa_positions = np.round(gene_df["relative_aa_pos"].values * aa_length).astype(int)

        # Decode one-hot AA ref/alt back to amino acid names
        ref_aas = []
        alt_aas = []
        for _, row in gene_df.iterrows():
            ref_aa = "Unk"
            alt_aa = "Unk"
            for col in aa_ref_cols:
                if row[col] == 1:
                    ref_aa = col.replace("AA_ref_", "")
                    break
            for col in aa_alt_cols:
                if row[col] == 1:
                    alt_aa = col.replace("AA_alt_", "")
                    break
            ref_aas.append(ref_aa)
            alt_aas.append(alt_aa)

        # Determine domain membership from domain feature columns
        domain_cols = ["in_critical_repeat_region", "in_DNA_binding", "in_OB_folds", "in_NLS", "in_primary_interaction"]
        domains = []
        for _, row in gene_df.iterrows():
            domain = "uncharacterized"
            for dc in domain_cols:
                if dc in row and row[dc] == 1:
                    domain = dc
                    break
            domains.append(domain)

        _TRAINING_INDEX[gene] = {
            "aa_positions": aa_positions,
            "ref_aas": ref_aas,
            "alt_aas": alt_aas,
            "labels": gene_df["Label"].values.tolist(),
            "domains": domains,
        }
        logger.info(f"[DATA-SCARCITY] {gene}: indexed {len(aa_positions)} training variants")

    logger.info(f"[DATA-SCARCITY] Training index built for {len(_TRAINING_INDEX)} genes")


def compute_data_support(gene_name: str, aa_pos: int, aa_ref: str, aa_alt: str,
                         in_domain_name: str = "uncharacterized") -> dict:
    """Compute data scarcity quantification for a prediction.

    Returns dict with nearby_variants, same_substitution_type, in_known_domain, level.
    """
    gene = gene_name.upper()
    if gene not in _TRAINING_INDEX:
        return {"nearby_variants": 0, "same_substitution_type": 0,
                "in_known_domain": False, "level": "LOW"}

    idx = _TRAINING_INDEX[gene]
    positions = idx["aa_positions"]

    # Count training variants within +/-50 AA positions
    nearby_mask = np.abs(positions - aa_pos) <= DATA_SCARCITY_WINDOW
    nearby_count = int(nearby_mask.sum())

    # Count same substitution type (same ref+alt AA)
    same_subst_count = 0
    for i in range(len(idx["ref_aas"])):
        if idx["ref_aas"][i] == aa_ref and idx["alt_aas"][i] == aa_alt:
            same_subst_count += 1

    # Check if the variant's domain has training data
    in_known_domain = in_domain_name != "uncharacterized"

    # Compute support level
    if nearby_count > 100:
        level = "HIGH"
    elif nearby_count >= 10:
        level = "MODERATE"
    else:
        level = "LOW"

    return {
        "nearby_variants": nearby_count,
        "same_substitution_type": same_subst_count,
        "in_known_domain": in_known_domain,
        "level": level,
    }


# ─── Contrastive Explanation Pairs (Item 43) ─────────────────────────────────
_CONTRASTIVE_INDEX: dict = {}  # gene -> {"tree_path": KDTree, "tree_benign": KDTree, ...}

def _build_contrastive_index():
    """Build per-gene KD-trees on scaled training features for contrastive explanations."""
    csv_path = os.path.join(DATA_DIR, "master_training_dataset.csv")
    if not os.path.exists(csv_path):
        logger.warning("[CONTRASTIVE] master_training_dataset.csv not found -- contrastive explanations disabled")
        return

    models = _get_universal_models()
    model_scaler = models.get("scaler")
    model_feature_names = models.get("feature_names")
    if model_scaler is None or not model_feature_names:
        logger.warning("[CONTRASTIVE] Scaler or feature names not available -- contrastive explanations disabled")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.warning("[CONTRASTIVE] Failed to load training CSV: %s" % str(e))
        return

    if "Gene" not in df.columns or "Label" not in df.columns:
        logger.warning("[CONTRASTIVE] Required columns (Gene, Label) missing from training CSV")
        return

    # Decode one-hot AA columns for variant identifiers
    aa_ref_cols = [c for c in df.columns if c.startswith("AA_ref_")]
    aa_alt_cols = [c for c in df.columns if c.startswith("AA_alt_")]

    for gene in SUPPORTED_GENES:
        gene_mask = df["Gene"] == gene
        gene_df = df[gene_mask].copy()
        if len(gene_df) < 10:
            logger.info("[CONTRASTIVE] %s: too few variants (%d) -- skipping" % (gene, len(gene_df)))
            continue

        # Build feature matrix in model feature order
        feat_matrix = np.zeros((len(gene_df), len(model_feature_names)), dtype=np.float32)
        for j, fname in enumerate(model_feature_names):
            if fname in gene_df.columns:
                feat_matrix[:, j] = gene_df[fname].values

        # Scale using the production scaler
        try:
            scaled_matrix = model_scaler.transform(feat_matrix)
        except Exception as e:
            logger.warning("[CONTRASTIVE] %s: scaler.transform failed: %s" % (gene, str(e)))
            continue

        labels = gene_df["Label"].values.astype(int)

        # Reconstruct AA position and amino acid identifiers
        aa_length = GENE_MAX_AA.get(gene, 3418)
        aa_positions = np.round(gene_df["relative_aa_pos"].values * aa_length).astype(int)

        ref_aas = []
        alt_aas = []
        for _, row in gene_df.iterrows():
            ref_aa = "Unk"
            alt_aa = "Unk"
            for col in aa_ref_cols:
                if row.get(col, 0) == 1:
                    ref_aa = col.replace("AA_ref_", "")
                    break
            for col in aa_alt_cols:
                if row.get(col, 0) == 1:
                    alt_aa = col.replace("AA_alt_", "")
                    break
            ref_aas.append(ref_aa)
            alt_aas.append(alt_aa)

        # Build separate KD-trees for pathogenic (label=1) and benign (label=0)
        path_mask = labels == 1
        ben_mask = labels == 0

        gene_entry = {
            "scaled_features": scaled_matrix,
            "labels": labels,
            "aa_positions": aa_positions,
            "ref_aas": ref_aas,
            "alt_aas": alt_aas,
            "tree_pathogenic": None,
            "tree_benign": None,
            "path_indices": np.where(path_mask)[0],
            "ben_indices": np.where(ben_mask)[0],
        }

        if path_mask.sum() >= 1:
            gene_entry["tree_pathogenic"] = KDTree(scaled_matrix[path_mask])
        if ben_mask.sum() >= 1:
            gene_entry["tree_benign"] = KDTree(scaled_matrix[ben_mask])

        _CONTRASTIVE_INDEX[gene] = gene_entry
        logger.info("[CONTRASTIVE] %s: indexed %d variants (path=%d, ben=%d)" % (
            gene, len(gene_df), int(path_mask.sum()), int(ben_mask.sum())))

    logger.info("[CONTRASTIVE] Contrastive index built for %d genes" % len(_CONTRASTIVE_INDEX))


def find_contrastive_explanation(gene_name: str, scaled_vector: np.ndarray,
                                 probability: float, feature_names_list: list,
                                 nice_names: dict = None) -> dict:
    """Find the nearest opposite-class training variant and compute key feature differences.

    Returns dict with contrast_variant, contrast_class, contrast_distance,
    key_differences (top 5 features by difference magnitude), or None if unavailable.

    Args:
        nice_names: Optional dict mapping feature keys to human-readable names.
                    If None, raw feature names are used.
    """
    gene = gene_name.upper()
    if gene not in _CONTRASTIVE_INDEX:
        return None

    idx = _CONTRASTIVE_INDEX[gene]
    query_class = "Pathogenic" if probability > 0.5 else "Benign"

    # Find nearest variant of the OPPOSITE class
    if query_class == "Pathogenic":
        # Query is pathogenic -> find nearest benign
        tree = idx["tree_benign"]
        opposite_indices = idx["ben_indices"]
        contrast_class = "Benign"
    else:
        # Query is benign -> find nearest pathogenic
        tree = idx["tree_pathogenic"]
        opposite_indices = idx["path_indices"]
        contrast_class = "Pathogenic"

    if tree is None or len(opposite_indices) == 0:
        return None

    # Query the KD-tree for the nearest neighbor
    query_vec = scaled_vector.reshape(1, -1) if scaled_vector.ndim == 1 else scaled_vector
    try:
        dist, local_idx = tree.query(query_vec[0], k=1)
    except Exception:
        return None

    # Map local tree index back to global index in the gene's dataset
    global_idx = opposite_indices[local_idx]

    # Get the contrast variant's info
    contrast_ref = idx["ref_aas"][global_idx]
    contrast_alt = idx["alt_aas"][global_idx]
    contrast_pos = int(idx["aa_positions"][global_idx])
    contrast_variant = "%s%d%s" % (contrast_ref, contrast_pos, contrast_alt)

    # Compute feature-wise differences (on scaled features)
    contrast_features = idx["scaled_features"][global_idx]
    query_features = query_vec[0]
    feature_diffs = np.abs(query_features - contrast_features)

    # Rank by difference magnitude, take top 5
    top_indices = np.argsort(feature_diffs)[::-1][:5]

    _nice_names = nice_names or {}

    key_differences = []
    for fi in top_indices:
        if fi >= len(feature_names_list):
            continue
        fname = feature_names_list[fi]
        nice_name = _nice_names.get(fname, fname)
        diff_val = float(feature_diffs[fi])
        # Skip near-zero differences
        if diff_val < 0.01:
            continue
        # Determine importance level based on scaled difference magnitude
        if diff_val > 2.0:
            importance = "high"
        elif diff_val > 1.0:
            importance = "moderate"
        else:
            importance = "low"

        key_differences.append({
            "feature": nice_name,
            "feature_key": fname,
            "query_value": round(float(query_features[fi]), 3),
            "contrast_value": round(float(contrast_features[fi]), 3),
            "difference": round(diff_val, 3),
            "importance": importance,
        })

    if not key_differences:
        return None

    return {
        "contrast_variant": contrast_variant,
        "contrast_class": contrast_class,
        "contrast_distance": round(float(dist), 4),
        "key_differences": key_differences,
    }


# ─── Bootstrap Confidence Intervals (Item 39) ────────────────────────────────

def compute_bootstrap_ci(scaled_vector: np.ndarray, feature_names_list: list) -> dict:
    """Compute bootstrap confidence interval from pre-trained models.

    Returns dict with ci_lower, ci_upper, ci_width, n_models, or None if unavailable.
    """
    if not _BOOTSTRAP_MODELS:
        return None
    dmat = xgb.DMatrix(scaled_vector, feature_names=feature_names_list)
    preds = np.array([m.predict(dmat)[0] for m in _BOOTSTRAP_MODELS])
    ci_lower = float(np.percentile(preds, 5))
    ci_upper = float(np.percentile(preds, 95))
    return {
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "ci_width": round(ci_upper - ci_lower, 4),
        "n_models": len(_BOOTSTRAP_MODELS),
    }


# ─── Split Conformal Prediction (Item 5.1) ────────────────────────────────────

def compute_conformal_set(probability: float, gene_name: str) -> dict:
    """Compute the conformal prediction set for a given probability and gene.

    Returns dict with:
        conformal_set: list of class labels included at the coverage level
        conformal_coverage: the target coverage (e.g. 0.90)
        conformal_alpha: the alpha level (e.g. 0.10)
        set_size: number of classes in the set (1 or 2)
    Returns None if conformal thresholds are not loaded.
    """
    if not _CONFORMAL_THRESHOLDS:
        return None

    gene_upper = gene_name.upper()
    gene_info = _CONFORMAL_THRESHOLDS.get(gene_upper, _CONFORMAL_THRESHOLDS.get("_global"))
    if not gene_info:
        return None

    q = gene_info["quantile"]
    alpha = gene_info.get("alpha", 0.10)

    p_pathogenic = probability
    p_benign = 1.0 - probability

    pred_set = []
    if p_pathogenic >= 1.0 - q:
        pred_set.append("Pathogenic")
    if p_benign >= 1.0 - q:
        pred_set.append("Benign")

    # Safety: if empty set (shouldn't happen), include highest-probability class
    if not pred_set:
        pred_set.append("Pathogenic" if probability >= 0.5 else "Benign")

    return {
        "conformal_set": pred_set,
        "conformal_coverage": round(1 - alpha, 2),
        "conformal_alpha": alpha,
        "set_size": len(pred_set),
    }
