"""
SteppeDNA - Explanation & Uncertainty Quantification
=====================================================

When a machine learning model predicts that a DNA variant is "pathogenic" (disease-causing)
or "benign" (harmless), clinicians need more than just a number. They need to understand
WHY the model made that call and HOW CONFIDENT the model really is.

This module provides four explainability and uncertainty features that accompany every
prediction, helping users trust (or appropriately distrust) the model's output:

1. DATA SCARCITY (compute_data_support)
   Answers: "How much training data did the model have near this variant?"
   Method:  Counts how many training variants fall within +/-50 amino acid positions
            of the query variant in the same gene. More nearby data = more reliable prediction.
   Output:  A support level of LOW / MODERATE / HIGH.

2. CONTRASTIVE EXPLANATIONS (find_contrastive_explanation)
   Answers: "What is the nearest opposite-class variant and how does it differ?"
   Method:  Uses KD-trees (a spatial search structure -- like a phonebook for feature space
            that lets you quickly find the closest point) to locate the nearest benign
            training variant to a pathogenic prediction (or vice versa). Then it shows the
            top 5 features where the query and contrast variant differ most.
   Output:  The contrast variant identity, distance, and ranked feature differences.

3. BOOTSTRAP CONFIDENCE INTERVALS (compute_bootstrap_ci)
   Answers: "How stable is this prediction across different training samples?"
   Method:  50 XGBoost models were each trained on a randomly resampled (bootstrapped)
            version of the training data. The query variant is run through all 50 models,
            and the 5th-95th percentile spread of their predictions gives a confidence
            interval. A narrow interval means high agreement; wide means uncertainty.
   Output:  ci_lower, ci_upper, ci_width (the 90% confidence interval).

4. CONFORMAL PREDICTION SETS (compute_conformal_set)
   Answers: "With 90% statistical confidence, which classes could this variant belong to?"
   Method:  Split conformal prediction uses a held-out calibration set to compute a
            quantile threshold. If only {Pathogenic} is in the set, the model is confident.
            If both {Pathogenic, Benign} are in the set, the model is uncertain about this
            variant. This provides a formal statistical guarantee on coverage.
   Output:  A prediction set (e.g., ["Pathogenic"] or ["Pathogenic", "Benign"]).
"""

import os
import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.spatial import KDTree  # Spatial search tree for fast nearest-neighbor lookups

from backend.models import (
    DATA_DIR,                # Path to the data/ directory where models and datasets live
    SUPPORTED_GENES,         # List of the 5 HR genes: BRCA1, BRCA2, PALB2, RAD51C, RAD51D
    GENE_MAX_AA,             # Dict mapping each gene to its protein length in amino acids
    _get_universal_models,   # Returns the loaded XGBoost/MLP models, scaler, feature names
    _BOOTSTRAP_MODELS,       # List of 50 XGBoost models trained on bootstrapped data
    _CONFORMAL_THRESHOLDS,   # Dict of per-gene conformal quantile thresholds
)

logger = logging.getLogger("steppedna")


# ─── Training Data Index for Data Scarcity Quantification (Item 41) ──────────
#
# PURPOSE: When the model predicts a variant, we want to know if there was enough
# training data nearby to make that prediction reliable. A prediction in a region
# with 200 training variants is much more trustworthy than one in a region with 2.
#
# HOW IT WORKS:
# At startup, we load the entire training dataset and build a per-gene index of
# amino acid positions. When a new variant comes in for prediction, we count how
# many training variants fall within a +/-50 amino acid window around it.
#
# The index stores: positions, reference/alt amino acids, labels, and domain info
# for every training variant, organized by gene.

_TRAINING_INDEX: dict = {}  # Per-gene lookup: gene -> {"aa_positions": array, "ref_aas": list, ...}
DATA_SCARCITY_WINDOW = 50   # How far to look in each direction (in amino acid positions)


def _build_training_index():
    """Build per-gene training data index from master dataset for data scarcity computation.

    Called once at server startup. Reads the master training CSV, and for each of the
    5 supported genes, extracts amino acid positions, amino acid identities, labels,
    and domain memberships. These are stored in _TRAINING_INDEX for fast lookup later.
    """
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

    # The training data stores positions as a fraction (0.0 to 1.0) of the protein length.
    # We need to convert back to absolute amino acid positions by multiplying by the
    # protein length. For example, relative_aa_pos=0.5 in BRCA2 (3418 AA) -> position 1709.
    aa_ref_cols = [c for c in df.columns if c.startswith("AA_ref_")]
    aa_alt_cols = [c for c in df.columns if c.startswith("AA_alt_")]

    for gene in SUPPORTED_GENES:
        gene_mask = df["Gene"] == gene
        gene_df = df[gene_mask]
        if len(gene_df) == 0:
            continue

        # Convert relative position (0.0-1.0) back to absolute AA position
        aa_length = GENE_MAX_AA.get(gene, 3418)
        aa_positions = np.round(gene_df["relative_aa_pos"].values * aa_length).astype(int)

        # Decode one-hot encoded amino acid columns back to amino acid names.
        # The training data stores amino acids as one-hot vectors (e.g., AA_ref_Ala=1,
        # AA_ref_Val=0, ...). We reverse this to get the actual amino acid name ("Ala").
        ref_aas = []
        alt_aas = []
        for _, row in gene_df.iterrows():
            ref_aa = "Unk"  # Default if no one-hot column is active
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

        # Determine which protein domain each training variant falls in.
        # Domains are functional regions of the protein (e.g., DNA binding site, NLS).
        # Variants in characterized domains have more context for interpretation.
        domain_cols = ["in_critical_repeat_region", "in_DNA_binding", "in_OB_folds", "in_NLS", "in_primary_interaction"]
        domains = []
        for _, row in gene_df.iterrows():
            domain = "uncharacterized"
            for dc in domain_cols:
                if dc in row and row[dc] == 1:
                    domain = dc
                    break
            domains.append(domain)

        # Store everything in the index for this gene
        _TRAINING_INDEX[gene] = {
            "aa_positions": aa_positions,
            "ref_aas": ref_aas,
            "alt_aas": alt_aas,
            "labels": gene_df["Label"].values.tolist(),  # 0=benign, 1=pathogenic
            "domains": domains,
        }
        logger.info(f"[DATA-SCARCITY] {gene}: indexed {len(aa_positions)} training variants")

    logger.info(f"[DATA-SCARCITY] Training index built for {len(_TRAINING_INDEX)} genes")


def compute_data_support(gene_name: str, aa_pos: int, aa_ref: str, aa_alt: str,
                         in_domain_name: str = "uncharacterized") -> dict:
    """Compute data scarcity quantification for a prediction.

    This tells the user how much training data the model had near their variant.
    Think of it like a "reliability meter" -- predictions in well-studied protein regions
    (HIGH support) are more trustworthy than those in poorly-studied regions (LOW support).

    Args:
        gene_name: Which gene the variant is in (e.g., "BRCA2").
        aa_pos:    The amino acid position of the variant in the protein.
        aa_ref:    The original (reference) amino acid at this position.
        aa_alt:    The new (altered) amino acid caused by the mutation.
        in_domain_name: Which protein domain this position falls in, if any.

    Returns:
        Dict with: nearby_variants (count), same_substitution_type (count),
                   in_known_domain (bool), level ("LOW"/"MODERATE"/"HIGH").
    """
    gene = gene_name.upper()
    if gene not in _TRAINING_INDEX:
        # If we have no index for this gene, report minimal support
        return {"nearby_variants": 0, "same_substitution_type": 0,
                "in_known_domain": False, "level": "LOW"}

    idx = _TRAINING_INDEX[gene]
    positions = idx["aa_positions"]

    # Count training variants within +/-50 AA positions of the query variant.
    # For example, if the query is at position 500, we count all training variants
    # between positions 450 and 550.
    nearby_mask = np.abs(positions - aa_pos) <= DATA_SCARCITY_WINDOW
    nearby_count = int(nearby_mask.sum())

    # Count how many training variants have the exact same amino acid substitution
    # (e.g., Ala->Val). This is even more specific than just being nearby.
    same_subst_count = 0
    for i in range(len(idx["ref_aas"])):
        if idx["ref_aas"][i] == aa_ref and idx["alt_aas"][i] == aa_alt:
            same_subst_count += 1

    # A variant in a known functional domain (like DNA binding) has more biological
    # context than one in an uncharacterized region.
    in_known_domain = in_domain_name != "uncharacterized"

    # Assign a human-readable support level based on the count of nearby variants:
    #   > 100 nearby training variants  -> HIGH   (very well-studied region)
    #   10-100 nearby training variants -> MODERATE (some data available)
    #   < 10 nearby training variants   -> LOW    (sparse data, prediction less reliable)
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
#
# PURPOSE: Help users understand predictions by showing "the closest counterexample."
# If the model says a variant is Pathogenic, we find the most similar Benign variant
# from training data and show which features differ most. This gives users an intuitive
# sense of WHY the model classified it as pathogenic -- "your variant differs from the
# nearest benign variant mainly in conservation score and structural stability."
#
# HOW IT WORKS:
# We use KD-trees, which are a spatial data structure for fast nearest-neighbor search.
# Think of a KD-tree like a phonebook for multi-dimensional feature space: instead of
# searching alphabetically by name, it organizes points so you can quickly find the
# closest one in 120-dimensional feature space without checking every single point.
#
# We build TWO KD-trees per gene: one containing all pathogenic training variants and
# one containing all benign training variants. When a new variant is predicted as
# pathogenic, we search the BENIGN tree for its nearest neighbor (and vice versa).

_CONTRASTIVE_INDEX: dict = {}  # Per-gene lookup: gene -> {"tree_pathogenic": KDTree, "tree_benign": KDTree, ...}


def _build_contrastive_index():  # noqa: C901
    """Build per-gene KD-trees on scaled training features for contrastive explanations.

    Called once at server startup. For each gene, this function:
    1. Loads the training dataset and extracts the 120 model features.
    2. Scales the features using the same scaler used during prediction (important
       so that distances are comparable across features with different units).
    3. Splits variants into pathogenic vs. benign groups.
    4. Builds a KD-tree for each group to enable fast nearest-neighbor search later.
    """
    csv_path = os.path.join(DATA_DIR, "master_training_dataset.csv")
    if not os.path.exists(csv_path):
        logger.warning("[CONTRASTIVE] master_training_dataset.csv not found -- contrastive explanations disabled")
        return

    # We need the same scaler and feature names that the production model uses,
    # so that the training variants are in the same "coordinate system" as new predictions.
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

    # Collect column names for one-hot encoded amino acids (used to recover variant identity)
    aa_ref_cols = [c for c in df.columns if c.startswith("AA_ref_")]
    aa_alt_cols = [c for c in df.columns if c.startswith("AA_alt_")]

    for gene in SUPPORTED_GENES:
        gene_mask = df["Gene"] == gene
        gene_df = df[gene_mask].copy()
        if len(gene_df) < 10:
            logger.info("[CONTRASTIVE] %s: too few variants (%d) -- skipping" % (gene, len(gene_df)))
            continue

        # Build a feature matrix with columns in the exact same order as the model expects.
        # If a feature column is missing from the CSV, it stays as 0 (safe default).
        feat_matrix = np.zeros((len(gene_df), len(model_feature_names)), dtype=np.float32)
        for j, fname in enumerate(model_feature_names):
            if fname in gene_df.columns:
                feat_matrix[:, j] = gene_df[fname].values

        # Scale features using the production StandardScaler so that all features
        # are on the same scale (mean=0, std=1). Without this, a feature ranging
        # 0-1000 would dominate distance calculations over one ranging 0-1.
        try:
            scaled_matrix = model_scaler.transform(feat_matrix)
        except Exception as e:
            logger.warning("[CONTRASTIVE] %s: scaler.transform failed: %s" % (gene, str(e)))
            continue

        labels = gene_df["Label"].values.astype(int)  # 0=benign, 1=pathogenic

        # Reconstruct amino acid position and identity from the training data
        # (same decoding as in the data scarcity index above)
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

        # Split into two groups: pathogenic variants and benign variants.
        # Each group gets its own KD-tree for nearest-neighbor search.
        path_mask = labels == 1
        ben_mask = labels == 0

        gene_entry = {
            "scaled_features": scaled_matrix,   # All variants' features (for computing diffs later)
            "labels": labels,
            "aa_positions": aa_positions,
            "ref_aas": ref_aas,
            "alt_aas": alt_aas,
            "tree_pathogenic": None,             # KD-tree of pathogenic variants only
            "tree_benign": None,                 # KD-tree of benign variants only
            "path_indices": np.where(path_mask)[0],  # Mapping from KD-tree index -> global index
            "ben_indices": np.where(ben_mask)[0],
        }

        # Build the KD-trees. Each tree only contains variants of one class,
        # so when we search for the nearest neighbor we're guaranteed to find
        # a variant of the opposite class.
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

    This is the core of the contrastive explanation system. Given a new variant that the
    model predicted as (say) Pathogenic with probability 0.85, this function finds the
    closest Benign variant from the training data and shows which features differ most.

    Example output for a Pathogenic prediction:
        "The nearest benign variant is Val600Ala (distance: 1.23).
         Key differences: your variant has much higher conservation score (+2.1),
         lower structural stability (-1.5), ..."

    This helps clinicians understand the decision boundary -- what would need to change
    for the model to flip its prediction.

    Args:
        gene_name:          Which gene (e.g., "BRCA2").
        scaled_vector:      The query variant's 120 features, already scaled by StandardScaler.
        probability:        The model's predicted probability of pathogenicity (0.0 to 1.0).
        feature_names_list: List of 120 feature names in the same order as scaled_vector.
        nice_names:         Optional dict mapping internal feature keys (e.g., "phylop_score")
                            to human-readable names (e.g., "Conservation (PhyloP)").

    Returns:
        Dict with contrast_variant, contrast_class, contrast_distance,
        key_differences (top 5 features by difference magnitude), or None if unavailable.
    """
    gene = gene_name.upper()
    if gene not in _CONTRASTIVE_INDEX:
        return None

    idx = _CONTRASTIVE_INDEX[gene]

    # Determine what class the model predicted, so we can search for the OPPOSITE class
    query_class = "Pathogenic" if probability > 0.5 else "Benign"

    # Select the appropriate KD-tree: if the query is pathogenic, search among benign
    # variants (and vice versa). This ensures we find the nearest COUNTEREXAMPLE.
    if query_class == "Pathogenic":
        tree = idx["tree_benign"]
        opposite_indices = idx["ben_indices"]
        contrast_class = "Benign"
    else:
        tree = idx["tree_pathogenic"]
        opposite_indices = idx["path_indices"]
        contrast_class = "Pathogenic"

    if tree is None or len(opposite_indices) == 0:
        return None

    # Query the KD-tree to find the single nearest neighbor (k=1).
    # The KD-tree makes this O(log n) instead of O(n) -- much faster than checking
    # every training variant one by one.
    query_vec = scaled_vector.reshape(1, -1) if scaled_vector.ndim == 1 else scaled_vector
    try:
        dist, local_idx = tree.query(query_vec[0], k=1)
    except Exception:
        return None

    # The KD-tree returns an index relative to just the pathogenic (or benign) subset.
    # We need to map it back to the global index in the full gene dataset so we can
    # look up the variant's amino acid info from the stored arrays.
    global_idx = opposite_indices[local_idx]

    # Reconstruct the contrast variant's human-readable identity (e.g., "Val600Ala")
    contrast_ref = idx["ref_aas"][global_idx]
    contrast_alt = idx["alt_aas"][global_idx]
    contrast_pos = int(idx["aa_positions"][global_idx])
    contrast_variant = "%s%d%s" % (contrast_ref, contrast_pos, contrast_alt)

    # Now compute WHICH FEATURES differ most between the query and the contrast variant.
    # We compare the scaled feature values (not raw) because scaling ensures all features
    # contribute equally to the comparison.
    contrast_features = idx["scaled_features"][global_idx]
    query_features = query_vec[0]
    feature_diffs = np.abs(query_features - contrast_features)

    # Sort features by how much they differ, take the top 5 most different ones.
    # These are the features that best explain WHY the two variants got different predictions.
    top_indices = np.argsort(feature_diffs)[::-1][:5]

    _nice_names = nice_names or {}

    key_differences = []
    for fi in top_indices:
        if fi >= len(feature_names_list):
            continue
        fname = feature_names_list[fi]
        nice_name = _nice_names.get(fname, fname)  # Use human-readable name if available
        diff_val = float(feature_diffs[fi])

        # Skip near-zero differences (not informative)
        if diff_val < 0.01:
            continue

        # Classify how important this difference is, based on scaled magnitude.
        # Since features are standardized (mean=0, std=1), a difference of 2.0
        # means the features are 2 standard deviations apart -- very significant.
        if diff_val > 2.0:
            importance = "high"      # > 2 std devs apart: major difference
        elif diff_val > 1.0:
            importance = "moderate"  # 1-2 std devs apart: notable difference
        else:
            importance = "low"       # < 1 std dev apart: minor difference

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
#
# PURPOSE: Measure how STABLE a prediction is across different training samples.
# If the model was trained on slightly different data, would it give the same answer?
#
# HOW IT WORKS:
# During training, we created 50 "bootstrapped" versions of the training dataset
# (each is a random sample WITH replacement -- some variants appear multiple times,
# others are left out). We trained a separate XGBoost model on each version.
#
# At prediction time, we run the query variant through ALL 50 models and collect
# their predictions. If all 50 models agree (e.g., all predict ~0.9), the confidence
# interval is narrow and the prediction is robust. If they disagree (e.g., predictions
# range from 0.3 to 0.8), the interval is wide and we should be cautious.
#
# We report the 5th and 95th percentiles, giving a 90% confidence interval.
# Example: ci_lower=0.82, ci_upper=0.91 means "90% of bootstrap models predict
# between 0.82 and 0.91" -- a tight, reliable prediction.

def compute_bootstrap_ci(scaled_vector: np.ndarray, feature_names_list: list) -> dict:
    """Compute bootstrap confidence interval from pre-trained models.

    Runs the query variant through all 50 bootstrapped XGBoost models and takes
    the 5th-95th percentile spread of their predictions.

    Args:
        scaled_vector:      The variant's 120 features, already scaled.
        feature_names_list: List of 120 feature names (needed by XGBoost DMatrix).

    Returns:
        Dict with ci_lower, ci_upper, ci_width, n_models, or None if unavailable.
        Gracefully returns None if bootstrap models have a different feature count
        (e.g., trained on v5.3 with 103 features but current model uses 120).
    """
    if not _BOOTSTRAP_MODELS:
        return None
    try:
        # Create an XGBoost DMatrix (the input format XGBoost expects)
        dmat = xgb.DMatrix(scaled_vector, feature_names=feature_names_list)

        # Run the variant through each of the 50 bootstrap models and collect predictions
        preds = np.array([m.predict(dmat)[0] for m in _BOOTSTRAP_MODELS])

        # Take the 5th and 95th percentiles to form a 90% confidence interval
        ci_lower = float(np.percentile(preds, 5))
        ci_upper = float(np.percentile(preds, 95))
        return {
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "ci_width": round(ci_upper - ci_lower, 4),  # Narrow = stable, wide = uncertain
            "n_models": len(_BOOTSTRAP_MODELS),
        }
    except Exception as e:
        logger.warning(f"[BOOTSTRAP-CI] Failed: {e}. Falling back to Beta approximation.")
        return None


# ─── Split Conformal Prediction (Item 5.1) ────────────────────────────────────
#
# PURPOSE: Provide a statistical GUARANTEE on prediction reliability.
# Unlike bootstrap CIs (which measure model stability), conformal prediction gives
# a formal coverage guarantee: "with 90% probability, the true class is in this set."
#
# HOW IT WORKS:
# During training, we held out a calibration set (separate from training and test data).
# We ran the calibration set through the model and looked at how "wrong" the model was
# on those examples. Specifically, we computed "nonconformity scores" (1 - probability
# of the true class) for each calibration example, then took a quantile (e.g., the 90th
# percentile) of those scores. This quantile becomes our threshold.
#
# At prediction time, we include a class in the prediction set if its predicted
# probability is high enough (>= 1 - quantile). The logic is:
#   - If the model says P(Pathogenic) = 0.95 and the quantile threshold is 0.85,
#     then 0.95 >= (1 - 0.85) = 0.15, so Pathogenic is included.
#     And P(Benign) = 0.05 < 0.15, so Benign is NOT included.
#     Result: {"Pathogenic"} -- the model is confident.
#
#   - If the model says P(Pathogenic) = 0.55 and the quantile is 0.85,
#     then 0.55 >= 0.15 (Pathogenic included) and 0.45 >= 0.15 (Benign also included).
#     Result: {"Pathogenic", "Benign"} -- the model is uncertain, both classes are plausible.
#
# Set size interpretation:
#   - Size 1: Model is confident in one class (reliable prediction).
#   - Size 2: Model is uncertain; both classes are plausible (use caution).

def compute_conformal_set(probability: float, gene_name: str) -> dict:
    """Compute the conformal prediction set for a given probability and gene.

    Uses pre-computed quantile thresholds (from the calibration set) to determine
    which class labels should be included in the prediction set at the target
    coverage level (typically 90%).

    Args:
        probability: The model's predicted probability of pathogenicity (0.0 to 1.0).
        gene_name:   Which gene (e.g., "BRCA2"). Per-gene thresholds are used when
                     available; otherwise falls back to a global threshold.

    Returns:
        Dict with:
            conformal_set:      List of class labels included (e.g., ["Pathogenic"])
            conformal_coverage: The target coverage level (e.g., 0.90)
            conformal_alpha:    The significance level (e.g., 0.10)
            set_size:           Number of classes in the set (1 = confident, 2 = uncertain)
        Returns None if conformal thresholds are not loaded.
    """
    if not _CONFORMAL_THRESHOLDS:
        return None

    # Try gene-specific threshold first, fall back to global threshold
    gene_upper = gene_name.upper()
    gene_info = _CONFORMAL_THRESHOLDS.get(gene_upper, _CONFORMAL_THRESHOLDS.get("_global"))
    if not gene_info:
        return None

    # The quantile (q) was computed on the calibration set as the (1-alpha) quantile
    # of nonconformity scores. Alpha is typically 0.10 for 90% coverage.
    q = gene_info["quantile"]
    alpha = gene_info.get("alpha", 0.10)

    p_pathogenic = probability
    p_benign = 1.0 - probability

    # Include a class in the prediction set if its probability meets the threshold.
    # The threshold is (1 - q): a class is included if p_class >= 1 - q.
    # Higher q (from a well-calibrated model) means a lower bar to include a class,
    # which means more classes get included (larger sets, more conservative).
    pred_set = []
    if p_pathogenic >= 1.0 - q:
        pred_set.append("Pathogenic")
    if p_benign >= 1.0 - q:
        pred_set.append("Benign")

    # Safety fallback: if neither class meets the threshold (theoretically shouldn't
    # happen with properly calibrated thresholds), include the most likely class.
    if not pred_set:
        pred_set.append("Pathogenic" if probability >= 0.5 else "Benign")

    return {
        "conformal_set": pred_set,
        "conformal_coverage": round(1 - alpha, 2),
        "conformal_alpha": alpha,
        "set_size": len(pred_set),
    }
