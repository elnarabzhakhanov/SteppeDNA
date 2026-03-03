"""
SteppeDNA: Active Learning Priority Ranker (Item 42)
=====================================================
Ranks VUS variants by their value for functional validation using a
query-by-committee approach combined with gene scarcity and positional novelty.

Priority = QBC_score * scarcity_weight * novelty_weight

Where:
  QBC_score = |xgb_prob - mlp_prob|  (model disagreement)
  scarcity_weight = 1/sqrt(gene_training_size)  (rarer genes weighted higher)
  novelty_weight = 1/(1 + nearby_training_variants_within_50aa)

Output:
  data/active_learning_priorities.json

Run from project root:
  python scripts/active_learning_ranker.py
"""

import os
import sys
import math
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.feature_engineering import engineer_features

DATA_DIR = "data"
RANDOM_STATE = 42
VUS_CSV = os.path.join(DATA_DIR, "vus_predictions_multigene.csv")
MASTER_CSV = os.path.join(DATA_DIR, "master_training_dataset.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "active_learning_priorities.json")
TOP_N_PER_GENE = 50
NEARBY_WINDOW = 50  # +/- AA positions

GENES = ["BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D"]

# Gene training sizes (from master dataset counts)
GENE_TRAINING_SIZES = {
    "BRCA2": 10085,
    "BRCA1": 4849,
    "PALB2": 2456,
    "RAD51C": 1048,
    "RAD51D": 785,
}

GENE_CDS_LENGTH = {
    "BRCA1": 5592, "BRCA2": 10257, "PALB2": 3561,
    "RAD51C": 1131, "RAD51D": 987,
}

GENE_MAX_AA = {
    "BRCA1": 1863, "BRCA2": 3418, "PALB2": 1186,
    "RAD51C": 376, "RAD51D": 328,
}

print("=" * 65, flush=True)
print("  SteppeDNA: Active Learning Priority Ranker (Item 42)", flush=True)
print("=" * 65, flush=True)

# ---------------------------------------------------------------------------
# 1. Load Models
# ---------------------------------------------------------------------------
print("\n[1/5] Loading models...", flush=True)

with open(os.path.join(DATA_DIR, "universal_feature_names.pkl"), "rb") as f:
    feature_names = pickle.load(f)
with open(os.path.join(DATA_DIR, "universal_scaler_ensemble.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(DATA_DIR, "universal_calibrator_ensemble.pkl"), "rb") as f:
    calibrator = pickle.load(f)

xgb_model = xgb.Booster()
xgb_model.load_model(os.path.join(DATA_DIR, "universal_xgboost_final.json"))

nn_model = None
try:
    import tensorflow as tf
    nn_model = tf.keras.models.load_model(
        os.path.join(DATA_DIR, "universal_nn.h5"), compile=False
    )
    print("  Loaded XGBoost + MLP ensemble", flush=True)
except Exception:
    print("  XGBoost-only mode (TensorFlow not available)", flush=True)

# ---------------------------------------------------------------------------
# 2. Load VUS predictions
# ---------------------------------------------------------------------------
print("\n[2/5] Loading VUS predictions...", flush=True)

if not os.path.exists(VUS_CSV):
    print("  ERROR: %s not found. Run vus_reclassification_multigene.py first." % VUS_CSV, flush=True)
    sys.exit(1)

vus_df = pd.read_csv(VUS_CSV)
print("  Loaded %d VUS variants" % len(vus_df), flush=True)

# ---------------------------------------------------------------------------
# 3. Build training position index for novelty scoring
# ---------------------------------------------------------------------------
print("\n[3/5] Building training position index...", flush=True)

master_df = pd.read_csv(MASTER_CSV)
training_positions = {}  # gene -> numpy array of AA positions

for gene in GENES:
    gene_mask = master_df["Gene"] == gene
    gene_df = master_df[gene_mask]
    if len(gene_df) == 0:
        training_positions[gene] = np.array([])
        continue
    aa_length = GENE_MAX_AA.get(gene, 3418)
    positions = np.round(gene_df["relative_aa_pos"].values * aa_length).astype(int)
    training_positions[gene] = positions
    print("  %s: %d training variants indexed" % (gene, len(positions)), flush=True)

# ---------------------------------------------------------------------------
# 4. Load per-gene lookup data for feature engineering
# ---------------------------------------------------------------------------
print("\n[4/5] Loading per-gene data and computing priorities...", flush=True)


def load_pkl(fname):
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def load_gene_lookups(gene):
    """Load per-gene lookup tables for feature engineering."""
    data = {}
    for key, gene_file, global_file in [
        ("phylop", "%s_phylop_scores.pkl" % gene.lower(), "phylop_scores.pkl"),
        ("mave", "%s_mave_scores.pkl" % gene.lower(), "mave_scores.pkl"),
        ("am", "%s_alphamissense_scores.pkl" % gene.lower(), "alphamissense_scores.pkl"),
        ("structural", "%s_structural_features.pkl" % gene.lower(), "structural_features.pkl"),
        ("gnomad", "%s_gnomad_frequencies.pkl" % gene.lower(), "gnomad_frequencies.pkl"),
        ("spliceai", "%s_spliceai_scores.pkl" % gene.lower(), "spliceai_scores.pkl"),
        ("esm2", "%s_esm2_embeddings.pkl" % gene.lower(), "esm2_embeddings.pkl"),
    ]:
        gene_path = os.path.join(DATA_DIR, gene_file)
        global_path = os.path.join(DATA_DIR, global_file)
        if os.path.exists(gene_path):
            raw = load_pkl(gene_file)
        elif os.path.exists(global_path):
            raw = load_pkl(global_file)
        else:
            raw = {}
        if key == "esm2" and isinstance(raw, dict) and "embeddings" in raw:
            raw = raw["embeddings"]
        data[key] = raw
    return data


def compute_nearby_count(gene, aa_pos):
    """Count training variants within +/- NEARBY_WINDOW of aa_pos."""
    positions = training_positions.get(gene, np.array([]))
    if len(positions) == 0:
        return 0
    return int(np.sum(np.abs(positions - aa_pos) <= NEARBY_WINDOW))


# Process all VUS variants per gene using batch feature engineering
results = {}
total_processed = 0
total_skipped = 0

for gene in GENES:
    gene_vus = vus_df[vus_df["gene"] == gene].copy()
    if len(gene_vus) == 0:
        print("  %s: 0 VUS variants - skipping" % gene, flush=True)
        results[gene] = []
        continue

    print("  %s: processing %d VUS variants..." % (gene, len(gene_vus)), flush=True)

    # Load gene-specific lookup data
    gene_data = load_gene_lookups(gene)
    cds_length = GENE_CDS_LENGTH.get(gene, 10257)

    # Scarcity weight for this gene
    train_size = GENE_TRAINING_SIZES.get(gene, 1000)
    scarcity_weight = 1.0 / math.sqrt(train_size)

    # Build DataFrame for batch feature engineering (same as vus_reclassification_multigene.py)
    rows = []
    vus_list = []
    for _, row in gene_vus.iterrows():
        aa_ref = str(row["aa_ref"])
        aa_pos = int(row["aa_pos"])
        aa_alt = str(row["aa_alt"])
        cdna_pos_est = aa_pos * 3 - 1
        rows.append({
            "cDNA_pos": cdna_pos_est,
            "AA_pos": aa_pos,
            "AA_ref": aa_ref,
            "AA_alt": aa_alt,
            "Ref_nt": "N",
            "Alt_nt": "N",
            "Mutation": "N>N",
        })
        vus_list.append({
            "aa_ref": aa_ref,
            "aa_pos": aa_pos,
            "aa_alt": aa_alt,
            "variant_key": "%s%d%s" % (aa_ref, aa_pos, aa_alt),
        })

    mut_df = pd.DataFrame(rows)

    try:
        features_df = engineer_features(
            mut_df,
            phylop_scores=gene_data.get("phylop"),
            mave_data=gene_data.get("mave"),
            am_data=gene_data.get("am"),
            structural_data=gene_data.get("structural"),
            gnomad_data=gene_data.get("gnomad"),
            spliceai_data=gene_data.get("spliceai"),
            esm2_data=gene_data.get("esm2"),
            gene_name=gene,
        )
    except Exception as e:
        print("    [ERROR] engineer_features failed: %s" % str(e), flush=True)
        results[gene] = []
        total_skipped += len(gene_vus)
        continue

    # Align to model's feature order
    feature_matrix = np.zeros((len(features_df), len(feature_names)))
    for j, fname in enumerate(feature_names):
        if fname in features_df.columns:
            feature_matrix[:, j] = features_df[fname].values

    # Scale
    scaled_matrix = scaler.transform(feature_matrix)

    # Get separate XGBoost and MLP predictions
    dmat = xgb.DMatrix(scaled_matrix, feature_names=feature_names)
    xgb_probs = xgb_model.predict(dmat)

    if nn_model is not None:
        mlp_probs = nn_model.predict(scaled_matrix, verbose=0).flatten()
    else:
        mlp_probs = xgb_probs.copy()  # No disagreement if MLP unavailable

    # Calibrated blended predictions
    raw_probs = 0.6 * xgb_probs + 0.4 * mlp_probs
    cal_probs = calibrator.predict(raw_probs)

    # Compute priorities for each variant
    gene_priorities = []

    for i, vus in enumerate(vus_list):
        aa_pos = vus["aa_pos"]
        xgb_prob = float(xgb_probs[i])
        mlp_prob = float(mlp_probs[i])
        calibrated = float(cal_probs[i])

        # Query-by-Committee score
        qbc_score = abs(xgb_prob - mlp_prob)

        # Novelty weight
        nearby_count = compute_nearby_count(gene, aa_pos)
        novelty_weight = 1.0 / (1.0 + nearby_count)

        # Combined priority
        priority = qbc_score * scarcity_weight * novelty_weight

        # Determine reason for high priority
        reasons = []
        if qbc_score > 0.3:
            reasons.append("high model disagreement (%.3f)" % qbc_score)
        if scarcity_weight > 0.02:
            reasons.append("data-scarce gene (%d training)" % train_size)
        if novelty_weight > 0.5:
            reasons.append("low positional coverage (%d nearby)" % nearby_count)
        reason_str = "; ".join(reasons) if reasons else "moderate priority"

        gene_priorities.append({
            "variant": "%s:p.%s" % (gene, vus["variant_key"]),
            "aa_pos": aa_pos,
            "aa_ref": vus["aa_ref"],
            "aa_alt": vus["aa_alt"],
            "priority_score": round(priority, 6),
            "qbc_score": round(qbc_score, 4),
            "xgb_prob": round(xgb_prob, 4),
            "mlp_prob": round(mlp_prob, 4),
            "current_prediction": round(calibrated, 4),
            "nearby_training": nearby_count,
            "scarcity_weight": round(scarcity_weight, 6),
            "novelty_weight": round(novelty_weight, 4),
            "reason": reason_str,
        })
        total_processed += 1

    # Sort by priority descending
    gene_priorities.sort(key=lambda x: x["priority_score"], reverse=True)

    # Keep top N
    results[gene] = gene_priorities[:TOP_N_PER_GENE]

    if gene_priorities:
        top = gene_priorities[0]
        print("    Top priority: %s (score=%.6f, QBC=%.4f)" % (
            top["variant"], top["priority_score"], top["qbc_score"]
        ), flush=True)
        print("    Total ranked: %d, kept top %d" % (
            len(gene_priorities), min(len(gene_priorities), TOP_N_PER_GENE)
        ), flush=True)

# ---------------------------------------------------------------------------
# 5. Save results
# ---------------------------------------------------------------------------
print("\n[5/5] Saving results...", flush=True)

output = {
    "metadata": {
        "nearby_window": NEARBY_WINDOW,
        "top_n_per_gene": TOP_N_PER_GENE,
        "total_processed": total_processed,
        "total_skipped": total_skipped,
        "method": "query_by_committee * gene_scarcity * positional_novelty",
        "description": "Variants ranked by their value for functional validation. "
                       "High-priority variants are those where the ML models disagree "
                       "(indicating uncertain classification), from data-scarce genes, "
                       "and in under-represented protein regions.",
    },
    "gene_training_sizes": GENE_TRAINING_SIZES,
    "priorities": results,
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print("  Saved to %s" % OUTPUT_PATH, flush=True)

# Summary table
print("\n  Summary:", flush=True)
print("  %-8s  %6s  %10s  %10s  %10s" % (
    "Gene", "Ranked", "Top Score", "Med Score", "Scarcity"
), flush=True)
print("  " + "-" * 50, flush=True)

for gene in GENES:
    entries = results.get(gene, [])
    if entries:
        scores = [e["priority_score"] for e in entries]
        print("  %-8s  %6d  %10.6f  %10.6f  %10.6f" % (
            gene, len(entries), max(scores), float(np.median(scores)),
            1.0 / math.sqrt(GENE_TRAINING_SIZES[gene])
        ), flush=True)
    else:
        print("  %-8s  %6d  %10s  %10s  %10.6f" % (
            gene, 0, "N/A", "N/A",
            1.0 / math.sqrt(GENE_TRAINING_SIZES[gene])
        ), flush=True)

print("\n  Processed: %d, Skipped: %d" % (total_processed, total_skipped), flush=True)
print("\nDone. Active learning priorities ready.", flush=True)
