"""
SteppeDNA: Split Conformal Prediction Training (Item 5.1)

Computes gene-stratified nonconformity scores on the calibration split
and saves per-gene quantile thresholds to data/conformal_thresholds.json.

Split conformal prediction provides distribution-free coverage guarantees:
at prediction time, the resulting prediction SET is guaranteed to contain
the true class with probability >= 1 - alpha (default 90%).

Uses the same 60/20/20 split logic and RANDOM_STATE=42 as
train_universal_model.py to ensure identical data partitioning.

Run from project root:
    python scripts/train_conformal.py
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
XGB_WEIGHT = 0.6
NN_WEIGHT = 0.4
ALPHA = 0.10  # 90% coverage guarantee
MIN_CONFORMAL_SAMPLES = 20  # minimum calibration samples per gene

print("=" * 60)
print("  SteppeDNA: Split Conformal Prediction Training")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data and models
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/5] Loading data and pre-trained models...")

df = pd.read_csv("data/master_training_dataset.csv")
with open("data/universal_feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

feature_names = [f for f in feature_names if f in df.columns]
X = df[feature_names].values
y = df["Label"].values
genes = df["Gene"].values

print(f"  Variants: {len(X)} across {len(set(genes))} genes")

# Load pre-trained models
with open("data/universal_scaler_ensemble.pkl", "rb") as f:
    scaler = pickle.load(f)

from tensorflow.keras.models import load_model
nn_model = load_model("data/universal_nn.h5")

xgb_clf = xgb.XGBClassifier()
xgb_clf.load_model("data/universal_xgboost_final.json")

# Load gene-specific ensemble weights if available
gene_weights_path = "data/gene_ensemble_weights.json"
gene_ensemble_weights = {}
if os.path.exists(gene_weights_path):
    with open(gene_weights_path, "r") as f:
        raw_weights = json.load(f)
    for gene, w in raw_weights.items():
        gene_ensemble_weights[gene.upper()] = {
            "xgb_weight": w.get("xgb_weight", XGB_WEIGHT),
            "mlp_weight": w.get("mlp_weight", NN_WEIGHT),
        }
    print(f"  Gene-specific ensemble weights loaded for {len(gene_ensemble_weights)} genes.")
else:
    print("  Using default ensemble weights (XGB 60% / MLP 40%).")

# Load gene-specific calibrators if available
gene_calibrators = {}
for gene in sorted(set(genes)):
    cal_path = f"data/calibrator_{gene.lower()}.pkl"
    if os.path.exists(cal_path):
        with open(cal_path, "rb") as f:
            gene_calibrators[gene.upper()] = pickle.load(f)
if gene_calibrators:
    print(f"  Gene-specific calibrators loaded for: {', '.join(sorted(gene_calibrators.keys()))}")

# Load universal calibrator as fallback
with open("data/universal_calibrator_ensemble.pkl", "rb") as f:
    universal_calibrator = pickle.load(f)

print("  Models loaded OK.")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Reproduce the exact same split as train_universal_model.py
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Reproducing 60/20/20 split...")

strata = np.array([f"{g}_{l}" for g, l in zip(genes, y)])
X_trainval, X_test, y_trainval, y_test, genes_trainval, genes_test, strata_tv, _ = train_test_split(
    X, y, genes, strata, test_size=0.20, random_state=RANDOM_STATE, stratify=strata
)
strata_tv_inner = np.array([f"{g}_{l}" for g, l in zip(genes_trainval, y_trainval)])
X_train, X_cal, y_train, y_cal, genes_train, genes_cal = train_test_split(
    X_trainval, y_trainval, genes_trainval, test_size=0.25, random_state=RANDOM_STATE, stratify=strata_tv_inner
)

print(f"  Train: {len(X_train)}, Calibration: {len(X_cal)}, Test: {len(X_test)}")
for gene in sorted(set(genes_cal)):
    mask = genes_cal == gene
    n_p = int((y_cal[mask] == 1).sum())
    n_b = int((y_cal[mask] == 0).sum())
    print(f"    {gene:8s}: {mask.sum():5d} cal samples ({n_p}P / {n_b}B)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Compute calibrated probabilities on calibration set
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Computing calibrated probabilities on calibration set...")

X_cal_s = scaler.transform(X_cal)
nn_preds_cal = nn_model.predict(X_cal_s, verbose=0).flatten()
xgb_preds_cal = xgb_clf.predict_proba(X_cal_s)[:, 1]

# Apply gene-specific ensemble weights
blended_cal = np.zeros(len(X_cal))
for i in range(len(X_cal)):
    gene = genes_cal[i].upper()
    gw = gene_ensemble_weights.get(gene)
    if gw:
        blended_cal[i] = gw["xgb_weight"] * xgb_preds_cal[i] + gw["mlp_weight"] * nn_preds_cal[i]
    else:
        blended_cal[i] = XGB_WEIGHT * xgb_preds_cal[i] + NN_WEIGHT * nn_preds_cal[i]

# Apply gene-specific calibration
calibrated_cal = np.zeros(len(X_cal))
for i in range(len(X_cal)):
    gene = genes_cal[i].upper()
    cal = gene_calibrators.get(gene, universal_calibrator)
    calibrated_cal[i] = cal.predict([blended_cal[i]])[0]

# Clip to [0.005, 0.995] as in production
calibrated_cal = np.clip(calibrated_cal, 0.005, 0.995)

print(f"  Calibrated probabilities: mean={calibrated_cal.mean():.4f}, std={calibrated_cal.std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Compute nonconformity scores and per-gene quantile thresholds
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Computing nonconformity scores and quantile thresholds...")


def compute_nonconformity_scores(probs, labels):
    """Nonconformity score = 1 - predicted probability of the true class.

    For binary classification:
      - If true label is 1 (pathogenic): score = 1 - p
      - If true label is 0 (benign): score = 1 - (1 - p) = p
    """
    scores = np.where(labels == 1, 1.0 - probs, probs)
    return scores


def compute_quantile_threshold(scores, alpha):
    """Compute the conformal quantile with finite-sample correction.

    q = ceil((n+1)*(1-alpha)) / n  percentile of the scores.
    This guarantees >= (1-alpha) marginal coverage.
    """
    n = len(scores)
    quantile_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, quantile_level))


# Global (all genes combined)
all_scores = compute_nonconformity_scores(calibrated_cal, y_cal)
global_q = compute_quantile_threshold(all_scores, ALPHA)

print(f"\n  Global quantile threshold (alpha={ALPHA}): {global_q:.4f}")
print(f"  Global nonconformity scores: mean={all_scores.mean():.4f}, "
      f"median={np.median(all_scores):.4f}, max={all_scores.max():.4f}")

# Per-gene thresholds
thresholds = {
    "_global": {
        "quantile": round(global_q, 6),
        "alpha": ALPHA,
        "n_calibration": int(len(all_scores)),
        "coverage_target": round(1 - ALPHA, 2),
    }
}

for gene in sorted(set(genes_cal)):
    mask = genes_cal == gene
    gene_upper = gene.upper()
    n_gene = int(mask.sum())

    if n_gene < MIN_CONFORMAL_SAMPLES:
        print(f"  {gene:8s}: {n_gene:4d} samples (< {MIN_CONFORMAL_SAMPLES}) — using global threshold")
        thresholds[gene_upper] = {
            "quantile": round(global_q, 6),
            "alpha": ALPHA,
            "n_calibration": n_gene,
            "coverage_target": round(1 - ALPHA, 2),
            "fallback": "global",
        }
        continue

    gene_scores = compute_nonconformity_scores(calibrated_cal[mask], y_cal[mask])
    gene_q = compute_quantile_threshold(gene_scores, ALPHA)

    thresholds[gene_upper] = {
        "quantile": round(gene_q, 6),
        "alpha": ALPHA,
        "n_calibration": n_gene,
        "coverage_target": round(1 - ALPHA, 2),
    }
    print(f"  {gene:8s}: {n_gene:4d} cal samples, quantile={gene_q:.4f}, "
          f"scores: mean={gene_scores.mean():.4f}, max={gene_scores.max():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Validate coverage on test set
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Validating coverage on test set...")

X_test_s = scaler.transform(X_test)
nn_preds_test = nn_model.predict(X_test_s, verbose=0).flatten()
xgb_preds_test = xgb_clf.predict_proba(X_test_s)[:, 1]

# Apply gene-specific ensemble weights
blended_test = np.zeros(len(X_test))
for i in range(len(X_test)):
    gene = genes_test[i].upper()
    gw = gene_ensemble_weights.get(gene)
    if gw:
        blended_test[i] = gw["xgb_weight"] * xgb_preds_test[i] + gw["mlp_weight"] * nn_preds_test[i]
    else:
        blended_test[i] = XGB_WEIGHT * xgb_preds_test[i] + NN_WEIGHT * nn_preds_test[i]

# Apply gene-specific calibration
calibrated_test = np.zeros(len(X_test))
for i in range(len(X_test)):
    gene = genes_test[i].upper()
    cal = gene_calibrators.get(gene, universal_calibrator)
    calibrated_test[i] = cal.predict([blended_test[i]])[0]

calibrated_test = np.clip(calibrated_test, 0.005, 0.995)

# Check coverage: does the true class fall within the conformal set?
CLASS_NAMES = {0: "Benign", 1: "Pathogenic"}

print(f"\n  {'Gene':8s} {'N':>5s} {'Coverage':>8s} {'Avg Set Size':>12s} {'Singleton%':>10s}")
print("  " + "-" * 50)

overall_covered = 0
overall_total = 0
set_size_dist = {1: 0, 2: 0}

for gene in sorted(set(genes_test)):
    mask = genes_test == gene
    gene_upper = gene.upper()
    gene_probs = calibrated_test[mask]
    gene_labels = y_test[mask]

    # Get the quantile for this gene
    gene_q = thresholds.get(gene_upper, thresholds["_global"])["quantile"]

    covered = 0
    total = 0
    set_sizes = []

    for prob, true_label in zip(gene_probs, gene_labels):
        # Build prediction set: include class c if P(c) >= 1 - q
        pred_set = []
        p_pathogenic = prob
        p_benign = 1.0 - prob
        if p_pathogenic >= 1.0 - gene_q:
            pred_set.append("Pathogenic")
        if p_benign >= 1.0 - gene_q:
            pred_set.append("Benign")
        # If empty (shouldn't happen with proper calibration), include the highest-probability class
        if not pred_set:
            pred_set.append("Pathogenic" if prob >= 0.5 else "Benign")

        true_class = CLASS_NAMES[int(true_label)]
        if true_class in pred_set:
            covered += 1
        total += 1
        set_sizes.append(len(pred_set))

        size_key = min(len(pred_set), 2)
        set_size_dist[size_key] = set_size_dist.get(size_key, 0) + 1

    cov = covered / total if total > 0 else 0
    avg_size = np.mean(set_sizes) if set_sizes else 0
    singleton_pct = (sum(1 for s in set_sizes if s == 1) / len(set_sizes) * 100) if set_sizes else 0
    overall_covered += covered
    overall_total += total

    thresholds[gene_upper]["test_coverage"] = round(cov, 4)
    thresholds[gene_upper]["test_n"] = total
    thresholds[gene_upper]["avg_set_size"] = round(avg_size, 3)
    thresholds[gene_upper]["singleton_rate"] = round(singleton_pct / 100, 4)

    cov_ok = "OK" if cov >= (1 - ALPHA) else "LOW"
    print(f"  {gene:8s} {total:5d} {cov:8.1%} [{cov_ok:3s}] {avg_size:12.3f} {singleton_pct:9.1f}%")

overall_cov = overall_covered / overall_total if overall_total > 0 else 0
print(f"\n  {'OVERALL':8s} {overall_total:5d} {overall_cov:8.1%}")
print(f"\n  Set size distribution: singleton={set_size_dist.get(1, 0)}, "
      f"pair={set_size_dist.get(2, 0)}")

# ─────────────────────────────────────────────────────────────────────────────
# Save thresholds
# ─────────────────────────────────────────────────────────────────────────────
output_path = "data/conformal_thresholds.json"
with open(output_path, "w") as f:
    json.dump(thresholds, f, indent=2)

print(f"\n  Saved conformal thresholds to {output_path}")
print("\n" + "=" * 60)
print("  Conformal prediction training complete!")
print("=" * 60)
