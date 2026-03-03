"""
SteppeDNA: Per-Gene Calibrator Training (Item 45)

Trains separate IsotonicRegression calibrators for each gene on the
held-out calibration split. Falls back to the universal calibrator for
genes with fewer than 50 calibration samples.

Uses the same 60/20/20 split logic and RANDOM_STATE=42 as
train_universal_model.py to ensure identical data partitioning.

Run from project root:
    python scripts/train_gene_calibrators.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

RANDOM_STATE = 42
XGB_WEIGHT = 0.6
NN_WEIGHT = 0.4
MIN_CALIBRATION_SAMPLES = 50

print("=" * 60)
print("  SteppeDNA: Per-Gene Calibrator Training")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data and models
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/4] Loading data and pre-trained models...")

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

with open("data/universal_calibrator_ensemble.pkl", "rb") as f:
    universal_calibrator = pickle.load(f)

print("  Models loaded OK.")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Reproduce the exact same split as train_universal_model.py
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/4] Reproducing 60/20/20 split...")

strata = np.array([f"{g}_{l}" for g, l in zip(genes, y)])
X_trainval, X_test, y_trainval, y_test, genes_trainval, genes_test, strata_tv, _ = train_test_split(
    X, y, genes, strata, test_size=0.20, random_state=RANDOM_STATE, stratify=strata
)
strata_tv_inner = np.array([f"{g}_{l}" for g, l in zip(genes_trainval, y_trainval)])
X_train, X_cal, y_train, y_cal, genes_train, genes_cal = train_test_split(
    X_trainval, y_trainval, genes_trainval, test_size=0.25, random_state=RANDOM_STATE, stratify=strata_tv_inner
)

print(f"  Calibration set: {len(X_cal)} samples")
for gene in sorted(set(genes_cal)):
    mask = genes_cal == gene
    n_p = int((y_cal[mask] == 1).sum())
    n_b = int((y_cal[mask] == 0).sum())
    print(f"    {gene:8s}: {mask.sum():5d} ({n_p}P / {n_b}B)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Compute blended probabilities on calibration set
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/4] Computing blended probabilities on calibration set...")

X_cal_s = scaler.transform(X_cal)
nn_preds_cal = nn_model.predict(X_cal_s, verbose=0).flatten()
xgb_preds_cal = xgb_clf.predict_proba(X_cal_s)[:, 1]
blended_cal = (XGB_WEIGHT * xgb_preds_cal) + (NN_WEIGHT * nn_preds_cal)

# Also compute on test set for evaluation
X_test_s = scaler.transform(X_test)
nn_preds_test = nn_model.predict(X_test_s, verbose=0).flatten()
xgb_preds_test = xgb_clf.predict_proba(X_test_s)[:, 1]
blended_test = (XGB_WEIGHT * xgb_preds_test) + (NN_WEIGHT * nn_preds_test)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Train per-gene calibrators
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/4] Training per-gene calibrators...")

results = {}
all_genes = sorted(set(genes))

for gene in all_genes:
    cal_mask = genes_cal == gene
    n_cal = int(cal_mask.sum())

    if n_cal < MIN_CALIBRATION_SAMPLES:
        print(f"  {gene:8s}: {n_cal:4d} cal samples < {MIN_CALIBRATION_SAMPLES} -> using universal calibrator (fallback)")
        results[gene] = {"method": "universal_fallback", "n_cal": n_cal}
        continue

    # Check we have both classes in calibration set
    n_pos = int((y_cal[cal_mask] == 1).sum())
    n_neg = int((y_cal[cal_mask] == 0).sum())
    if n_pos < 2 or n_neg < 2:
        print(f"  {gene:8s}: {n_cal:4d} cal samples but only {n_pos}P/{n_neg}B -> using universal (fallback)")
        results[gene] = {"method": "universal_fallback", "n_cal": n_cal, "n_pos": n_pos, "n_neg": n_neg}
        continue

    # Train gene-specific isotonic calibrator
    gene_blended = blended_cal[cal_mask]
    gene_labels = y_cal[cal_mask]

    gene_calibrator = IsotonicRegression(out_of_bounds='clip')
    gene_calibrator.fit(gene_blended, gene_labels)

    # Evaluate on test set for this gene
    test_mask = genes_test == gene
    n_test = int(test_mask.sum())

    if n_test > 0:
        gene_test_blended = blended_test[test_mask]
        gene_test_labels = y_test[test_mask]

        # Per-gene calibrated predictions
        gene_calibrated = gene_calibrator.predict(gene_test_blended)
        universal_calibrated = universal_calibrator.predict(gene_test_blended)

        brier_gene = brier_score_loss(gene_test_labels, gene_calibrated)
        brier_universal = brier_score_loss(gene_test_labels, universal_calibrated)

        improvement = brier_universal - brier_gene
        print(f"  {gene:8s}: {n_cal:4d} cal ({n_pos}P/{n_neg}B), "
              f"Brier gene={brier_gene:.4f} vs universal={brier_universal:.4f} "
              f"({'improved' if improvement > 0 else 'worse'} by {abs(improvement):.4f})")
    else:
        brier_gene = None
        brier_universal = None
        improvement = None
        print(f"  {gene:8s}: {n_cal:4d} cal ({n_pos}P/{n_neg}B), no test samples")

    # Save per-gene calibrator
    cal_path = f"data/calibrator_{gene.lower()}.pkl"
    with open(cal_path, "wb") as f:
        pickle.dump(gene_calibrator, f)

    results[gene] = {
        "method": "gene_specific",
        "n_cal": n_cal,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_test": n_test,
        "brier_gene": round(brier_gene, 6) if brier_gene is not None else None,
        "brier_universal": round(brier_universal, 6) if brier_universal is not None else None,
        "improvement": round(improvement, 6) if improvement is not None else None,
    }

# Save results summary
import json
with open("data/gene_calibrator_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("  Summary")
print("=" * 60)
for gene, res in sorted(results.items()):
    if res["method"] == "gene_specific":
        status = f"gene-specific (Brier: {res['brier_gene']:.4f})"
    else:
        status = "universal fallback"
    print(f"  {gene:8s}: {status}")

print(f"\nResults saved to data/gene_calibrator_results.json")
print("Per-gene calibrators saved to data/calibrator_<gene>.pkl")
