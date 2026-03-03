"""
mave_ablation.py — Quantify MAVE Feature Impact (Data Leakage Assessment)
===========================================================================

Tests the impact of zeroing out MAVE features on the trained model
to assess whether MAVE scores represent genuine signal leakage.

Approach: Zero out MAVE features in the test set and compare predictions
with and without MAVE. If MAVE features drive pathogenicity predictions
for MAVE-scored variants, this indicates potential leakage.

Output: data/mave_ablation_results.json

Run from project root:
  python scripts/mave_ablation.py
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, balanced_accuracy_score
)

import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
XGB_WEIGHT = 0.6
NN_WEIGHT = 0.4

print("=" * 65)
print("  MAVE Feature Ablation (Data Leakage Assessment)")
print("=" * 65)

# ─── Load model and data ──────────────────────────────────────────────
print("\n[1] Loading model and reproducing split...", flush=True)

df = pd.read_csv("data/master_training_dataset.csv")
with open("data/universal_feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)
with open("data/universal_scaler_ensemble.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("data/universal_calibrator_ensemble.pkl", "rb") as f:
    calibrator = pickle.load(f)
with open("data/universal_threshold_ensemble.pkl", "rb") as f:
    threshold = pickle.load(f)

feature_names = [f for f in feature_names if f in df.columns]
X = df[feature_names].values
y = df["Label"].values
genes = df["Gene"].values
strata = np.array([f"{g}_{l}" for g, l in zip(genes, y)])

indices = np.arange(len(df))
idx_traincal, idx_test = train_test_split(
    indices, test_size=0.20, random_state=RANDOM_STATE, stratify=strata
)

test_df = df.iloc[idx_test].copy().reset_index(drop=True)
X_test_orig = test_df[feature_names].values
y_test = test_df["Label"].values
genes_test = test_df["Gene"].values

# Load models
xgb_clf = xgb.XGBClassifier()
xgb_clf.load_model("data/universal_xgboost_final.json")
from tensorflow.keras.models import load_model
nn_model = load_model("data/universal_nn.h5", compile=False)

# ─── Identify MAVE feature indices ────────────────────────────────────
MAVE_FEATURES = ["mave_score", "has_mave", "mave_abnormal", "mave_x_blosum"]
mave_indices = [i for i, f in enumerate(feature_names) if f in MAVE_FEATURES]
print(f"  MAVE features: {[feature_names[i] for i in mave_indices]}", flush=True)
print(f"  MAVE feature indices: {mave_indices}", flush=True)

# Count variants with MAVE scores
mave_score_idx = feature_names.index("mave_score") if "mave_score" in feature_names else None
if mave_score_idx is not None:
    n_with_mave = np.sum(X_test_orig[:, mave_score_idx] != 0)
    print(f"  Test variants with MAVE scores: {n_with_mave}/{len(y_test)} "
          f"({n_with_mave/len(y_test)*100:.1f}%)", flush=True)


def predict_ensemble(X_test):
    """Run ensemble prediction."""
    X_s = scaler.transform(X_test)
    nn_preds = nn_model.predict(X_s, verbose=0).flatten()
    xgb_preds = xgb_clf.predict_proba(X_s)[:, 1]
    blended = XGB_WEIGHT * xgb_preds + NN_WEIGHT * nn_preds
    return calibrator.predict(blended)


# ─── Baseline predictions (with MAVE) ─────────────────────────────────
print("\n[2] Baseline predictions (with MAVE features)...", flush=True)
probs_with_mave = predict_ensemble(X_test_orig)

auc_with = roc_auc_score(y_test, probs_with_mave)
preds_with = (probs_with_mave >= threshold).astype(int)
mcc_with = matthews_corrcoef(y_test, preds_with)
bal_acc_with = balanced_accuracy_score(y_test, preds_with)
print(f"  ROC-AUC: {auc_with:.4f}  MCC: {mcc_with:.4f}  BalAcc: {bal_acc_with:.4f}", flush=True)

# ─── Ablation predictions (MAVE zeroed out) ───────────────────────────
print("\n[3] Ablation predictions (MAVE features zeroed out)...", flush=True)
X_test_no_mave = X_test_orig.copy()
for idx in mave_indices:
    X_test_no_mave[:, idx] = 0.0

probs_no_mave = predict_ensemble(X_test_no_mave)

auc_without = roc_auc_score(y_test, probs_no_mave)
preds_without = (probs_no_mave >= threshold).astype(int)
mcc_without = matthews_corrcoef(y_test, preds_without)
bal_acc_without = balanced_accuracy_score(y_test, preds_without)
print(f"  ROC-AUC: {auc_without:.4f}  MCC: {mcc_without:.4f}  BalAcc: {bal_acc_without:.4f}", flush=True)

# ─── Impact analysis ──────────────────────────────────────────────────
print(f"\n{'=' * 65}", flush=True)
print("MAVE ABLATION RESULTS", flush=True)
print(f"{'=' * 65}", flush=True)

delta_auc = auc_without - auc_with
delta_mcc = mcc_without - mcc_with

print(f"\n  {'Metric':<20s} {'With MAVE':>12s} {'Without MAVE':>14s} {'Delta':>10s}", flush=True)
print(f"  {'-' * 58}", flush=True)
print(f"  {'ROC-AUC':<20s} {auc_with:>12.4f} {auc_without:>14.4f} {delta_auc:>+10.4f}", flush=True)
print(f"  {'MCC':<20s} {mcc_with:>12.4f} {mcc_without:>14.4f} {delta_mcc:>+10.4f}", flush=True)
print(f"  {'Balanced Acc':<20s} {bal_acc_with:>12.4f} {bal_acc_without:>14.4f} {bal_acc_without-bal_acc_with:>+10.4f}", flush=True)

# Per-gene impact
print(f"\n  Per-gene ROC-AUC impact:", flush=True)
per_gene_results = {}
for gene in ["BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D"]:
    mask = genes_test == gene
    if mask.sum() < 20:
        continue
    y_g = y_test[mask]
    if len(set(y_g)) < 2:
        continue
    auc_g_with = roc_auc_score(y_g, probs_with_mave[mask])
    auc_g_without = roc_auc_score(y_g, probs_no_mave[mask])
    delta = auc_g_without - auc_g_with
    per_gene_results[gene] = {
        "auc_with_mave": float(auc_g_with),
        "auc_without_mave": float(auc_g_without),
        "delta": float(delta),
        "n": int(mask.sum()),
    }
    print(f"    {gene}: {auc_g_with:.4f} -> {auc_g_without:.4f} (delta={delta:+.4f}, n={mask.sum()})", flush=True)

# Prediction shift for MAVE-scored variants
if mave_score_idx is not None:
    mave_mask = X_test_orig[:, mave_score_idx] != 0
    if mave_mask.sum() > 0:
        print(f"\n  Prediction shift for MAVE-scored variants (n={mave_mask.sum()}):", flush=True)
        shift = np.abs(probs_with_mave[mave_mask] - probs_no_mave[mave_mask])
        print(f"    Mean |dP(pathogenic)|:   {shift.mean():.4f}", flush=True)
        print(f"    Median |dP(pathogenic)|: {np.median(shift):.4f}", flush=True)
        print(f"    Max |dP(pathogenic)|:    {shift.max():.4f}", flush=True)

        # How many predictions change class?
        class_changes = (preds_with[mave_mask] != preds_without[mave_mask]).sum()
        print(f"    Predictions that change class: {class_changes}/{mave_mask.sum()}", flush=True)

# Overall assessment
print(f"\n  ASSESSMENT:", flush=True)
if abs(delta_auc) < 0.005:
    print(f"  [OK] MAVE features have MINIMAL impact on model performance (dAUC={delta_auc:+.4f})", flush=True)
    print(f"    The model does not rely on MAVE for classification.", flush=True)
    assessment = "minimal_impact"
elif abs(delta_auc) < 0.02:
    print(f"  [~] MAVE features have MODERATE impact on model performance (dAUC={delta_auc:+.4f})", flush=True)
    print(f"    The model benefits slightly from MAVE but is not dependent on it.", flush=True)
    assessment = "moderate_impact"
else:
    print(f"  [X] MAVE features have SIGNIFICANT impact on model performance (dAUC={delta_auc:+.4f})", flush=True)
    print(f"    Consider retraining without MAVE features.", flush=True)
    assessment = "significant_impact"

# Save results
results = {
    "with_mave": {
        "roc_auc": float(auc_with),
        "mcc": float(mcc_with),
        "balanced_accuracy": float(bal_acc_with),
    },
    "without_mave": {
        "roc_auc": float(auc_without),
        "mcc": float(mcc_without),
        "balanced_accuracy": float(bal_acc_without),
    },
    "delta": {
        "roc_auc": float(delta_auc),
        "mcc": float(delta_mcc),
    },
    "per_gene": per_gene_results,
    "assessment": assessment,
    "mave_feature_count": len(mave_indices),
    "total_feature_count": len(feature_names),
    "test_variants_with_mave": int(n_with_mave) if mave_score_idx else 0,
    "total_test_variants": len(y_test),
}

with open("data/mave_ablation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n  Results saved to data/mave_ablation_results.json", flush=True)
