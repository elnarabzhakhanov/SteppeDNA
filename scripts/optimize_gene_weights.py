"""
SteppeDNA: Gene-Adaptive Ensemble Weight Optimization (Item 38)

For each gene, grid-searches XGBoost/MLP weight ratios from 0.0 to 1.0
(step 0.05) on the calibration set to find the blend that maximizes AUC.

Since AUC is rank-invariant under monotonic transforms, we compute AUC
on raw blended probabilities (pre-calibration). The optimal weight ratio
is selected on the calibration set and validated on the held-out test set.

Uses the same 60/20/20 split logic and RANDOM_STATE=42 as
train_universal_model.py and train_gene_calibrators.py.

Output:
    data/gene_ensemble_weights.json

Run from project root:
    set PYTHONUNBUFFERED=1 && python scripts/optimize_gene_weights.py
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42
DEFAULT_XGB_WEIGHT = 0.6
DEFAULT_NN_WEIGHT = 0.4
WEIGHT_STEP = 0.05
MIN_SAMPLES_FOR_AUC = 10  # Need enough samples for meaningful AUC

print("=" * 65, flush=True)
print("  SteppeDNA: Gene-Adaptive Ensemble Weight Optimization", flush=True)
print("=" * 65, flush=True)

# -----------------------------------------------------------------------
# 1. Load data and models
# -----------------------------------------------------------------------
print("\n[1/5] Loading data and pre-trained models...", flush=True)

df = pd.read_csv("data/master_training_dataset.csv")
with open("data/universal_feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

feature_names = [f for f in feature_names if f in df.columns]
X = df[feature_names].values
y = df["Label"].values
genes = df["Gene"].values

print(f"  Variants: {len(X)} across {len(set(genes))} genes", flush=True)

# Load pre-trained models
with open("data/universal_scaler_ensemble.pkl", "rb") as f:
    scaler = pickle.load(f)

from tensorflow.keras.models import load_model
nn_model = load_model("data/universal_nn.h5")

xgb_clf = xgb.XGBClassifier()
xgb_clf.load_model("data/universal_xgboost_final.json")

print("  Models loaded OK.", flush=True)

# -----------------------------------------------------------------------
# 2. Reproduce the exact same 60/20/20 split
# -----------------------------------------------------------------------
print("\n[2/5] Reproducing 60/20/20 split...", flush=True)

strata = np.array([f"{g}_{l}" for g, l in zip(genes, y)])
X_trainval, X_test, y_trainval, y_test, genes_trainval, genes_test, strata_tv, _ = train_test_split(
    X, y, genes, strata, test_size=0.20, random_state=RANDOM_STATE, stratify=strata
)
strata_tv_inner = np.array([f"{g}_{l}" for g, l in zip(genes_trainval, y_trainval)])
X_train, X_cal, y_train, y_cal, genes_train, genes_cal = train_test_split(
    X_trainval, y_trainval, genes_trainval, test_size=0.25, random_state=RANDOM_STATE, stratify=strata_tv_inner
)

print(f"  Calibration set: {len(X_cal)} samples", flush=True)
print(f"  Test set: {len(X_test)} samples", flush=True)

# -----------------------------------------------------------------------
# 3. Get raw predictions from both models on calibration + test sets
# -----------------------------------------------------------------------
print("\n[3/5] Computing individual model predictions...", flush=True)

X_cal_s = scaler.transform(X_cal)
nn_preds_cal = nn_model.predict(X_cal_s, verbose=0).flatten()
xgb_preds_cal = xgb_clf.predict_proba(X_cal_s)[:, 1]

X_test_s = scaler.transform(X_test)
nn_preds_test = nn_model.predict(X_test_s, verbose=0).flatten()
xgb_preds_test = xgb_clf.predict_proba(X_test_s)[:, 1]

# Individual model AUCs for reference
for label, preds_cal, preds_test in [
    ("XGBoost", xgb_preds_cal, xgb_preds_test),
    ("MLP", nn_preds_cal, nn_preds_test),
]:
    try:
        auc_cal = roc_auc_score(y_cal, preds_cal)
        auc_test = roc_auc_score(y_test, preds_test)
        print(f"  {label:8s} cal AUC: {auc_cal:.4f}, test AUC: {auc_test:.4f}", flush=True)
    except ValueError:
        print(f"  {label:8s} AUC: could not compute", flush=True)

# -----------------------------------------------------------------------
# 4. Grid-search optimal weight per gene (raw blended, no calibration)
# -----------------------------------------------------------------------
print("\n[4/5] Grid-searching optimal XGB/MLP weights per gene...", flush=True)
print("  (AUC computed on raw blended scores, pre-calibration)", flush=True)

weight_grid = np.arange(0.0, 1.0 + WEIGHT_STEP / 2, WEIGHT_STEP)
all_genes = sorted(set(genes))
results = {}

for gene in all_genes:
    cal_mask = genes_cal == gene
    test_mask = genes_test == gene
    n_cal = int(cal_mask.sum())
    n_test = int(test_mask.sum())

    n_cal_pos = int((y_cal[cal_mask] == 1).sum())
    n_cal_neg = int((y_cal[cal_mask] == 0).sum())
    n_test_pos = int((y_test[test_mask] == 1).sum())
    n_test_neg = int((y_test[test_mask] == 0).sum())

    if n_cal < MIN_SAMPLES_FOR_AUC or n_cal_pos < 2 or n_cal_neg < 2:
        print(f"  {gene:8s}: {n_cal:4d} cal samples (too few) -> default {DEFAULT_XGB_WEIGHT}/{DEFAULT_NN_WEIGHT}", flush=True)
        results[gene] = {
            "xgb_weight": DEFAULT_XGB_WEIGHT,
            "mlp_weight": DEFAULT_NN_WEIGHT,
            "method": "default_fallback",
            "reason": f"only {n_cal} cal samples ({n_cal_pos}P/{n_cal_neg}B)",
            "auc": None,
        }
        continue

    can_eval_test = n_test >= MIN_SAMPLES_FOR_AUC and n_test_pos >= 2 and n_test_neg >= 2

    gene_xgb_cal = xgb_preds_cal[cal_mask]
    gene_nn_cal = nn_preds_cal[cal_mask]
    gene_y_cal = y_cal[cal_mask]

    gene_xgb_test = xgb_preds_test[test_mask]
    gene_nn_test = nn_preds_test[test_mask]
    gene_y_test = y_test[test_mask]

    best_auc_cal = -1.0
    best_w_xgb = DEFAULT_XGB_WEIGHT
    all_aucs = []

    for w_xgb in weight_grid:
        w_nn = round(1.0 - w_xgb, 2)
        # Raw blended (no calibration) -- AUC is rank-invariant
        blended = w_xgb * gene_xgb_cal + w_nn * gene_nn_cal

        try:
            auc = roc_auc_score(gene_y_cal, blended)
        except ValueError:
            auc = 0.5

        all_aucs.append((round(w_xgb, 2), round(w_nn, 2), round(auc, 6)))

        if auc > best_auc_cal:
            best_auc_cal = auc
            best_w_xgb = round(w_xgb, 2)

    best_w_nn = round(1.0 - best_w_xgb, 2)

    # Evaluate optimal and default weights on test set
    test_auc = None
    default_test_auc = None
    if can_eval_test:
        try:
            blended_test_opt = best_w_xgb * gene_xgb_test + best_w_nn * gene_nn_test
            test_auc = round(roc_auc_score(gene_y_test, blended_test_opt), 6)
        except ValueError:
            pass
        try:
            blended_test_def = DEFAULT_XGB_WEIGHT * gene_xgb_test + DEFAULT_NN_WEIGHT * gene_nn_test
            default_test_auc = round(roc_auc_score(gene_y_test, blended_test_def), 6)
        except ValueError:
            pass

    delta_str = ""
    if test_auc is not None and default_test_auc is not None:
        delta = test_auc - default_test_auc
        delta_str = f", test delta: {delta:+.4f}"

    print(f"  {gene:8s}: best XGB={best_w_xgb:.2f}/MLP={best_w_nn:.2f} "
          f"(cal AUC={best_auc_cal:.4f}, test AUC={test_auc}{delta_str})", flush=True)

    results[gene] = {
        "xgb_weight": best_w_xgb,
        "mlp_weight": best_w_nn,
        "method": "optimized",
        "cal_auc": round(best_auc_cal, 6),
        "test_auc": test_auc,
        "default_test_auc": default_test_auc,
        "n_cal": n_cal,
        "n_test": n_test,
        "all_aucs": all_aucs,
    }

# -----------------------------------------------------------------------
# 5. Save results
# -----------------------------------------------------------------------
print("\n[5/5] Saving results...", flush=True)

output_path = "data/gene_ensemble_weights.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'=' * 65}", flush=True)
print("  GENE-ADAPTIVE ENSEMBLE WEIGHT SUMMARY", flush=True)
print("=" * 65, flush=True)
print(f"  {'Gene':<10s} {'XGB':>6s} {'MLP':>6s} {'Cal AUC':>10s} {'Test AUC':>10s} {'Dflt Test':>10s} {'Delta':>8s}", flush=True)
print(f"  {'-'*10} {'-'*6} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*8}", flush=True)

for gene in all_genes:
    r = results[gene]
    cal_auc_s = f"{r.get('cal_auc', 0):.4f}" if r.get('cal_auc') is not None else "N/A"
    test_auc_s = f"{r.get('test_auc', 0):.4f}" if r.get('test_auc') is not None else "N/A"
    dflt_auc_s = f"{r.get('default_test_auc', 0):.4f}" if r.get('default_test_auc') is not None else "N/A"
    if r.get('test_auc') is not None and r.get('default_test_auc') is not None:
        delta = r['test_auc'] - r['default_test_auc']
        delta_s = f"{delta:+.4f}"
    else:
        delta_s = "N/A"
    print(f"  {gene:<10s} {r['xgb_weight']:>5.2f} {r['mlp_weight']:>5.2f}  {cal_auc_s:>9s} {test_auc_s:>9s}  {dflt_auc_s:>9s} {delta_s:>8s}", flush=True)

print(f"\nSaved -> {output_path}", flush=True)
print("Done!", flush=True)
