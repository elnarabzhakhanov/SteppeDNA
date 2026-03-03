"""
SteppeDNA: Bootstrap Model Generation for Confidence Intervals (Item 39)
=========================================================================
Generates 50 bootstrap XGBoost models by resampling the training set with
replacement.  At prediction time, running input through all 50 models produces
a distribution of predictions whose 5th/95th percentiles form a 90% CI.

This directly quantifies the equity gap:
  - BRCA2 variants (data-rich) -> narrow CIs
  - PALB2/RAD51D variants (data-scarce) -> wide CIs

Output:
  data/bootstrap_models/bootstrap_0.json  ...  bootstrap_49.json

Run from project root:
  python scripts/generate_bootstrap_models.py
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
N_BOOTSTRAP = 50

# Ensure unbuffered output on Windows
os.environ["PYTHONUNBUFFERED"] = "1"

print("=" * 65, flush=True)
print("  SteppeDNA: Bootstrap Model Generation (Item 39)", flush=True)
print("=" * 65, flush=True)

# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
print("\n[1/4] Loading master training dataset...", flush=True)

df = pd.read_csv("data/master_training_dataset.csv")
with open("data/universal_feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

missing = [feat for feat in feature_names if feat not in df.columns]
if missing:
    print("  [WARN] %d features missing from dataset" % len(missing), flush=True)
    feature_names = [feat for feat in feature_names if feat in df.columns]

X = df[feature_names].values
y = df["Label"].values
genes = df["Gene"].values

print("  Variants: %d across %d genes" % (len(X), len(set(genes))), flush=True)
print("  Features: %d" % len(feature_names), flush=True)

# ---------------------------------------------------------------------------
# 2. Reproduce the exact 60/20/20 split from production training
# ---------------------------------------------------------------------------
print("\n[2/4] Reproducing production train/cal/test split...", flush=True)

strata = np.array(["%s_%d" % (g, l) for g, l in zip(genes, y)])
X_trainval, X_test, y_trainval, y_test, genes_tv, genes_test, strata_tv, _ = train_test_split(
    X, y, genes, strata, test_size=0.20, random_state=RANDOM_STATE, stratify=strata
)
strata_tv_inner = np.array(["%s_%d" % (g, l) for g, l in zip(genes_tv, y_trainval)])
X_train, X_cal, y_train, y_cal, genes_train, genes_cal = train_test_split(
    X_trainval, y_trainval, genes_tv, test_size=0.25, random_state=RANDOM_STATE, stratify=strata_tv_inner
)

print("  Train: %d  Cal: %d  Test: %d" % (len(X_train), len(X_cal), len(X_test)), flush=True)

# Scale using same approach as production
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)

# Save bootstrap scaler (should match production but kept separate for safety)
with open("data/bootstrap_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("  Scaler saved to data/bootstrap_scaler.pkl", flush=True)

# ---------------------------------------------------------------------------
# 3. Read production XGBoost hyperparameters
# ---------------------------------------------------------------------------
print("\n[3/4] Reading production XGBoost hyperparameters...", flush=True)

# Use same hyperparameters as production training script
# (from scripts/train_universal_model.py)
cw = float(np.sum(y_train == 0)) / max(np.sum(y_train == 1), 1)

base_params = {
    'n_estimators': 400,
    'max_depth': 7,
    'learning_rate': 0.05,
    'scale_pos_weight': cw,
    'eval_metric': 'logloss',
    'n_jobs': -1,
}

print("  Base params: max_depth=%d, lr=%.2f, n_estimators=%d" % (
    base_params['max_depth'], base_params['learning_rate'], base_params['n_estimators']
), flush=True)
print("  scale_pos_weight: %.4f" % cw, flush=True)

# ---------------------------------------------------------------------------
# 4. Generate 50 bootstrap models
# ---------------------------------------------------------------------------
print("\n[4/4] Training %d bootstrap XGBoost models..." % N_BOOTSTRAP, flush=True)

out_dir = os.path.join("data", "bootstrap_models")
os.makedirs(out_dir, exist_ok=True)

rng = np.random.RandomState(RANDOM_STATE)

for i in range(N_BOOTSTRAP):
    # Sample with replacement, same size as training set
    indices = rng.choice(len(X_train_s), size=len(X_train_s), replace=True)
    X_boot = X_train_s[indices]
    y_boot = y_train[indices]

    # Recompute class weight for this bootstrap sample
    n_neg = int(np.sum(y_boot == 0))
    n_pos = int(np.sum(y_boot == 1))
    boot_cw = float(n_neg) / max(n_pos, 1)

    params = dict(base_params)
    params['random_state'] = RANDOM_STATE + i
    params['scale_pos_weight'] = boot_cw

    clf = xgb.XGBClassifier(**params)
    clf.fit(X_boot, y_boot, verbose=False)

    model_path = os.path.join(out_dir, "bootstrap_%d.json" % i)
    clf.save_model(model_path)

    if (i + 1) % 10 == 0 or i == 0:
        # Quick OOB accuracy check
        oob_mask = np.ones(len(X_train_s), dtype=bool)
        oob_mask[indices] = False
        if oob_mask.sum() > 0:
            oob_preds = clf.predict_proba(X_train_s[oob_mask])[:, 1]
            oob_acc = np.mean((oob_preds > 0.5) == y_train[oob_mask])
            print("  [%d/%d] OOB accuracy: %.3f (%d OOB samples)" % (
                i + 1, N_BOOTSTRAP, oob_acc, int(oob_mask.sum())
            ), flush=True)
        else:
            print("  [%d/%d] trained (no OOB samples)" % (i + 1, N_BOOTSTRAP), flush=True)

# ---------------------------------------------------------------------------
# 5. Validation: compute CIs on test set and show equity gap
# ---------------------------------------------------------------------------
print("\n[Validation] Computing bootstrap CIs on test set...", flush=True)

X_test_s = scaler.transform(X_test)

# Load all 50 models
bootstrap_models = []
for i in range(N_BOOTSTRAP):
    model_path = os.path.join(out_dir, "bootstrap_%d.json" % i)
    booster = xgb.Booster()
    booster.load_model(model_path)
    bootstrap_models.append(booster)

# Get predictions from all models
dmat_test = xgb.DMatrix(X_test_s, feature_names=feature_names)
all_preds = np.array([m.predict(dmat_test) for m in bootstrap_models])  # (50, n_test)

ci_lower = np.percentile(all_preds, 5, axis=0)
ci_upper = np.percentile(all_preds, 95, axis=0)
ci_width = ci_upper - ci_lower
median_preds = np.median(all_preds, axis=0)

# Per-gene CI width analysis
print("\n  Per-gene CI width (90%% CI = 5th-95th percentile):", flush=True)
print("  %-8s  %6s  %6s  %6s  %6s  %s" % (
    "Gene", "N", "Med.CI", "Mean", "Std", "Equity Thesis"
), flush=True)
print("  " + "-" * 60, flush=True)

gene_stats = {}
for gene in sorted(set(genes_test)):
    mask = genes_test == gene
    widths = ci_width[mask]
    med_width = float(np.median(widths))
    mean_width = float(np.mean(widths))
    std_width = float(np.std(widths))
    n = int(mask.sum())
    thesis = "NARROW (data-rich)" if med_width < 0.10 else ("MODERATE" if med_width < 0.20 else "WIDE (data-scarce)")
    print("  %-8s  %6d  %6.3f  %6.3f  %6.3f  %s" % (
        gene, n, med_width, mean_width, std_width, thesis
    ), flush=True)
    gene_stats[gene] = {
        "n_test": n,
        "median_ci_width": round(med_width, 4),
        "mean_ci_width": round(mean_width, 4),
        "std_ci_width": round(std_width, 4),
    }

# Save summary stats
summary = {
    "n_bootstrap": N_BOOTSTRAP,
    "n_train": len(X_train),
    "n_test": len(X_test),
    "per_gene": gene_stats,
    "overall_median_ci_width": round(float(np.median(ci_width)), 4),
    "overall_mean_ci_width": round(float(np.mean(ci_width)), 4),
}
with open(os.path.join(out_dir, "bootstrap_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

total_size_mb = sum(
    os.path.getsize(os.path.join(out_dir, "bootstrap_%d.json" % i))
    for i in range(N_BOOTSTRAP)
) / (1024 * 1024)

print("\n  Total model size: %.1f MB" % total_size_mb, flush=True)
print("  Summary saved to data/bootstrap_models/bootstrap_summary.json", flush=True)
print("\nDone. Bootstrap models ready for production use.", flush=True)
