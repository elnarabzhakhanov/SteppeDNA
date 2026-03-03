"""
SteppeDNA: Expanded XGBoost Ablation Study
============================================
Removes each biological evidence source one at a time and measures the
AUC drop on a held-out test set (and via 5-fold CV on the training set).

Groups tested:
  1. Full model (baseline)
  2. No AlphaMissense         -> tests ClinVar label circularity concern
  3. No MAVE                  -> tests wet-lab feature contribution
  4. No PhyloP                -> tests conservation signal
  5. No Structure             -> tests 3D structural features
  6. No gnomAD                -> tests population frequency signal
  7. Sequence Only            -> keeps ONLY BLOSUM62/volume/hydro/charge/position/type

Output:
  visual_proofs/10_Ablation_Study_XGB.pdf   (bar chart + table)
  data/ablation_results_xgb.pkl

Run from project root:
  python scripts/ablation_study_xgb.py
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.feature_engineering import (
    load_phylop_scores, load_mave_scores, load_alphamissense_scores,
    load_structural_features, load_gnomad_frequencies,
    engineer_features,
)

sns.set_theme(style="whitegrid", font_scale=1.0)
os.makedirs("visual_proofs", exist_ok=True)

RANDOM_STATE = 42
DATA_DIR = "data"
N_CV_SPLITS = 5

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  SteppeDNA: Expanded XGBoost Ablation Study")
print("=" * 65)
print("\n[1] Loading data and engineering full feature set...")

df = pd.read_csv("brca2_missense_dataset_2.csv")
phylop  = load_phylop_scores(data_dir=DATA_DIR)
mave    = load_mave_scores(data_dir=DATA_DIR)
am      = load_alphamissense_scores(data_dir=DATA_DIR)
struct  = load_structural_features(data_dir=DATA_DIR)
gnomad  = load_gnomad_frequencies(data_dir=DATA_DIR)

try:
    from backend.feature_engineering import load_spliceai_scores
    spliceai = load_spliceai_scores(data_dir=DATA_DIR)
    X_full = engineer_features(df, phylop, mave, am, struct, gnomad, spliceai_data=spliceai)
    print("  -> SpliceAI included")
except (ImportError, AttributeError, TypeError):
    X_full = engineer_features(df, phylop, mave, am, struct, gnomad)

y = df["Label"].values
print(f"  -> {len(df)} variants, {X_full.shape[1]} features")
print(f"  -> Classes: {int(y.sum())} pathogenic | {int((y==0).sum())} benign")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Define Ablation Groups
# ─────────────────────────────────────────────────────────────────────────────
all_cols = list(X_full.columns)

AM_COLS      = [c for c in all_cols if any(k in c for k in ["am_score", "am_pathogenic", "am_x"])]
MAVE_COLS    = [c for c in all_cols if any(k in c for k in ["mave_score", "has_mave", "mave_abnormal", "mave_x"])]
PHYLOP_COLS  = [c for c in all_cols if any(k in c for k in ["phylop_score", "high_conservation", "ultra_conservation", "conserv_x"])]
STRUCT_COLS  = [c for c in all_cols if any(k in c for k in [
    "rsa", "is_buried", "bfactor", "dist_dna", "dist_palb2",
    "is_dna_contact", "ss_helix", "ss_sheet", "buried_x", "dna_contact_x"
])]
GNOMAD_COLS  = [c for c in all_cols if any(k in c for k in ["gnomad_af", "is_rare", "af_x"])]

# Sequence-only: keep only physicochemical + position features, nothing database-derived
SEQUENCE_KEEP = [
    "blosum62_score", "volume_diff", "hydro_diff", "charge_changed",
    "is_nonsense", "cDNA_pos", "AA_pos", "relative_cdna_pos", "relative_aa_pos",
    "is_transition", "is_transversion", "same_charge", "same_hydro_class",
    "aa_ref_hydro", "aa_alt_hydro", "aa_ref_volume", "aa_alt_volume",
]
SEQUENCE_ONLY = [c for c in all_cols if c in SEQUENCE_KEEP]

ABLATION_GROUPS = {
    "Full Model":        [],                  # nothing removed
    "No AlphaMissense":  AM_COLS,
    "No MAVE":           MAVE_COLS,
    "No PhyloP":         PHYLOP_COLS,
    "No Structure":      STRUCT_COLS,
    "No gnomAD":         GNOMAD_COLS,
    "Sequence Only":     [c for c in all_cols if c not in SEQUENCE_ONLY],
}

print(f"\n[2] Ablation group feature counts:")
for name, removed in ABLATION_GROUPS.items():
    n_kept = X_full.shape[1] - len(removed)
    print(f"  {name:<22}: {n_kept:3d} features ({len(removed)} removed)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Hyperparameters (same as production model)
# ─────────────────────────────────────────────────────────────────────────────
BASE_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "eval_metric": "logloss",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

optuna_path = os.path.join(DATA_DIR, "optuna_best_params.json")
if os.path.exists(optuna_path):
    with open(optuna_path) as f:
        opt = json.load(f)
    BASE_PARAMS.update(opt)
    BASE_PARAMS["random_state"] = RANDOM_STATE
    BASE_PARAMS["n_jobs"] = -1
    print(f"  -> Loaded Optuna hyperparameters")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Holdout + 5-Fold CV for each ablation group
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[3] Running holdout + {N_CV_SPLITS}-fold CV for each group...")

# Fixed holdout split (same as production)
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

skf = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

def train_and_evaluate(X_tr, y_tr, X_te, y_te, params):
    """Train XGBoost with SMOTE and return holdout AUC + PR-AUC."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_sm, y_sm = smote.fit_resample(X_tr_s, y_tr)

    cw = float(np.sum(y_sm == 0)) / np.sum(y_sm == 1)
    p = params.copy()
    p["scale_pos_weight"] = cw

    clf = xgb.XGBClassifier(**p)
    clf.fit(X_sm, y_sm, verbose=False)
    y_prob = clf.predict_proba(X_te_s)[:, 1]

    roc = roc_auc_score(y_te, y_prob)
    prec, rec, _ = precision_recall_curve(y_te, y_prob)
    pr = auc(rec, prec)
    return roc, pr

ablation_results = {}

for group_name, cols_to_remove in ABLATION_GROUPS.items():
    print(f"\n  [{group_name}]")
    cols_to_remove_valid = [c for c in cols_to_remove if c in X_full.columns]
    X_group = X_full.drop(columns=cols_to_remove_valid)
    n_feats = X_group.shape[1]

    # Holdout evaluation
    X_tr_g = X_group.iloc[X_train_full.index] if hasattr(X_train_full, 'index') else X_group.loc[X_train_full.index]
    X_te_g = X_group.iloc[X_test_full.index] if hasattr(X_test_full, 'index') else X_group.loc[X_test_full.index]

    holdout_roc, holdout_pr = train_and_evaluate(
        X_tr_g.values, y_train, X_te_g.values, y_test, BASE_PARAMS
    )
    print(f"    Holdout  -> ROC-AUC={holdout_roc:.4f}  PR-AUC={holdout_pr:.4f}")

    # 5-fold CV
    fold_rocs = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_group, y), start=1):
        fold_roc, _ = train_and_evaluate(
            X_group.iloc[tr_idx].values, y[tr_idx],
            X_group.iloc[val_idx].values, y[val_idx],
            BASE_PARAMS,
        )
        fold_rocs.append(fold_roc)
        print(f"    CV Fold {fold}: ROC-AUC={fold_roc:.4f}", end="")

    cv_mean = np.mean(fold_rocs)
    cv_std  = np.std(fold_rocs)
    print(f"\n    CV Mean  -> {cv_mean:.4f} ± {cv_std:.4f}")

    ablation_results[group_name] = {
        "n_features": n_feats,
        "n_removed": len(cols_to_remove_valid),
        "removed_cols": cols_to_remove_valid,
        "holdout_roc": holdout_roc,
        "holdout_pr": holdout_pr,
        "cv_roc_mean": cv_mean,
        "cv_roc_std": cv_std,
        "cv_fold_rocs": fold_rocs,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 5. Compute AUC Drop vs Full Model
# ─────────────────────────────────────────────────────────────────────────────
full_roc = ablation_results["Full Model"]["holdout_roc"]
full_pr  = ablation_results["Full Model"]["holdout_pr"]

print("\n" + "=" * 65)
print("  ABLATION RESULTS")
print("=" * 65)
print(f"\n  {'Group':<22} {'Feats':>6} {'ROC-AUC':>9} {'Drop':>8} {'PR-AUC':>9} {'CV Mean':>9}")
print(f"  {'-'*65}")

for name, res in ablation_results.items():
    drop = full_roc - res["holdout_roc"]
    marker = ""
    if name == "Full Model":
        marker = " <-- baseline"
    elif drop > 0.05:
        marker = " *** HIGH IMPACT"
    elif drop > 0.02:
        marker = " ** MODERATE"
    elif drop > 0.005:
        marker = " * SMALL"
    print(f"  {name:<22} {res['n_features']:>6} {res['holdout_roc']:>9.4f} {drop:>+8.4f} "
          f"{res['holdout_pr']:>9.4f} {res['cv_roc_mean']:>9.4f}{marker}")

# Circularity verdict
am_drop = full_roc - ablation_results["No AlphaMissense"]["holdout_roc"]
print(f"\n  ClinVar Circularity Verdict (AlphaMissense removal):")
if am_drop < 0.01:
    print(f"  [PASS] AUC drop = {am_drop:.4f} < 0.01 — minimal AlphaMissense dependence.")
    print(f"         Other features (MAVE, PhyloP) drive predictions. Circularity risk LOW.")
elif am_drop < 0.05:
    print(f"  [INFO] AUC drop = {am_drop:.4f} — moderate AlphaMissense contribution.")
    print(f"         Model works without AM. Circularity risk MODERATE. Defendable.")
else:
    print(f"  [WARN] AUC drop = {am_drop:.4f} > 0.05 — high AlphaMissense reliance.")
    print(f"         Consider retraining with AM removed as a robustness ablation.")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Save Results
# ─────────────────────────────────────────────────────────────────────────────
with open(os.path.join(DATA_DIR, "ablation_results_xgb.pkl"), "wb") as f:
    pickle.dump(ablation_results, f)
print(f"\n  Saved -> {DATA_DIR}/ablation_results_xgb.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Publication Figure
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Generating figure...")

group_names = list(ablation_results.keys())
holdout_rocs = [ablation_results[g]["holdout_roc"] for g in group_names]
drops = [full_roc - r for r in holdout_rocs]
cv_means = [ablation_results[g]["cv_roc_mean"] for g in group_names]
cv_stds  = [ablation_results[g]["cv_roc_std"]  for g in group_names]

BRAND = "#6260FF"
RED   = "#FF3B30"

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel A: Holdout ROC-AUC per group
ax1 = axes[0]
colors_bar = [BRAND if g == "Full Model" else RED for g in group_names]
bars = ax1.barh(group_names, holdout_rocs, color=colors_bar, alpha=0.8)
ax1.axvline(full_roc, color=BRAND, ls="--", lw=1.5, alpha=0.6, label=f"Full = {full_roc:.4f}")
for i, v in enumerate(holdout_rocs):
    ax1.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9)
ax1.set_xlim(max(0, min(holdout_rocs) - 0.05), 1.02)
ax1.set_xlabel("Holdout ROC-AUC", fontweight="bold")
ax1.set_title("A. Holdout ROC-AUC\nby Ablation Group", fontweight="bold")
ax1.legend(fontsize=9)

# Panel B: AUC Drop (positive = worse without this group)
ax2 = axes[1]
drop_colors = ["#8E8E93" if g == "Full Model" else (RED if d > 0.02 else "#FF9500" if d > 0.005 else "#34C759")
               for g, d in zip(group_names, drops)]
ax2.barh(group_names, drops, color=drop_colors, alpha=0.85)
ax2.axvline(0, color="black", lw=0.8)
ax2.axvline(0.05, color=RED, ls=":", lw=1.5, alpha=0.7, label="Critical threshold (0.05)")
ax2.axvline(0.02, color="#FF9500", ls=":", lw=1.2, alpha=0.7, label="Moderate threshold (0.02)")
for i, d in enumerate(drops):
    ax2.text(d + 0.001 if d >= 0 else d - 0.001, i, f"{d:+.4f}", va="center",
             ha="left" if d >= 0 else "right", fontsize=9)
ax2.set_xlabel("AUC Drop vs Full Model (+ve = feature group matters)", fontweight="bold")
ax2.set_title("B. AUC Drop by Removed Feature Group\n(Higher = that source is more important)", fontweight="bold")
ax2.legend(fontsize=8, loc="lower right")

plt.suptitle(
    f"SteppeDNA: Expanded Ablation Study (XGBoost, n={len(df)} variants)\n"
    f"Demonstrates contribution of each biological evidence source",
    fontsize=12, fontweight="bold", y=1.02
)

plt.tight_layout()
plt.savefig("visual_proofs/10_Ablation_Study_XGB.pdf", bbox_inches="tight")
plt.savefig("visual_proofs/10_Ablation_Study_XGB.png", bbox_inches="tight", dpi=150)
plt.close()

print("  Saved -> visual_proofs/10_Ablation_Study_XGB.pdf")
print("  Saved -> visual_proofs/10_Ablation_Study_XGB.png")
print("\n  Done.")
