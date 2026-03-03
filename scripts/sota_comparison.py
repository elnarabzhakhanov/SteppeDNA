"""
SteppeDNA vs Standalone Predictors (v4.1 - Universal Model)
=============================================================
Compares SteppeDNA ensemble against:
  A) Individual input features used as standalone predictors
  B) Real independent SOTA tools (REVEL, CADD, BayesDel) fetched
     from myvariant.info/dbNSFP — NOT used as input features

Generates:
  visual_proofs/7_SOTA_Comparison_ROC.png/pdf
  visual_proofs/7_SOTA_Comparison_PR.png/pdf
  visual_proofs/7_SOTA_Comparison_Summary.png/pdf

Run from project root:
  python scripts/sota_comparison.py
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc, roc_auc_score,
    average_precision_score, matthews_corrcoef, balanced_accuracy_score
)

import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", font_scale=1.1)
os.makedirs("visual_proofs", exist_ok=True)

RANDOM_STATE = 42
XGB_WEIGHT = 0.6
NN_WEIGHT = 0.4

# Standalone comparators: feature name, display name, direction, color, citation
COMPARATORS = [
    ("am_score",       "AlphaMissense",  True,  "#34C759", "Cheng et al., Nature 2023"),
    ("phylop_score",   "PhyloP",         True,  "#FF9500", "Pollard et al. 2010"),
    ("blosum62_score", "BLOSUM62",       False, "#E63946", "Henikoff & Henikoff 1992"),
    ("mave_score",     "MAVE",           False, "#457B9D", "Findlay et al. 2018"),
    ("spliceai_score", "SpliceAI",       True,  "#2A9D8F", "Jaganathan et al. 2019"),
    ("esm2_cosine_sim","ESM-2 (cosine)", False, "#9B59B6", "Lin et al. 2023"),
]
# direction: True = higher is pathogenic, False = lower is pathogenic (we invert)

print("=" * 65)
print("  SteppeDNA vs Standalone Predictors (Universal Model v4.1)")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load and reproduce exact split
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Loading data and reproducing train/test split...")

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
X_trainval, X_test, y_trainval, y_test, genes_trainval, genes_test, _, _ = train_test_split(
    X, y, genes, strata, test_size=0.20, random_state=RANDOM_STATE, stratify=strata
)
strata_tv = np.array([f"{g}_{l}" for g, l in zip(genes_trainval, y_trainval)])
X_train, X_cal, y_train, y_cal, _, _ = train_test_split(
    X_trainval, y_trainval, genes_trainval, test_size=0.25, random_state=RANDOM_STATE, stratify=strata_tv
)

# Also split the full dataframe to get raw feature values for comparators
df_trainval, df_test_raw, _, _ = train_test_split(
    df, strata, test_size=0.20, random_state=RANDOM_STATE, stratify=strata
)

print(f"  Test set: {len(y_test)} variants across {len(set(genes_test))} genes")
print(f"  Test pathogenic: {(y_test == 1).sum()} | benign: {(y_test == 0).sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Get SteppeDNA ensemble predictions
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Loading SteppeDNA ensemble predictions...")

xgb_clf = xgb.XGBClassifier()
xgb_clf.load_model("data/universal_xgboost_final.json")

from tensorflow.keras.models import load_model
nn_model = load_model("data/universal_nn.h5", compile=False)

X_test_s = scaler.transform(X_test)
nn_preds = nn_model.predict(X_test_s, verbose=0).flatten()
xgb_preds = xgb_clf.predict_proba(X_test_s)[:, 1]
blended = XGB_WEIGHT * xgb_preds + NN_WEIGHT * nn_preds
steppedna_probs = calibrator.predict(blended)

print(f"  SteppeDNA ROC-AUC: {roc_auc_score(y_test, steppedna_probs):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Extract standalone comparator scores on test set
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Extracting standalone predictor scores on test set...")

results = {}

# SteppeDNA first
fpr_sd, tpr_sd, _ = roc_curve(y_test, steppedna_probs)
prec_sd, rec_sd, _ = precision_recall_curve(y_test, steppedna_probs)
y_pred_sd = (steppedna_probs >= threshold).astype(int)
results["SteppeDNA (Ensemble)"] = {
    "roc_auc": auc(fpr_sd, tpr_sd),
    "pr_auc": auc(rec_sd, prec_sd),
    "mcc": matthews_corrcoef(y_test, y_pred_sd),
    "bal_acc": balanced_accuracy_score(y_test, y_pred_sd),
    "fpr": fpr_sd, "tpr": tpr_sd, "prec": prec_sd, "rec": rec_sd,
    "n_scored": len(y_test),
    "color": "#6260FF",
    "citation": "This work",
}

for feat_name, display_name, higher_is_path, color, citation in COMPARATORS:
    if feat_name not in df_test_raw.columns:
        print(f"  {display_name:>20}: SKIPPED (not in dataset)")
        continue

    scores = df_test_raw[feat_name].values.astype(float)

    # Count valid (non-NaN, non-zero for some)
    valid_mask = ~np.isnan(scores)
    if feat_name == "mave_score":
        # MAVE has many zeros for variants without data
        valid_mask = valid_mask & (df_test_raw["has_mave"].values == 1) if "has_mave" in df_test_raw.columns else valid_mask

    n_valid = valid_mask.sum()
    coverage = 100 * n_valid / len(scores)

    if n_valid < 50:
        print(f"  {display_name:>20}: SKIPPED (only {n_valid} valid scores)")
        continue

    y_valid = y_test[valid_mask]
    s_valid = scores[valid_mask]

    if len(set(y_valid)) < 2:
        print(f"  {display_name:>20}: SKIPPED (single class)")
        continue

    # Invert if lower = more pathogenic
    if not higher_is_path:
        s_valid = -s_valid  # negate for ROC/PR computation

    fpr_c, tpr_c, _ = roc_curve(y_valid, s_valid)
    prec_c, rec_c, _ = precision_recall_curve(y_valid, s_valid)
    roc_val = auc(fpr_c, tpr_c)
    pr_val = auc(rec_c, prec_c)

    # MCC at median threshold
    med = np.median(s_valid)
    pred_c = (s_valid >= med).astype(int)
    mcc_c = matthews_corrcoef(y_valid, pred_c)
    bal_c = balanced_accuracy_score(y_valid, pred_c)

    results[display_name] = {
        "roc_auc": roc_val, "pr_auc": pr_val,
        "mcc": mcc_c, "bal_acc": bal_c,
        "fpr": fpr_c, "tpr": tpr_c, "prec": prec_c, "rec": rec_c,
        "n_scored": int(n_valid), "color": color, "citation": citation,
    }
    print(f"  {display_name:>20}: ROC-AUC={roc_val:.4f}  PR-AUC={pr_val:.4f}  (n={n_valid}, {coverage:.0f}% coverage)")

# ─────────────────────────────────────────────────────────────────────────────
# 3b. REAL SOTA: REVEL, CADD, BayesDel (from myvariant.info / dbNSFP)
# ─────────────────────────────────────────────────────────────────────────────
dbnsfp_path = "data/dbnsfp_sota_scores.csv"
if os.path.exists(dbnsfp_path):
    print("\n[3b] Loading REAL SOTA scores (REVEL, CADD, BayesDel from dbNSFP)...")
    dbnsfp = pd.read_csv(dbnsfp_path)

    REAL_SOTA = [
        ("revel_score",   "REVEL",    True,  "#FF2D55", "Ioannidis et al. 2016"),
        ("cadd_phred",    "CADD",     True,  "#FF6B35", "Rentzsch et al. 2019"),
        ("bayesdel_score","BayesDel", True,  "#00C7BE", "Feng 2017"),
    ]

    for col, display_name, higher_is_path, color, citation in REAL_SOTA:
        if col not in dbnsfp.columns:
            continue
        scores = dbnsfp[col].values.astype(float)
        labels = dbnsfp["Label"].values

        valid_mask = ~np.isnan(scores)
        n_valid = valid_mask.sum()
        if n_valid < 50:
            print(f"  {display_name:>20}: SKIPPED (only {n_valid} valid)")
            continue

        y_valid = labels[valid_mask]
        s_valid = scores[valid_mask]
        if len(set(y_valid)) < 2:
            continue

        if not higher_is_path:
            s_valid = -s_valid

        fpr_c, tpr_c, _ = roc_curve(y_valid, s_valid)
        prec_c, rec_c, _ = precision_recall_curve(y_valid, s_valid)
        roc_val = auc(fpr_c, tpr_c)
        pr_val = auc(rec_c, prec_c)

        # MCC at optimal threshold (Youden's J)
        j_scores = tpr_c - fpr_c
        best_idx = np.argmax(j_scores)
        thresholds = np.sort(s_valid)
        if best_idx < len(thresholds):
            best_thresh = thresholds[min(best_idx, len(thresholds)-1)]
        else:
            best_thresh = np.median(s_valid)
        pred_c = (s_valid >= best_thresh).astype(int)
        mcc_c = matthews_corrcoef(y_valid, pred_c)
        bal_c = balanced_accuracy_score(y_valid, pred_c)

        coverage = 100 * n_valid / len(y_test)
        results[f"{display_name} (REAL)"] = {
            "roc_auc": roc_val, "pr_auc": pr_val,
            "mcc": mcc_c, "bal_acc": bal_c,
            "fpr": fpr_c, "tpr": tpr_c, "prec": prec_c, "rec": rec_c,
            "n_scored": int(n_valid), "color": color,
            "citation": citation,
            "is_real_sota": True,
        }
        print(f"  {display_name:>20}: ROC-AUC={roc_val:.4f}  PR-AUC={pr_val:.4f}  MCC={mcc_c:.3f}  (n={n_valid}, {coverage:.0f}% coverage)")
else:
    print("\n[3b] No dbNSFP scores found. Run: python data_pipelines/fetch_dbnsfp_scores.py")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Overall + BRCA2-only comparison
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] BRCA2-specific comparison...")

brca2_mask = genes_test == "BRCA2"
y_brca2 = y_test[brca2_mask]
sp_brca2 = steppedna_probs[brca2_mask]

fpr_b2, tpr_b2, _ = roc_curve(y_brca2, sp_brca2)
brca2_auc = auc(fpr_b2, tpr_b2)
print(f"  SteppeDNA BRCA2-only ROC-AUC: {brca2_auc:.4f}")

brca2_results = {"SteppeDNA (BRCA2)": {
    "roc_auc": brca2_auc, "fpr": fpr_b2, "tpr": tpr_b2,
    "color": "#6260FF", "n_scored": brca2_mask.sum(),
}}

for feat_name, display_name, higher_is_path, color, citation in COMPARATORS:
    if feat_name not in df_test_raw.columns:
        continue
    scores = df_test_raw[feat_name].values[brca2_mask].astype(float)
    valid = ~np.isnan(scores)
    if feat_name == "mave_score" and "has_mave" in df_test_raw.columns:
        valid = valid & (df_test_raw["has_mave"].values[brca2_mask] == 1)
    if valid.sum() < 20:
        continue
    y_b2v = y_brca2[valid]
    s_b2v = scores[valid]
    if len(set(y_b2v)) < 2:
        continue
    if not higher_is_path:
        s_b2v = -s_b2v
    fpr_c, tpr_c, _ = roc_curve(y_b2v, s_b2v)
    b2_auc = auc(fpr_c, tpr_c)
    brca2_results[f"{display_name} (BRCA2)"] = {
        "roc_auc": b2_auc, "fpr": fpr_c, "tpr": tpr_c,
        "color": color, "n_scored": valid.sum(),
    }
    print(f"  {display_name:>20} BRCA2: ROC-AUC={b2_auc:.4f} (n={valid.sum()})")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Generate figures
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Generating comparison figures...")

sorted_names = sorted(results.keys(), key=lambda n: results[n]["roc_auc"], reverse=True)

# --- Figure 7a: ROC Curves (Overall) ---
fig, ax = plt.subplots(figsize=(10, 8))
for name in sorted_names:
    r = results[name]
    lw = 3.0 if "SteppeDNA" in name else 1.8
    ls = '-' if "SteppeDNA" in name else '--'
    ax.plot(r["fpr"], r["tpr"], color=r["color"], lw=lw, linestyle=ls,
            label=f'{name} (AUC={r["roc_auc"]:.3f}, n={r["n_scored"]})')

ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('SteppeDNA Ensemble vs Standalone Predictors\n'
             f'(ROC Curves, Universal Model, {len(set(genes_test))} genes, n={len(y_test)})')
ax.legend(loc='lower right', fontsize=9)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
fig.tight_layout()
fig.savefig('visual_proofs/7_SOTA_Comparison_ROC.png', dpi=150, bbox_inches='tight')
fig.savefig('visual_proofs/7_SOTA_Comparison_ROC.pdf', bbox_inches='tight')
plt.close()
print("  Saved 7_SOTA_Comparison_ROC")

# --- Figure 7b: PR Curves (Overall) ---
fig, ax = plt.subplots(figsize=(10, 8))
prevalence = y_test.mean()
for name in sorted_names:
    r = results[name]
    lw = 3.0 if "SteppeDNA" in name else 1.8
    ls = '-' if "SteppeDNA" in name else '--'
    ax.plot(r["rec"], r["prec"], color=r["color"], lw=lw, linestyle=ls,
            label=f'{name} (AP={r["pr_auc"]:.3f})')

ax.axhline(prevalence, color='gray', ls=':', alpha=0.5, label=f'Baseline ({prevalence:.2f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('SteppeDNA Ensemble vs Standalone Predictors\n'
             f'(Precision-Recall Curves, Universal Model)')
ax.legend(loc='lower left', fontsize=9)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([0, 1.05])
fig.tight_layout()
fig.savefig('visual_proofs/7_SOTA_Comparison_PR.png', dpi=150, bbox_inches='tight')
fig.savefig('visual_proofs/7_SOTA_Comparison_PR.pdf', bbox_inches='tight')
plt.close()
print("  Saved 7_SOTA_Comparison_PR")

# --- Figure 7c: Summary bar chart + table ---
fig = plt.figure(figsize=(15, 7))
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35, width_ratios=[1.3, 1])

# Panel A: horizontal bar chart
ax1 = fig.add_subplot(gs[0])
names_disp = [n for n in sorted_names]
aucs_vals = [results[n]["roc_auc"] for n in sorted_names]
colors_vals = [results[n]["color"] for n in sorted_names]
bars = ax1.barh(range(len(sorted_names)), aucs_vals, color=colors_vals, edgecolor='white', height=0.6)

for i, name in enumerate(sorted_names):
    if "SteppeDNA" in name:
        bars[i].set_edgecolor('#333')
        bars[i].set_linewidth(2)

ax1.set_yticks(range(len(sorted_names)))
ax1.set_yticklabels(names_disp, fontsize=10)
ax1.set_xlabel('ROC-AUC')
ax1.set_title('A. ROC-AUC Comparison', fontsize=12, fontweight='bold')
ax1.set_xlim([0.4, 1.02])
ax1.invert_yaxis()

for i, (v, name) in enumerate(zip(aucs_vals, sorted_names)):
    bold = 'bold' if 'SteppeDNA' in name else 'normal'
    ax1.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=9, fontweight=bold)
ax1.grid(axis='x', alpha=0.3)

# Panel B: metrics table
ax2 = fig.add_subplot(gs[1])
ax2.axis('off')

table_data = []
for name in sorted_names:
    r = results[name]
    table_data.append([
        name,
        f'{r["roc_auc"]:.4f}',
        f'{r["pr_auc"]:.4f}',
        f'{r["mcc"]:.3f}',
        f'{r["n_scored"]:,}',
    ])

table = ax2.table(
    cellText=table_data,
    colLabels=['Predictor', 'ROC-AUC', 'PR-AUC', 'MCC', 'n'],
    loc='center', cellLoc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.4)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(fontweight='bold')
        cell.set_facecolor('#e8e8e8')
    if row > 0 and 'SteppeDNA' in table_data[row - 1][0]:
        cell.set_facecolor('#e8e0ff')
        cell.set_text_props(fontweight='bold')

ax2.set_title('B. Performance Summary', fontsize=12, fontweight='bold', pad=20)

fig.suptitle('SteppeDNA: Comparison Against Standalone Variant Predictors\n'
             '(Universal multi-gene model, held-out test set)',
             fontsize=12, y=1.02)
fig.savefig('visual_proofs/7_SOTA_Comparison_Summary.png', dpi=150, bbox_inches='tight')
fig.savefig('visual_proofs/7_SOTA_Comparison_Summary.pdf', bbox_inches='tight')
plt.close()
print("  Saved 7_SOTA_Comparison_Summary")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Save comparison metrics to JSON
# ─────────────────────────────────────────────────────────────────────────────
comparison_json = {}
for name in sorted_names:
    r = results[name]
    comparison_json[name] = {
        "roc_auc": round(r["roc_auc"], 4),
        "pr_auc": round(r["pr_auc"], 4),
        "mcc": round(r["mcc"], 3),
        "bal_acc": round(r["bal_acc"], 4),
        "n_scored": r["n_scored"],
        "citation": r.get("citation", ""),
    }

with open("data/sota_comparison.json", "w") as f:
    json.dump(comparison_json, f, indent=2)
print("\n  Metrics saved to data/sota_comparison.json")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Final report
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  FINAL COMPARISON REPORT (Universal Model v4.1)")
print("=" * 65)

# Separate real SOTA from input features for clarity
real_sota_names = [n for n in sorted_names if results[n].get("is_real_sota")]
input_feat_names = [n for n in sorted_names if not results[n].get("is_real_sota") and "SteppeDNA" not in n]

print(f"\n  {'Predictor':<25} {'ROC-AUC':>8} {'PR-AUC':>8} {'MCC':>6} {'n':>6}")
print(f"  {'-'*57}")

# SteppeDNA first
sd = results["SteppeDNA (Ensemble)"]
print(f"  {'SteppeDNA (Ensemble)':<25} {sd['roc_auc']:>8.4f} {sd['pr_auc']:>8.4f} {sd['mcc']:>6.3f} {sd['n_scored']:>6} <--")

if real_sota_names:
    print(f"\n  --- Real SOTA (independent, NOT used as input features) ---")
    for name in real_sota_names:
        r = results[name]
        print(f"  {name:<25} {r['roc_auc']:>8.4f} {r['pr_auc']:>8.4f} {r['mcc']:>6.3f} {r['n_scored']:>6}")

if input_feat_names:
    print(f"\n  --- Input features as standalone predictors ---")
    for name in input_feat_names:
        r = results[name]
        print(f"  {name:<25} {r['roc_auc']:>8.4f} {r['pr_auc']:>8.4f} {r['mcc']:>6.3f} {r['n_scored']:>6}")

sd_auc = results["SteppeDNA (Ensemble)"]["roc_auc"]
n_beat = sum(1 for n, r in results.items() if n != "SteppeDNA (Ensemble)" and r["roc_auc"] < sd_auc)
n_comp = len(results) - 1
n_real = sum(1 for n in real_sota_names if results[n]["roc_auc"] < sd_auc)
n_real_total = len(real_sota_names)

print(f"\n  SteppeDNA outperforms {n_beat}/{n_comp} total predictors by ROC-AUC")
if n_real_total > 0:
    print(f"  Including {n_real}/{n_real_total} independent SOTA tools (REVEL, CADD, BayesDel)")

print(f"\n{'=' * 65}")
print(f"  DONE")
print(f"{'=' * 65}")
