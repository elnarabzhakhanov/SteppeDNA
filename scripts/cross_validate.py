"""
SteppeDNA: 10-Fold Gene-Stratified Cross-Validation (v4 - Universal Model)
==========================================================================
Validates the universal model using 10-fold stratified CV with gene x label
stratification to ensure each fold has all genes and both classes.

Uses XGBoost only (faster, more stable) with same hyperparameters as
production train_universal_model.py.

Outputs:
  - visual_proofs/9_CrossValidation_Results.pdf/png
  - data/cv_results.pkl
  - Printed summary: AUC, PR-AUC, MCC, Balanced Accuracy per fold

Run from project root:
  python scripts/cross_validate.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, precision_recall_curve,
    balanced_accuracy_score, matthews_corrcoef, confusion_matrix
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)
os.makedirs("visual_proofs", exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 10
N_BOOTSTRAP = 1000

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  SteppeDNA: 10-Fold Cross-Validation (Universal Model v4)")
print("=" * 65)
print("\n[1] Loading master training dataset...")

df = pd.read_csv("data/master_training_dataset.csv")
with open("data/universal_feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

feature_names = [f for f in feature_names if f in df.columns]
X = df[feature_names].values
y = df["Label"].values
genes = df["Gene"].values

print(f"  Variants: {len(X)} across {len(set(genes))} genes")
print(f"  Features: {len(feature_names)}")
print(f"  Classes:  {int((y==0).sum())} Benign / {int((y==1).sum())} Pathogenic")

# Use gene x label strata for stratified splitting
strata = np.array([f"{g}_{l}" for g, l in zip(genes, y)])

# ─────────────────────────────────────────────────────────────────────────────
# 2. 10-Fold Stratified CV
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[2] Running {N_SPLITS}-fold gene x label stratified CV...")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

fold_results = []
all_y_true = []
all_y_pred = []
all_genes_val = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, strata), start=1):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    g_val = genes[val_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr_s, y_tr)

    cw = float(np.sum(y_tr_sm == 0)) / np.sum(y_tr_sm == 1)
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=7, learning_rate=0.05,
        scale_pos_weight=cw, random_state=RANDOM_STATE,
        eval_metric='logloss', n_jobs=-1
    )
    clf.fit(X_tr_sm, y_tr_sm, verbose=False)

    y_prob = clf.predict_proba(X_val_s)[:, 1]

    # Optimal threshold from PR curve
    prec, rec, thresholds = precision_recall_curve(y_val, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    if len(thresholds) > 0:
        opt_thresh = float(thresholds[np.argmax(f1[:-1])])
    else:
        opt_thresh = 0.5

    y_bin = (y_prob >= opt_thresh).astype(int)

    fold_roc = roc_auc_score(y_val, y_prob)
    fold_pr = auc(rec, prec)
    fold_mcc = matthews_corrcoef(y_val, y_bin)
    fold_bal = balanced_accuracy_score(y_val, y_bin)

    tn, fp, fn, tp = confusion_matrix(y_val, y_bin).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    fold_results.append({
        "fold": fold, "roc_auc": fold_roc, "pr_auc": fold_pr,
        "mcc": fold_mcc, "balanced_accuracy": fold_bal,
        "sensitivity": sens, "specificity": spec,
        "threshold": opt_thresh, "n_val": len(y_val),
    })

    all_y_true.extend(y_val.tolist())
    all_y_pred.extend(y_prob.tolist())
    all_genes_val.extend(g_val.tolist())

    print(f"  Fold {fold:2d}: AUC={fold_roc:.4f}  PR={fold_pr:.4f}  "
          f"MCC={fold_mcc:.3f}  BalAcc={fold_bal:.3f}  Sens={sens:.3f}  Spec={spec:.3f}  (n={len(y_val)})")

results_df = pd.DataFrame(fold_results)
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)
all_genes_val = np.array(all_genes_val)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Bootstrap Confidence Intervals
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[3] Computing bootstrap 95% CIs ({N_BOOTSTRAP} resamples)...")

rng = np.random.default_rng(RANDOM_STATE)
boot_roc, boot_pr = [], []

for _ in range(N_BOOTSTRAP):
    idx = rng.integers(0, len(all_y_true), size=len(all_y_true))
    yt, yp = all_y_true[idx], all_y_pred[idx]
    if len(np.unique(yt)) < 2:
        continue
    boot_roc.append(roc_auc_score(yt, yp))
    prec_b, rec_b, _ = precision_recall_curve(yt, yp)
    boot_pr.append(auc(rec_b, prec_b))

boot_roc = np.array(boot_roc)
boot_pr = np.array(boot_pr)
roc_ci = (np.percentile(boot_roc, 2.5), np.percentile(boot_roc, 97.5))
pr_ci = (np.percentile(boot_pr, 2.5), np.percentile(boot_pr, 97.5))

mean_roc = results_df["roc_auc"].mean()
std_roc = results_df["roc_auc"].std()
mean_pr = results_df["pr_auc"].mean()
std_pr = results_df["pr_auc"].std()
mean_mcc = results_df["mcc"].mean()
mean_bal = results_df["balanced_accuracy"].mean()
mean_sens = results_df["sensitivity"].mean()
mean_spec = results_df["specificity"].mean()

print(f"  ROC-AUC: {mean_roc:.4f} +/- {std_roc:.4f}  (95% CI: {roc_ci[0]:.4f}-{roc_ci[1]:.4f})")
print(f"  PR-AUC:  {mean_pr:.4f} +/- {std_pr:.4f}  (95% CI: {pr_ci[0]:.4f}-{pr_ci[1]:.4f})")
print(f"  MCC:     {mean_mcc:.4f}  |  BalAcc: {mean_bal:.4f}")
print(f"  Sens:    {mean_sens:.4f}  |  Spec:   {mean_spec:.4f}")

# Per-gene OOF performance
print(f"\n  Per-gene out-of-fold performance:")
print(f"  {'Gene':8s} {'n':>5s}  {'ROC-AUC':>8s}  {'MCC':>6s}  {'BalAcc':>7s}")
print(f"  {'-'*38}")
for gene in sorted(set(all_genes_val)):
    mask = all_genes_val == gene
    y_g = all_y_true[mask]
    p_g = all_y_pred[mask]
    if len(set(y_g)) < 2:
        print(f"  {gene:8s} {mask.sum():5d}  (single class)")
        continue
    gene_auc = roc_auc_score(y_g, p_g)
    # Use median as threshold for MCC
    gene_pred = (p_g >= 0.5).astype(int)
    gene_mcc = matthews_corrcoef(y_g, gene_pred)
    gene_bal = balanced_accuracy_score(y_g, gene_pred)
    print(f"  {gene:8s} {mask.sum():5d}  {gene_auc:8.4f}  {gene_mcc:6.3f}  {gene_bal:7.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Save results
# ─────────────────────────────────────────────────────────────────────────────
cv_output = {
    "fold_results": results_df.to_dict("records"),
    "mean_roc_auc": mean_roc, "std_roc_auc": std_roc, "roc_ci_95": roc_ci,
    "mean_pr_auc": mean_pr, "std_pr_auc": std_pr, "pr_ci_95": pr_ci,
    "mean_mcc": mean_mcc, "mean_balanced_accuracy": mean_bal,
    "oof_y_true": all_y_true.tolist(),
    "oof_y_pred": all_y_pred.tolist(),
    "oof_genes": all_genes_val.tolist(),
}
with open("data/cv_results.pkl", "wb") as f:
    pickle.dump(cv_output, f)
print(f"\n  Saved data/cv_results.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Publication Figure
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Generating publication figure...")

fig = plt.figure(figsize=(18, 6))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

BRAND = "#6260FF"
RED = "#FF3B30"

# Panel A: Per-fold ROC-AUC boxplot
ax1 = fig.add_subplot(gs[0])
bp = ax1.boxplot(
    results_df["roc_auc"].values, patch_artist=True, widths=0.4,
    boxprops=dict(facecolor=BRAND, alpha=0.6),
    medianprops=dict(color="white", linewidth=2.5),
    whiskerprops=dict(color=BRAND), capprops=dict(color=BRAND),
)
ax1.scatter([1] * N_SPLITS, results_df["roc_auc"].values,
            color=BRAND, zorder=5, s=40, alpha=0.8)
ax1.axhline(mean_roc, color=RED, lw=1.5, ls="--", label=f"Mean = {mean_roc:.4f}")
ax1.fill_between([0.6, 1.4], roc_ci[0], roc_ci[1], alpha=0.15, color=RED,
                 label=f"95% CI [{roc_ci[0]:.4f}, {roc_ci[1]:.4f}]")
ax1.set_xlim(0.5, 1.5)
ax1.set_ylim(max(0, mean_roc - 0.1), min(1.0, mean_roc + 0.05))
ax1.set_xticks([])
ax1.set_ylabel("ROC-AUC", fontweight="bold")
ax1.set_title("A. 10-Fold CV ROC-AUC\nDistribution", fontweight="bold")
ax1.legend(fontsize=9, loc="lower right")

# Panel B: Metrics bar chart
ax2 = fig.add_subplot(gs[1])
metric_names = ["ROC-AUC", "PR-AUC", "MCC", "BalAcc", "Sens", "Spec"]
vals = [mean_roc, mean_pr, mean_mcc, mean_bal, mean_sens, mean_spec]
errs = [std_roc, std_pr, results_df["mcc"].std(), results_df["balanced_accuracy"].std(),
        results_df["sensitivity"].std(), results_df["specificity"].std()]
colors_bar = [BRAND, RED, "#9B59B6", "#2ECC71", "#F39C12", "#3498DB"]

bars = ax2.barh(metric_names, vals, xerr=errs, color=colors_bar, alpha=0.8,
                error_kw=dict(ecolor="black", lw=1.5, capsize=5))
for i, (v, e) in enumerate(zip(vals, errs)):
    ax2.text(v + e + 0.005, i, f"{v:.3f}+/-{e:.3f}", va="center", fontsize=8.5)
ax2.set_xlim(0, 1.15)
ax2.set_xlabel("Score", fontweight="bold")
ax2.set_title("B. Mean +/- Std\n(10-Fold CV)", fontweight="bold")

# Panel C: OOF ROC curve
ax3 = fig.add_subplot(gs[2])
oof_fpr, oof_tpr, _ = roc_curve(all_y_true, all_y_pred)
oof_roc = auc(oof_fpr, oof_tpr)
ax3.plot(oof_fpr, oof_tpr, color=BRAND, lw=2.5, label=f"OOF ROC (AUC = {oof_roc:.4f})")
ax3.fill_between(oof_fpr, oof_tpr, alpha=0.08, color=BRAND)
ax3.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC=0.50)")
ax3.text(0.55, 0.15,
         f"95% CI: [{roc_ci[0]:.4f}, {roc_ci[1]:.4f}]\n"
         f"n={len(df)} variants | {N_SPLITS} folds\n"
         f"{len(set(genes))} genes | {len(feature_names)} features",
         fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
ax3.set_xlabel("False Positive Rate", fontweight="bold")
ax3.set_ylabel("True Positive Rate", fontweight="bold")
ax3.set_title("C. Out-of-Fold ROC Curve\n(Pooled Predictions)", fontweight="bold")
ax3.legend(loc="lower right", fontsize=9)
ax3.set_xlim([0, 1])
ax3.set_ylim([0, 1.05])

plt.suptitle(
    f"SteppeDNA: {N_SPLITS}-Fold Gene-Stratified Cross-Validation (Universal XGBoost, n={len(df)})\n"
    f"ROC-AUC = {mean_roc:.4f} +/- {std_roc:.4f}  |  95% CI: [{roc_ci[0]:.4f}, {roc_ci[1]:.4f}]",
    fontsize=12, fontweight="bold", y=1.03
)

plt.savefig("visual_proofs/9_CrossValidation_Results.pdf", bbox_inches="tight")
plt.savefig("visual_proofs/9_CrossValidation_Results.png", bbox_inches="tight", dpi=150)
plt.close()

print("\n" + "=" * 65)
print("  CROSS-VALIDATION SUMMARY (Universal Model)")
print("=" * 65)
print(f"\n  {'Metric':<15} {'Mean':>8} {'Std':>8}")
print(f"  {'-'*33}")
print(f"  {'ROC-AUC':<15} {mean_roc:>8.4f} {std_roc:>8.4f}")
print(f"  {'PR-AUC':<15} {mean_pr:>8.4f} {std_pr:>8.4f}")
print(f"  {'MCC':<15} {mean_mcc:>8.4f} {results_df['mcc'].std():>8.4f}")
print(f"  {'BalAcc':<15} {mean_bal:>8.4f} {results_df['balanced_accuracy'].std():>8.4f}")
print(f"  {'Sensitivity':<15} {mean_sens:>8.4f} {results_df['sensitivity'].std():>8.4f}")
print(f"  {'Specificity':<15} {mean_spec:>8.4f} {results_df['specificity'].std():>8.4f}")
print(f"\n  Figures: visual_proofs/9_CrossValidation_Results.pdf/png")
print(f"  Done.")
