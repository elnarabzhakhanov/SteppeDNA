"""
SteppeDNA: Universal Model Visual Proofs (v4)
==============================================
Generates publication-quality figures using the trained universal model artifacts.
Produces per-gene ROC/PR curves, calibration plot, SHAP analysis, and summary charts.

Run from project root:
  python scripts/generate_figures.py
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import json
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, roc_auc_score, matthews_corrcoef, balanced_accuracy_score
)
from sklearn.calibration import calibration_curve

import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = {
    'BRCA1': '#E74C3C', 'BRCA2': '#3498DB', 'PALB2': '#2ECC71',
    'RAD51C': '#9B59B6', 'RAD51D': '#F39C12', 'overall': '#1A1A2E'
}

os.makedirs("visual_proofs", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load artifacts
# ─────────────────────────────────────────────────────────────────────────────
print("Loading model artifacts...")

df = pd.read_csv("data/master_training_dataset.csv")
with open("data/universal_feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)
with open("data/universal_scaler_ensemble.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("data/universal_calibrator_ensemble.pkl", "rb") as f:
    calibrator = pickle.load(f)
with open("data/universal_threshold_ensemble.pkl", "rb") as f:
    threshold = pickle.load(f)
with open("data/model_metrics.json") as f:
    metrics = json.load(f)

xgb_clf = xgb.XGBClassifier()
xgb_clf.load_model("data/universal_xgboost_final.json")

from tensorflow.keras.models import load_model
nn_model = load_model("data/universal_nn.h5", compile=False)

# Reproduce the exact train/test split
from sklearn.model_selection import train_test_split
RANDOM_STATE = 42
XGB_WEIGHT = 0.6
NN_WEIGHT = 0.4

feature_names = [f for f in feature_names if f in df.columns]
X = df[feature_names].values
y = df["Label"].values
genes = df["Gene"].values

strata = np.array([f"{g}_{l}" for g, l in zip(genes, y)])
X_trainval, X_test, y_trainval, y_test, genes_trainval, genes_test, _, _ = train_test_split(
    X, y, genes, strata, test_size=0.20, random_state=RANDOM_STATE, stratify=strata
)
strata_tv = np.array([f"{g}_{l}" for g, l in zip(genes_trainval, y_trainval)])
X_train, X_cal, y_train, y_cal, genes_train, genes_cal = train_test_split(
    X_trainval, y_trainval, genes_trainval, test_size=0.25, random_state=RANDOM_STATE, stratify=strata_tv
)

X_test_s = scaler.transform(X_test)

# Generate predictions
nn_preds = nn_model.predict(X_test_s, verbose=0).flatten()
xgb_preds = xgb_clf.predict_proba(X_test_s)[:, 1]
blended_raw = XGB_WEIGHT * xgb_preds + NN_WEIGHT * nn_preds
y_probs = calibrator.predict(blended_raw)
y_pred = (y_probs >= threshold).astype(int)

print(f"  Test set: {len(y_test)} variants, {len(set(genes_test))} genes")
print(f"  Overall ROC-AUC: {roc_auc_score(y_test, y_probs):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Per-Gene ROC Curves
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/6] Per-Gene ROC Curves...")

fig, ax = plt.subplots(figsize=(9, 7))

# Overall
fpr_all, tpr_all, _ = roc_curve(y_test, y_probs)
auc_all = auc(fpr_all, tpr_all)
ax.plot(fpr_all, tpr_all, color=COLORS['overall'], lw=2.5,
        label=f'Overall (AUC={auc_all:.3f}, n={len(y_test)})')

# Per gene
for gene in sorted(set(genes_test)):
    mask = genes_test == gene
    y_g = y_test[mask]
    p_g = y_probs[mask]
    if len(set(y_g)) < 2:
        continue
    fpr_g, tpr_g, _ = roc_curve(y_g, p_g)
    auc_g = auc(fpr_g, tpr_g)
    ax.plot(fpr_g, tpr_g, color=COLORS.get(gene, '#888'),
            lw=1.8, label=f'{gene} (AUC={auc_g:.3f}, n={mask.sum()})')

ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Random (AUC=0.500)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('SteppeDNA Universal Model: Per-Gene ROC Curves\n(Held-out test set, gene x label stratified split)')
ax.legend(loc='lower right', fontsize=9)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
fig.tight_layout()
fig.savefig('visual_proofs/1_PerGene_ROC_Curves.png', dpi=150, bbox_inches='tight')
fig.savefig('visual_proofs/1_PerGene_ROC_Curves.pdf', bbox_inches='tight')
plt.close()
print("  Saved 1_PerGene_ROC_Curves")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Per-Gene Precision-Recall Curves
# ─────────────────────────────────────────────────────────────────────────────
print("[2/6] Per-Gene Precision-Recall Curves...")

fig, ax = plt.subplots(figsize=(9, 7))

# Overall
prec_all, rec_all, _ = precision_recall_curve(y_test, y_probs)
ap_all = average_precision_score(y_test, y_probs)
ax.plot(rec_all, prec_all, color=COLORS['overall'], lw=2.5,
        label=f'Overall (AP={ap_all:.3f}, n={len(y_test)})')

for gene in sorted(set(genes_test)):
    mask = genes_test == gene
    y_g = y_test[mask]
    p_g = y_probs[mask]
    if len(set(y_g)) < 2:
        continue
    prec_g, rec_g, _ = precision_recall_curve(y_g, p_g)
    ap_g = average_precision_score(y_g, p_g)
    ax.plot(rec_g, prec_g, color=COLORS.get(gene, '#888'),
            lw=1.8, label=f'{gene} (AP={ap_g:.3f}, n={mask.sum()})')

prevalence = y_test.mean()
ax.axhline(y=prevalence, color='gray', linestyle=':', alpha=0.5, label=f'Baseline (prev={prevalence:.2f})')
ax.set_xlabel('Recall (Sensitivity)')
ax.set_ylabel('Precision')
ax.set_title('SteppeDNA Universal Model: Per-Gene Precision-Recall Curves\n(Held-out test set)')
ax.legend(loc='lower left', fontsize=9)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([0, 1.05])
fig.tight_layout()
fig.savefig('visual_proofs/2_PerGene_PR_Curves.png', dpi=150, bbox_inches='tight')
fig.savefig('visual_proofs/2_PerGene_PR_Curves.pdf', bbox_inches='tight')
plt.close()
print("  Saved 2_PerGene_PR_Curves")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Calibration (Raw vs Isotonic)
# ─────────────────────────────────────────────────────────────────────────────
print("[3/6] Calibration Reliability Diagram...")

fig, ax = plt.subplots(figsize=(8, 7))

# Raw blended
prob_true_raw, prob_pred_raw = calibration_curve(y_test, blended_raw, n_bins=10, strategy='uniform')
ax.plot(prob_pred_raw, prob_true_raw, 's-', color='#E74C3C', lw=2,
        label='Raw Ensemble (before calibration)', markersize=8)

# Calibrated
prob_true_cal, prob_pred_cal = calibration_curve(y_test, y_probs, n_bins=10, strategy='uniform')
ax.plot(prob_pred_cal, prob_true_cal, 'o-', color='#3498DB', lw=2,
        label='Isotonic Calibrated', markersize=8)

ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Reliability Diagram\n(Isotonic regression on held-out calibration set)')
ax.legend(loc='upper left', fontsize=10)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
fig.tight_layout()
fig.savefig('visual_proofs/3_Calibration_Curve.png', dpi=150, bbox_inches='tight')
fig.savefig('visual_proofs/3_Calibration_Curve.pdf', bbox_inches='tight')
plt.close()
print("  Saved 3_Calibration_Curve")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: SHAP Feature Importances (XGBoost)
# ─────────────────────────────────────────────────────────────────────────────
print("[4/6] SHAP Feature Importances...")

try:
    import shap
    booster = xgb_clf.get_booster()
    dmatrix = xgb.DMatrix(X_test_s, feature_names=feature_names)
    shap_vals = booster.predict(dmatrix, pred_contribs=True)[:, :-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_test_s, feature_names=feature_names,
                      show=False, max_display=20)
    plt.title("SteppeDNA: Global SHAP Feature Importances (XGBoost)\nUniversal model, test set")
    fig.savefig('visual_proofs/4_SHAP_Global_Beeswarm.png', dpi=150, bbox_inches='tight')
    fig.savefig('visual_proofs/4_SHAP_Global_Beeswarm.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved 4_SHAP_Global_Beeswarm")
except Exception as e:
    print(f"  SHAP skipped: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Per-Gene Performance Summary (Bar Chart)
# ─────────────────────────────────────────────────────────────────────────────
print("[5/6] Per-Gene Performance Summary...")

gene_names = []
gene_aucs = []
gene_mccs = []
gene_bals = []
gene_ns = []

for gene in sorted(set(genes_test)):
    mask = genes_test == gene
    y_g = y_test[mask]
    p_g = y_probs[mask]
    if len(set(y_g)) < 2:
        continue
    gene_pred_g = (p_g >= threshold).astype(int)
    gene_names.append(gene)
    gene_aucs.append(roc_auc_score(y_g, p_g))
    gene_mccs.append(matthews_corrcoef(y_g, gene_pred_g))
    gene_bals.append(balanced_accuracy_score(y_g, gene_pred_g))
    gene_ns.append(mask.sum())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

x = np.arange(len(gene_names))
bar_colors = [COLORS.get(g, '#888') for g in gene_names]

# ROC-AUC
axes[0].bar(x, gene_aucs, color=bar_colors, edgecolor='white', linewidth=0.5)
axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
axes[0].set_xticks(x)
axes[0].set_xticklabels([f'{g}\n(n={n})' for g, n in zip(gene_names, gene_ns)], fontsize=9)
axes[0].set_ylabel('ROC-AUC')
axes[0].set_title('ROC-AUC by Gene')
axes[0].set_ylim([0, 1.05])
for i, v in enumerate(gene_aucs):
    axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

# MCC
axes[1].bar(x, gene_mccs, color=bar_colors, edgecolor='white', linewidth=0.5)
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1].set_xticks(x)
axes[1].set_xticklabels([f'{g}\n(n={n})' for g, n in zip(gene_names, gene_ns)], fontsize=9)
axes[1].set_ylabel('MCC')
axes[1].set_title("Matthews Correlation Coefficient")
axes[1].set_ylim([-0.15, 1.05])
for i, v in enumerate(gene_mccs):
    axes[1].text(i, max(v + 0.03, 0.05), f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

# Balanced Accuracy
axes[2].bar(x, gene_bals, color=bar_colors, edgecolor='white', linewidth=0.5)
axes[2].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
axes[2].set_xticks(x)
axes[2].set_xticklabels([f'{g}\n(n={n})' for g, n in zip(gene_names, gene_ns)], fontsize=9)
axes[2].set_ylabel('Balanced Accuracy')
axes[2].set_title('Balanced Accuracy by Gene')
axes[2].set_ylim([0, 1.05])
for i, v in enumerate(gene_bals):
    axes[2].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

fig.suptitle('SteppeDNA Universal Model: Per-Gene Performance Breakdown\n'
             'BRCA2 dominates overall metrics; non-BRCA2 genes lack discriminative features in ClinVar',
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig('visual_proofs/5_PerGene_Performance_Summary.png', dpi=150, bbox_inches='tight')
fig.savefig('visual_proofs/5_PerGene_Performance_Summary.pdf', bbox_inches='tight')
plt.close()
print("  Saved 5_PerGene_Performance_Summary")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Confusion Matrix Heatmaps (Overall + Per-Gene)
# ─────────────────────────────────────────────────────────────────────────────
print("[6/6] Confusion Matrix Heatmaps...")

unique_genes = sorted(set(genes_test))
n_panels = 1 + len(unique_genes)
fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))

# Overall
cm_all = confusion_matrix(y_test, y_pred)
sns.heatmap(cm_all, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Benign', 'Path.'], yticklabels=['Benign', 'Path.'])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')
axes[0].set_title(f'Overall (n={len(y_test)})\nMCC={metrics["mcc"]:.3f}')

# Per gene
for idx, gene in enumerate(unique_genes):
    mask = genes_test == gene
    y_g = y_test[mask]
    pred_g = y_pred[mask]
    cm_g = confusion_matrix(y_g, pred_g)
    gene_mcc = matthews_corrcoef(y_g, pred_g)
    ax = axes[idx + 1]
    sns.heatmap(cm_g, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Ben.', 'Path.'], yticklabels=['Ben.', 'Path.'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{gene} (n={mask.sum()})\nMCC={gene_mcc:.3f}')

fig.suptitle('SteppeDNA: Confusion Matrices by Gene (threshold={:.3f})'.format(threshold),
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig('visual_proofs/6_Confusion_Matrices.png', dpi=150, bbox_inches='tight')
fig.savefig('visual_proofs/6_Confusion_Matrices.pdf', bbox_inches='tight')
plt.close()
print("  Saved 6_Confusion_Matrices")

print("\nAll figures saved to visual_proofs/")
print("Done.")
