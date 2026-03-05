# External validation against MAVE wet-lab data specifically for the Stacking Ensemble
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import sys
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    accuracy_score, confusion_matrix
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.feature_engineering import (
    load_phylop_scores, load_mave_scores, load_alphamissense_scores,
    load_structural_features, load_gnomad_frequencies, load_spliceai_scores,
    engineer_features
)

import warnings
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", font_scale=1.1)

DATA_DIR = "data"

print("=" * 60)
print(" SteppeDNA: MAVE External Biological Validation (ENSEMBLE)")
print("=" * 60)

print("\n[1] Loading MAVE Wet-Lab Ground Truth...")
mave_df = pd.read_csv(f"{DATA_DIR}/mave_scores.csv")

MAVE_PATH_THRESHOLD = 1.5
MAVE_BENIGN_THRESHOLD = 2.5

mave_df['mave_label'] = np.where(
    mave_df['mave_score'] < MAVE_PATH_THRESHOLD, 1,
    np.where(mave_df['mave_score'] > MAVE_BENIGN_THRESHOLD, 0, np.nan)
)
mave_df = mave_df.dropna(subset=['mave_label'])
mave_df['mave_label'] = mave_df['mave_label'].astype(int)

print("\n[2] Loading SteppeDNA Stacking Ensemble (XGB+MLP+SVM)...")
with open(f"{DATA_DIR}/scaler_ensemble_blind.pkl", "rb") as f:
    scaler = pickle.load(f)
with open(f"{DATA_DIR}/calibrator_ensemble_blind.pkl", "rb") as f:
    calibrator = pickle.load(f)
with open(f"{DATA_DIR}/feature_names_ensemble_blind.pkl", "rb") as f:
    feature_names = pickle.load(f)

training_mutation_df = pd.read_csv("brca2_missense_dataset_2.csv")
phylop = load_phylop_scores(data_dir=DATA_DIR)
mave_feat = load_mave_scores(data_dir=DATA_DIR)
am = load_alphamissense_scores(data_dir=DATA_DIR)
struct = load_structural_features(data_dir=DATA_DIR)
gnomad = load_gnomad_frequencies(data_dir=DATA_DIR)
spliceai = load_spliceai_scores(data_dir=DATA_DIR)

print("\n[3] Mapping MAVE Variants Through Feature Pipeline...")

def parse_hgvs_pro(hgvs):
    try:
        p = hgvs.replace("p.", "")
        return p[:3], int(''.join(filter(str.isdigit, p))), p[-3:]
    except:
        return None, None, None

X_full = engineer_features(training_mutation_df, phylop, mave_feat, am, struct, gnomad, spliceai_data=spliceai)

row_lookup = {}
for i, (_, row) in enumerate(training_mutation_df.iterrows()):
    try:
        row_lookup[(str(row['AA_ref']), int(row['AA_pos']), str(row['AA_alt']))] = i
    except:
        continue

matched, y_true, y_pred_proba = [], [], []

for _, mrow in mave_df.iterrows():
    aa_ref, aa_pos, aa_alt = parse_hgvs_pro(mrow['hgvs_pro'])
    if aa_ref is None: continue
    feat_idx = row_lookup.get((aa_ref, aa_pos, aa_alt))
    if feat_idx is None: continue
    
    feature_vec = X_full.iloc[[feat_idx]][feature_names].values
    scaled_vec = scaler.transform(feature_vec)
    prob = float(calibrator.predict_proba(scaled_vec)[0, 1])
    
    y_true.append(mrow['mave_label'])
    y_pred_proba.append(prob)
    matched.append({'hgvs_pro': mrow['hgvs_pro'], 'mave_score': mrow['mave_score'], 'mave_label': mrow['mave_label'], 'prob': prob})

y_true = np.array(y_true)
y_pred_proba = np.array(y_pred_proba)

SENSITIVITY_THRESHOLD = 0.10
y_pred_label = (y_pred_proba >= SENSITIVITY_THRESHOLD).astype(int)

fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)
prec, rec, _ = precision_recall_curve(y_true, y_pred_proba)
pr_auc = auc(rec, prec)

sensitivity = float(np.sum((y_pred_label == 1) & (y_true == 1))) / float(np.sum(y_true == 1))
specificity = float(np.sum((y_pred_label == 0) & (y_true == 0))) / float(np.sum(y_true == 0))
accuracy = accuracy_score(y_true, y_pred_label)

cm = confusion_matrix(y_true, y_pred_label)

print(f"\n  ROC-AUC:     {roc_auc:.4f}")
print(f"  PR-AUC:      {pr_auc:.4f}")
print(f"  Sensitivity: {sensitivity*100:.1f}%")
print(f"  Specificity: {specificity*100:.1f}%")
print(f"  Accuracy:    {accuracy*100:.1f}%")

fig = plt.figure(figsize=(16, 6))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

ax1 = fig.add_subplot(gs[0])
ax1.plot(fpr, tpr, color='#9C27B0', lw=2.5, label=f'SteppeDNA Ensemble\n(AUC = {roc_auc:.4f})')
ax1.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax1.fill_between(fpr, tpr, alpha=0.08, color='#9C27B0')
ax1.set_xlabel('False Positive Rate', fontweight='bold')
ax1.set_ylabel('True Positive Rate', fontweight='bold')
ax1.set_title('A. ROC Curve (Ensemble)', fontweight='bold')
ax1.legend(loc='lower right')
ax1.set_xlim([0, 1]); ax1.set_ylim([0, 1.05])

ax2 = fig.add_subplot(gs[1])
ax2.plot(rec, prec, color='#FF9800', lw=2.5, label=f'SteppeDNA Ensemble\n(AUC = {pr_auc:.4f})')
prevalence = np.sum(y_true) / len(y_true)
ax2.axhline(prevalence, color='k', ls='--', lw=1, alpha=0.5)
ax2.fill_between(rec, prec, alpha=0.08, color='#FF9800')
ax2.set_xlabel('Recall (Sensitivity)', fontweight='bold')
ax2.set_ylabel('Precision', fontweight='bold')
ax2.set_title('B. PR Curve (Ensemble)', fontweight='bold')
ax2.legend(loc='lower left')
ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.05])

ax3 = fig.add_subplot(gs[2])
matched_df = pd.DataFrame(matched)
b_mask = matched_df['mave_label'] == 0
p_mask = matched_df['mave_label'] == 1
ax3.scatter(matched_df.loc[b_mask, 'prob'], matched_df.loc[b_mask, 'mave_score'], c='#34C759', alpha=0.6, s=30, label='Benign (MAVE)')
ax3.scatter(matched_df.loc[p_mask, 'prob'], matched_df.loc[p_mask, 'mave_score'], c='#FF3B30', alpha=0.6, s=30, label='Pathogenic (MAVE)')
ax3.axvline(SENSITIVITY_THRESHOLD, color='#9C27B0', ls='--', lw=1.5, label=f'Ensemble Threshold')
ax3.axhline(MAVE_PATH_THRESHOLD, color='gray', ls=':', lw=1, alpha=0.6, label='MAVE Path Threshold')
ax3.set_xlabel('SteppeDNA Ensemble Probability', fontweight='bold')
ax3.set_ylabel('MAVE Functional Score', fontweight='bold')
ax3.set_title('C. Predicted Prob vs Wet-Lab', fontweight='bold')
ax3.legend(fontsize=8)

plt.suptitle('SteppeDNA: External Biological Validation (Stacking Ensemble: XGBoost + MLP + SVM)', fontsize=12, fontweight='bold', y=1.02)
plt.savefig('visual_proofs/8_MAVE_Ensemble_Validation.pdf', bbox_inches='tight')
plt.savefig('visual_proofs/8_MAVE_Ensemble_Validation.png', bbox_inches='tight', dpi=150)
plt.close()

print(f"\n[SUCCESS] Ensemble validated. ROC-AUC: {roc_auc:.4f}")
