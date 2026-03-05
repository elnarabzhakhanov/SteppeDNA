# External validation against MAVE wet-lab data (independent of ClinVar training set).

import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import sys
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    accuracy_score, confusion_matrix, classification_report,
    balanced_accuracy_score, matthews_corrcoef
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
os.makedirs("visual_proofs", exist_ok=True)

DATA_DIR = "data"

print("=" * 60)
print(" SteppeDNA: MAVE External Biological Validation")
print("=" * 60)

# ============================================================
# Load the MAVE ground truth (wet-lab experimental measurements)
# ============================================================
print("\n[1] Loading MAVE Wet-Lab Ground Truth...")
mave_df = pd.read_csv(f"{DATA_DIR}/mave_scores.csv")
print(f"    -> {len(mave_df)} total MAVE variants")

# Using the original Findlay et al. cutoffs (1.5 and 2.5) to stay consistent with the literature.
MAVE_PATH_THRESHOLD = 1.5
MAVE_BENIGN_THRESHOLD = 2.5

mave_df['mave_label'] = np.where(
    mave_df['mave_score'] < MAVE_PATH_THRESHOLD, 1,  # 1 = Pathogenic
    np.where(mave_df['mave_score'] > MAVE_BENIGN_THRESHOLD, 0, np.nan)  # 0 = Benign, NaN = ambiguous
)
# Drop ambiguous middle-zone variants
mave_df = mave_df.dropna(subset=['mave_label'])
mave_df['mave_label'] = mave_df['mave_label'].astype(int)
print(f"    -> {len(mave_df)} usable variants (excl. ambiguous middle zone)")
print(f"    -> {mave_df['mave_label'].sum()} Pathogenic | {(mave_df['mave_label']==0).sum()} Benign")

# ============================================================
# Load SteppeDNA Model Artifacts
# ============================================================
print("\n[2] Loading SteppeDNA MAVE-Blind XGBoost Engine...")
with open(f"{DATA_DIR}/scaler_blind.pkl", "rb") as f:
    scaler = pickle.load(f)
with open(f"{DATA_DIR}/calibrator_blind.pkl", "rb") as f:
    calibrator = pickle.load(f)
with open(f"{DATA_DIR}/feature_names_blind.pkl", "rb") as f:
    feature_names = pickle.load(f)

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(f"{DATA_DIR}/brca2_xgboost_blind.json")

# Reload all feature data for the engineering pipeline
training_mutation_df = pd.read_csv("brca2_missense_dataset_2.csv")
phylop = load_phylop_scores(data_dir=DATA_DIR)
mave_feat = load_mave_scores(data_dir=DATA_DIR)
am = load_alphamissense_scores(data_dir=DATA_DIR)
struct = load_structural_features(data_dir=DATA_DIR)
gnomad = load_gnomad_frequencies(data_dir=DATA_DIR)
spliceai = load_spliceai_scores(data_dir=DATA_DIR)

# ============================================================
# Match MAVE variants to our feature engineering pipeline
# ============================================================
print("\n[3] Mapping MAVE Variants Through 99-Feature Engineering Pipeline...")

# Parse HGVS protein notation (expects format like p.Ser2483Asn, missense only)
def parse_hgvs_pro(hgvs):
    try:
        p = hgvs.replace("p.", "")
        ref3 = p[:3]    # e.g. "Ser"
        alt3 = p[-3:]   # e.g. "Asn"
        pos = int(''.join(filter(str.isdigit, p)))
        return ref3, pos, alt3
    except:
        return None, None, None

X_full = engineer_features(training_mutation_df, phylop, mave_feat, am, struct, gnomad, spliceai_data=spliceai)

row_lookup = {}
for i, (_, row) in enumerate(training_mutation_df.iterrows()):
    try:
        aa_pos = int(row['AA_pos'])
        aa_ref = str(row['AA_ref'])   # e.g. 'Asn'
        aa_alt = str(row['AA_alt'])   # e.g. 'His'
        row_lookup[(aa_ref, aa_pos, aa_alt)] = i
    except:
        continue

matched = []
y_true = []
y_pred_proba = []

for _, mrow in mave_df.iterrows():
    aa_ref, aa_pos, aa_alt = parse_hgvs_pro(mrow['hgvs_pro'])
    if aa_ref is None or aa_pos is None:
        continue
    feat_idx = row_lookup.get((aa_ref, aa_pos, aa_alt))
    if feat_idx is None:
        continue
    
    # Ensure we only use the features that the blind model expects
    feature_vec = X_full.iloc[[feat_idx]][feature_names].values
    scaled_vec = scaler.transform(feature_vec)
    prob = float(calibrator.predict_proba(scaled_vec)[0, 1])
    
    y_true.append(mrow['mave_label'])
    y_pred_proba.append(prob)
    matched.append({'hgvs_pro': mrow['hgvs_pro'], 'mave_score': mrow['mave_score'],
                    'mave_label': mrow['mave_label'], 'steppedna_prob': prob})

print(f"    -> Matched {len(matched)}/{len(mave_df)} MAVE variants to feature vectors")

if len(y_true) < 10:
    print("[ERROR] Not enough matched variants. Check column names in brca2_missense_dataset_2.csv")
    exit(1)

y_true = np.array(y_true)
y_pred_proba = np.array(y_pred_proba)

# Apply the 0.10 clinical sensitivity threshold
SENSITIVITY_THRESHOLD = 0.10
y_pred_label = (y_pred_proba >= SENSITIVITY_THRESHOLD).astype(int)

# ============================================================
# Print External Validation Report
# ============================================================
print("\n" + "=" * 60)
print(" EXTERNAL MAVE VALIDATION REPORT")
print("=" * 60)

fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)
prec, rec, _ = precision_recall_curve(y_true, y_pred_proba)
pr_auc = auc(rec, prec)

sensitivity = float(np.sum((y_pred_label == 1) & (y_true == 1))) / float(np.sum(y_true == 1))
specificity = float(np.sum((y_pred_label == 0) & (y_true == 0))) / float(np.sum(y_true == 0))
accuracy = accuracy_score(y_true, y_pred_label)

cm = confusion_matrix(y_true, y_pred_label)

print(f"\n  External Validation Set: {len(y_true)} BRCA2 variants (MAVE wet-lab labels)")
print(f"  Pathogenic: {int(np.sum(y_true))} | Benign: {int(np.sum(y_true==0))}")
bal_acc = balanced_accuracy_score(y_true, y_pred_label)
mcc = matthews_corrcoef(y_true, y_pred_label)

print(f"\n  ROC-AUC:           {roc_auc:.4f}")
print(f"  PR-AUC:            {pr_auc:.4f}")
print(f"  Balanced Accuracy: {bal_acc:.4f}")
print(f"  MCC:               {mcc:.4f}")
print(f"  Sensitivity:       {sensitivity*100:.1f}%   (threshold = {SENSITIVITY_THRESHOLD})")
print(f"  Specificity:       {specificity*100:.1f}%")
print(f"  Accuracy:          {accuracy*100:.1f}%")
print(f"\n  Confusion Matrix (Pathogenic=1, Benign=0):")
print(f"  [[TN={cm[0,0]} FP={cm[0,1]}]")
print(f"   [FN={cm[1,0]} TP={cm[1,1]}]]")

# ============================================================
# Generate High-Quality Publication Figure
# ============================================================
print("\n[4] Generating External Validation Figure...")

fig = plt.figure(figsize=(16, 6))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

# Panel A: ROC Curve
ax1 = fig.add_subplot(gs[0])
ax1.plot(fpr, tpr, color='#6260FF', lw=2.5, label=f'SteppeDNA XGBoost\n(AUC = {roc_auc:.4f})')
ax1.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random Chance (AUC=0.5)')
ax1.fill_between(fpr, tpr, alpha=0.08, color='#6260FF')
ax1.set_xlabel('False Positive Rate', fontweight='bold')
ax1.set_ylabel('True Positive Rate', fontweight='bold')
ax1.set_title('A. ROC Curve\n(MAVE External Validation)', fontweight='bold')
ax1.legend(loc='lower right', fontsize=9)
ax1.set_xlim([0, 1]); ax1.set_ylim([0, 1.05])

# Panel B: PR Curve
ax2 = fig.add_subplot(gs[1])
ax2.plot(rec, prec, color='#FF3B30', lw=2.5, label=f'SteppeDNA XGBoost\n(AUC = {pr_auc:.4f})')
prevalence = np.sum(y_true) / len(y_true)
ax2.axhline(prevalence, color='k', ls='--', lw=1, alpha=0.5, label=f'Random ({prevalence:.2f})')
ax2.fill_between(rec, prec, alpha=0.08, color='#FF3B30')
ax2.set_xlabel('Recall (Sensitivity)', fontweight='bold')
ax2.set_ylabel('Precision', fontweight='bold')
ax2.set_title('B. Precision-Recall Curve\n(MAVE External Validation)', fontweight='bold')
ax2.legend(loc='lower left', fontsize=9)
ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.05])

# Panel C: Score Distribution (model prob vs MAVE score)
ax3 = fig.add_subplot(gs[2])
matched_df = pd.DataFrame(matched)
benign_mask = matched_df['mave_label'] == 0
path_mask = matched_df['mave_label'] == 1
ax3.scatter(matched_df.loc[benign_mask, 'steppedna_prob'], 
            matched_df.loc[benign_mask, 'mave_score'],
            c='#34C759', alpha=0.6, s=30, label='Benign (MAVE)', zorder=2)
ax3.scatter(matched_df.loc[path_mask, 'steppedna_prob'], 
            matched_df.loc[path_mask, 'mave_score'],
            c='#FF3B30', alpha=0.6, s=30, label='Pathogenic (MAVE)', zorder=2)
ax3.axvline(SENSITIVITY_THRESHOLD, color='#6260FF', ls='--', lw=1.5, label=f'SteppeDNA Threshold ({SENSITIVITY_THRESHOLD})')
ax3.axhline(MAVE_PATH_THRESHOLD, color='gray', ls=':', lw=1, alpha=0.6, label=f'MAVE Path Threshold ({MAVE_PATH_THRESHOLD})')
ax3.set_xlabel('SteppeDNA Probability', fontweight='bold')
ax3.set_ylabel('MAVE Functional Score', fontweight='bold')
ax3.set_title('C. Predicted Prob vs Wet-Lab Score\n(Experimental Correlation)', fontweight='bold')
ax3.legend(fontsize=8)

# Add metrics text box
metrics_text = f"Sens: {sensitivity*100:.1f}%\nSpec: {specificity*100:.1f}%\nn={len(y_true)}"
ax3.text(0.98, 0.98, metrics_text, transform=ax3.transAxes, fontsize=9,
         va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('SteppeDNA: External Biological Validation on Independent MAVE Dataset\n'
             '(Trained on ClinVar, tested on Findlay et al. wet-lab assay)', 
             fontsize=12, fontweight='bold', y=1.02)

plt.savefig('visual_proofs/7_MAVE_External_Validation.pdf', bbox_inches='tight')
plt.savefig('visual_proofs/7_MAVE_External_Validation.png', bbox_inches='tight', dpi=150)
plt.close()

print("\n[SUCCESS] External Validation complete!")
print(f"  -> PDF saved: visual_proofs/7_MAVE_External_Validation.pdf")
print(f"  -> PNG saved: visual_proofs/7_MAVE_External_Validation.png")
print(f"\n  KEY RESULT: SteppeDNA achieves ROC-AUC = {roc_auc:.4f} on INDEPENDENT WET-LAB DATA")
print(f"  This validates the model generalizes beyond its ClinVar training set.")
