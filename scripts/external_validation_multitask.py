# External Validation: Multi-Task Neural Network vs. MAVE Wet-Lab
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.feature_engineering import (
    load_phylop_scores, load_mave_scores, load_alphamissense_scores,
    load_structural_features, load_gnomad_frequencies, load_spliceai_scores,
    engineer_features
)

print("\n============================================================")
print(" SteppeDNA: True Out-of-Distribution Validation (Multi-Task)")
print("============================================================\n")

# 1. Load the Strict MAVE Holdout Set
print("[1] Loading Independent MAVE Biological Assays...")
with open("data/mave_scores.pkl", "rb") as f:
    mave_dict = pickle.load(f)

mave_by_variant = mave_dict.get("by_variant", {})
print(f"    -> Found {len(mave_by_variant)} unique MAVE variant assays.")

# 2. Re-construct the dataset from the ground up for these exact variants
mutation_dfs = []
for hgvs, mave_data in mave_by_variant.items():
    try:
        p_str = hgvs.replace("p.", "")
        ref = p_str[:3]
        pos = int(''.join(filter(str.isdigit, p_str)))
        alt = p_str[-3:]
    except:
        continue
        
    # In the pickle file, mave_data is already just the raw float score
    score = float(mave_data)
    
    # Define ground truth based on MAVE threshold (Apples to Apples with Ensemble script)
    if score < 1.5:
        is_pathogenic = 1
    elif score > 2.5:
        is_pathogenic = 0
    else:
        continue # Drop intermediate/uncertain assays for clean evaluation
    
    mutation_dfs.append({
        "AA_ref": ref,
        "AA_pos": pos,
        "AA_alt": alt,
        "Mutation": f"{ref}{pos}{alt}",
        "cDNA_pos": 0,  # Proxy
        "Label": is_pathogenic,
        "mave_true_score": score
    })

holdout_df = pd.DataFrame(mutation_dfs)
print(f"    -> Constructed clean holdout dataframe: {len(holdout_df)} missense variants.")

# 3. Load Feature Modules
print("\n[2] Loading SteppeDNA Feature Tensors...")
phylop = load_phylop_scores(data_dir="data")
mave = load_mave_scores(data_dir="data")
am = load_alphamissense_scores(data_dir="data")
struct = load_structural_features(data_dir="data")
gnomad = load_gnomad_frequencies(data_dir="data")
spliceai = load_spliceai_scores(data_dir="data")

# 4. Feature Engineering
print("\n[3] Calculating Physical Features for the MAVE variants...")
X_holdout = engineer_features(holdout_df, phylop, mave, am, struct, gnomad, spliceai_data=spliceai)

# Mask out MAVE data so the model is truly blind during evaluation
mave_cols = ['mave_score', 'has_mave', 'mave_abnormal', 'mave_x_blosum']
X_holdout = X_holdout.drop(columns=mave_cols, errors='ignore')

noise_cols = [c for c in X_holdout.columns if c.startswith('AA_ref_') or c.startswith('AA_alt_') or c.startswith('Mutation_')]
X_holdout = X_holdout.drop(columns=noise_cols, errors='ignore')

# 5. Load the Multi-Task Blind Weights & Scaler
print("\n[4] Loading Keras Multi-Task Brain (MAVE-Blind)...")

with open("data/scaler_multitask_blind.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("data/feature_names_multitask_blind.pkl", "rb") as f:
    feature_names = pickle.load(f)

def masked_mse(y_true, y_pred):
    target = y_true[:, 0]
    mask = y_true[:, 1]
    pred = tf.reshape(y_pred, [-1])
    squared_diff = tf.square(target - pred)
    masked_diff = squared_diff * mask
    unmasked_count = tf.reduce_sum(mask) + 1e-8
    return tf.reduce_sum(masked_diff) / unmasked_count

# Realign columns strictly to match what the Keras model expects
missing_cols = set(feature_names) - set(X_holdout.columns)
for col in missing_cols:
    X_holdout[col] = 0

X_holdout = X_holdout[feature_names]

# Crucially, fill any remaining NaNs (e.g., missing AlphaMissense) with 0 before scaling
X_holdout = X_holdout.fillna(0.0)

# Scale
X_holdout_s = scaler.transform(X_holdout.values)

# Load Model
model = tf.keras.models.load_model("data/brca2_multitask_blind.h5", custom_objects={'masked_mse': masked_mse})

# 6. Execute Predictions
print("\n[5] Executing Predictions (Classification & Regression Heads)...")
preds = model.predict(X_holdout_s, verbose=0)
class_probs = preds[0].flatten()
regress_scores = preds[1].flatten()

y_true_class = holdout_df["Label"].values
y_true_regress = holdout_df["mave_true_score"].values

# 7. Evaluate Performance
print("\n[6] Evaluating True Generalization Metrics...")

# Classification Evaluation
SENSITIVITY_THRESHOLD = 0.50 # Standard Neural Network boundary
y_pred_label = (class_probs >= SENSITIVITY_THRESHOLD).astype(int)

fpr, tpr, _ = roc_curve(y_true_class, class_probs)
roc_auc = auc(fpr, tpr)
prec, rec, _ = precision_recall_curve(y_true_class, class_probs)
pr_auc = auc(rec, prec)

sensitivity = float(np.sum((y_pred_label == 1) & (y_true_class == 1))) / float(np.sum(y_true_class == 1))
specificity = float(np.sum((y_pred_label == 0) & (y_true_class == 0))) / float(np.sum(y_true_class == 0))
accuracy = accuracy_score(y_true_class, y_pred_label)

cm = confusion_matrix(y_true_class, y_pred_label)

print(f"\n  Classification Head (Pathogenic vs Benign):")
print(f"  -------------------------------------------")
print(f"  ROC-AUC:     {roc_auc:.4f}")
print(f"  PR-AUC:      {pr_auc:.4f}")
print(f"  Sensitivity: {sensitivity*100:.1f}%")
print(f"  Specificity: {specificity*100:.1f}%")
print(f"  Accuracy:    {accuracy*100:.1f}%")

# Regression Evaluation
mse = np.mean((y_true_regress - regress_scores)**2)
rmse = np.sqrt(mse)
correlation = np.corrcoef(y_true_regress, regress_scores)[0, 1]

print(f"\n  Regression Head (Physical MAVE Score):")
print(f"  -------------------------------------------")
print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"  Pearson Correlation (r):        {correlation:.4f}")

# 8. Visually Map the Curves
os.makedirs("visual_proofs", exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
# 1. ROC Curve
axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'Multi-Task ROC (AUC = {roc_auc:.4f})')
axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[0].set_xlim([0.0, 1.0])
axes[0].set_ylim([0.0, 1.05])
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('Strict MAVE Validation: ROC Curve')
axes[0].legend(loc="lower right")

# 2. PR Curve
axes[1].plot(rec, prec, color='blue', lw=2, label=f'Multi-Task PR (AUC = {pr_auc:.4f})')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('Recall (Sensitivity)')
axes[1].set_ylabel('Precision (PPV)')
axes[1].set_title('Strict MAVE Validation: Precision-Recall')
axes[1].legend(loc="lower left")

# 3. Regression Scatter
axes[2].scatter(y_true_regress, regress_scores, alpha=0.5, color='purple', s=10)
# Fit a line for visual aid
m, b = np.polyfit(y_true_regress, regress_scores, 1)
axes[2].plot(y_true_regress, m*y_true_regress + b, color='red', lw=2, label=f'Trend (r={correlation:.2f})')
axes[2].set_xlabel("True Biological MAVE Score")
axes[2].set_ylabel("Neural Network Predicted MAVE Score")
axes[2].set_title("Regression Head Latent Prediction Accuracy")
axes[2].legend()

plt.tight_layout()
plt.savefig("visual_proofs/9_MAVE_MultiTask_Validation.png", dpi=300)

print(f"\n[SUCCESS] ROC/PR Curves and Regression Scatter plotted to visual_proofs/9_MAVE_MultiTask_Validation.png")
