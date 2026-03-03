# SteppeDNA: Comprehensive Model Metrics Evaluation
# Evaluates the universal ensemble model on a held-out test set using proper stratified splits.
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, classification_report, balanced_accuracy_score,
    cohen_kappa_score, log_loss, brier_score_loss
)

RANDOM_STATE = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

print("=" * 60)
print("  SteppeDNA: Comprehensive Model Metrics")
print("=" * 60)

# 1. Load Data
df = pd.read_csv(os.path.join(DATA_DIR, "master_training_dataset.csv"))
with open(os.path.join(DATA_DIR, "universal_feature_names.pkl"), "rb") as f:
    feature_names = pickle.load(f)

X = df[feature_names].values
y = df["Label"].values
genes = df["Gene"].values

print(f"\n  Dataset:       {len(X)} missense variants")
print(f"  Features:      {len(feature_names)}")
print(f"  Genes:         {', '.join(sorted(set(genes)))}")
print(f"  Class Balance: {sum(y==0)} Benign / {sum(y==1)} Pathogenic")

# 2. Stratified Train/Test Split (same as training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
genes_test = df["Gene"].values[
    train_test_split(np.arange(len(y)), test_size=0.2, random_state=RANDOM_STATE, stratify=y)[1]
]

print(f"\n  Test Set:      {len(y_test)} samples ({sum(y_test==0)} B / {sum(y_test==1)} P)")

# 3. Load Model Artifacts
print("\n  Loading universal model artifacts...")
with open(os.path.join(DATA_DIR, "universal_scaler_ensemble.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(DATA_DIR, "universal_threshold_ensemble.pkl"), "rb") as f:
    threshold = pickle.load(f)
with open(os.path.join(DATA_DIR, "universal_calibrator_ensemble.pkl"), "rb") as f:
    calibrator = pickle.load(f)

# Load XGBoost
import xgboost as xgb
xgb_clf = xgb.XGBClassifier()
xgb_clf.load_model(os.path.join(DATA_DIR, "universal_xgboost_final.json"))

# Load Neural Network
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
nn_model = tf.keras.models.load_model(os.path.join(DATA_DIR, "universal_nn.h5"), compile=False)

# 4. Predict on Test Set
X_test_scaled = scaler.transform(X_test)

nn_preds = nn_model.predict(X_test_scaled, verbose=0).flatten()
xgb_preds = xgb_clf.predict_proba(X_test_scaled)[:, 1]
blended = (0.6 * xgb_preds) + (0.4 * nn_preds)
y_probs = calibrator.predict(blended)
y_pred = (y_probs >= threshold).astype(int)

# 5. Core Metrics
print("\n" + "=" * 60)
print("  CORE PERFORMANCE METRICS")
print("=" * 60)
print(f"  Optimal Threshold:       {threshold:.4f}")
print("-" * 60)
print(f"  Accuracy:                {accuracy_score(y_test, y_pred):.4f}")
print(f"  Balanced Accuracy:       {balanced_accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision:               {precision_score(y_test, y_pred):.4f}")
print(f"  Recall (Sensitivity):    {recall_score(y_test, y_pred):.4f}")
print(f"  Specificity:             {recall_score(y_test, y_pred, pos_label=0):.4f}")
print(f"  F1 Score:                {f1_score(y_test, y_pred):.4f}")
print(f"  ROC-AUC:                 {roc_auc_score(y_test, y_probs):.4f}")
print(f"  PR-AUC (Avg Precision):  {average_precision_score(y_test, y_probs):.4f}")
print(f"  Matthews Corr. Coeff:    {matthews_corrcoef(y_test, y_pred):.4f}")
print(f"  Cohen's Kappa:           {cohen_kappa_score(y_test, y_pred):.4f}")
print(f"  Log Loss:                {log_loss(y_test, y_probs):.4f}")
print(f"  Brier Score:             {brier_score_loss(y_test, y_probs):.4f}")
print("-" * 60)

# 6. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print("\n" + "=" * 60)
print("  CONFUSION MATRIX")
print("=" * 60)
print(f"                        Predicted")
print(f"                     Benign    Pathogenic")
print(f"  Actual Benign:    {tn:>6}      {fp:>6}")
print(f"  Actual Pathogenic:{fn:>6}      {tp:>6}")
print()
print(f"  True Positives:  {tp}     False Positives: {fp}")
print(f"  True Negatives:  {tn}     False Negatives: {fn}")

# 7. Per-Class Classification Report
print("\n" + "=" * 60)
print("  CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=["Benign", "Pathogenic"]))

# 8. Per-Gene Breakdown
print("=" * 60)
print("  PER-GENE PERFORMANCE")
print("=" * 60)
for gene in sorted(set(genes_test)):
    mask = genes_test == gene
    if sum(mask) < 2:
        continue
    g_y = y_test[mask]
    g_probs = y_probs[mask]
    g_pred = y_pred[mask]
    
    n = sum(mask)
    acc = accuracy_score(g_y, g_pred)
    
    # ROC-AUC only if both classes present
    try:
        auc = roc_auc_score(g_y, g_probs)
        auc_str = f"{auc:.4f}"
    except ValueError:
        auc_str = "N/A (single class)"
        
    try:
        g_mcc = matthews_corrcoef(g_y, g_pred)
        g_bal = balanced_accuracy_score(g_y, g_pred)
    except Exception:
        g_mcc = 0.0
        g_bal = 0.0

    print(f"\n  {gene} ({n} test samples, {sum(g_y==0)}B/{sum(g_y==1)}P):")
    print(f"    ROC-AUC: {auc_str}  |  MCC: {g_mcc:.4f}  |  BalAcc: {g_bal:.4f}  |  F1: {f1_score(g_y, g_pred, zero_division=0):.4f}")

print("\n" + "=" * 60)
print("  [EVALUATION COMPLETE]")
print("=" * 60)
