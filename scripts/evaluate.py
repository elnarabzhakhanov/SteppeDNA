# Loads the trained XGBoost model and dumps evaluation metrics on the untouched test set.
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    balanced_accuracy_score, matthews_corrcoef
)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backend.feature_engineering import (
    engineer_features, load_phylop_scores, load_mave_scores,
    load_alphamissense_scores, load_structural_features, load_gnomad_frequencies
)

DATA_PATH = "brca2_missense_dataset_2.csv"
DATA_DIR = "data"

print("Loading and engineering features...")
df = pd.read_csv(DATA_PATH)
phylop = load_phylop_scores(data_dir=DATA_DIR)
mave = load_mave_scores(data_dir=DATA_DIR)
am = load_alphamissense_scores(data_dir=DATA_DIR)
struct = load_structural_features(data_dir=DATA_DIR)
gnomad = load_gnomad_frequencies(data_dir=DATA_DIR)
X = engineer_features(df, phylop, mave, am, struct, gnomad)
y = df["Label"].values

# Same stratified split as training to isolate the exact test set
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42, stratify=y
)

print("Loading saved artifacts...")
with open(f"{DATA_DIR}/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open(f"{DATA_DIR}/threshold.pkl", "rb") as f:
    threshold = pickle.load(f)
with open(f"{DATA_DIR}/calibrator.pkl", "rb") as f:
    calibrator = pickle.load(f)

# Scale using the training-fitted scaler
X_test_scaled = scaler.transform(X_test)

# Predict probabilities using the calibrator
y_probs = calibrator.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_probs >= threshold).astype(int)

print("=" * 50)
print("  SteppeDNA Model Evaluation (Untouched Test Set Only)")
print("=" * 50)
print()
print(f"  Dataset total: {len(y)} samples")
print(f"  Test set:      {len(y_test)} samples ({sum(y_test==0)} benign, {sum(y_test==1)} pathogenic)")
print(f"  Threshold:     {threshold:.3f}")
print()
print("-" * 50)
print(f"  Balanced Accuracy:  {balanced_accuracy_score(y_test, y_pred):.4f}")
print(f"  MCC:                {matthews_corrcoef(y_test, y_pred):.4f}")
print(f"  Accuracy:           {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision:          {precision_score(y_test, y_pred):.4f}")
print(f"  Recall:             {recall_score(y_test, y_pred):.4f}")
print(f"  F1 Score:           {f1_score(y_test, y_pred):.4f}")
print(f"  ROC-AUC:            {roc_auc_score(y_test, y_probs):.4f}")
print(f"  Avg Precision (PR): {average_precision_score(y_test, y_probs):.4f}")
print("-" * 50)
print()

cm = confusion_matrix(y_test, y_pred)
print("  Confusion Matrix:")
print(f"                    Predicted Benign   Predicted Pathogenic")
print(f"  Actual Benign:    {cm[0][0]:>10}          {cm[0][1]:>10}")
print(f"  Actual Pathogenic:{cm[1][0]:>10}          {cm[1][1]:>10}")
print()

print("  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Benign", "Pathogenic"]))
