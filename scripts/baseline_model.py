# Logistic regression on 5 hand-picked features as a baseline comparison.
# Proves the full 99-feature XGBoost adds value over a naive approach.

import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, average_precision_score
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backend.feature_engineering import BLOSUM62, get_blosum62

df = pd.read_csv("brca2_missense_dataset_2.csv")
print(f"Dataset: {len(df)} samples")

# Build 5 features
print("Engineering 5 features...")

df['AA_pos'] = df['cDNA_pos'] // 3
df['blosum62_score'] = df.apply(lambda r: get_blosum62(r['AA_ref'], r['AA_alt']), axis=1)
df['in_DNA_binding'] = ((df['AA_pos'] >= 2402) & (df['AA_pos'] <= 3190)).astype(int)
df['is_nonsense'] = (df['AA_alt'] == 'Ter').astype(int)

FEATURE_COLS = ['cDNA_pos', 'blosum62_score', 'in_DNA_binding', 'is_nonsense', 'AA_pos']
TARGET = 'Label'

X = df[FEATURE_COLS].values
y = df[TARGET].values

print(f"Features:  {FEATURE_COLS}")
print(f"Positive (pathogenic): {y.sum()}  |  Negative (benign): {(1-y).sum()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)

y_prob = lr.predict_proba(X_test_s)[:, 1]
y_pred = lr.predict(X_test_s)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_prob)
ap   = average_precision_score(y_test, y_prob)
cm   = confusion_matrix(y_test, y_pred)

print("\n" + "="*50)
print("  BASELINE: Logistic Regression (5 features)")
print("="*50)
print(f"  Accuracy:          {acc:.3f}")
print(f"  Precision:         {prec:.3f}")
print(f"  Recall:            {rec:.3f}")
print(f"  F1 Score:          {f1:.3f}")
print(f"  ROC-AUC:           {auc:.3f}")
print(f"  Avg Precision:     {ap:.3f}")
print(f"\n  Confusion Matrix:")
print(f"    TN={cm[0][0]:4d}   FP={cm[0][1]:4d}")
print(f"    FN={cm[1][0]:4d}   TP={cm[1][1]:4d}")

print("\n  Feature coefficients (Logistic Regression):")
for name, coef in zip(FEATURE_COLS, lr.coef_[0]):
    direction = "pathogenic" if coef > 0 else "benign"
    print(f"    {name:20s}  {coef:+.4f}  ({direction})")

print("\n[DONE]")
