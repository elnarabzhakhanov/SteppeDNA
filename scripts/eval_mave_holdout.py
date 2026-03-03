import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backend.feature_engineering import (
    engineer_features, load_phylop_scores, load_mave_scores,
    load_alphamissense_scores, load_structural_features, load_gnomad_frequencies,
    load_spliceai_scores
)

print("Loading Data for MAVE-Blind True Holdout Test...")
mutation_df = pd.read_csv("brca2_missense_dataset_2.csv")

mave_df = pd.read_csv("data/mave_scores.csv")
def parse_hgvs_pro(hgvs):
    try:
        p = hgvs.replace("p.", "")
        return p[:3], int(''.join(filter(str.isdigit, p))), p[-3:]
    except:
        return None, None, None

mave_set = set()
for _, row in mave_df.iterrows():
    r, p, a = parse_hgvs_pro(row['hgvs_pro'])
    if r:
        mave_set.add((r, p, a))

# Identify TRUE MAVE variants
is_mave = mutation_df.apply(lambda r: (str(r['AA_ref']).strip(), int(r['AA_pos']), str(r['AA_alt']).strip()) in mave_set, axis=1)

mave_holdout_df = mutation_df[is_mave].copy()
print(f"Total True MAVE variants held out: {len(mave_holdout_df)}")

phylop = load_phylop_scores(data_dir="data")
mave = load_mave_scores(data_dir="data")
am = load_alphamissense_scores(data_dir="data")
struct = load_structural_features(data_dir="data")
gnomad = load_gnomad_frequencies(data_dir="data")
spliceai = load_spliceai_scores(data_dir="data")

X_mave = engineer_features(mave_holdout_df, phylop, mave, am, struct, gnomad, spliceai_data=spliceai)

mave_cols = ['mave_score', 'has_mave', 'mave_abnormal', 'mave_x_blosum']
X_mave = X_mave.drop(columns=mave_cols, errors='ignore')
noise_cols = [c for c in X_mave.columns if c.startswith('AA_ref_') or c.startswith('AA_alt_') or c.startswith('Mutation_')]
X_mave = X_mave.drop(columns=noise_cols, errors='ignore')

y_mave = mave_holdout_df["Label"].values

print(f"Class Balance in MAVE holdout: {sum(y_mave==0)} B / {sum(y_mave==1)} P")

print("\nLoading pre-trained MAVE-blind XGBoost models and tools...")
with open("data/scaler_blind.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("data/threshold_blind.pkl", "rb") as f:
    threshold = pickle.load(f)
with open("data/calibrator_blind.pkl", "rb") as f:
    calibrator = pickle.load(f)

# The model expects specific features, make sure X_mave aligns
with open("data/feature_names_blind.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Reorder/fill missing
for col in feature_names:
    if col not in X_mave.columns:
        X_mave[col] = 0
X_mave = X_mave[feature_names]

X_mave_s = scaler.transform(X_mave.values)

y_probs = calibrator.predict_proba(X_mave_s)[:, 1]
y_pred = (y_probs >= threshold).astype(int)

auc = roc_auc_score(y_mave, y_probs)
f1 = f1_score(y_mave, y_pred)
acc = accuracy_score(y_mave, y_pred)
prec = precision_score(y_mave, y_pred)
rec = recall_score(y_mave, y_pred)

print("\n===== MAVE STRICT INDEPENDENT HOLDOUT PERFORMANCE =====")
print(f"ROC-AUC:   {auc:.4f}")
print(f"Accuracy:  {acc:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")

cm = confusion_matrix(y_mave, y_pred)
print("\nConfusion Matrix:")
print(cm)
