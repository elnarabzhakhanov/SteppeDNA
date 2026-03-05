# Train a "MAVE-blind" XGBoost model that explicitly drops MAVE features.
# Used exclusively for unbiased external validation against the MAVE dataset.
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
import pickle
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.feature_engineering import (
    load_phylop_scores, load_mave_scores, load_alphamissense_scores,
    load_structural_features, load_gnomad_frequencies, load_spliceai_scores,
    engineer_features
)

RANDOM_STATE = 42

print("Loading Data for MAVE-Blind XGBoost Model...")
mutation_df = pd.read_csv("brca2_missense_dataset_2.csv")

# ----- HOLD OUT MAVE VARIANTS FROM TRAINING ENTIRELY -----
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

is_mave = mutation_df.apply(lambda r: (str(r['AA_ref']).strip(), int(r['AA_pos']), str(r['AA_alt']).strip()) in mave_set, axis=1)
print(f"Holding out {is_mave.sum()} MAVE variants from training to ensure TRUE unseen validation.")

# Only train on the non-MAVE variants
mutation_train_df = mutation_df[~is_mave].copy()

phylop = load_phylop_scores(data_dir="data")
mave = load_mave_scores(data_dir="data")
am = load_alphamissense_scores(data_dir="data")
struct = load_structural_features(data_dir="data")
gnomad = load_gnomad_frequencies(data_dir="data")
spliceai = load_spliceai_scores(data_dir="data")

X = engineer_features(mutation_train_df, phylop, mave, am, struct, gnomad, spliceai_data=spliceai)

# ----- DROP MAVE FEATURES FOR BLIND VALIDATION -----
mave_cols = ['mave_score', 'has_mave', 'mave_abnormal', 'mave_x_blosum']
print(f"Dropping {len(mave_cols)} MAVE features to prevent data leakage: {mave_cols}")
X = X.drop(columns=mave_cols, errors='ignore')

# ----- DROP SPARSE ONE-HOT ENCODED FEATURES TO PREVENT OVERFITTING -----
noise_cols = [c for c in X.columns if c.startswith('AA_ref_') or c.startswith('AA_alt_') or c.startswith('Mutation_')]
print(f"Dropping {len(noise_cols)} sparse identity features to force generalization...")
X = X.drop(columns=noise_cols, errors='ignore')
# ---------------------------------------------------

y = mutation_train_df["Label"].values

print("Splitting Data (80% Train, 20% Test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("Scaling Data based strictly on Train Set...")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)  

with open("data/scaler_blind.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Calculating class weights directly from training distribution (bypassing SMOTE)...")
cw = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

print("Training underlying XGBoost model...")
params = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.05,
    'scale_pos_weight': cw,
    'random_state': RANDOM_STATE,
    'eval_metric': 'logloss',
    'n_jobs': -1
}

if os.path.exists("data/optuna_blind_params.json"):
    print("  -> Loading optimal hyperparameters from aggressive regularized Optuna tuning...")
    with open("data/optuna_blind_params.json", "r") as f:
        opt_params = json.load(f)
        params.update(opt_params)
        params['scale_pos_weight'] = cw  # Preserve calculated class weight
        params['random_state'] = RANDOM_STATE
        params['n_jobs'] = -1

xgb_clf = xgb.XGBClassifier(**params)
xgb_clf.fit(X_train_s, y_train)

print("Saving standard underlying model to data/brca2_xgboost_blind.json...")
xgb_clf.save_model("data/brca2_xgboost_blind.json")
with open("data/feature_names_blind.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("Fitting isotonic calibrator via 5-fold CV on training set...")
calibrator = CalibratedClassifierCV(xgb_clf, method='isotonic', cv=5)
calibrator.fit(X_train_s, y_train)

with open("data/calibrator_blind.pkl", "wb") as f:
    pickle.dump(calibrator, f)

print("Finding optimal threshold to maximize F1 on the holdout Test set...")
y_probs = calibrator.predict_proba(X_test_s)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores[:-1])]

print(f"Optimal threshold found: {best_threshold:.3f}")
with open("data/threshold_blind.pkl", "wb") as f:
    pickle.dump(best_threshold, f)

print("\n[BLIND MODEL SAVED SUCCESSFULLY]")
