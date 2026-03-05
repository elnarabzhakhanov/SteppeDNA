# Train a "MAVE-blind" Stacking Ensemble (XGBoost + MLP + SVM)
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.feature_engineering import (
    load_phylop_scores, load_mave_scores, load_alphamissense_scores,
    load_structural_features, load_gnomad_frequencies, load_spliceai_scores,
    engineer_features
)

RANDOM_STATE = 42

print("\n============================================================")
print(" SteppeDNA: Heterogeneous Ensemble Training (MAVE-Blind)")
print("============================================================\n")

print("[1] Loading Data & Constructing True Holdout...")
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

is_mave = mutation_df.apply(lambda r: (str(r['AA_ref']).strip(), int(r['AA_pos']), str(r['AA_alt']).strip()) in mave_set, axis=1)
mutation_train_df = mutation_df[~is_mave].copy()

phylop = load_phylop_scores(data_dir="data")
mave = load_mave_scores(data_dir="data")
am = load_alphamissense_scores(data_dir="data")
struct = load_structural_features(data_dir="data")
gnomad = load_gnomad_frequencies(data_dir="data")
spliceai = load_spliceai_scores(data_dir="data")

print("[2] Engineering Features (Strict Generalization Set)...")
X = engineer_features(mutation_train_df, phylop, mave, am, struct, gnomad, spliceai_data=spliceai)

mave_cols = ['mave_score', 'has_mave', 'mave_abnormal', 'mave_x_blosum']
X = X.drop(columns=mave_cols, errors='ignore')

noise_cols = [c for c in X.columns if c.startswith('AA_ref_') or c.startswith('AA_alt_') or c.startswith('Mutation_')]
X = X.drop(columns=noise_cols, errors='ignore')
y = mutation_train_df["Label"].values

print("    -> Splitting Data (80% Train, 20% Test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("    -> Scaling Data based strictly on Train Set...")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)  

with open("data/scaler_ensemble_blind.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("data/feature_names_ensemble_blind.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

cw = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

print("\n[3] Building Heterogeneous Base Estimators...")

# Model 1: The Non-Linear Tree Specialist
xgb_params = {
    'scale_pos_weight': cw,
    'random_state': RANDOM_STATE,
    'eval_metric': 'logloss',
    'n_jobs': -1
}
if os.path.exists("data/optuna_blind_params.json"):
    print("    -> Loading aggressive Optuna parameters for XGBoost")
    with open("data/optuna_blind_params.json", "r") as f:
        xgb_params.update(json.load(f))
        xgb_params['scale_pos_weight'] = cw

xgb_model = xgb.XGBClassifier(**xgb_params)

# Model 2: The Continuous Probability Deep Network
print("    -> Initializing Multi-Layer Perceptron (128x64x32)")
mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.01,          # Strong L2 regularization
    batch_size=256,
    learning_rate='adaptive',
    max_iter=500,
    early_stopping=True, # Prevent memorization
    random_state=RANDOM_STATE
)

# Model 3: The Hyperplane Margin Maximizer
print("    -> Initializing Support Vector Machine (RBF Kernel)")
svm_model = SVC(
    kernel='rbf',
    C=1.0,
    class_weight='balanced',
    probability=True,    # Required for Stacking
    random_state=RANDOM_STATE
)

estimators = [
    ('xgb', xgb_model),
    ('mlp', mlp_model),
    ('svm', svm_model)
]

print("\n[4] Training the Stacking Ensemble Meta-Model...")
# Logistic Regression automatically learns which model to trust based on internal CV performance
ensemble_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE),
    cv=5,
    n_jobs=-1
)

ensemble_model.fit(X_train_s, y_train)

print("\n[5] Fitting Isotonic Calibrator & Maximizing F1 Threshold...")
calibrator = CalibratedClassifierCV(ensemble_model, method='isotonic', cv=5)
calibrator.fit(X_train_s, y_train)

y_probs = calibrator.predict_proba(X_test_s)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores[:-1])]

print(f"    -> Optimal Operational Threshold: {best_threshold:.3f}")

print("\n[6] Serializing Ensemble to Disk...")
with open("data/brca2_ensemble_blind.pkl", "wb") as f:
    pickle.dump(ensemble_model, f)
with open("data/calibrator_ensemble_blind.pkl", "wb") as f:
    pickle.dump(calibrator, f)
with open("data/threshold_ensemble_blind.pkl", "wb") as f:
    pickle.dump(best_threshold, f)

print("\n[SUCCESS] MAVE-Blind Stacking Architecture Built and Saved.")
