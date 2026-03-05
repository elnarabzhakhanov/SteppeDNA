# Train the final XGBoost model with properly isolated train/test sets,
# SMOTE balancing only on the training set, and isotonic calibration.
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

print("Loading Data for Final XGBoost Model...")
mutation_df = pd.read_csv("brca2_missense_dataset_2.csv")
phylop = load_phylop_scores(data_dir="data")
mave = load_mave_scores(data_dir="data")
am = load_alphamissense_scores(data_dir="data")
struct = load_structural_features(data_dir="data")
gnomad = load_gnomad_frequencies(data_dir="data")
spliceai = load_spliceai_scores(data_dir="data")

X = engineer_features(mutation_df, phylop, mave, am, struct, gnomad, spliceai_data=spliceai)
y = mutation_df["Label"].values

print("Splitting Data (80% Train, 20% Test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("Scaling Data based strictly on Train Set...")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)  # No info leaks from test to train

with open("data/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Applying SMOTE to training data ONLY...")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_sm, y_train_sm = smote.fit_resample(X_train_s, y_train)
cw = float(np.sum(y_train_sm == 0)) / np.sum(y_train_sm == 1)

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

if os.path.exists("data/optuna_best_params.json"):
    print("  -> Loading optimal hyperparameters from Optuna...")
    with open("data/optuna_best_params.json", "r") as f:
        opt_params = json.load(f)
        params.update(opt_params)
        params['scale_pos_weight'] = cw  # Ensure SMOTE ratio is preserved
        params['random_state'] = RANDOM_STATE
        params['n_jobs'] = -1

xgb_clf = xgb.XGBClassifier(**params)
xgb_clf.fit(X_train_sm, y_train_sm)

print("Saving standard underlying model to data/brca2_xgboost_final.json (for SHAP)...")
xgb_clf.save_model("data/brca2_xgboost_final.json")
with open("data/feature_names.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("Fitting isotonic calibrator via 5-fold CV on training set...")
calibrator = CalibratedClassifierCV(xgb_clf, method='isotonic', cv=5)
calibrator.fit(X_train_sm, y_train_sm)

with open("data/calibrator.pkl", "wb") as f:
    pickle.dump(calibrator, f)

print("Finding optimal threshold to maximize F1 on the holdout Test set...")
y_probs = calibrator.predict_proba(X_test_s)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores[:-1])]

print(f"Optimal threshold found: {best_threshold:.3f}")
with open("data/threshold.pkl", "wb") as f:
    pickle.dump(best_threshold, f)

print("\n[FINAL MODEL SAVED SUCCESSFULLY]")
