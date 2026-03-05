import os
import sys
import json
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.feature_engineering import (
    load_phylop_scores, load_mave_scores, load_alphamissense_scores,
    load_structural_features, load_gnomad_frequencies, load_spliceai_scores,
    load_esm2_embeddings, engineer_features
)

RANDOM_STATE = 42

print("Loading Data for Optuna Tuning...")
mutation_df = pd.read_csv("brca2_missense_dataset_2.csv")
phylop = load_phylop_scores(data_dir="data")
mave = load_mave_scores(data_dir="data")
am = load_alphamissense_scores(data_dir="data")
struct = load_structural_features(data_dir="data")
gnomad = load_gnomad_frequencies(data_dir="data")
spliceai = load_spliceai_scores(data_dir="data")
esm2 = load_esm2_embeddings(data_dir="data")

print("[2] Engineering Features...")
X = engineer_features(mutation_df, phylop, mave, am, struct, gnomad, spliceai_data=spliceai, esm2_data=esm2)

# Drop MAVE features to ensure hyperparameters generalize to unseen data (Blind Strategy)
mave_cols = ['mave_score', 'has_mave', 'mave_abnormal', 'mave_x_blosum']
print(f"Dropping MAVE features for pure generalization tuning: {mave_cols}")
X = X.drop(columns=mave_cols, errors='ignore')

y = mutation_df["Label"].values

def objective(trial):
    # Search space for XGBoost
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'eval_metric': 'logloss',
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    auc_scores = []
    
    for train_idx, val_idx in cv.split(X.values, y):
        X_tr, X_val = X.values[train_idx], X.values[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # Scale
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        
        # Apply SMOTE only to the training fold
        smote = SMOTE(random_state=RANDOM_STATE)
        X_tr_sm, y_tr_sm = smote.fit_resample(X_tr_s, y_tr)
        
        # Class weighting for XGBoost
        cw = float(np.sum(y_tr_sm == 0)) / np.sum(y_tr_sm == 1)
        params['scale_pos_weight'] = cw
        
        # Train
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr_sm, y_tr_sm)
        
        # Predict on validation fold
        preds = model.predict_proba(X_val_s)[:, 1]
        auc = roc_auc_score(y_val, preds)
        auc_scores.append(auc)
        
    return np.mean(auc_scores)

if __name__ == "__main__":
    print("\n============================================================")
    print(" SteppeDNA: Optuna Hyperparameter Optimization")
    print("============================================================\n")
    
    os.makedirs('data', exist_ok=True)
    # Using TPE sampler which is the default
    study = optuna.create_study(direction="maximize")
    print("Starting 50 Optimization Trials (using 5-Fold CV)...")
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    
    print("\n[OPTIMIZATION COMPLETE]")
    trial = study.best_trial
    print(f"  Best cross-validation ROC-AUC: {trial.value:.4f}")
    print("  Best Hyperparameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    out_path = "data/optuna_best_params.json"
    with open(out_path, "w") as f:
        json.dump(trial.params, f, indent=4)
        
    print(f"\nSaved optimal parameters -> {out_path}")
