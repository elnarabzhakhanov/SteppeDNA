import os
import sys
import json
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.feature_engineering import (
    load_phylop_scores, load_mave_scores, load_alphamissense_scores,
    load_structural_features, load_gnomad_frequencies, load_spliceai_scores,
    engineer_features
)

RANDOM_STATE = 42

print("Loading Data for Blind Optuna Tuning...")
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
mutation_train_df = mutation_df[~is_mave].copy()

phylop = load_phylop_scores(data_dir="data")
mave = load_mave_scores(data_dir="data")
am = load_alphamissense_scores(data_dir="data")
struct = load_structural_features(data_dir="data")
gnomad = load_gnomad_frequencies(data_dir="data")
spliceai = load_spliceai_scores(data_dir="data")

X = engineer_features(mutation_train_df, phylop, mave, am, struct, gnomad, spliceai_data=spliceai)

# ----- DROP MAVE FEATURES -----
mave_cols = ['mave_score', 'has_mave', 'mave_abnormal', 'mave_x_blosum']
X = X.drop(columns=mave_cols, errors='ignore')

# ----- DROP NOISE FEATURES -----
noise_cols = [c for c in X.columns if c.startswith('AA_ref_') or c.startswith('AA_alt_') or c.startswith('Mutation_')]
X = X.drop(columns=noise_cols, errors='ignore')

y = mutation_train_df["Label"].values

def objective(trial):
    # Aggressively regularized search space for generalization
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200, step=25),
        'max_depth': trial.suggest_int('max_depth', 2, 5), # Shallower trees prevent memorization
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9), # High row sampling prevents overfitting
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.8), # Very high feature sampling
        'gamma': trial.suggest_float('gamma', 1.0, 10.0), # Forces node splits to be highly profitable
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 15), # Require more records per leaf
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True), # L1 regularization
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True), # L2 regularization
        'eval_metric': 'logloss',
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    auc_scores = []
    
    for train_idx, val_idx in cv.split(X.values, y):
        X_tr, X_val = X.values[train_idx], X.values[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        
        cw = float(np.sum(y_tr == 0)) / np.sum(y_tr == 1)
        params['scale_pos_weight'] = cw
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr_s, y_tr)
        
        preds = model.predict_proba(X_val_s)[:, 1]
        auc = roc_auc_score(y_val, preds)
        auc_scores.append(auc)
        
    return np.mean(auc_scores)

if __name__ == "__main__":
    import logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    print("Running 100 heavily-regularized Optuna trials to boost MAVE holdout Generalization...")
    study.optimize(objective, n_trials=100, show_progress_bar=False)
    
    trial = study.best_trial
    print(f"\n[BLIND OPT_PARAM SEARCH COMPLETE] ROC-AUC: {trial.value:.4f}")
    
    out_path = "data/optuna_blind_params.json"
    with open(out_path, "w") as f:
        json.dump(trial.params, f, indent=4)
