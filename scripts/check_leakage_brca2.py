import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
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

print("Loading Data for BRCA2-Only True 5-Fold CV Test...")
mutation_df = pd.read_csv("brca2_missense_dataset_2.csv")

phylop = load_phylop_scores(data_dir="data")
mave = load_mave_scores(data_dir="data")
am = load_alphamissense_scores(data_dir="data")
struct = load_structural_features(data_dir="data")
gnomad = load_gnomad_frequencies(data_dir="data")
spliceai = load_spliceai_scores(data_dir="data")

X = engineer_features(mutation_df, phylop, mave, am, struct, gnomad, spliceai_data=spliceai)
y = mutation_df["Label"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_scores = []
f1_scores_list = []
acc_scores = []

print("Running strict 5-Fold Cross Validation on BRCA2 ONLY (with MAVE features)...")
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.values[train_idx], y[train_idx]
    X_test, y_test = X.values[test_idx], y[test_idx]
    
    # Strictly apply SMOTE to training data ONLY
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    cw = float(np.sum(y_train_sm == 0)) / np.sum(y_train_sm == 1)
    
    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=cw, random_state=42, eval_metric='logloss', n_jobs=-1
    )
    clf.fit(X_train_sm, y_train_sm)
    
    # Predict on unseen test set
    y_probs = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_probs > 0.5).astype(int) 
    
    auc_scores.append(roc_auc_score(y_test, y_probs))
    f1_scores_list.append(f1_score(y_test, y_pred))
    acc_scores.append(accuracy_score(y_test, y_pred))

print(f"\nREAL Performance across 5 Folds for BRCA2 ONLY:")
print(f"ROC-AUC:   {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
print(f"Accuracy:  {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")

