import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("data/master_training_dataset.csv")
with open("data/universal_feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

X = df[feature_names].values
y = df["Label"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_scores = []
f1_scores_list = []
acc_scores = []
prec_scores = []
rec_scores = []

print("Running strict 5-Fold Cross Validation (no leakage)...")
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
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
    y_pred = (y_probs > 0.5).astype(int) # Default 0.5 threshold for speed
    
    auc_scores.append(roc_auc_score(y_test, y_probs))
    f1_scores_list.append(f1_score(y_test, y_pred))
    acc_scores.append(accuracy_score(y_test, y_pred))
    prec_scores.append(precision_score(y_test, y_pred))
    rec_scores.append(recall_score(y_test, y_pred))

print(f"\nREAL Performance across 5 Folds:")
print(f"ROC-AUC:   {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
print(f"Accuracy:  {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
print(f"F1 Score:  {np.mean(f1_scores_list):.4f} ± {np.std(f1_scores_list):.4f}")
print(f"Precision: {np.mean(prec_scores):.4f} ± {np.std(prec_scores):.4f}")
print(f"Recall:    {np.mean(rec_scores):.4f} ± {np.std(rec_scores):.4f}")

# Also let's check Feature Importances to see if there's feature leakage
clf.fit(X, y)
feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': clf.feature_importances_})
feat_imp = feat_imp.sort_values(by='Importance', ascending=False)
print("\nTop 10 Features (Checking for Feature Leakage):")
print(feat_imp.head(10).to_string(index=False))

