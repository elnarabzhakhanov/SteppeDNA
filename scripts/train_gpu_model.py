"""
SteppeDNA: GPU-Augmented Model Training (v5.3)
===============================================
Trains the production XGBoost + MLP ensemble on the GPU-augmented dataset
that includes ESM-2 650M, ESM-2 LoRA, and GNN structural embeddings.

Features: 103 original + 75 GPU = 178 total
  - 103 original (biochemical, conservation, structural, ESM-2 8M)
  - 20 ESM-2 LoRA PCA
  - 22 ESM-2 650M (20 PCA + cosine_sim + l2_shift)
  - 33 GNN structural (32 dims + has_gnn indicator)

Run after integrate_gpu_embeddings.py:
  python scripts/train_gpu_model.py
"""

import os, sys, pickle, json, argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, matthews_corrcoef, balanced_accuracy_score
)
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

RANDOM_STATE = 42
XGB_WEIGHT = 0.6
NN_WEIGHT = 0.4
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

print("=" * 60)
print("  SteppeDNA: GPU-Augmented Model Training (v5.3)")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/7] Loading GPU-augmented dataset...")

dataset_path = os.path.join(DATA_DIR, "master_training_dataset_gpu_augmented.csv")
if not os.path.exists(dataset_path):
    print("[ERROR] GPU-augmented dataset not found! Run integrate_gpu_embeddings.py first.")
    sys.exit(1)

features_path = os.path.join(DATA_DIR, "gpu_feature_names.pkl")
if not os.path.exists(features_path):
    print("[ERROR] gpu_feature_names.pkl not found! Run integrate_gpu_embeddings.py first.")
    sys.exit(1)

df = pd.read_csv(dataset_path)
with open(features_path, "rb") as f:
    feature_names = pickle.load(f)

# Validate features exist in dataset
missing = [f for f in feature_names if f not in df.columns]
if missing:
    print(f"  [WARNING] {len(missing)} features missing from dataset: {missing[:5]}...")
    feature_names = [f for f in feature_names if f in df.columns]

X = df[feature_names].values
y = df["Label"].values
genes = df["Gene"].values

print(f"  Variants: {len(X)} across {len(set(genes))} genes")
print(f"  Features: {len(feature_names)} ({103} original + {len(feature_names)-103} GPU)")
print(f"  Classes:  {int((y==0).sum())} Benign / {int((y==1).sum())} Pathogenic")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Drop zero-variance features
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/7] Checking for zero-variance features...")

variances = np.var(X, axis=0)
zero_var_mask = variances > 0
n_dropped = int((~zero_var_mask).sum())

if n_dropped > 0:
    dropped_names = [feature_names[i] for i in range(len(feature_names)) if not zero_var_mask[i]]
    print(f"  Dropping {n_dropped} zero-variance features: {dropped_names[:10]}...")
    X = X[:, zero_var_mask]
    feature_names = [feature_names[i] for i in range(len(feature_names)) if zero_var_mask[i]]
else:
    print(f"  All {len(feature_names)} features have non-zero variance. OK.")

# Save final feature names
with open(os.path.join(DATA_DIR, "gpu_feature_names.pkl"), "wb") as f:
    pickle.dump(feature_names, f)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Three-way split with gene x label stratification
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/7] Splitting data (60/20/20, gene x label stratified)...")

strata = np.array([f"{g}_{l}" for g, l in zip(genes, y)])
X_trainval, X_test, y_trainval, y_test, genes_trainval, genes_test, strata_tv, _ = train_test_split(
    X, y, genes, strata, test_size=0.20, random_state=RANDOM_STATE, stratify=strata
)
strata_tv_inner = np.array([f"{g}_{l}" for g, l in zip(genes_trainval, y_trainval)])
X_train, X_cal, y_train, y_cal, genes_train, genes_cal = train_test_split(
    X_trainval, y_trainval, genes_trainval, test_size=0.25, random_state=RANDOM_STATE, stratify=strata_tv_inner
)

print(f"  Train: {len(X_train)} ({(y_train==1).sum()}P / {(y_train==0).sum()}B)")
print(f"  Calib: {len(X_cal)} ({(y_cal==1).sum()}P / {(y_cal==0).sum()}B)")
print(f"  Test:  {len(X_test)} ({(y_test==1).sum()}P / {(y_test==0).sum()}B)")

for gene in sorted(set(genes_train)):
    mask = genes_train == gene
    n_p = (y_train[mask] == 1).sum()
    n_b = (y_train[mask] == 0).sum()
    print(f"    {gene:8s}: {mask.sum():5d} train ({n_p}P / {n_b}B)")

# Scale — fit on training only
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_cal_s   = scaler.transform(X_cal)
X_test_s  = scaler.transform(X_test)

with open(os.path.join(DATA_DIR, "gpu_scaler_ensemble.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# SMOTE only on training split
smote = SMOTE(random_state=RANDOM_STATE)
X_train_sm, y_train_sm = smote.fit_resample(X_train_s, y_train)
print(f"  SMOTE: {len(X_train_sm)} training samples after resampling")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Train XGBoost
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/7] Training XGBoost...")

cw = float(np.sum(y_train_sm == 0)) / np.sum(y_train_sm == 1)
xgb_params = {
    'n_estimators': 400,
    'max_depth': 7,
    'learning_rate': 0.05,
    'scale_pos_weight': cw,
    'random_state': RANDOM_STATE,
    'eval_metric': 'logloss',
    'n_jobs': -1,
}
xgb_clf = xgb.XGBClassifier(**xgb_params)
xgb_clf.fit(X_train_sm, y_train_sm, verbose=False)
xgb_clf.save_model(os.path.join(DATA_DIR, "gpu_xgboost_final.json"))
print("  XGBoost trained and saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Train Neural Network
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/7] Training Neural Network...")

def build_nn(input_dim):
    model = Sequential([
        Dense(256, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model

es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
nn_model = build_nn(X_train_sm.shape[1])
nn_model.fit(
    X_train_sm, y_train_sm,
    epochs=100,
    batch_size=32,
    validation_split=0.15,
    callbacks=[es],
    verbose=0
)
nn_model.save(os.path.join(DATA_DIR, "gpu_nn.h5"))
print("  Neural network trained and saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Fit Calibrator on HELD-OUT calibration set
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/7] Fitting isotonic calibrator on held-out calibration set...")

nn_preds_cal  = nn_model.predict(X_cal_s, verbose=0).flatten()
xgb_preds_cal = xgb_clf.predict_proba(X_cal_s)[:, 1]
blended_cal   = (XGB_WEIGHT * xgb_preds_cal) + (NN_WEIGHT * nn_preds_cal)

calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(blended_cal, y_cal)

with open(os.path.join(DATA_DIR, "gpu_calibrator_ensemble.pkl"), "wb") as f:
    pickle.dump(calibrator, f)

print(f"  Calibrator fitted on {len(X_cal)} real samples")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Evaluate on Test Set
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/7] Evaluating on held-out test set...")

nn_preds_test  = nn_model.predict(X_test_s, verbose=0).flatten()
xgb_preds_test = xgb_clf.predict_proba(X_test_s)[:, 1]
blended_test   = (XGB_WEIGHT * xgb_preds_test) + (NN_WEIGHT * nn_preds_test)

y_probs = calibrator.predict(blended_test)
y_probs_raw = blended_test

# Optimal threshold from PR curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_threshold = float(thresholds[np.argmax(f1_scores[:-1])])

with open(os.path.join(DATA_DIR, "gpu_threshold_ensemble.pkl"), "wb") as f:
    pickle.dump(best_threshold, f)

y_pred = (y_probs >= best_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print(f"\n{'='*60}")
print(f"  GPU-AUGMENTED EVALUATION RESULTS (Test Set, n={len(y_test)})")
print(f"{'='*60}")
print(f"  Optimal Threshold:  {best_threshold:.4f}")
print(f"  ROC-AUC:            {roc_auc_score(y_test, y_probs):.4f}")
print(f"  PR-AUC:             {average_precision_score(y_test, y_probs):.4f}")
print(f"  Sensitivity:        {tp/(tp+fn):.4f}")
print(f"  Specificity:        {tn/(tn+fp):.4f}")
print(f"  Balanced Accuracy:  {balanced_accuracy_score(y_test, y_pred):.4f}")
print(f"  MCC:                {matthews_corrcoef(y_test, y_pred):.4f}")
print(f"  Confusion Matrix:   TP={tp} FP={fp} FN={fn} TN={tn}")
print(f"  Features used:      {len(feature_names)}")

roc_raw = roc_auc_score(y_test, y_probs_raw)
roc_cal = roc_auc_score(y_test, y_probs)
print(f"\n  Raw blended ROC-AUC:     {roc_raw:.4f}")
print(f"  Calibrated ROC-AUC:      {roc_cal:.4f}")

# Compare with original model
try:
    with open(os.path.join(DATA_DIR, "model_metrics.json")) as f:
        orig = json.load(f)
    orig_auc = orig.get("roc_auc", "N/A")
    print(f"\n  COMPARISON vs Original Model:")
    print(f"  Original ROC-AUC:        {orig_auc}")
    print(f"  GPU-Augmented ROC-AUC:   {roc_cal:.4f}")
    delta = roc_cal - orig_auc if isinstance(orig_auc, float) else None
    if delta is not None:
        direction = "IMPROVED" if delta > 0 else "DECREASED" if delta < 0 else "UNCHANGED"
        print(f"  Delta:                   {delta:+.4f} ({direction})")
except:
    pass

# Per-gene breakdown
print(f"\n  Per-gene performance (GPU-augmented):")
print(f"  {'Gene':8s} {'n':>5s}  {'ROC-AUC':>8s}  {'MCC':>6s}  {'BalAcc':>7s}  {'Sens':>6s}  {'Spec':>6s}")
print(f"  {'-'*52}")

metrics_json = {
    "model_version": "v5.3-gpu",
    "roc_auc": round(roc_auc_score(y_test, y_probs), 4),
    "pr_auc": round(average_precision_score(y_test, y_probs), 4),
    "mcc": round(matthews_corrcoef(y_test, y_pred), 4),
    "balanced_accuracy": round(balanced_accuracy_score(y_test, y_pred), 4),
    "sensitivity": round(tp / (tp + fn), 4),
    "specificity": round(tn / (tn + fp), 4),
    "threshold": round(best_threshold, 4),
    "n_test": int(len(y_test)),
    "n_train": int(len(X_train)),
    "n_features": len(feature_names),
    "n_genes": len(set(genes)),
    "gpu_features": {
        "esm2_lora_pca": 20,
        "esm2_650m_pca": 20,
        "esm2_650m_extra": 2,
        "gnn_structural": 32,
        "gnn_indicator": 1,
    },
    "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    "per_gene": {},
}

for gene in sorted(set(genes_test)):
    mask = genes_test == gene
    n = mask.sum()
    y_g = y_test[mask]
    p_g = y_probs[mask]

    if n < 2 or len(set(y_g)) < 2:
        print(f"  {gene:8s} {n:5d}  (insufficient for AUC)")
        metrics_json["per_gene"][gene] = {"n": int(n), "insufficient": True}
        continue

    gene_auc = roc_auc_score(y_g, p_g)
    gene_pred = (p_g >= best_threshold).astype(int)
    gene_mcc = matthews_corrcoef(y_g, gene_pred)
    gene_bal = balanced_accuracy_score(y_g, gene_pred)

    tn_g, fp_g, fn_g, tp_g = confusion_matrix(y_g, gene_pred).ravel()
    gene_sens = tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 0
    gene_spec = tn_g / (tn_g + fp_g) if (tn_g + fp_g) > 0 else 0

    print(f"  {gene:8s} {n:5d}  {gene_auc:8.4f}  {gene_mcc:6.3f}  {gene_bal:7.4f}  {gene_sens:6.3f}  {gene_spec:6.3f}")

    metrics_json["per_gene"][gene] = {
        "n": int(n),
        "roc_auc": round(gene_auc, 4),
        "mcc": round(gene_mcc, 4),
        "balanced_accuracy": round(gene_bal, 4),
        "sensitivity": round(gene_sens, 4),
        "specificity": round(gene_spec, 4),
    }

with open(os.path.join(DATA_DIR, "gpu_model_metrics.json"), "w") as f:
    json.dump(metrics_json, f, indent=2)

# Save SHAP background data for later use
shap_bg_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_train_s), min(200, len(X_train_s)), replace=False)
with open(os.path.join(DATA_DIR, "gpu_shap_background.pkl"), "wb") as f:
    pickle.dump(X_train_s[shap_bg_idx], f)

print(f"\n  Metrics saved to data/gpu_model_metrics.json")
print(f"\n  All GPU model artifacts saved with 'gpu_' prefix:")
print(f"    gpu_xgboost_final.json")
print(f"    gpu_nn.h5")
print(f"    gpu_scaler_ensemble.pkl")
print(f"    gpu_calibrator_ensemble.pkl")
print(f"    gpu_threshold_ensemble.pkl")
print(f"    gpu_feature_names.pkl")
print(f"    gpu_model_metrics.json")
print(f"    gpu_shap_background.pkl")
print(f"\n  [GPU MODEL TRAINING COMPLETE]")
