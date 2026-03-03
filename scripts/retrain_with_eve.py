"""
SteppeDNA: Retrain with EVE Score as 104th Feature (Item 44)

Adds EVE (Evolutionary model of Variant Effect) score as a new feature
and retrains XGBoost + MLP to measure per-gene improvement.

IMPORTANT: Does NOT overwrite production models. Saves with _eve suffix
for A/B comparison.

Uses RANDOM_STATE=42 and the same 60/20/20 split.

Output:
    data/universal_xgboost_eve.json     (XGBoost with 104 features)
    data/universal_nn_eve.h5            (MLP with 104 features)
    data/universal_scaler_eve.pkl       (Scaler for 104 features)
    data/universal_calibrator_eve.pkl   (Calibrator for EVE ensemble)
    data/universal_feature_names_eve.pkl
    data/eve_retrain_results.json       (Comparison report)

Run from project root:
    set PYTHONUNBUFFERED=1 && python scripts/retrain_with_eve.py
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, matthews_corrcoef, balanced_accuracy_score

RANDOM_STATE = 42
XGB_WEIGHT = 0.6
NN_WEIGHT = 0.4
GENE_AA_LENGTH = {'BRCA1': 1863, 'BRCA2': 3418, 'PALB2': 1186, 'RAD51C': 376, 'RAD51D': 328}

AA_3LETTER = ['Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly', 'His', 'Ile',
              'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser', 'Ter', 'Thr', 'Trp', 'Tyr', 'Val']


def decode_onehot(row, prefix, aa_list=AA_3LETTER):
    """Decode one-hot encoded amino acid columns back to 3-letter code."""
    for aa in aa_list:
        col = f"{prefix}_{aa}"
        if col in row.index and row[col] == 1:
            return aa
    return None


print("=" * 65, flush=True)
print("  SteppeDNA: Retrain with EVE Score (Item 44)", flush=True)
print("=" * 65, flush=True)

# -----------------------------------------------------------------------
# 1. Load data and EVE scores
# -----------------------------------------------------------------------
print("\n[1/7] Loading training data and EVE scores...", flush=True)

df = pd.read_csv("data/master_training_dataset.csv")
with open("data/universal_feature_names.pkl", "rb") as f:
    feature_names_orig = pickle.load(f)

feature_names_orig = [f for f in feature_names_orig if f in df.columns]
print(f"  Variants: {len(df)}, Original features: {len(feature_names_orig)}", flush=True)

# Load EVE scores
with open("data/eve_scores.pkl", "rb") as f:
    eve_data = pickle.load(f)
by_variant = eve_data.get("by_variant", {})
by_position = eve_data.get("by_position", {})
print(f"  EVE variants: {len(by_variant)}, positions: {len(by_position)}", flush=True)

# -----------------------------------------------------------------------
# 2. Add EVE score as 104th feature
# -----------------------------------------------------------------------
print("\n[2/7] Computing EVE scores for training variants...", flush=True)

# Decode amino acids from one-hot columns
df["_aa_ref_3"] = df.apply(lambda r: decode_onehot(r, "AA_ref"), axis=1)
df["_aa_alt_3"] = df.apply(lambda r: decode_onehot(r, "AA_alt"), axis=1)

# Compute AA position from relative_aa_pos
df["_aa_pos"] = df.apply(
    lambda r: max(1, round(r["relative_aa_pos"] * GENE_AA_LENGTH.get(r["Gene"], 1000))),
    axis=1
).astype(int)

# Look up EVE score for each variant
def lookup_eve(row):
    ref3 = row["_aa_ref_3"]
    alt3 = row["_aa_alt_3"]
    pos = row["_aa_pos"]
    gene = row["Gene"]

    if ref3 is None or alt3 is None:
        return 0.0

    # Try exact variant key first
    vkey = f"{ref3}{pos}{alt3}"
    if vkey in by_variant:
        return by_variant[vkey]

    # Fallback to position-based (max score at that position)
    if pos in by_position:
        return by_position[pos]

    return 0.0

df["eve_score"] = df.apply(lookup_eve, axis=1)

# Coverage stats per gene
print("\n  EVE coverage per gene:", flush=True)
for gene in sorted(df["Gene"].unique()):
    mask = df["Gene"] == gene
    n_total = mask.sum()
    n_nonzero = (df.loc[mask, "eve_score"] > 0).sum()
    mean_score = df.loc[mask, "eve_score"].mean()
    print(f"    {gene:8s}: {n_nonzero:5d}/{n_total:5d} ({n_nonzero/n_total*100:.1f}%) nonzero, mean={mean_score:.4f}", flush=True)

# Build feature matrix with EVE
feature_names_eve = feature_names_orig + ["eve_score"]
X = df[feature_names_eve].values
y = df["Label"].values
genes = df["Gene"].values

print(f"\n  New feature count: {len(feature_names_eve)} (was {len(feature_names_orig)})", flush=True)

# -----------------------------------------------------------------------
# 3. Reproduce the exact same 60/20/20 split
# -----------------------------------------------------------------------
print("\n[3/7] Reproducing 60/20/20 split...", flush=True)

strata = np.array([f"{g}_{l}" for g, l in zip(genes, y)])
X_trainval, X_test, y_trainval, y_test, genes_trainval, genes_test, strata_tv, _ = train_test_split(
    X, y, genes, strata, test_size=0.20, random_state=RANDOM_STATE, stratify=strata
)
strata_tv_inner = np.array([f"{g}_{l}" for g, l in zip(genes_trainval, y_trainval)])
X_train, X_cal, y_train, y_cal, genes_train, genes_cal = train_test_split(
    X_trainval, y_trainval, genes_trainval, test_size=0.25, random_state=RANDOM_STATE, stratify=strata_tv_inner
)

print(f"  Train: {len(X_train)}, Cal: {len(X_cal)}, Test: {len(X_test)}", flush=True)

# -----------------------------------------------------------------------
# 4. Train XGBoost with EVE feature
# -----------------------------------------------------------------------
print("\n[4/7] Training XGBoost with EVE feature...", flush=True)

# Use same hyperparameters as production model
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': RANDOM_STATE,
    'use_label_encoder': False,
}

xgb_clf = xgb.XGBClassifier(**xgb_params)

# Scale features first
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_cal_s = scaler.transform(X_cal)
X_test_s = scaler.transform(X_test)

xgb_clf.fit(
    X_train_s, y_train,
    eval_set=[(X_cal_s, y_cal)],
    verbose=False,
)

xgb_preds_cal = xgb_clf.predict_proba(X_cal_s)[:, 1]
xgb_preds_test = xgb_clf.predict_proba(X_test_s)[:, 1]

xgb_auc_test = roc_auc_score(y_test, xgb_preds_test)
print(f"  XGBoost test AUC: {xgb_auc_test:.4f}", flush=True)

# -----------------------------------------------------------------------
# 5. Train MLP with EVE feature
# -----------------------------------------------------------------------
print("\n[5/7] Training MLP with EVE feature...", flush=True)

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

n_features = X_train_s.shape[1]
nn_model = Sequential([
    Dense(256, activation='relu', input_shape=(n_features,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid'),
])

nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

nn_model.fit(
    X_train_s, y_train,
    validation_data=(X_cal_s, y_cal),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=0,
)

nn_preds_cal = nn_model.predict(X_cal_s, verbose=0).flatten()
nn_preds_test = nn_model.predict(X_test_s, verbose=0).flatten()

nn_auc_test = roc_auc_score(y_test, nn_preds_test)
print(f"  MLP test AUC: {nn_auc_test:.4f}", flush=True)

# -----------------------------------------------------------------------
# 6. Calibrate ensemble
# -----------------------------------------------------------------------
print("\n[6/7] Calibrating ensemble...", flush=True)

blended_cal = (XGB_WEIGHT * xgb_preds_cal) + (NN_WEIGHT * nn_preds_cal)
blended_test = (XGB_WEIGHT * xgb_preds_test) + (NN_WEIGHT * nn_preds_test)

calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(blended_cal, y_cal)

calibrated_test = calibrator.predict(blended_test)
ensemble_auc_test = roc_auc_score(y_test, calibrated_test)
print(f"  Ensemble test AUC (calibrated): {ensemble_auc_test:.4f}", flush=True)

# -----------------------------------------------------------------------
# 7. Compare with production model per gene
# -----------------------------------------------------------------------
print("\n[7/7] Comparing with production model...", flush=True)

# Load production model for comparison
with open("data/universal_scaler_ensemble.pkl", "rb") as f:
    prod_scaler = pickle.load(f)
from tensorflow.keras.models import load_model
prod_nn = load_model("data/universal_nn.h5")
prod_xgb = xgb.XGBClassifier()
prod_xgb.load_model("data/universal_xgboost_final.json")
with open("data/universal_calibrator_ensemble.pkl", "rb") as f:
    prod_calibrator = pickle.load(f)

# Reproduce same split with original features for production comparison
X_orig = df[feature_names_orig].values
_, X_test_orig, _, _, _, _, _, _ = train_test_split(
    X_orig, y, genes, strata, test_size=0.20, random_state=RANDOM_STATE, stratify=strata
)
X_test_orig_s = prod_scaler.transform(X_test_orig)

prod_xgb_preds = prod_xgb.predict_proba(X_test_orig_s)[:, 1]
prod_nn_preds = prod_nn.predict(X_test_orig_s, verbose=0).flatten()
prod_blended = (XGB_WEIGHT * prod_xgb_preds) + (NN_WEIGHT * prod_nn_preds)
prod_calibrated = prod_calibrator.predict(prod_blended)

prod_auc_overall = roc_auc_score(y_test, prod_calibrated)

# Per-gene comparison
print(f"\n  {'Gene':<10s} {'Prod AUC':>10s} {'EVE AUC':>10s} {'Delta':>8s} {'Improved':>10s}", flush=True)
print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*10}", flush=True)

results = {
    "overall": {
        "prod_auc": round(prod_auc_overall, 6),
        "eve_auc": round(ensemble_auc_test, 6),
        "delta": round(ensemble_auc_test - prod_auc_overall, 6),
    },
    "per_gene": {},
}

any_improved = False
for gene in sorted(set(genes)):
    test_mask = genes_test == gene
    if test_mask.sum() < 10:
        continue

    n_pos = int((y_test[test_mask] == 1).sum())
    n_neg = int((y_test[test_mask] == 0).sum())
    if n_pos < 2 or n_neg < 2:
        print(f"  {gene:10s}  (skipped - insufficient class balance)", flush=True)
        continue

    try:
        prod_gene_auc = roc_auc_score(y_test[test_mask], prod_calibrated[test_mask])
    except ValueError:
        prod_gene_auc = 0.5

    try:
        eve_gene_auc = roc_auc_score(y_test[test_mask], calibrated_test[test_mask])
    except ValueError:
        eve_gene_auc = 0.5

    delta = eve_gene_auc - prod_gene_auc
    improved = "YES" if delta > 0.001 else ("no" if delta < -0.001 else "~same")
    if delta > 0.001:
        any_improved = True

    print(f"  {gene:<10s} {prod_gene_auc:>9.4f} {eve_gene_auc:>9.4f} {delta:>+7.4f} {improved:>10s}", flush=True)

    results["per_gene"][gene] = {
        "prod_auc": round(prod_gene_auc, 6),
        "eve_auc": round(eve_gene_auc, 6),
        "delta": round(delta, 6),
        "improved": delta > 0.001,
        "n_test": int(test_mask.sum()),
    }

delta_overall = ensemble_auc_test - prod_auc_overall
print(f"\n  {'OVERALL':<10s} {prod_auc_overall:>9.4f} {ensemble_auc_test:>9.4f} {delta_overall:>+7.4f}", flush=True)

# -----------------------------------------------------------------------
# Save EVE models (never overwrite production)
# -----------------------------------------------------------------------
print("\n  Saving EVE models (separate from production)...", flush=True)

# Save XGBoost
xgb_clf.save_model("data/universal_xgboost_eve.json")

# Save MLP
nn_model.save("data/universal_nn_eve.h5")

# Save scaler
with open("data/universal_scaler_eve.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save calibrator
with open("data/universal_calibrator_eve.pkl", "wb") as f:
    pickle.dump(calibrator, f)

# Save feature names
with open("data/universal_feature_names_eve.pkl", "wb") as f:
    pickle.dump(feature_names_eve, f)

# Save results
results["feature_count"] = len(feature_names_eve)
results["eve_coverage"] = {
    gene: {
        "nonzero": int((df.loc[df["Gene"] == gene, "eve_score"] > 0).sum()),
        "total": int((df["Gene"] == gene).sum()),
    }
    for gene in sorted(set(genes))
}

with open("data/eve_retrain_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Check XGBoost feature importance for EVE
booster = xgb_clf.get_booster()
importance = booster.get_score(importance_type='gain')
eve_importance = importance.get(f'f{len(feature_names_orig)}', 0)  # EVE is the last feature
top_features = sorted(importance.items(), key=lambda x: -x[1])[:10]

print(f"\n  EVE feature importance (gain): {eve_importance:.1f}", flush=True)
print(f"  Top 10 features by XGBoost gain:", flush=True)
for fname, gain in top_features:
    # Convert f-index to name
    idx = int(fname.replace('f', ''))
    if idx < len(feature_names_eve):
        name = feature_names_eve[idx]
    else:
        name = fname
    print(f"    {name:<30s}: {gain:.1f}", flush=True)

print(f"\n{'=' * 65}", flush=True)
print("  EVE INTEGRATION SUMMARY", flush=True)
print("=" * 65, flush=True)
print(f"  Production AUC: {prod_auc_overall:.4f}", flush=True)
print(f"  EVE model AUC:  {ensemble_auc_test:.4f} ({delta_overall:+.4f})", flush=True)
print(f"  Any gene improved: {'YES' if any_improved else 'No'}", flush=True)
print(f"\n  Models saved with _eve suffix (production NOT overwritten):", flush=True)
print(f"    data/universal_xgboost_eve.json", flush=True)
print(f"    data/universal_nn_eve.h5", flush=True)
print(f"    data/universal_scaler_eve.pkl", flush=True)
print(f"    data/universal_calibrator_eve.pkl", flush=True)
print(f"    data/universal_feature_names_eve.pkl", flush=True)
print(f"    data/eve_retrain_results.json", flush=True)
print("\nDone!", flush=True)
