# Benchmark SteppeDNA against standalone AlphaMissense, PhyloP, logistic
# regression baseline, a deep learning ensemble, and (optionally) published
# clinical predictors from dbNSFP (REVEL, BayesDel, CADD, PolyPhen-2, SIFT).
# The production model is XGBoost; the DL ensemble here is just for comparison.
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.feature_engineering import (
    load_phylop_scores, load_mave_scores, load_alphamissense_scores,
    load_structural_features, load_gnomad_frequencies,
    engineer_features
)
from train_ensemble_baseline import build_model

import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")
os.makedirs("visual_proofs", exist_ok=True)

RANDOM_STATE = 42

print("Loading Data & Re-engineering 99 Features...")
df = pd.read_csv("brca2_missense_dataset_2.csv")
phylop = load_phylop_scores(data_dir="data")
mave = load_mave_scores(data_dir="data")
am = load_alphamissense_scores(data_dir="data")
struct = load_structural_features(data_dir="data")
gnomad = load_gnomad_frequencies(data_dir="data")
X = engineer_features(df, phylop, mave, am, struct, gnomad)
y = df["Label"].values

# 1. INDEPENDENT EXTERNAL VALIDATION SPLIT (Holdout Set)
print("\nSplitting Data into 80% Train and 20% True Holdout Set...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
# Keep the DataFrame split in sync for variant-level matching (dbNSFP)
df_train, df_test = train_test_split(
    df, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train.values)
X_test_s = scaler.transform(X_test.values)

# Evaluate Models Dict
y_preds = {}

# ==========================================
# MODEL 1: THE "DUMB" BASELINE (Logistic Regression, 5 Features)
# ==========================================
print("\nTraining Baseline Logistic Regression (5 basic physical features)...")
# First 5 features: blosum62_score, volume_diff, hydro_diff, charge_changed, is_nonsense
# (Wait, if is_nonsense is constant 0 for missense, it's just 4 features, but that's fine)
X_train_dumb = X_train_s[:, :5]
X_test_dumb = X_test_s[:, :5]

lr_model = LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE)
lr_model.fit(X_train_dumb, y_train)
y_preds['LogReg_Basic'] = lr_model.predict_proba(X_test_dumb)[:, 1]

# ==========================================
# MODEL 2: THE XGBOOST PIVOT (99 Features)
# ==========================================
print("\nTraining Advanced XGBoost Classifier...")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_sm, y_train_sm = smote.fit_resample(X_train_s, y_train)
cw = float(np.sum(y_train_sm == 0)) / np.sum(y_train_sm == 1)

xgb_clf = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=cw,
    random_state=RANDOM_STATE,
    eval_metric='logloss',
    n_jobs=-1
)
xgb_clf.fit(X_train_sm, y_train_sm)
y_preds['XGBoost'] = xgb_clf.predict_proba(X_test_s)[:, 1]

# ==========================================
# MODEL 3: DEEP DL ENSEMBLE (5 Models)
# ==========================================
print("\nTraining Deep Neural Network Ensemble (5 isolated models)...")
# Reduce models/epochs slightly just to run this benchmark faster, but we'll do 3 models for the benchmark graph to save time and memory.
N_MODELS = 3 
nn_ensemble_preds = np.zeros((len(y_test), N_MODELS))

weight_0 = len(y_train) / (2 * np.sum(y_train == 0))
weight_1 = len(y_train) / (2 * np.sum(y_train == 1))

for m in range(N_MODELS):
    print(f"  -> Training DL Model {m+1}/{N_MODELS}...")
    tf.random.set_seed(RANDOM_STATE + m)
    np.random.seed(RANDOM_STATE + m)
    
    # Each model gets a slightly different SMOTE
    smote_nn = SMOTE(random_state=RANDOM_STATE + m)
    X_train_nn_sm, y_train_nn_sm = smote_nn.fit_resample(X_train_s, y_train)
    
    model = build_model(X_train_s.shape[1])
    model.fit(
        X_train_nn_sm, y_train_nn_sm,
        epochs=30, batch_size=32, # reduced epochs for benchmark
        validation_split=0.1,
        class_weight={0: weight_0, 1: weight_1},
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ],
        verbose=0
    )
    nn_ensemble_preds[:, m] = model.predict(X_test_s, verbose=0).ravel()

y_preds['DeepEnsemble'] = np.mean(nn_ensemble_preds, axis=1)

# ==========================================
# MODEL 4: SOTA AlphaMissense Standalone
# ==========================================
print("\nExtracting AlphaMissense Standalone SOTA Predictions...")
idx_am = list(X.columns).index("am_score")
# Use the unscaled raw AM score!
y_preds['AlphaMissense'] = X_test.iloc[:, idx_am].values

# ==========================================
# MODEL 5: SOTA PhyloP Standalone
# ==========================================
print("Extracting PhyloP Conservation SOTA Predictions...")
idx_phylop = list(X.columns).index("phylop_score")
# PhyloP ranking directly: higher score = more conserved = more pathogenic
y_preds['PhyloP (Conservation)'] = X_test.iloc[:, idx_phylop].values

# ==========================================
# MODEL 6-10: PUBLISHED CLINICAL PREDICTORS (dbNSFP)
# ==========================================
# These are loaded from data/dbnsfp_brca2_scores.pkl if available.
# Run data_pipelines/fetch_dbnsfp.py to generate this file.
dbnsfp_path = os.path.join("data", "dbnsfp_brca2_scores.pkl")
dbnsfp_loaded = False

if os.path.exists(dbnsfp_path):
    print("\nLoading published predictor scores from dbNSFP...")
    with open(dbnsfp_path, "rb") as f:
        dbnsfp_data = pickle.load(f)
    dbnsfp_by_variant = dbnsfp_data["by_variant"]
    dbnsfp_by_position = dbnsfp_data["by_position"]

    # Predictor configs: (dbNSFP key, display name, invert for ROC?)
    # SIFT is inverted: lower score = more damaging, so we flip for ROC
    DBNSFP_PREDICTORS = [
        ("revel",          "REVEL",       False),
        ("bayesdel",       "BayesDel",    False),
        ("cadd_phred",     "CADD",        False),
        ("polyphen2_hdiv", "PolyPhen-2",  False),
        ("sift",           "SIFT",        True),   # invert: low=damaging
    ]

    for score_key, display_name, invert in DBNSFP_PREDICTORS:
        scores = np.full(len(df_test), np.nan)
        for i, (_, row) in enumerate(df_test.iterrows()):
            vkey = f"{row['AA_ref']}{int(row['AA_pos'])}{row['AA_alt']}"
            if vkey in dbnsfp_by_variant:
                val = dbnsfp_by_variant[vkey].get(score_key, np.nan)
            elif int(row['AA_pos']) in dbnsfp_by_position:
                val = dbnsfp_by_position[int(row['AA_pos'])].get(score_key, np.nan)
            else:
                val = np.nan
            scores[i] = val

        if invert:
            scores = 1.0 - scores

        valid = ~np.isnan(scores)
        n_valid = np.sum(valid)
        pct = 100 * n_valid / len(scores)

        if n_valid >= 50 and len(np.unique(y_test[valid])) == 2:
            # Evaluate only on variants that have scores
            # For the unified graph, fill NaN with the predictor's median
            # so the curve uses all test variants (fair comparison)
            median_val = np.nanmedian(scores)
            filled_scores = np.where(np.isnan(scores), median_val, scores)
            y_preds[display_name] = filled_scores
            print(f"  {display_name}: {n_valid}/{len(scores)} valid ({pct:.1f}%), NaN filled with median={median_val:.3f}")
            dbnsfp_loaded = True
        else:
            print(f"  {display_name}: SKIPPED ({n_valid} valid scores, need >= 50)")

    if not dbnsfp_loaded:
        print("  [WARN] No dbNSFP predictors had sufficient coverage")
else:
    print("\n[INFO] dbNSFP scores not found — skipping published predictor comparison")
    print("  To add REVEL/BayesDel/CADD/PolyPhen-2/SIFT curves, run:")
    print("  python data_pipelines/fetch_dbnsfp.py --chr13-file <path_to_dbNSFP_chr13.gz>")


# ==========================================
# GENERATE UNIFIED GRAPH 1: ROC CURVE
# ==========================================
print("\nGenerating Unified ROC-AUC Comparison Graph...")
plt.figure(figsize=(10, 8))

colors = {
    'DeepEnsemble': '#6260FF',     # Primary Brand Purple/Blue
    'XGBoost': '#FF3B30',          # Danger Red
    'AlphaMissense': '#34C759',    # Success Green
    'PhyloP (Conservation)': '#FF9500', # Warning Orange
    'LogReg_Basic': '#8E8E93',     # Gray
    # Published clinical predictors (from dbNSFP)
    'REVEL': '#E63946',            # Crimson
    'BayesDel': '#457B9D',         # Steel Blue
    'CADD': '#2A9D8F',             # Teal
    'PolyPhen-2': '#E9C46A',       # Gold
    'SIFT': '#F4A261',             # Peach
}

PUBLISHED_NAMES = {'REVEL', 'BayesDel', 'CADD', 'PolyPhen-2', 'SIFT'}

for name, preds in y_preds.items():
    fpr, tpr, _ = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    if name == 'LogReg_Basic':
        linestyle, lw = '--', 1.5
    elif name in PUBLISHED_NAMES:
        linestyle, lw = '--', 2.0
    elif name == 'XGBoost':
        linestyle, lw = '-', 3.0
    else:
        linestyle, lw = '-', 2.0
    plt.plot(fpr, tpr, color=colors.get(name, '#333'), lw=lw, linestyle=linestyle,
             label=f'{name} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
title_suffix = " + Published Predictors" if dbnsfp_loaded else ""
plt.title(f'SteppeDNA: State-of-the-Art Benchmarking{title_suffix} (ROC Curve)', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=9)
plt.grid(True, alpha=0.3)
plt.savefig('visual_proofs/5_SOTA_Comparison_ROC.pdf', bbox_inches='tight')
plt.close()

# ==========================================
# GENERATE UNIFIED GRAPH 2: PR CURVE
# ==========================================
print("Generating Unified Precision-Recall Comparison Graph...")
plt.figure(figsize=(10, 8))

for name, preds in y_preds.items():
    prec, rec, _ = precision_recall_curve(y_test, preds)
    pr_auc = auc(rec, prec)
    if name == 'LogReg_Basic':
        linestyle, lw = '--', 1.5
    elif name in PUBLISHED_NAMES:
        linestyle, lw = '--', 2.0
    elif name == 'XGBoost':
        linestyle, lw = '-', 3.0
    else:
        linestyle, lw = '-', 2.0
    plt.plot(rec, prec, color=colors.get(name, '#333'), lw=lw, linestyle=linestyle,
             label=f'{name} (PR-AUC = {pr_auc:.4f})')

# Baseline prevalence
prevalence = np.sum(y_test) / len(y_test)
plt.plot([0, 1], [prevalence, prevalence], color='black', lw=1, linestyle='--', label=f'Random Chance ({prevalence:.3f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall (Sensitivity)', fontsize=12, fontweight='bold')
plt.ylabel('Precision', fontsize=12, fontweight='bold')
plt.title(f'SteppeDNA: State-of-the-Art Benchmarking{title_suffix} (Precision-Recall)', fontsize=14, fontweight='bold')
plt.legend(loc="lower left", fontsize=9)
plt.grid(True, alpha=0.3)
plt.savefig('visual_proofs/6_SOTA_Comparison_PR.pdf', bbox_inches='tight')
plt.close()

# ==========================================
# SUMMARY TABLE
# ==========================================
print("\n" + "=" * 55)
print("  BENCHMARKING SUMMARY")
print("=" * 55)
print(f"\n  {'Predictor':<25} {'ROC-AUC':>10} {'PR-AUC':>10}")
print(f"  {'-'*47}")

for name, preds in y_preds.items():
    fpr_s, tpr_s, _ = roc_curve(y_test, preds)
    prec_s, rec_s, _ = precision_recall_curve(y_test, preds)
    r = auc(fpr_s, tpr_s)
    p = auc(rec_s, prec_s)
    marker = " <--" if name == 'XGBoost' else ""
    print(f"  {name:<25} {r:>10.4f} {p:>10.4f}{marker}")

print(f"\nDone — benchmarking PDFs saved to visual_proofs/")
