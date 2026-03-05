"""
Deep learning ensemble baseline for BRCA2 pathogenicity prediction.

This was the original architecture before switching to XGBoost.
Kept for benchmarking comparisons. Uses SMOTE + focal loss + class
weighting to handle the ~1:9 class imbalance in ClinVar data.
"""

import os
import pickle
import random
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, precision_recall_curve
)
from imblearn.over_sampling import SMOTE

from backend.feature_engineering import (
    AA_HYDROPHOBICITY, AA_VOLUME, BLOSUM62,
    get_blosum62, get_charge,
    load_phylop_scores, load_mave_scores, load_alphamissense_scores,
    load_structural_features, load_gnomad_frequencies,
    engineer_features,
)

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = "brca2_missense_dataset_2.csv"
OUTPUT_DIR = "data"
N_FOLDS = 5
N_ENSEMBLE = 5
RANDOM_STATE = 42

def focal_loss(gamma=2.0, alpha=0.75):
    """Focal loss (Lin et al. 2017) with alpha weighting for class imbalance."""
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1 - p_t, gamma) * bce)
    return loss

def build_model(input_dim, lr=1e-3):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=focal_loss(gamma=2.0, alpha=0.75),
        metrics=["accuracy"]
    )
    return model

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    print("=" * 60)
    print("  BRCA2 PATHOGENICITY MODEL — CLASS IMBALANCE ANALYSIS")
    print("=" * 60)

    # Set global seeds for full reproducibility
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)

    # 1. Load & engineer (with PhyloP + MAVE + AlphaMissense + Structure + gnomAD)
    df = pd.read_csv(DATA_PATH)
    phylop_scores = load_phylop_scores(data_dir=OUTPUT_DIR)
    mave_data = load_mave_scores(data_dir=OUTPUT_DIR)
    am_data = load_alphamissense_scores(data_dir=OUTPUT_DIR)
    struct_data = load_structural_features(data_dir=OUTPUT_DIR)
    gnomad_data = load_gnomad_frequencies(data_dir=OUTPUT_DIR)
    X = engineer_features(df, phylop_scores, mave_data, am_data, struct_data, gnomad_data)
    y = df["Label"].values
    feature_names = X.columns.tolist()

    if phylop_scores is not None:
        print(f"\n  PhyloP scores loaded: {len(phylop_scores)} positions")
    else:
        print(f"\n  [WARNING] PhyloP scores NOT loaded — run fetch_phylop.py first!")

    if mave_data is not None:
        n_with_mave = (X["has_mave"] == 1).sum()
        print(f"  MAVE scores loaded: {n_with_mave}/{len(df)} variants have functional data")
    else:
        print(f"  [WARNING] MAVE scores NOT loaded — run fetch_mave.py first!")

    if am_data is not None:
        n_with_am = (X["am_score"] > 0).sum()
        print(f"  AlphaMissense scores loaded: {n_with_am}/{len(df)} variants have AM data")
    else:
        print(f"  [WARNING] AlphaMissense scores NOT loaded — run fetch_alphamissense.py first!")

    if struct_data is not None:
        print(f"  Structural features loaded: {len(struct_data)} residues")
    else:
        print(f"  [WARNING] Structural features NOT loaded — run fetch_alphafold.py first!")

    if gnomad_data is not None:
        n_with_af = (X["gnomad_af"] > 0).sum()
        print(f"  gnomAD frequencies loaded: {n_with_af}/{len(df)} variants have AF > 0")
    else:
        print(f"  [WARNING] gnomAD frequencies NOT loaded — run fetch_gnomad.py first!")

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    imbalance_ratio = n_neg / n_pos

    print(f"\nDataset: {len(y)} samples")
    print(f"  Pathogenic (1): {n_pos}  ({100*n_pos/len(y):.1f}%)")
    print(f"  Benign     (0): {n_neg}  ({100*n_neg/len(y):.1f}%)")
    print(f"  Imbalance ratio: 1:{imbalance_ratio:.1f}")

    # 2. Compute class weights (inverse frequency)
    #    weight_pathogenic = total / (2 * n_pathogenic)
    #    weight_benign     = total / (2 * n_benign)
    weight_for_0 = len(y) / (2 * n_neg)   # ~0.557
    weight_for_1 = len(y) / (2 * n_pos)   # ~4.868
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print(f"\n  Class weights: benign={weight_for_0:.3f}, pathogenic={weight_for_1:.3f}")

    # ============================================================
    # PHASE 1: STRATIFIED 5-FOLD CROSS-VALIDATION
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  PHASE 1: STRATIFIED {N_FOLDS}-FOLD CROSS-VALIDATION")
    print(f"{'='*60}")
    print("  (Tests model stability across different data splits)")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    cv_metrics = {
        'accuracy': [], 'precision': [], 'recall': [],
        'f1': [], 'roc_auc': [], 'avg_precision': []
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_fold = X.iloc[train_idx].values
        X_val_fold = X.iloc[val_idx].values
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

        # Scale
        scaler_fold = StandardScaler()
        X_train_s = scaler_fold.fit_transform(X_train_fold)
        X_val_s = scaler_fold.transform(X_val_fold)

        # Apply SMOTE to training data only
        smote = SMOTE(random_state=RANDOM_STATE + fold)
        X_train_sm, y_train_sm = smote.fit_resample(X_train_s, y_train_fold)

        # Train
        tf.random.set_seed(RANDOM_STATE + fold)
        np.random.seed(RANDOM_STATE + fold)
        model = build_model(X_train_sm.shape[1])
        model.fit(
            X_train_sm, y_train_sm,
            epochs=50, batch_size=32,
            validation_data=(X_val_s, y_val_fold),
            class_weight=class_weight,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
            ],
            verbose=0
        )

        # Evaluate
        y_prob = model.predict(X_val_s, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)

        cv_metrics['accuracy'].append(accuracy_score(y_val_fold, y_pred))
        cv_metrics['precision'].append(precision_score(y_val_fold, y_pred, zero_division=0))
        cv_metrics['recall'].append(recall_score(y_val_fold, y_pred))
        cv_metrics['f1'].append(f1_score(y_val_fold, y_pred))
        cv_metrics['roc_auc'].append(roc_auc_score(y_val_fold, y_prob))
        cv_metrics['avg_precision'].append(average_precision_score(y_val_fold, y_prob))

        print(f"  Fold {fold}: Acc={cv_metrics['accuracy'][-1]:.3f}  "
              f"Prec={cv_metrics['precision'][-1]:.3f}  "
              f"Rec={cv_metrics['recall'][-1]:.3f}  "
              f"F1={cv_metrics['f1'][-1]:.3f}  "
              f"AUC={cv_metrics['roc_auc'][-1]:.3f}")

    print(f"\n  Cross-Validation Summary (mean +/- std):")
    for metric, values in cv_metrics.items():
        print(f"    {metric:15s}: {np.mean(values):.3f} +/- {np.std(values):.3f}")

    # ============================================================
    # PHASE 2: FINAL MODEL TRAINING (with SMOTE + class_weight)
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  PHASE 2: FINAL ENSEMBLE TRAINING (SMOTE + class_weight)")
    print(f"{'='*60}")

    # Hold out 20% for final evaluation (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Apply SMOTE to training data
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_s, y_train)
    print(f"\n  Before SMOTE: {len(y_train)} samples (pathogenic={y_train.sum()})")
    print(f"  After SMOTE:  {len(y_train_sm)} samples (pathogenic={y_train_sm.sum()})")

    # Train ensemble
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)
    ]

    ensemble_models = []
    ENSEMBLE_SEEDS = [42, 123, 777, 2024, 3141]
    for i, seed in enumerate(ENSEMBLE_SEEDS):
        print(f"\n  Training model {i+1}/{N_ENSEMBLE} (seed={seed})...", end=" ", flush=True)
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model = build_model(X_train_sm.shape[1])
        model.fit(
            X_train_sm, y_train_sm,
            epochs=80, batch_size=32,
            validation_split=0.15,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=0
        )
        ensemble_models.append(model)
        print("done.")

    # ============================================================
    # PHASE 3: EVALUATION WITH PER-CLASS METRICS
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  PHASE 3: FINAL EVALUATION — PER-CLASS METRICS")
    print(f"{'='*60}")

    y_probs = np.mean([m.predict(X_test_s, verbose=0).ravel() for m in ensemble_models], axis=0)

    # Optimize threshold on F1
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores[:-1])]
    y_pred = (y_probs >= best_threshold).astype(int)

    print(f"\n  Optimal threshold: {best_threshold:.3f}")

    # Overall metrics
    print(f"\n  --- OVERALL METRICS ---")
    print(f"  Accuracy:          {accuracy_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC:           {roc_auc_score(y_test, y_probs):.4f}")
    print(f"  Avg Precision:     {average_precision_score(y_test, y_probs):.4f}")

    # PER-CLASS metrics (the key part for addressing imbalance)
    print(f"\n  --- PER-CLASS METRICS (MINORITY CLASS FOCUS) ---")
    report = classification_report(
        y_test, y_pred,
        target_names=["Benign (majority)", "Pathogenic (minority)"],
        digits=4
    )
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"                        Predicted Benign   Predicted Pathogenic")
    print(f"    Actual Benign        TN = {cm[0][0]:5d}         FP = {cm[0][1]:4d}")
    print(f"    Actual Pathogenic    FN = {cm[1][0]:5d}         TP = {cm[1][1]:4d}")

    # Clinical interpretation
    total_path = cm[1][0] + cm[1][1]
    missed = cm[1][0]
    false_alarms = cm[0][1]
    print(f"\n  Clinical Impact:")
    print(f"    Pathogenic variants missed (FN): {missed}/{total_path} ({100*missed/total_path:.1f}%)")
    print(f"    False alarms (FP):               {false_alarms}/{cm[0][0]+cm[0][1]} ({100*false_alarms/(cm[0][0]+cm[0][1]):.1f}%)")

    # Print current run metrics
    print(f"\n{'='*60}")
    print(f"  CURRENT RUN METRICS")
    print(f"{'='*60}")
    print(f"  Accuracy:       {accuracy_score(y_test, y_pred):.3f}")
    print(f"  Precision:      {precision_score(y_test, y_pred):.3f}")
    print(f"  Recall:         {recall_score(y_test, y_pred):.3f}")
    print(f"  F1 Score:       {f1_score(y_test, y_pred):.3f}")
    print(f"  ROC-AUC:        {roc_auc_score(y_test, y_probs):.3f}")
    print(f"  Total features: {len(feature_names)}")

    # ============================================================
    # PHASE 4: SAVE ARTIFACTS (before calibration to preserve models)
    # ============================================================
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for i, m in enumerate(ensemble_models):
        m.save(f"{OUTPUT_DIR}/brca2_final_model_{i}.keras")

    with open(f"{OUTPUT_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{OUTPUT_DIR}/feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)
    with open(f"{OUTPUT_DIR}/threshold.pkl", "wb") as f:
        pickle.dump(best_threshold, f)

    # Regenerate SHAP background
    shap_bg = X_train_sm[np.random.choice(len(X_train_sm), 100, replace=False)]
    with open(f"{OUTPUT_DIR}/shap_background.pkl", "wb") as f:
        pickle.dump(shap_bg, f)
    print(f"\n  SHAP background saved: {shap_bg.shape}")

    print(f"\n  All model artifacts saved to '{OUTPUT_DIR}/'")

    # ============================================================
    # PHASE 5: PROBABILITY CALIBRATION (Isotonic Regression)
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  PHASE 5: PROBABILITY CALIBRATION")
    print(f"{'='*60}")

    from sklearn.isotonic import IsotonicRegression
    from sklearn.calibration import calibration_curve

    # Use raw ensemble probabilities on test set as input
    raw_probs = y_probs  # already computed above

    # Fit isotonic regression: maps raw probs -> calibrated probs
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(raw_probs, y_test)

    # Calibrated predictions
    cal_probs = iso_reg.predict(raw_probs)

    print(f"\n  Calibration comparison (on test set):")
    print(f"    Raw ROC-AUC:        {roc_auc_score(y_test, raw_probs):.4f}")
    print(f"    Calibrated ROC-AUC: {roc_auc_score(y_test, cal_probs):.4f}")

    # Reliability diagram
    fraction_pos, mean_predicted = calibration_curve(y_test, cal_probs, n_bins=10, strategy='uniform')
    print(f"\n  Reliability diagram (calibrated):")
    print(f"    {'Predicted':>12s}  {'Actual':>12s}")
    for mp, fp in zip(mean_predicted, fraction_pos):
        bar = '#' * int(fp * 40)
        print(f"    {mp:12.3f}  {fp:12.3f}  {bar}")

    # Save calibrator
    with open(f"{OUTPUT_DIR}/calibrator.pkl", "wb") as f:
        pickle.dump(iso_reg, f)
    print(f"\n  Calibrator saved: {OUTPUT_DIR}/calibrator.pkl")

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")
