# Ablation study: remove AlphaMissense features and measure the AUC drop.
# A drop >5% would suggest over-reliance on AM (potential leakage concern).
import os
import random
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import from our training script
from train_ensemble_baseline import (
    load_phylop_scores, load_mave_scores, load_alphamissense_scores,
    engineer_features, build_model, focal_loss
)

RANDOM_STATE = 42
DATA_PATH = "brca2_missense_dataset_2.csv"
OUTPUT_DIR = "ablation_results"

def run_cv(X, y, name="Baseline"):
    print(f"\nRunning 5-Fold CV for: {name}")
    print(f"Features: {X.shape[1]}")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        # Split
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        
        # SMOTE (only on training data)
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = smote.fit_resample(X_train_s, y_train)
        
        # Train
        model = build_model(X_train_res.shape[1])
        # Silent training
        model.fit(
            X_train_res, y_train_res,
            validation_data=(X_val_s, y_val),
            epochs=50, batch_size=32, verbose=0,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5)
            ]
        )
        
        # Evaluate
        y_pred_prob = model.predict(X_val_s, verbose=0).ravel()
        
        auc = roc_auc_score(y_val, y_pred_prob)
        fold_metrics.append(auc)
        print(f"  Fold {fold}: AUC={auc:.4f}")
        
    mean_auc = np.mean(fold_metrics)
    std_auc = np.std(fold_metrics)
    print(f"  {name} Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
    return mean_auc, std_auc

if __name__ == "__main__":
    # Fix seeds
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    
    # Load Data
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    phylop = load_phylop_scores()
    mave = load_mave_scores()
    am = load_alphamissense_scores()
    
    # 1. Full Features (Baseline)
    print("\n--- Preparing Baseline Feature Set ---")
    X_full = engineer_features(df, phylop, mave, am)
    y = df["Label"].values
    
    # 2. Ablated Features (No AlphaMissense)
    print("\n--- Preparing Ablated Feature Set (No AM) ---")
    # Identify AM columns
    am_cols = ["am_score", "am_pathogenic", "am_x_phylop"]
    # Verify columns exist
    present_cols = [c for c in am_cols if c in X_full.columns]
    print(f"Removing columns: {present_cols}")
    X_ablated = X_full.drop(columns=present_cols)
    
    # Run Comparison
    auc_full, std_full = run_cv(X_full, y, "Full Model (with AM)")
    auc_abl, std_abl = run_cv(X_ablated, y, "Ablated Model (NO AM)")
    
    # Report
    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS")
    print("="*60)
    print(f"Full Model AUC:    {auc_full:.4f}")
    print(f"No-AM Model AUC:   {auc_abl:.4f}")
    print(f"Performance Drop:  {(auc_full - auc_abl):.4f}")
    
    delta = auc_full - auc_abl
    if delta > 0.05:
        print("\n[CRITICAL WARNING] Drop > 0.05 indicate massive reliance on AlphaMissense.")
        print("Data leakage is highly likely. The model fails without AM.")
    elif delta < 0.01:
        print("\n[SUCCESS] Minimal drop. Model is robust without AlphaMissense.")
        print("Leakage risk is managed; other features (MAVE, PhyloP) are driving predictions.")
    else:
        print("\n[INFO] Moderate drop. AlphaMissense is useful but not the sole driver.")
    
    # Save results
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    with open(f"{OUTPUT_DIR}/ablation_summary.txt", "w") as f:
        f.write(f"Full: {auc_full:.4f}\nNo-AM: {auc_abl:.4f}\nDelta: {delta:.4f}")
