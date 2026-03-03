# Saves a small background sample for SHAP explainability.
# Picks 100 random samples, scales them, saves as pickle.
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backend.feature_engineering import (
    engineer_features, load_phylop_scores, load_mave_scores,
    load_alphamissense_scores, load_structural_features, load_gnomad_frequencies
)

DATA_PATH = "brca2_missense_dataset_2.csv"
DATA_DIR = "data"

# Load and process
df = pd.read_csv(DATA_PATH)
phylop = load_phylop_scores(data_dir=DATA_DIR)
mave = load_mave_scores(data_dir=DATA_DIR)
am = load_alphamissense_scores(data_dir=DATA_DIR)
struct = load_structural_features(data_dir=DATA_DIR)
gnomad = load_gnomad_frequencies(data_dir=DATA_DIR)
X = engineer_features(df, phylop, mave, am, struct, gnomad)
y = df["Label"].values

# Load scaler
with open(f"{DATA_DIR}/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

X_scaled = scaler.transform(X)

# Pick 100 random samples as SHAP background
np.random.seed(42)
indices = np.random.choice(len(X_scaled), 100, replace=False)
background = X_scaled[indices]

with open(f"{DATA_DIR}/shap_background.pkl", "wb") as f:
    pickle.dump(background, f)

print(f"Saved SHAP background: {background.shape} samples to {DATA_DIR}/shap_background.pkl")
