"""
Shared helper to reproduce the exact 60/20/20 train/cal/test split from retrain_v54.ipynb.
Used by acmg_comparison.py, ancestry_bias_evaluation.py, af_recalibration_analysis.py.
"""
import os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RS = 42
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def get_split():
    """Returns (X_tr, X_ca, X_te, y_tr, y_ca, y_te, g_tr, g_ca, g_te, feature_cols, df)"""
    df = pd.read_csv(os.path.join(DATA_DIR, "master_training_dataset.csv"))

    with open(os.path.join(DATA_DIR, "universal_feature_names.pkl"), "rb") as f:
        fn = pickle.load(f)
    fn = [f for f in fn if f in df.columns]

    X = df[fn].values
    y = df["Label"].values
    genes = df["Gene"].values

    st = np.array([f"{g}_{l}" for g, l in zip(genes, y)])
    Xtv, Xte, ytv, yte, gtv, gte, idx_tv, idx_te = train_test_split(
        X, y, genes, np.arange(len(X)),
        test_size=0.2, random_state=RS, stratify=st
    )
    st2 = np.array([f"{g}_{l}" for g, l in zip(gtv, ytv)])
    Xtr, Xca, ytr, yca, gtr, gca, idx_tr, idx_ca = train_test_split(
        Xtv, ytv, gtv, idx_tv,
        test_size=0.25, random_state=RS, stratify=st2
    )

    return {
        "X_tr": Xtr, "X_ca": Xca, "X_te": Xte,
        "y_tr": ytr, "y_ca": yca, "y_te": yte,
        "g_tr": gtr, "g_ca": gca, "g_te": gte,
        "idx_tr": idx_tr, "idx_ca": idx_ca, "idx_te": idx_te,
        "feature_cols": fn, "df": df,
    }
