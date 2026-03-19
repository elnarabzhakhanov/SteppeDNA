"""
Precompute UMAP 2D coordinates from the training dataset.
Produces data/umap_coordinates.json used by the frontend variant landscape.

Usage:
    python scripts/precompute_umap.py

Requires: umap-learn, scikit-learn, pandas, numpy
    pip install umap-learn
"""
import os, sys, json
import numpy as np
import pandas as pd

RANDOM_STATE = 42
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def main():
    print("Loading master dataset...")
    df = pd.read_csv(os.path.join(DATA_DIR, "master_training_dataset.csv"))
    print(f"  {len(df)} variants loaded")

    # Identify feature columns (exclude metadata)
    meta_cols = {"gene_name", "label", "cDNA_pos", "AA_ref", "AA_alt", "Mutation",
                 "AA_pos", "source", "variant_id", "clinical_significance"}
    feat_cols = [c for c in df.columns if c not in meta_cols and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
    print(f"  {len(feat_cols)} feature columns")

    X = df[feat_cols].fillna(0).values
    label_col = "Label" if "Label" in df.columns else "label"
    gene_col = "Gene" if "Gene" in df.columns else "gene_name"
    labels = df[label_col].values  # 1 = pathogenic, 0 = benign
    genes = df[gene_col].values if gene_col in df.columns else ["Unknown"] * len(df)

    # Subsample if too large for browser rendering (max ~5000 points)
    MAX_POINTS = 5000
    if len(X) > MAX_POINTS:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(len(X), MAX_POINTS, replace=False)
        X_sub = X[idx]
        labels_sub = labels[idx]
        genes_sub = genes[idx] if hasattr(genes, '__getitem__') else [genes[i] for i in idx]
    else:
        X_sub = X
        labels_sub = labels
        genes_sub = genes

    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)

    # Run UMAP
    print("Computing UMAP projection...")
    try:
        from umap import UMAP
    except ImportError:
        print("ERROR: umap-learn not installed. Run: pip install umap-learn")
        sys.exit(1)

    reducer = UMAP(n_components=2, n_neighbors=30, min_dist=0.3, metric='euclidean',
                   random_state=RANDOM_STATE, n_jobs=1)
    coords = reducer.fit_transform(X_scaled)
    print(f"  UMAP done: {coords.shape}")

    # Normalize to [0, 1] for easy frontend rendering
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    coords[:, 0] = (coords[:, 0] - x_min) / (x_max - x_min + 1e-8)
    coords[:, 1] = (coords[:, 1] - y_min) / (y_max - y_min + 1e-8)

    # Build JSON output (compact)
    points = []
    for i in range(len(coords)):
        points.append({
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "l": int(labels_sub[i]),  # 0=benign, 1=pathogenic
            "g": str(genes_sub[i])[:6],  # gene abbreviation
        })

    # Also save the scaler params so we can project new variants
    out = {
        "points": points,
        "n_total": len(df),
        "n_displayed": len(points),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist(),
        "feature_cols": feat_cols,
    }

    out_path = os.path.join(DATA_DIR, "umap_coordinates.json")
    with open(out_path, "w") as f:
        json.dump(out, f, separators=(',', ':'))

    file_size = os.path.getsize(out_path)
    print(f"Saved {out_path} ({file_size:,} bytes, {len(points)} points)")
    print("Done! The frontend will load this file for the variant landscape visualization.")

if __name__ == "__main__":
    main()
