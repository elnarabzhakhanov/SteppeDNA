"""
SteppeDNA — Integrate GPU-generated embeddings into training dataset.

The GPU notebooks (Colab) produced embeddings indexed by row position
in the master training dataset. This script merges them by row index.

Usage: python scripts/integrate_gpu_embeddings.py

Inputs (from data/):
  - master_training_dataset.csv
  - esm2_lora_embeddings.pkl   (19,223 entries, 20D PCA, keyed by row idx)
  - esm2_650m_embeddings.pkl   (19,223 entries, 20D PCA + cosine_sim + l2_shift)
  - gnn_structural_embeddings.pkl (9,138 entries, 32D, BRCA1/PALB2/RAD51C/RAD51D only)

Output:
  - master_training_dataset_gpu_augmented.csv
  - gpu_feature_names.pkl
"""
import os, sys, pickle
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def load_pkl(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        print(f"  [SKIP] {name} not found")
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"  [OK] {name} loaded ({os.path.getsize(path):,} bytes)")
    return data

def main():
    print("=" * 60)
    print("SteppeDNA GPU Embedding Integration (Row-Index Mode)")
    print("=" * 60)

    # Load master dataset
    csv_path = os.path.join(DATA_DIR, "master_training_dataset.csv")
    if not os.path.exists(csv_path):
        print("[ERROR] master_training_dataset.csv not found!")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    n_rows = len(df)
    print(f"\nDataset: {n_rows} variants, {len(df.columns)} columns")
    print(f"Genes: {df['Gene'].value_counts().to_dict()}")

    new_feature_names = []

    # 1. ESM-2 LoRA embeddings (row-indexed, 20 PCA components)
    print("\n--- ESM-2 LoRA Fine-tuned Embeddings ---")
    lora = load_pkl("esm2_lora_embeddings.pkl")
    if lora is not None:
        lora_emb = lora.get("embeddings", {})
        n_lora = 20
        for i in range(n_lora):
            col = f"esm2_lora_pca_{i}"
            df[col] = 0.0
            new_feature_names.append(col)

        matched = 0
        for row_idx in range(n_rows):
            if row_idx in lora_emb:
                emb = lora_emb[row_idx]
                if hasattr(emb, '__len__'):
                    for i in range(min(n_lora, len(emb))):
                        df.loc[row_idx, f"esm2_lora_pca_{i}"] = float(emb[i])
                matched += 1
        print(f"  Matched: {matched}/{n_rows} ({matched/n_rows*100:.1f}%)")

    # 2. ESM-2 650M embeddings (row-indexed, 20 PCA + cosine_sim + l2_shift)
    print("\n--- ESM-2 650M Embeddings ---")
    esm650 = load_pkl("esm2_650m_embeddings.pkl")
    if esm650 is not None:
        esm650_pca = esm650.get("embeddings_pca", {})
        esm650_cos = esm650.get("cosine_sim", {})
        esm650_l2 = esm650.get("l2_shift", {})
        n_pca = 20

        for i in range(n_pca):
            col = f"esm2_650m_pca_{i}"
            df[col] = 0.0
            new_feature_names.append(col)

        df["esm2_650m_cosine_sim"] = 0.0
        df["esm2_650m_l2_shift"] = 0.0
        new_feature_names.extend(["esm2_650m_cosine_sim", "esm2_650m_l2_shift"])

        matched = 0
        for row_idx in range(n_rows):
            if row_idx in esm650_pca:
                pca = esm650_pca[row_idx]
                if hasattr(pca, '__len__'):
                    for i in range(min(n_pca, len(pca))):
                        df.loc[row_idx, f"esm2_650m_pca_{i}"] = float(pca[i])
                if row_idx in esm650_cos:
                    df.loc[row_idx, "esm2_650m_cosine_sim"] = float(esm650_cos[row_idx])
                if row_idx in esm650_l2:
                    df.loc[row_idx, "esm2_650m_l2_shift"] = float(esm650_l2[row_idx])
                matched += 1
        print(f"  Matched: {matched}/{n_rows} ({matched/n_rows*100:.1f}%)")

    # 3. GNN structural embeddings (sparse — no BRCA2, 32D)
    print("\n--- GNN AlphaFold Structural Embeddings ---")
    gnn = load_pkl("gnn_structural_embeddings.pkl")
    if gnn is not None:
        gnn_emb = gnn.get("embeddings", {})
        gnn_dim = gnn.get("embedding_dim", 32)
        gene_coverage = gnn.get("gene_coverage", [])

        for i in range(gnn_dim):
            col = f"gnn_struct_{i}"
            df[col] = 0.0
            new_feature_names.append(col)

        # Also add a binary indicator for GNN coverage
        df["has_gnn_features"] = 0
        new_feature_names.append("has_gnn_features")

        matched = 0
        gene_counts = {}
        for row_idx in range(n_rows):
            if row_idx in gnn_emb:
                emb_vec = gnn_emb[row_idx]
                if hasattr(emb_vec, '__len__'):
                    for i in range(min(gnn_dim, len(emb_vec))):
                        df.loc[row_idx, f"gnn_struct_{i}"] = float(emb_vec[i])
                df.loc[row_idx, "has_gnn_features"] = 1
                matched += 1
                gene = df.loc[row_idx, "Gene"]
                gene_counts[gene] = gene_counts.get(gene, 0) + 1

        print(f"  Total matched: {matched}/{n_rows} ({matched/n_rows*100:.1f}%)")
        print(f"  Gene coverage: {gene_coverage}")
        for gene, cnt in sorted(gene_counts.items()):
            total_gene = (df["Gene"] == gene).sum()
            print(f"    {gene}: {cnt}/{total_gene} ({cnt/total_gene*100:.1f}%)")
        print(f"  NOTE: BRCA2 has no GNN data (AlphaFold structure too large for GNN)")

    # Save augmented dataset
    out_path = os.path.join(DATA_DIR, "master_training_dataset_gpu_augmented.csv")
    df.to_csv(out_path, index=False)

    # Save combined feature names (original + new GPU features)
    with open(os.path.join(DATA_DIR, "universal_feature_names.pkl"), "rb") as f:
        orig_features = pickle.load(f)

    all_features = orig_features + new_feature_names
    with open(os.path.join(DATA_DIR, "gpu_feature_names.pkl"), "wb") as f:
        pickle.dump(all_features, f)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"GPU Augmented Dataset saved: {out_path}")
    print(f"Total rows: {n_rows}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Original features: {len(orig_features)}")
    print(f"New GPU features: {len(new_feature_names)}")
    print(f"Total features: {len(all_features)}")
    print(f"  - ESM-2 LoRA: 20 PCA components")
    print(f"  - ESM-2 650M: 20 PCA + cosine_sim + l2_shift = 22")
    print(f"  - GNN structural: 32 dimensions + has_gnn indicator = 33")
    print(f"Feature names saved: gpu_feature_names.pkl ({len(all_features)} features)")
    print(f"\nTo retrain with GPU features:")
    print(f"  python scripts/train_gpu_model.py")

if __name__ == "__main__":
    main()
