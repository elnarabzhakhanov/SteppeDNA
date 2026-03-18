"""
SteppeDNA: Universal Multi-Gene Master Dataset Builder (v4 — gnomAD Augmentation)
=================================================================================
Builds the master training dataset from all 5 HR pathway genes.

Fixes over v3:
  7. Incorporates gnomAD proxy-benign variants (AC >= 2 in gnomAD v4) to
     augment severely imbalanced non-BRCA2 genes. These common population
     variants are very likely benign for high-penetrance cancer genes.

Previous fixes:
  5-6. Drops gene-identifying features, keeps relative positions (v3).
  1-4. BRCA2 inclusion, feature alignment, zero-variance, feature names (v2).

Run from project root:
  python scripts/build_master_dataset.py
"""

import os
import sys
import pandas as pd
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.feature_engineering import (
    engineer_features,
    load_phylop_scores, load_mave_scores, load_alphamissense_scores,
    load_structural_features, load_gnomad_frequencies, load_spliceai_scores,
    load_esm2_embeddings,
)

GENES = ["brca1", "brca2", "palb2", "rad51c", "rad51d"]

# BRCA2 uses a separate main dataset file, not the unified one from prepare_all_genes.py
BRCA2_MAIN_DATASET = "brca2_missense_dataset_2.csv"

print("SteppeDNA: Universal Feature Engineer (v2)")
print("=" * 55)

def _load_pickle(gene_prefix, filename):
    """Load a pickle file, trying gene-specific path first, then global fallback."""
    gene_path = os.path.join("data", f"{gene_prefix}_{filename}")
    global_path = os.path.join("data", filename)
    for p in [gene_path, global_path]:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    return None


all_dfs = []
total_missense = 0

for g in GENES:
    # ── Locate the dataset ──────────────────────────────────────────────
    if g == "brca2":
        # BRCA2 uses the main dataset which has real nucleotide info
        dataset_path = BRCA2_MAIN_DATASET
    else:
        dataset_path = os.path.join("data", g, f"{g}_missense_dataset_unified.csv")

    if not os.path.exists(dataset_path):
        print(f"  [SKIP] {g.upper()} dataset not found at {dataset_path}")
        continue

    try:
        df = pd.read_csv(dataset_path)
    except pd.errors.EmptyDataError:
        print(f"  [SKIP] {g.upper()} dataset is empty.")
        continue

    if len(df) == 0:
        print(f"  [SKIP] {g.upper()} dataset has 0 rows.")
        continue

    # BRCA2 main dataset doesn't have a 'Gene' column — add it
    if "Gene" not in df.columns:
        df["Gene"] = g.upper()

    # ── Load gnomAD proxy-benign variants ──────────────────────────────
    gnomad_proxy_path = os.path.join("data", g, f"{g}_gnomad_proxy_benign.csv")
    if os.path.exists(gnomad_proxy_path):
        gnomad_proxy = pd.read_csv(gnomad_proxy_path)
        n_proxy = len(gnomad_proxy)
        if n_proxy > 0:
            # Build a dataframe compatible with engineer_features()
            proxy_df = pd.DataFrame({
                "AA_ref": gnomad_proxy["AA_ref"],
                "AA_pos": gnomad_proxy["AA_pos"].astype(int),
                "AA_alt": gnomad_proxy["AA_alt"],
                "cDNA_pos": (gnomad_proxy["AA_pos"].astype(int) * 3).clip(lower=1),  # approximate
                "Mutation": "N>N",  # placeholder — gets dropped as gene-leak feature
                "Label": 0,
                "Gene": g.upper(),
            })
            df = pd.concat([df, proxy_df], ignore_index=True)
            print(f"  + Added {n_proxy} gnomAD proxy-benign variants")

    total_missense += len(df)
    print(f"\n[{g.upper()}] Engineering {len(df)} variants...")

    # ── Load biological data sources ────────────────────────────────────
    phylop   = _load_pickle(g, "phylop_scores.pkl")
    mave     = _load_pickle(g, "mave_scores.pkl")
    am       = _load_pickle(g, "alphamissense_scores.pkl")
    struct   = _load_pickle(g, "structural_features.pkl")
    gnomad   = _load_pickle(g, "gnomad_frequencies.pkl")
    spliceai = _load_pickle(g, "spliceai_scores.pkl")

    esm2_raw = _load_pickle(g, "esm2_embeddings.pkl")
    esm2 = esm2_raw.get("embeddings", {}) if isinstance(esm2_raw, dict) else None
    eve      = _load_pickle(g, "eve_scores.pkl")

    loaded = []
    if phylop is not None: loaded.append("PhyloP")
    if mave is not None: loaded.append("MAVE")
    if am is not None: loaded.append("AlphaMissense")
    if struct is not None: loaded.append("Structure")
    if gnomad is not None: loaded.append("gnomAD")
    if esm2 is not None: loaded.append("ESM-2")
    if eve is not None: loaded.append("EVE")
    print(f"  Data sources loaded: {', '.join(loaded) if loaded else 'NONE'}")

    # ── Run feature engineering ─────────────────────────────────────────
    X_features = engineer_features(
        df, phylop, mave, am, struct, gnomad, spliceai, esm2, gene_name=g,
        eve_data=eve,
    )

    # Attach metadata for training
    X_features["Label"] = df["Label"].values
    X_features["Gene"] = df["Gene"].values
    X_features["Mutation_Str"] = df["Mutation"].values

    print(f"  Engineered {X_features.shape[1] - 3} features for {len(X_features)} variants")
    all_dfs.append(X_features)

    # Free memory before next gene
    del phylop, mave, am, struct, gnomad, spliceai, esm2_raw, esm2, eve, df, X_features
    import gc; gc.collect()

if not all_dfs:
    print("\n[ERROR] No variants engineered. Check data paths.")
    sys.exit(1)

# ── Concatenate all genes ───────────────────────────────────────────────
master_df = pd.concat(all_dfs, ignore_index=True)
print(f"\n{'='*55}")
print(f"  Combined: {len(master_df)} variants from {len(all_dfs)} genes")

# ── Drop gene-identifying features ────────────────────────────────────
# These features encode gene identity (not variant biology) and let the
# model shortcut: "BRCA1/PALB2 → pathogenic, BRCA2 → benign" instead of
# learning actual pathogenicity signals.
GENE_LEAK_FEATURES = {"cDNA_pos", "AA_pos", "is_transition", "is_transversion"}
# Mutation one-hots are also gene-identifiers: nucleotide data is "N>N" for
# all non-BRCA2 genes, so Mutation_* columns are always 0 for 4 of 5 genes.
mutation_onehots = [c for c in master_df.columns if c.startswith("Mutation_")]
gene_leak_cols = [c for c in master_df.columns if c in GENE_LEAK_FEATURES or c in mutation_onehots]

if gene_leak_cols:
    print(f"\n  Dropping {len(gene_leak_cols)} gene-identifying features:")
    for c in sorted(gene_leak_cols)[:10]:
        print(f"    - {c}")
    if len(gene_leak_cols) > 10:
        print(f"    ... and {len(gene_leak_cols) - 10} more")
    master_df.drop(columns=gene_leak_cols, inplace=True)

# ── Drop zero-variance features ────────────────────────────────────────
# These carry no discriminative information and add noise to the model.
feature_cols = [c for c in master_df.columns if c not in ["Label", "Gene", "Mutation_Str"]]

zero_var_cols = []
for col in feature_cols:
    if master_df[col].std() == 0 or master_df[col].nunique() <= 1:
        zero_var_cols.append(col)

if zero_var_cols:
    print(f"\n  Dropping {len(zero_var_cols)} zero-variance features:")
    for c in sorted(zero_var_cols):
        print(f"    - {c} (constant = {master_df[c].iloc[0]})")
    master_df.drop(columns=zero_var_cols, inplace=True)

# ── Save ────────────────────────────────────────────────────────────────
final_feature_cols = [c for c in master_df.columns if c not in ["Label", "Gene", "Mutation_Str"]]

out_path = "data/master_training_dataset.csv"
master_df.to_csv(out_path, index=False)

with open("data/universal_feature_names.pkl", "wb") as f:
    pickle.dump(final_feature_cols, f)

print(f"\n{'='*55}")
print(f"  [SAVED] {out_path}")
print(f"  Total variants: {len(master_df)}")
print(f"  Final feature count: {len(final_feature_cols)}")
print(f"  Feature names saved to: data/universal_feature_names.pkl")

# ── Summary statistics ──────────────────────────────────────────────────
print(f"\n  Class distribution:")
for label, count in master_df["Label"].value_counts().sort_index().items():
    name = "Pathogenic" if label == 1 else "Benign"
    print(f"    {name}: {count} ({count/len(master_df)*100:.1f}%)")

print(f"\n  Per-gene breakdown:")
for gene, group in master_df.groupby("Gene"):
    n_p = (group["Label"] == 1).sum()
    n_b = (group["Label"] == 0).sum()
    print(f"    {gene}: {len(group)} variants ({n_p}P / {n_b}B)")

print(f"\n  Done.")
