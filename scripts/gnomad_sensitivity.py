"""
SteppeDNA: gnomAD Proxy-Benign Sensitivity Analysis
=====================================================
Evaluates how dependent model performance is on the 485 gnomAD proxy-benign
variants added to augment class balance in non-BRCA2 genes.

Motivation (Section 1.7 of maximizing_project.md):
  Using common gnomAD variants (AC >= 2) as benign proxies introduces potential
  label noise. Some common variants may be low-penetrance pathogenic or have
  population-specific effects. This script quantifies the impact.

What this script does:
  1. Loads master_training_dataset.csv (19,223 variants)
  2. Loads gnomad_proxy_benign_all.csv (485 variants)
  3. Reports per-gene breakdown of gnomAD proxy-benign variants
  4. Reports class balance before/after removal
  5. Documents retraining steps (actual retraining requires model artifacts)

What a full run would do (TODO when model files available):
  - Retrain XGBoost + MLP on the remaining 18,738 ClinVar-only variants
  - Same 60/20/20 gene-stratified split, same hyperparameters
  - Evaluate on same test set
  - Report per-gene AUC delta
  - If delta < 0.01: gnomAD augmentation helps without introducing noise
  - If delta > 0.01: investigate which gnomAD variants are borderline
  - Save results to data/gnomad_sensitivity_results.json

Run from project root:
  python scripts/gnomad_sensitivity.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

MASTER_CSV = os.path.join(DATA_DIR, "master_training_dataset.csv")
GNOMAD_CSV = os.path.join(DATA_DIR, "gnomad_proxy_benign_all.csv")
OUTPUT_JSON = os.path.join(DATA_DIR, "gnomad_sensitivity_results.json")


def load_datasets():
    """Load master training dataset and gnomAD proxy-benign catalog."""
    if not os.path.exists(MASTER_CSV):
        print(f"ERROR: {MASTER_CSV} not found")
        sys.exit(1)
    if not os.path.exists(GNOMAD_CSV):
        print(f"ERROR: {GNOMAD_CSV} not found")
        sys.exit(1)

    master = pd.read_csv(MASTER_CSV)
    gnomad = pd.read_csv(GNOMAD_CSV)

    print(f"Master dataset: {len(master)} variants")
    print(f"gnomAD proxy-benign catalog: {len(gnomad)} variants")
    return master, gnomad


def analyze_gnomad_contribution(master, gnomad):
    """Report per-gene gnomAD contribution and class balance impact."""

    # --- Per-gene breakdown of gnomAD proxy-benign variants ---
    print("\n=== gnomAD Proxy-Benign Variants by Gene ===")
    gene_counts = gnomad["Gene"].value_counts().sort_index()
    for gene, count in gene_counts.items():
        print(f"  {gene}: {count} gnomAD proxy-benign variants")
    print(f"  TOTAL: {len(gnomad)}")

    # --- Class balance in master dataset ---
    print("\n=== Class Balance in Master Dataset (all 19,223) ===")
    if "Gene" in master.columns and "Label" in master.columns:
        for gene in sorted(master["Gene"].unique()):
            gene_mask = master["Gene"] == gene
            n_total = gene_mask.sum()
            n_pathogenic = (master.loc[gene_mask, "Label"] == 1).sum()
            n_benign = (master.loc[gene_mask, "Label"] == 0).sum()
            pct_path = 100 * n_pathogenic / n_total if n_total > 0 else 0
            print(f"  {gene}: {n_total} total ({n_pathogenic} P / {n_benign} B, "
                  f"{pct_path:.1f}% pathogenic)")

    # --- What class balance would look like without gnomAD ---
    n_gnomad = len(gnomad)
    n_clinvar_only = len(master) - n_gnomad
    print(f"\n=== After Removing {n_gnomad} gnomAD Variants ===")
    print(f"  Remaining: {n_clinvar_only} ClinVar-only variants")

    # Per-gene impact: gnomAD variants are all Label=0 (benign)
    print("\n  Per-gene impact (gnomAD variants removed = fewer benign):")
    for gene in sorted(gnomad["Gene"].unique()):
        gnomad_count = (gnomad["Gene"] == gene).sum()
        if "Gene" in master.columns:
            gene_mask = master["Gene"] == gene
            total_now = gene_mask.sum()
            benign_now = (master.loc[gene_mask, "Label"] == 0).sum()
            benign_after = benign_now - gnomad_count
            total_after = total_now - gnomad_count
            if total_after > 0:
                pct_path_after = 100 * (total_now - benign_now) / total_after
            else:
                pct_path_after = 0
            print(f"    {gene}: {total_now} -> {total_after} "
                  f"(benign: {benign_now} -> {benign_after}, "
                  f"pathogenic %: {pct_path_after:.1f}%)")

    return {
        "total_master": len(master),
        "total_gnomad": n_gnomad,
        "total_clinvar_only": n_clinvar_only,
        "gnomad_per_gene": gene_counts.to_dict(),
    }


def document_retraining_steps():
    """Print the steps needed for full sensitivity analysis (requires model files)."""
    print("\n" + "=" * 70)
    print("RETRAINING STEPS (not executed — requires model artifacts)")
    print("=" * 70)
    print("""
To complete the full sensitivity analysis:

1. IDENTIFY gnomAD rows in master dataset:
   - Match on Gene + relative_aa_pos (or reconstruct from gnomAD AA_pos / gene aa_length)
   - Or add a 'source' column to master_training_dataset.csv in build_master_dataset.py

2. SPLIT remaining 18,738 ClinVar-only variants:
   - 60/20/20 train/cal/test with gene x label stratification
   - Use same random seed as original split for reproducibility

3. RETRAIN models:
   - XGBoost: same hyperparameters (max_depth, learning_rate, n_estimators)
   - MLP: same architecture (hidden layers, dropout, epochs)
   - Isotonic calibration on calibration set

4. EVALUATE on the ORIGINAL test set (not the new split):
   - Per-gene ROC-AUC
   - Overall weighted AUC
   - Macro-averaged AUC

5. COMPARE:
   - Per-gene AUC delta (with_gnomad - without_gnomad)
   - If |delta| < 0.01: gnomAD augmentation is safe
   - If delta > 0.01 for any gene: investigate those gnomAD variants

6. SAVE results to data/gnomad_sensitivity_results.json
""")


def main():
    print("SteppeDNA gnomAD Proxy-Benign Sensitivity Analysis")
    print("=" * 50)

    master, gnomad = load_datasets()
    results = analyze_gnomad_contribution(master, gnomad)
    document_retraining_steps()

    # Save preliminary analysis
    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "analysis": "gnomad_sensitivity_preliminary",
            "status": "data_analysis_only",
            "note": "Full retraining not yet executed — requires model artifacts",
            **results,
        }, f, indent=2)
    print(f"\nPreliminary results saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
