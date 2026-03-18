#!/usr/bin/env python3
"""
B15: Population-Stratified gnomAD Analysis

Analyzes gnomAD population sub-frequencies across all 5 HR genes to:
1. Quantify representation gaps (especially Central Asian)
2. Identify population-specific allele frequency differences
3. Support SteppeDNA KZ population equity claims
"""

import os, json, pickle, numpy as np, pandas as pd
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
GENES = ["BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D"]
POPS = ["afr", "amr", "eas", "nfe", "sas"]

def load_gnomad(gene):
    for path in [f"{DATA_DIR}/{gene.lower()}_gnomad_frequencies.pkl", f"{DATA_DIR}/gnomad_frequencies.pkl"]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data
    return None

def analyze():
    print("=" * 60)
    print("Population-Stratified gnomAD AF Analysis")
    print("=" * 60)

    results = {}
    for gene in GENES:
        data = load_gnomad(gene)
        if data is None:
            print(f"  {gene}: No gnomAD data found")
            continue

        by_var = data.get("by_variant", {})
        by_pos = data.get("by_position", {})

        pop_counts = {p: 0 for p in POPS}
        pop_sums = {p: 0.0 for p in POPS}
        total = 0
        nonzero_any = 0

        for key, val in by_var.items():
            total += 1
            if isinstance(val, dict):
                has_any = False
                for p in POPS:
                    af = val.get(p, 0.0) or val.get(f"af_{p}", 0.0)
                    if af and af > 0:
                        pop_counts[p] += 1
                        pop_sums[p] += float(af)
                        has_any = True
                if has_any:
                    nonzero_any += 1
            elif isinstance(val, (int, float)) and val > 0:
                nonzero_any += 1

        results[gene] = {
            "total_variants": total,
            "nonzero_any_pop": nonzero_any,
            "per_pop_nonzero": pop_counts,
            "per_pop_mean_af": {p: pop_sums[p] / max(pop_counts[p], 1) for p in POPS},
        }

        NL = chr(10)
        print(f"{NL}  {gene} ({total} variants, {nonzero_any} with AF>0):")
        for p in POPS:
            mean_af = pop_sums[p] / max(pop_counts[p], 1)
            print(f"    {p.upper():4s}: {pop_counts[p]:5d} variants with AF>0  (mean={mean_af:.2e})")

    print(chr(10) + "=" * 60)
    print("Central Asian Representation Gap")
    print("=" * 60)
    print("  gnomAD v4 populations do NOT include a Central Asian (CAS) group.")
    print("  Kazakh population is partially represented under:")
    print("    - EAS (East Asian): closest proxy but genetically distinct")
    print("    - SAS (South Asian): some Central Asian overlap")
    print("  This creates a systematic blind spot for KZ-specific variants.")
    print("  SteppeDNA addresses this by integrating Kazakh founder mutations")
    print("  from published literature (Oncotarget 2023, CAJGH).")

    out_path = os.path.join(DATA_DIR, "population_analysis_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"{NL}Results saved to {out_path}")

if __name__ == "__main__":
    analyze()
