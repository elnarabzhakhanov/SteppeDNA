"""
integrate_brca1_dms.py - Integrate BRCA1 Findlay SGE (2018) DMS scores into training features.

The Findlay et al. 2018 saturation genome editing data for BRCA1 provides functional
scores for 1,837 single-nucleotide variants in exons 2-12 (RING + BRCT domains).
Currently used only for benchmarking, this script integrates it as a training feature
alongside the existing BRCA2 MAVE HDR scores.

Input: data/benchmark/BRCA1_HUMAN_Findlay_2018.csv
Output: data/brca1_dms_scores.pkl
  Format: {by_variant: {key: score}, by_position: {pos: avg_score}}

Run: python data_pipelines/integrate_brca1_dms.py
"""

import os
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict

DATA_DIR = "data"
INPUT_FILE = os.path.join(DATA_DIR, "benchmark", "BRCA1_HUMAN_Findlay_2018.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "brca1_dms_scores.pkl")

AA_1TO3 = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
    "Q": "Gln", "E": "Glu", "G": "Gly", "H": "His", "I": "Ile",
    "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
    "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
    "*": "Ter",
}


def main():
    print("=" * 60)
    print("Integrating BRCA1 Findlay SGE DMS scores")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} variants from Findlay 2018")

    by_variant = {}
    by_position = defaultdict(list)

    for _, row in df.iterrows():
        mutant = str(row.get("mutant", ""))
        dms_score = row.get("DMS_score", None)
        if pd.isna(dms_score) or not mutant:
            continue

        # Parse mutant format: e.g., "M1I" -> ref=M, pos=1, alt=I
        ref_1 = mutant[0]
        alt_1 = mutant[-1]
        try:
            pos = int(mutant[1:-1])
        except ValueError:
            continue

        ref_3 = AA_1TO3.get(ref_1, ref_1)
        alt_3 = AA_1TO3.get(alt_1, alt_1)

        # Store by variant key (3-letter codes to match training data format)
        variant_key = f"{ref_3}{pos}{alt_3}"
        by_variant[variant_key] = float(dms_score)
        by_position[pos].append(float(dms_score))

    # Average scores per position
    by_position_avg = {pos: np.mean(scores) for pos, scores in by_position.items()}

    output = {"by_variant": by_variant, "by_position": by_position_avg}
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(output, f)

    # Compute thresholds (Findlay: score < -1.328 = LOF, > -0.748 = functional)
    n_lof = sum(1 for v in by_variant.values() if v < -1.328)
    n_func = sum(1 for v in by_variant.values() if v > -0.748)
    n_inter = len(by_variant) - n_lof - n_func

    print("\nResults:")
    print(f"  Variants: {len(by_variant)}")
    print(f"  Positions: {len(by_position_avg)}")
    print(f"  LOF (< -1.328): {n_lof} ({100*n_lof/len(by_variant):.0f}%)")
    print(f"  Functional (> -0.748): {n_func} ({100*n_func/len(by_variant):.0f}%)")
    print(f"  Intermediate: {n_inter} ({100*n_inter/len(by_variant):.0f}%)")
    print(f"  Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
