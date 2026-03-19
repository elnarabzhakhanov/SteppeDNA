"""
AF Recalibration Impact Analysis.
Shows how many variants change ACMG classification (BA1/BS1/PM2) when switching
from global AF to population-specific AF.

Usage: python scripts/af_recalibration_analysis.py
"""
import os, sys, json, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.acmg_rules import GENE_BA1_THRESHOLDS, GENE_BS1_THRESHOLDS

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

GENES = ["BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D"]
POPULATIONS = ["afr", "amr", "eas", "nfe"]


def classify_af(af, gene):
    """Return the AF-based ACMG code: BA1, BS1, PM2, or None."""
    ba1_t = GENE_BA1_THRESHOLDS.get(gene.upper(), 0.01)
    bs1_t = GENE_BS1_THRESHOLDS.get(gene.upper(), 0.001)
    if af > ba1_t:
        return "BA1"
    elif af > bs1_t:
        return "BS1"
    elif af == 0.0:
        return "PM2"
    return None


def main():
    print("AF Recalibration Impact Analysis")
    print("=" * 60)

    all_results = {}
    total_reclassifications = {pop: 0 for pop in POPULATIONS}
    total_variants_with_pop_af = 0

    for gene in GENES:
        pkl_path = os.path.join(DATA_DIR, f"{gene.lower()}_gnomad_frequencies.pkl")
        if not os.path.exists(pkl_path):
            print(f"  {gene}: gnomAD pkl not found, skipping")
            continue

        with open(pkl_path, "rb") as f:
            gnomad_data = pickle.load(f)

        # Handle both tuple and dict formats
        if isinstance(gnomad_data, tuple):
            by_variant = gnomad_data[0] if gnomad_data[0] else {}
        elif isinstance(gnomad_data, dict):
            by_variant = gnomad_data.get("by_variant", gnomad_data)
        else:
            by_variant = {}

        gene_results = {"total": len(by_variant), "reclassifications": {}}

        for pop in POPULATIONS:
            changes = {"PM2_gained": 0, "PM2_lost": 0, "BS1_gained": 0, "BS1_lost": 0,
                        "BA1_gained": 0, "BA1_lost": 0, "total_changed": 0}

            for variant_key, vdata in by_variant.items():
                if not isinstance(vdata, dict):
                    continue

                global_af = vdata.get("af", 0.0) or 0.0
                pop_af = vdata.get(pop, 0.0) or 0.0

                global_code = classify_af(global_af, gene)
                pop_code = classify_af(pop_af, gene)

                if global_code != pop_code:
                    changes["total_changed"] += 1

                    # Track direction of change
                    if pop_code == "PM2" and global_code != "PM2":
                        changes["PM2_gained"] += 1
                    if global_code == "PM2" and pop_code != "PM2":
                        changes["PM2_lost"] += 1
                    if pop_code == "BS1" and global_code != "BS1":
                        changes["BS1_gained"] += 1
                    if global_code == "BS1" and pop_code != "BS1":
                        changes["BS1_lost"] += 1
                    if pop_code == "BA1" and global_code != "BA1":
                        changes["BA1_gained"] += 1
                    if global_code == "BA1" and pop_code != "BA1":
                        changes["BA1_lost"] += 1

            gene_results["reclassifications"][pop] = changes
            total_reclassifications[pop] += changes["total_changed"]

        all_results[gene] = gene_results
        total_variants_with_pop_af += len(by_variant)

    # Print summary
    print(f"\nTotal variants with gnomAD data: {total_variants_with_pop_af}")
    print(f"\nReclassifications by population (global -> population-specific AF):")
    print(f"{'Pop':>6s} {'Total':>8s} {'PM2+':>6s} {'PM2-':>6s} {'BS1+':>6s} {'BS1-':>6s} {'BA1+':>6s} {'BA1-':>6s}")
    print("-" * 55)

    pop_summary = {}
    for pop in POPULATIONS:
        totals = {"total_changed": 0, "PM2_gained": 0, "PM2_lost": 0,
                  "BS1_gained": 0, "BS1_lost": 0, "BA1_gained": 0, "BA1_lost": 0}
        for gene in GENES:
            if gene in all_results and pop in all_results[gene]["reclassifications"]:
                for key in totals:
                    totals[key] += all_results[gene]["reclassifications"][pop][key]
        pop_summary[pop] = totals
        print(f"{pop.upper():>6s} {totals['total_changed']:>8d} {totals['PM2_gained']:>6d} "
              f"{totals['PM2_lost']:>6d} {totals['BS1_gained']:>6d} {totals['BS1_lost']:>6d} "
              f"{totals['BA1_gained']:>6d} {totals['BA1_lost']:>6d}")

    print(f"\nPer-gene breakdown:")
    for gene in GENES:
        if gene not in all_results:
            continue
        gr = all_results[gene]
        print(f"\n  {gene} ({gr['total']} variants):")
        for pop in POPULATIONS:
            if pop in gr["reclassifications"]:
                ch = gr["reclassifications"][pop]
                if ch["total_changed"] > 0:
                    print(f"    {pop.upper()}: {ch['total_changed']} reclassifications "
                          f"(PM2 +{ch['PM2_gained']}/-{ch['PM2_lost']}, "
                          f"BS1 +{ch['BS1_gained']}/-{ch['BS1_lost']})")

    # Key finding for competition
    eas = pop_summary.get("eas", {})
    nfe = pop_summary.get("nfe", {})
    print(f"\n{'='*60}")
    print(f"KEY FINDING:")
    print(f"  EAS: {eas.get('PM2_gained', 0)} variants gain PM2 (become suspicious)")
    print(f"       {eas.get('PM2_lost', 0)} variants lose PM2 (become less suspicious)")
    print(f"  NFE: {nfe.get('PM2_gained', 0)} variants gain PM2")
    print(f"       {nfe.get('PM2_lost', 0)} variants lose PM2")
    print(f"{'='*60}")

    output = {
        "description": "AF recalibration impact: comparing global vs population-specific gnomAD AF for ACMG BA1/BS1/PM2",
        "total_variants_analyzed": total_variants_with_pop_af,
        "population_summary": pop_summary,
        "per_gene": all_results,
    }

    out_path = os.path.join(DATA_DIR, "af_recalibration_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
