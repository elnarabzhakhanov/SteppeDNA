"""Benchmark Deduplication — E1.13
Recompute benchmark metrics excluding training-set overlap.
Reports: test-only, novel-only, and training-excluded (test+novel) subsets.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, matthews_corrcoef, balanced_accuracy_score

DATA = Path(__file__).resolve().parent.parent / "data" / "benchmark"
PRED_CSV = DATA / "benchmark_v54_predictions.csv"
OUT_JSON = DATA / "benchmark_v54_deduped_results.json"


def safe_auc(y_true, y_score):
    if len(set(y_true)) < 2 or len(y_true) < 2:
        return None
    return round(roc_auc_score(y_true, y_score), 4)


def safe_mcc(y_true, y_pred_binary):
    if len(set(y_true)) < 2 or len(y_true) < 2:
        return None
    return round(matthews_corrcoef(y_true, y_pred_binary), 4)


def compute_metrics(df, threshold=0.5):
    """Compute AUC, MCC, balanced accuracy for a subset."""
    if len(df) < 2 or df["label"].nunique() < 2:
        return {"n": len(df), "roc_auc": None, "mcc": None, "bal_acc": None}
    auc = safe_auc(df["label"], df["steppedna_prob"])
    pred_binary = (df["steppedna_prob"] >= threshold).astype(int)
    mcc = safe_mcc(df["label"], pred_binary)
    bal = round(balanced_accuracy_score(df["label"], pred_binary), 4)
    return {"n": len(df), "roc_auc": auc, "mcc": mcc, "bal_acc": bal}


def compute_dms_metrics(df):
    """Compute DMS-specific metrics (Spearman correlation + AUC).
    DMS label convention: dms_score_bin=0 means loss-of-function (pathogenic).
    We flip to: 1=pathogenic, 0=benign for AUC computation.
    """
    if len(df) < 2:
        return {"n": len(df), "spearman_r": None, "roc_auc": None, "mcc": None}
    result = {"n": len(df)}
    valid = df.dropna(subset=["dms_score", "steppedna_prob"])
    if len(valid) >= 2:
        r, _ = spearmanr(valid["steppedna_prob"], valid["dms_score"])
        result["spearman_r"] = round(r, 4)
    else:
        result["spearman_r"] = None
    # Flip DMS labels: dms_score_bin=0 (loss-of-function) → pathogenic=1
    gdf_b = df[df["dms_score_bin"].notna()].copy()
    if len(gdf_b) >= 2:
        dl = (1 - gdf_b["dms_score_bin"]).astype(int)
        if dl.nunique() > 1:
            result["roc_auc"] = safe_auc(dl, gdf_b["steppedna_prob"])
            pred_binary = (gdf_b["steppedna_prob"] >= 0.5).astype(int)
            result["mcc"] = safe_mcc(dl, pred_binary)
        else:
            result["roc_auc"] = None
            result["mcc"] = None
    else:
        result["roc_auc"] = None
        result["mcc"] = None
    return result


def main():
    df = pd.read_csv(PRED_CSV)
    print(f"Total benchmark variants: {len(df)}")

    # Parse overlap flags (handle string 'True'/'False')
    df["in_training"] = df["in_training"].astype(str).str.strip().str.lower() == "true"
    df["in_test"] = df["in_test"].astype(str).str.strip().str.lower() == "true"

    n_train = df["in_training"].sum()
    n_test = df["in_test"].sum()
    n_novel = (~df["in_training"] & ~df["in_test"]).sum()
    print(f"  In training: {n_train} ({100*n_train/len(df):.1f}%)")
    print(f"  In test set: {n_test} ({100*n_test/len(df):.1f}%)")
    print(f"  Novel:       {n_novel} ({100*n_novel/len(df):.1f}%)")

    # Define subsets
    subsets = {
        "all": df,
        "no_training_overlap": df[~df["in_training"]],  # test + novel
        "test_only": df[df["in_test"]],
        "novel_only": df[~df["in_training"] & ~df["in_test"]],
    }

    results = {"overlap": {"training": int(n_train), "test": int(n_test), "novel": int(n_novel)}}

    for subset_name, sdf in subsets.items():
        print(f"\n{'='*60}")
        print(f"SUBSET: {subset_name} (n={len(sdf)})")
        print(f"{'='*60}")

        section = {}

        # DMS evaluation (ProteinGym_DMS source)
        dms = sdf[sdf["source"] == "ProteinGym_DMS"]
        if len(dms) > 0:
            section["dms"] = {}
            for gene in sorted(dms["gene"].unique()):
                g = dms[dms["gene"] == gene]
                metrics = compute_dms_metrics(g)
                section["dms"][gene] = metrics
                print(f"  DMS {gene}: n={metrics['n']}, AUC={metrics['roc_auc']}, r={metrics['spearman_r']}")
            # Overall DMS
            section["dms"]["overall"] = compute_dms_metrics(dms)

        # Expert panel evaluation (ClinVar_ExpertPanel source)
        expert = sdf[sdf["source"] == "ClinVar_ExpertPanel"]
        if len(expert) > 0:
            section["expert"] = {}
            section["expert"]["overall"] = compute_metrics(expert)
            print(f"  Expert overall: n={len(expert)}, AUC={section['expert']['overall']['roc_auc']}")
            for gene in sorted(expert["gene"].unique()):
                g = expert[expert["gene"] == gene]
                section["expert"][gene] = compute_metrics(g)
                print(f"  Expert {gene}: n={len(g)}, AUC={section['expert'][gene]['roc_auc']}")

        # Overall (all sources combined)
        section["overall"] = compute_metrics(sdf)
        print(f"  Overall: n={len(sdf)}, AUC={section['overall']['roc_auc']}")

        # Per-gene overall
        section["per_gene"] = {}
        for gene in sorted(sdf["gene"].unique()):
            g = sdf[sdf["gene"] == gene]
            section["per_gene"][gene] = compute_metrics(g)

        results[subset_name] = section

    # Save
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_JSON}")

    # Print comparison table
    print(f"\n{'='*60}")
    print("COMPARISON: All vs No-Training-Overlap")
    print(f"{'='*60}")
    for key in ["all", "no_training_overlap", "test_only", "novel_only"]:
        s = results[key]
        expert_auc = s.get("expert", {}).get("overall", {}).get("roc_auc", "N/A")
        dms_auc = s.get("dms", {}).get("overall", {}).get("roc_auc", "N/A")
        overall_auc = s.get("overall", {}).get("roc_auc", "N/A")
        n = s.get("overall", {}).get("n", 0)
        print(f"  {key:25s}: n={n:5d}, Expert AUC={expert_auc}, DMS AUC={dms_auc}, Overall AUC={overall_auc}")


if __name__ == "__main__":
    main()
