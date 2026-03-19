"""
B28: European ancestry bias evaluation.
Proxy analysis: assigns variants to population groups by highest gnomAD sub-population AF,
then computes XGBoost-only AUC per group.

NOTE: This is a PROXY, not true ancestry-stratified evaluation. ClinVar lacks ancestry metadata.

Usage: python scripts/ancestry_bias_evaluation.py
"""
import os, sys, json, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts._split_helper import get_split, DATA_DIR

import xgboost as xgb
from sklearn.metrics import roc_auc_score


def main():
    print("Loading data and reproducing test split...")
    s = get_split()
    X_te, y_te, g_te = s["X_te"], s["y_te"], s["g_te"]
    fn = s["feature_cols"]
    print(f"  Test set: {len(X_te)} variants")

    # Load XGBoost model + scaler
    booster = xgb.Booster()
    booster.load_model(os.path.join(DATA_DIR, "universal_xgboost_final.json"))
    with open(os.path.join(DATA_DIR, "universal_scaler_ensemble.pkl"), "rb") as f:
        scaler = pickle.load(f)

    X_te_s = scaler.transform(X_te)
    dtest = xgb.DMatrix(X_te_s, feature_names=fn)
    xgb_probs = booster.predict(dtest)

    # Get population AF column indices
    pop_cols = {
        "AFR": fn.index("gnomad_af_afr") if "gnomad_af_afr" in fn else None,
        "AMR": fn.index("gnomad_af_amr") if "gnomad_af_amr" in fn else None,
        "EAS": fn.index("gnomad_af_eas") if "gnomad_af_eas" in fn else None,
        "NFE": fn.index("gnomad_af_nfe") if "gnomad_af_nfe" in fn else None,
    }

    # Assign each variant to its dominant population (highest AF)
    pop_assignments = []
    for i in range(len(X_te)):
        best_pop = "NO_DATA"
        best_af = 0.0
        for pop, col_idx in pop_cols.items():
            if col_idx is not None:
                af_val = float(X_te[i, col_idx])
                if af_val > best_af:
                    best_af = af_val
                    best_pop = pop
        if best_af == 0.0:
            best_pop = "NO_DATA"
        pop_assignments.append(best_pop)

    pop_assignments = np.array(pop_assignments)

    # Compute AUC per population group
    print(f"\n{'='*60}")
    print(f"Ancestry Bias Evaluation (Proxy Analysis)")
    print(f"{'='*60}")

    pop_results = {}
    for pop in ["NFE", "EAS", "AFR", "AMR", "NO_DATA"]:
        mask = pop_assignments == pop
        n = mask.sum()
        n_path = y_te[mask].sum() if n > 0 else 0
        n_benign = n - n_path

        if n < 5 or len(set(y_te[mask])) < 2:
            auc = None
            print(f"  {pop:8s} n={n:5d} (path={n_path}, benign={n_benign}) — too few for AUC")
        else:
            auc = roc_auc_score(y_te[mask], xgb_probs[mask])
            print(f"  {pop:8s} n={n:5d} (path={n_path}, benign={n_benign}) AUC={auc:.4f}")

        pop_results[pop] = {
            "n": int(n),
            "n_pathogenic": int(n_path),
            "n_benign": int(n_benign),
            "auc": round(auc, 4) if auc else None,
        }

    # PM2 disparity analysis: count PM2 flags per population
    print(f"\nPM2 Disparity (variants with AF=0 per population):")
    pm2_counts = {}
    for pop, col_idx in pop_cols.items():
        if col_idx is not None:
            af_zero = (X_te[:, col_idx] == 0.0).sum()
            pm2_counts[pop] = int(af_zero)
            pct = af_zero / len(X_te) * 100
            print(f"  {pop:8s} PM2 count={af_zero:5d} ({pct:.1f}%)")

    # Per-gene AUC as diversity proxy
    print(f"\nPer-gene AUC (data volume proxy for diversity):")
    gene_results = {}
    for gene in sorted(set(g_te)):
        mask = g_te == gene
        n = mask.sum()
        if n < 5 or len(set(y_te[mask])) < 2:
            print(f"  {gene:8s} n={n:5d} — too few for AUC")
            gene_results[gene] = {"n": int(n), "auc": None}
        else:
            auc = roc_auc_score(y_te[mask], xgb_probs[mask])
            print(f"  {gene:8s} n={n:5d} AUC={auc:.4f}")
            gene_results[gene] = {"n": int(n), "auc": round(auc, 4)}

    # Overall XGBoost AUC
    overall_auc = roc_auc_score(y_te, xgb_probs)
    print(f"\nOverall XGBoost-only AUC: {overall_auc:.4f}")

    results = {
        "method": "Proxy analysis: variants assigned to population by highest gnomAD sub-population AF",
        "caveat": "This is NOT true ancestry-stratified evaluation. ClinVar lacks ancestry metadata.",
        "overall_xgb_auc": round(overall_auc, 4),
        "per_population": pop_results,
        "pm2_disparity": pm2_counts,
        "per_gene_auc": gene_results,
        "nfe_enrichment": {
            "nfe_variants": pop_results.get("NFE", {}).get("n", 0),
            "eas_variants": pop_results.get("EAS", {}).get("n", 0),
            "ratio": round(
                pop_results.get("NFE", {}).get("n", 1) /
                max(pop_results.get("EAS", {}).get("n", 1), 1), 2
            ),
        },
    }

    out_path = os.path.join(DATA_DIR, "ancestry_bias_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
