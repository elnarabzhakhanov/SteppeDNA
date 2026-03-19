"""
C5: Compare SteppeDNA ACMG rule engine classifications against ClinVar binary labels.
Runs evaluate_acmg_rules() on the held-out test set using XGBoost-only predictions.

Usage: python scripts/acmg_comparison.py
"""
import os, sys, json, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts._split_helper import get_split, DATA_DIR
from backend.acmg_rules import evaluate_acmg_rules, combine_acmg_evidence

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score


def main():
    print("Loading data and reproducing test split...")
    s = get_split()
    X_te, y_te, g_te = s["X_te"], s["y_te"], s["g_te"]
    idx_te = s["idx_te"]
    df = s["df"]
    fn = s["feature_cols"]
    print(f"  Test set: {len(X_te)} variants")

    # Load XGBoost model + scaler
    booster = xgb.Booster()
    booster.load_model(os.path.join(DATA_DIR, "universal_xgboost_final.json"))
    with open(os.path.join(DATA_DIR, "universal_scaler_ensemble.pkl"), "rb") as f:
        scaler = pickle.load(f)

    # Scale test set and get XGBoost predictions
    X_te_s = scaler.transform(X_te)
    dtest = xgb.DMatrix(X_te_s, feature_names=fn)
    xgb_probs = booster.predict(dtest)

    # Load pathogenic positions for PS1/PM5
    pp_path = os.path.join(DATA_DIR, "pathogenic_positions.json")
    pathogenic_positions = {}
    if os.path.exists(pp_path):
        with open(pp_path) as f:
            pathogenic_positions = json.load(f)

    # Build feature dict for each test variant and run ACMG
    print("Running ACMG evaluation on test set...")
    acmg_classifications = []
    code_counts = {}
    per_gene_results = {}

    for i in range(len(X_te)):
        row_idx = idx_te[i]
        gene = g_te[i]
        prob = float(xgb_probs[i])
        true_label = int(y_te[i])

        # Build features dict from dataset columns
        feat_dict = {}
        for col_idx, col_name in enumerate(fn):
            feat_dict[col_name] = float(X_te[i, col_idx])

        # Map dataset columns to ACMG-expected keys
        feat_dict.setdefault("gnomad_af", feat_dict.get("gnomad_af", 0.0))
        feat_dict.setdefault("dist_dna", feat_dict.get("dist_dna", 999.0))
        feat_dict.setdefault("dist_palb2", feat_dict.get("dist_palb2", 999.0))
        feat_dict["in_critical_domain"] = (
            feat_dict.get("in_critical_repeat_region", 0) > 0.5 or
            feat_dict.get("in_DNA_binding", 0) > 0.5 or
            feat_dict.get("in_OB_folds", 0) > 0.5
        )
        feat_dict["is_nonsense"] = feat_dict.get("is_nonsense", 0) > 0.5

        # Check pathogenic position for PS1/PM5
        aa_pos = int(round(feat_dict.get("relative_aa_pos", 0) * 3418))  # approx
        gene_positions = pathogenic_positions.get(gene, [])
        if isinstance(gene_positions, dict):
            gene_positions = gene_positions.get("positions", [])
        feat_dict["known_pathogenic_at_pos"] = aa_pos in gene_positions if gene_positions else False

        # Run ACMG evaluation
        met_codes = evaluate_acmg_rules(feat_dict, prob, gene_name=gene)
        classification = combine_acmg_evidence(met_codes)

        # Count codes
        for code in met_codes:
            code_counts[code] = code_counts.get(code, 0) + 1

        # Map 5-tier to binary
        if classification in ("Pathogenic", "Likely Pathogenic"):
            acmg_binary = 1
        elif classification in ("Benign", "Likely Benign"):
            acmg_binary = 0
        else:
            acmg_binary = -1  # VUS — unresolved

        acmg_classifications.append({
            "true_label": true_label,
            "acmg_class": classification,
            "acmg_binary": acmg_binary,
            "gene": gene,
            "xgb_prob": round(prob, 4),
            "codes": list(met_codes.keys()),
        })

        # Per-gene tracking
        if gene not in per_gene_results:
            per_gene_results[gene] = {"correct": 0, "total": 0, "vus": 0, "resolved": 0}
        per_gene_results[gene]["total"] += 1
        if acmg_binary == -1:
            per_gene_results[gene]["vus"] += 1
        else:
            per_gene_results[gene]["resolved"] += 1
            if acmg_binary == true_label:
                per_gene_results[gene]["correct"] += 1

    # Compute overall metrics
    resolved = [c for c in acmg_classifications if c["acmg_binary"] != -1]
    vus_count = len(acmg_classifications) - len(resolved)

    if resolved:
        y_true_r = [c["true_label"] for c in resolved]
        y_pred_r = [c["acmg_binary"] for c in resolved]
        agreement = sum(1 for t, p in zip(y_true_r, y_pred_r) if t == p) / len(resolved)
        kappa = cohen_kappa_score(y_true_r, y_pred_r)
    else:
        agreement = 0
        kappa = 0

    # Per-gene summary
    gene_summary = {}
    for gene, gr in sorted(per_gene_results.items()):
        acc = gr["correct"] / gr["resolved"] if gr["resolved"] > 0 else 0
        gene_summary[gene] = {
            "total": gr["total"],
            "resolved": gr["resolved"],
            "vus": gr["vus"],
            "correct": gr["correct"],
            "agreement": round(acc, 4),
            "vus_rate": round(gr["vus"] / gr["total"], 4),
        }

    # Classification distribution
    class_dist = {}
    for c in acmg_classifications:
        cls = c["acmg_class"]
        class_dist[cls] = class_dist.get(cls, 0) + 1

    results = {
        "total_variants": len(acmg_classifications),
        "resolved": len(resolved),
        "vus_count": vus_count,
        "vus_rate": round(vus_count / len(acmg_classifications), 4),
        "agreement_rate": round(agreement, 4),
        "cohen_kappa": round(kappa, 4),
        "classification_distribution": class_dist,
        "code_frequency": dict(sorted(code_counts.items(), key=lambda x: -x[1])),
        "per_gene": gene_summary,
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"ACMG Comparison Results (Test Set, XGBoost-only)")
    print(f"{'='*60}")
    print(f"Total variants:    {results['total_variants']}")
    print(f"Resolved (P/LP/B/LB): {results['resolved']} ({100-results['vus_rate']*100:.1f}%)")
    print(f"VUS (unresolved):  {results['vus_count']} ({results['vus_rate']*100:.1f}%)")
    print(f"Agreement rate:    {results['agreement_rate']*100:.1f}%")
    print(f"Cohen's kappa:     {results['cohen_kappa']:.4f}")
    print(f"\nClassification distribution:")
    for cls, cnt in sorted(class_dist.items(), key=lambda x: -x[1]):
        print(f"  {cls:25s} {cnt:5d}")
    print(f"\nACMG code frequency:")
    for code, cnt in sorted(code_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {code:10s} {cnt:5d}")
    print(f"\nPer-gene breakdown:")
    for gene, gs in sorted(gene_summary.items()):
        print(f"  {gene:8s} n={gs['total']:5d} resolved={gs['resolved']:5d} VUS={gs['vus']:4d} agreement={gs['agreement']*100:.1f}%")

    # Save results
    out_path = os.path.join(DATA_DIR, "acmg_comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
