"""
evaluate_benchmark.py — Evaluate SteppeDNA on the Gold-Standard Benchmark
==========================================================================

Loads the curated benchmark dataset and evaluates SteppeDNA predictions
against:
  1. DMS experimental scores (Spearman correlation)
  2. DMS binary labels (ROC-AUC, PR-AUC, MCC)
  3. Expert-panel classifications (ROC-AUC, PR-AUC, MCC)

Only evaluates variants present in the test set (to avoid training leakage).
Novel variants (not in training OR test) are reported separately.

Generates:
  visual_proofs/8_Benchmark_DMS_Correlation.png/pdf
  visual_proofs/8_Benchmark_Classification.png/pdf
  data/benchmark/benchmark_results.json

Run from project root:
  python scripts/evaluate_benchmark.py
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc, roc_auc_score,
    average_precision_score, matthews_corrcoef, balanced_accuracy_score,
    confusion_matrix
)
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", font_scale=1.1)
os.makedirs("visual_proofs", exist_ok=True)

RANDOM_STATE = 42
XGB_WEIGHT = 0.6
NN_WEIGHT = 0.4

BENCHMARK_FILE = "data/benchmark/gold_standard_benchmark.csv"
RESULTS_FILE = "data/benchmark/benchmark_results.json"

# Gene protein lengths
GENE_AA_LENGTH = {
    "BRCA1": 1863, "BRCA2": 3418, "PALB2": 1186,
    "RAD51C": 376, "RAD51D": 328,
}

AA_3TO1 = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
}


def load_model_and_predict():
    """Load SteppeDNA ensemble and generate test-set predictions with variant IDs."""
    print("\n[1] Loading model and reproducing test split...", flush=True)

    df = pd.read_csv("data/master_training_dataset.csv")
    with open("data/universal_feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    with open("data/universal_scaler_ensemble.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("data/universal_calibrator_ensemble.pkl", "rb") as f:
        calibrator = pickle.load(f)
    with open("data/universal_threshold_ensemble.pkl", "rb") as f:
        threshold = pickle.load(f)

    feature_names = [f for f in feature_names if f in df.columns]
    X = df[feature_names].values
    y = df["Label"].values
    genes = df["Gene"].values
    strata = np.array([f"{g}_{l}" for g, l in zip(genes, y)])

    indices = np.arange(len(df))
    idx_traincal, idx_test = train_test_split(
        indices, test_size=0.20, random_state=RANDOM_STATE, stratify=strata
    )

    test_df = df.iloc[idx_test].copy().reset_index(drop=True)
    X_test = test_df[feature_names].values
    y_test = test_df["Label"].values

    # Load models
    xgb_clf = xgb.XGBClassifier()
    xgb_clf.load_model("data/universal_xgboost_final.json")
    from tensorflow.keras.models import load_model
    nn_model = load_model("data/universal_nn.h5", compile=False)

    # Predict
    X_test_s = scaler.transform(X_test)
    nn_preds = nn_model.predict(X_test_s, verbose=0).flatten()
    xgb_preds = xgb_clf.predict_proba(X_test_s)[:, 1]
    blended = XGB_WEIGHT * xgb_preds + NN_WEIGHT * nn_preds
    probs = calibrator.predict(blended)

    # Decode AA from one-hot
    aa_names_3 = list(AA_3TO1.keys())

    def decode_onehot(row, prefix):
        for aa in aa_names_3:
            col = f"{prefix}_{aa}"
            if col in row.index and row[col] == 1:
                return AA_3TO1.get(aa)
        return None

    test_df["aa_ref_1"] = test_df.apply(lambda r: decode_onehot(r, "AA_ref"), axis=1)
    test_df["aa_alt_1"] = test_df.apply(lambda r: decode_onehot(r, "AA_alt"), axis=1)
    test_df["AA_pos"] = test_df.apply(
        lambda r: round(r["relative_aa_pos"] * GENE_AA_LENGTH.get(r["Gene"], 1000)),
        axis=1
    ).astype(int)
    test_df["steppedna_prob"] = probs
    test_df["steppedna_pred"] = (probs >= threshold).astype(int)

    print(f"  Test set: {len(test_df)} variants", flush=True)
    print(f"  SteppeDNA ROC-AUC: {roc_auc_score(y_test, probs):.4f}", flush=True)

    return test_df, threshold


def match_benchmark_to_predictions(benchmark_df, test_df):
    """Match benchmark variants to test-set predictions."""
    print("\n[2] Matching benchmark variants to test-set predictions...", flush=True)

    # Build test-set lookup: (gene, pos, ref, alt) -> row
    test_lookup = {}
    for idx, row in test_df.iterrows():
        key = (row["Gene"], int(row["AA_pos"]), row["aa_ref_1"], row["aa_alt_1"])
        test_lookup[key] = row

    matched = []
    unmatched = []
    for _, brow in benchmark_df.iterrows():
        key = (brow["gene"], int(brow["aa_pos"]), brow["aa_ref"], brow["aa_alt"])
        if key in test_lookup:
            trow = test_lookup[key]
            matched.append({
                **brow.to_dict(),
                "steppedna_prob": trow["steppedna_prob"],
                "steppedna_pred": trow["steppedna_pred"],
                "training_label": trow["Label"],
            })
        else:
            unmatched.append(brow.to_dict())

    matched_df = pd.DataFrame(matched)
    unmatched_df = pd.DataFrame(unmatched)

    print(f"  Matched to test set: {len(matched_df)}", flush=True)
    print(f"  Not in test set:     {len(unmatched_df)}", flush=True)

    return matched_df, unmatched_df


def evaluate_dms(matched_df):
    """Evaluate SteppeDNA against DMS experimental scores."""
    dms_df = matched_df[matched_df["source"] == "ProteinGym_DMS"].copy()
    if len(dms_df) == 0:
        print("  No DMS variants matched to test set", flush=True)
        return {}

    print(f"\n[3a] DMS Evaluation ({len(dms_df)} variants matched)", flush=True)

    results = {}
    for gene in sorted(dms_df["gene"].unique()):
        gene_df = dms_df[dms_df["gene"] == gene].copy()
        valid = gene_df["dms_score"].notna()
        gene_df = gene_df[valid]

        if len(gene_df) < 10:
            print(f"  {gene}: Too few matched variants ({len(gene_df)}), skipping", flush=True)
            continue

        # Spearman correlation between SteppeDNA prob and DMS score
        # Note: DMS lower score = loss of function = pathogenic
        # SteppeDNA higher prob = pathogenic
        # So we expect NEGATIVE correlation
        spearman_r, spearman_p = stats.spearmanr(
            gene_df["steppedna_prob"], gene_df["dms_score"]
        )

        # Binary classification using DMS_score_bin
        valid_bin = gene_df["dms_score_bin"].notna()
        gene_df_bin = gene_df[valid_bin]

        roc_auc_val = None
        mcc_val = None
        if len(gene_df_bin) > 10 and gene_df_bin["dms_score_bin"].nunique() > 1:
            # DMS_score_bin: 1 = functional/benign, 0 = loss-of-function/pathogenic
            # Invert: pathogenic = 1 - dms_score_bin
            dms_labels = (1 - gene_df_bin["dms_score_bin"]).astype(int)
            if dms_labels.nunique() > 1:
                roc_auc_val = roc_auc_score(dms_labels, gene_df_bin["steppedna_prob"])
                preds = gene_df_bin["steppedna_pred"].values
                mcc_val = matthews_corrcoef(dms_labels, preds)

        results[gene] = {
            "n_variants": len(gene_df),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "roc_auc": float(roc_auc_val) if roc_auc_val is not None else None,
            "mcc": float(mcc_val) if mcc_val is not None else None,
            "citation": gene_df["citation"].iloc[0],
        }

        print(f"  {gene}: n={len(gene_df)}, "
              f"Spearman r={spearman_r:.3f} (p={spearman_p:.2e})"
              f"{f', AUC={roc_auc_val:.3f}' if roc_auc_val else ''}", flush=True)

    return {"dms_results": results, "dms_df": dms_df}


def evaluate_expert(matched_df):
    """Evaluate SteppeDNA against expert-panel classifications."""
    expert_df = matched_df[matched_df["source"] == "ClinVar_ExpertPanel"].copy()
    if len(expert_df) == 0:
        print("  No expert-panel variants matched to test set", flush=True)
        return {}

    print(f"\n[3b] Expert-Panel Evaluation ({len(expert_df)} variants matched)", flush=True)

    valid = expert_df["label"].notna()
    expert_df = expert_df[valid]

    results = {}

    # Overall
    if len(expert_df) > 10 and expert_df["label"].nunique() > 1:
        expert_labels = expert_df["label"].astype(int)
        roc_auc_val = roc_auc_score(expert_labels, expert_df["steppedna_prob"])
        mcc_val = matthews_corrcoef(expert_labels, expert_df["steppedna_pred"])
        bal_acc = balanced_accuracy_score(expert_labels, expert_df["steppedna_pred"])

        results["overall"] = {
            "n_variants": len(expert_df),
            "roc_auc": float(roc_auc_val),
            "mcc": float(mcc_val),
            "balanced_accuracy": float(bal_acc),
        }
        print(f"  Overall: n={len(expert_df)}, AUC={roc_auc_val:.3f}, "
              f"MCC={mcc_val:.3f}, BalAcc={bal_acc:.3f}", flush=True)

    # Per gene
    for gene in sorted(expert_df["gene"].unique()):
        gene_df = expert_df[expert_df["gene"] == gene]
        if len(gene_df) < 5 or gene_df["label"].nunique() < 2:
            print(f"  {gene}: Too few variants or single class ({len(gene_df)})", flush=True)
            continue

        gene_labels = gene_df["label"].astype(int)
        roc_auc_val = roc_auc_score(gene_labels, gene_df["steppedna_prob"])
        mcc_val = matthews_corrcoef(gene_labels, gene_df["steppedna_pred"])

        results[gene] = {
            "n_variants": len(gene_df),
            "roc_auc": float(roc_auc_val),
            "mcc": float(mcc_val),
        }
        print(f"  {gene}: n={len(gene_df)}, AUC={roc_auc_val:.3f}, MCC={mcc_val:.3f}", flush=True)

    return {"expert_results": results, "expert_df": expert_df}


def plot_dms_correlation(dms_data, matched_df):
    """Plot SteppeDNA predictions vs DMS experimental scores."""
    dms_results = dms_data.get("dms_results", {})
    if not dms_results:
        print("  Skipping DMS correlation plots (no data)", flush=True)
        return

    dms_df = dms_data["dms_df"]
    genes_with_data = [g for g in dms_results if dms_results[g]["n_variants"] >= 10]
    n_genes = len(genes_with_data)

    if n_genes == 0:
        return

    fig, axes = plt.subplots(1, n_genes, figsize=(6 * n_genes, 5))
    if n_genes == 1:
        axes = [axes]

    gene_colors = {
        "BRCA1": "#E63946", "BRCA2": "#457B9D",
        "PALB2": "#2A9D8F", "RAD51C": "#E9C46A", "RAD51D": "#F4A261",
    }

    for ax, gene in zip(axes, genes_with_data):
        gene_df = dms_df[dms_df["gene"] == gene]
        valid = gene_df["dms_score"].notna()
        gene_df = gene_df[valid]

        color = gene_colors.get(gene, "#6260FF")
        r_val = dms_results[gene]["spearman_r"]
        p_val = dms_results[gene]["spearman_p"]

        ax.scatter(
            gene_df["dms_score"], gene_df["steppedna_prob"],
            alpha=0.5, s=20, color=color, edgecolors="none"
        )

        # Add regression line
        z = np.polyfit(gene_df["dms_score"], gene_df["steppedna_prob"], 1)
        p = np.poly1d(z)
        x_range = np.linspace(gene_df["dms_score"].min(), gene_df["dms_score"].max(), 100)
        ax.plot(x_range, p(x_range), "--", color="gray", alpha=0.7)

        ax.set_xlabel("DMS Fitness Score", fontsize=12)
        ax.set_ylabel("SteppeDNA P(pathogenic)", fontsize=12)
        ax.set_title(f"{gene}\n(n={len(gene_df)}, "
                     f"Spearman r={r_val:.3f}, p={p_val:.1e})", fontsize=11)
        ax.set_ylim(-0.05, 1.05)

    plt.suptitle("SteppeDNA vs DMS Experimental Fitness Scores", fontsize=14, y=1.02)
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(f"visual_proofs/8_Benchmark_DMS_Correlation.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved: visual_proofs/8_Benchmark_DMS_Correlation.png/pdf", flush=True)


def plot_classification_results(expert_data, dms_data, matched_df, threshold):
    """Plot benchmark classification performance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ─── Panel A: ROC curves by benchmark source ──────────────────────
    ax = axes[0]

    # Expert panel ROC
    expert_results = expert_data.get("expert_results", {})
    expert_df = expert_data.get("expert_df")
    if expert_df is not None and len(expert_df) > 10 and expert_df["label"].nunique() > 1:
        expert_labels = expert_df["label"].astype(int)
        fpr, tpr, _ = roc_curve(expert_labels, expert_df["steppedna_prob"])
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="#E63946", lw=2,
                label=f"Expert Panel (AUC={auc_val:.3f}, n={len(expert_df)})")

    # DMS binary ROC (per gene)
    dms_results = dms_data.get("dms_results", {})
    dms_df = dms_data.get("dms_df")
    dms_colors = {"BRCA1": "#457B9D", "BRCA2": "#2A9D8F"}
    if dms_df is not None:
        for gene in sorted(dms_results.keys()):
            if dms_results[gene].get("roc_auc") is None:
                continue
            gene_df = dms_df[dms_df["gene"] == gene]
            valid_bin = gene_df["dms_score_bin"].notna()
            gene_df = gene_df[valid_bin]
            if len(gene_df) < 10:
                continue
            dms_labels = (1 - gene_df["dms_score_bin"]).astype(int)
            if dms_labels.nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(dms_labels, gene_df["steppedna_prob"])
            auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=dms_colors.get(gene, "#9B59B6"), lw=2,
                    label=f"DMS {gene} (AUC={auc_val:.3f}, n={len(gene_df)})")

    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("SteppeDNA on Gold-Standard Benchmark", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)

    # ─── Panel B: Summary bar chart ───────────────────────────────────
    ax = axes[1]

    categories = []
    auc_values = []
    colors = []
    n_values = []

    if "overall" in expert_results:
        categories.append("Expert\nPanel")
        auc_values.append(expert_results["overall"]["roc_auc"])
        colors.append("#E63946")
        n_values.append(expert_results["overall"]["n_variants"])

    for gene in ["BRCA1", "BRCA2"]:
        if gene in dms_results and dms_results[gene].get("roc_auc") is not None:
            categories.append(f"DMS\n{gene}")
            auc_values.append(dms_results[gene]["roc_auc"])
            colors.append(dms_colors.get(gene, "#9B59B6"))
            n_values.append(dms_results[gene]["n_variants"])

    if categories:
        bars = ax.bar(categories, auc_values, color=colors, edgecolor="white", width=0.5)
        for bar, val, n in zip(bars, auc_values, n_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"AUC={val:.3f}\n(n={n})", ha="center", va="bottom", fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("ROC-AUC", fontsize=12)
        ax.set_title("Performance by Benchmark Source", fontsize=13)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, label="Random")
    else:
        ax.text(0.5, 0.5, "No benchmark data\navailable for evaluation",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(f"visual_proofs/8_Benchmark_Classification.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved: visual_proofs/8_Benchmark_Classification.png/pdf", flush=True)


def main():
    print("=" * 65, flush=True)
    print("  SteppeDNA Gold-Standard Benchmark Evaluation", flush=True)
    print("=" * 65, flush=True)

    # Check benchmark exists
    if not os.path.exists(BENCHMARK_FILE):
        print(f"\nERROR: Benchmark file not found: {BENCHMARK_FILE}", flush=True)
        print("Run first: python data_pipelines/build_gold_standard_benchmark.py", flush=True)
        sys.exit(1)

    benchmark_df = pd.read_csv(BENCHMARK_FILE)
    print(f"\nLoaded benchmark: {len(benchmark_df)} variants", flush=True)
    print(f"  Sources: {dict(benchmark_df['source'].value_counts())}", flush=True)
    print(f"  Genes:   {dict(benchmark_df['gene'].value_counts())}", flush=True)

    # Load model and generate predictions
    test_df, threshold = load_model_and_predict()

    # Match benchmark to test predictions
    matched_df, unmatched_df = match_benchmark_to_predictions(benchmark_df, test_df)

    if len(matched_df) == 0:
        print("\nNo benchmark variants matched to test set. Cannot evaluate.", flush=True)
        sys.exit(1)

    # Evaluate DMS
    print("\n[3] Evaluating against benchmark sources...", flush=True)
    print("-" * 50, flush=True)
    dms_data = evaluate_dms(matched_df)
    expert_data = evaluate_expert(matched_df)

    # Generate plots
    print("\n[4] Generating benchmark figures...", flush=True)
    print("-" * 50, flush=True)
    plot_dms_correlation(dms_data, matched_df)
    plot_classification_results(expert_data, dms_data, matched_df, threshold)

    # Save results
    results = {
        "benchmark_size": len(benchmark_df),
        "matched_to_test": len(matched_df),
        "unmatched": len(unmatched_df),
        "dms": dms_data.get("dms_results", {}),
        "expert": expert_data.get("expert_results", {}),
        "novel_variants_available": int(
            (~benchmark_df["in_training"] & ~benchmark_df["in_test"]).sum()
        ),
    }

    # Remove non-serializable data
    for key in list(results.get("dms", {}).keys()):
        results["dms"][key] = {
            k: v for k, v in results["dms"][key].items()
            if not isinstance(v, (np.ndarray, pd.Series))
        }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'=' * 65}", flush=True)
    print("BENCHMARK EVALUATION SUMMARY", flush=True)
    print(f"{'=' * 65}", flush=True)
    print(f"  Benchmark variants:    {len(benchmark_df)}", flush=True)
    print(f"  Matched to test set:   {len(matched_df)}", flush=True)
    print(f"  Novel (future eval):   {results['novel_variants_available']}", flush=True)

    if dms_data.get("dms_results"):
        print(f"\n  DMS Correlation:", flush=True)
        for gene, r in dms_data["dms_results"].items():
            auc_str = f", AUC={r['roc_auc']:.3f}" if r.get("roc_auc") else ""
            print(f"    {gene}: Spearman r={r['spearman_r']:.3f}{auc_str} "
                  f"(n={r['n_variants']})", flush=True)

    if expert_data.get("expert_results"):
        print(f"\n  Expert Panel:", flush=True)
        for key, r in expert_data["expert_results"].items():
            print(f"    {key}: AUC={r['roc_auc']:.3f}, MCC={r['mcc']:.3f} "
                  f"(n={r['n_variants']})", flush=True)

    print(f"\nResults saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
