"""
SteppeDNA: Temporal Validation — Predicting Future ClinVar Classifications
===========================================================================
Splits the training dataset temporally: trains on variants classified before
a cutoff date, tests on variants classified after. This answers the judge
question: "Can your model predict future variant classifications?"

Approach:
  1. Match training variants to ClinVar dates via AlleleID
  2. Split: before cutoff → train, after cutoff → test
  3. Retrain XGBoost + MLP ensemble on temporal-train split
  4. Evaluate on temporal-test split
  5. Compare per-gene performance

Run from project root:
  python scripts/temporal_validation.py [--cutoff 2024]
"""

import os, sys, gzip, json, pickle, re, argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
XGB_WEIGHT = 0.6
NN_WEIGHT = 0.4
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
VISUAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "visual_proofs")
os.makedirs(VISUAL_DIR, exist_ok=True)

TARGET_GENES = {"BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D"}
CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"

parser = argparse.ArgumentParser()
parser.add_argument("--cutoff", type=int, default=2024, help="Year cutoff for temporal split")
args = parser.parse_args()
CUTOFF_YEAR = args.cutoff

print("=" * 70)
print(f"  SteppeDNA: Temporal Validation (cutoff: before {CUTOFF_YEAR})")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load training data with clinvar_ids
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/8] Loading training data with ClinVar IDs...")

# Load master training dataset (has features but no clinvar_id)
master_df = pd.read_csv(os.path.join(DATA_DIR, "master_training_dataset.csv"))
with open(os.path.join(DATA_DIR, "universal_feature_names.pkl"), "rb") as f:
    feature_names = pickle.load(f)
print(f"  Master dataset: {len(master_df)} rows, {len(feature_names)} features")

# Load per-gene unified datasets to get clinvar_ids
unified_parts = []
for gene in ["brca1", "brca2", "palb2", "rad51c", "rad51d"]:
    path = os.path.join(DATA_DIR, gene, f"{gene}_missense_dataset_unified.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        unified_parts.append(df)

unified_df = pd.concat(unified_parts, ignore_index=True)
print(f"  Unified datasets: {len(unified_df)} rows (ClinVar variants with IDs)")

# Build AlleleID lookup
allele_ids = {}
for _, row in unified_df.iterrows():
    cid = str(row.get("clinvar_id", "")).strip()
    if cid and cid != "nan":
        allele_ids[cid] = True

print(f"  ClinVar AlleleIDs: {len(allele_ids)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Download and parse ClinVar dates
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/8] Loading ClinVar variant_summary for date info...")

import requests, time

cache_path = os.path.join(DATA_DIR, "variant_summary.txt.gz")
if not os.path.exists(cache_path):
    print(f"  Downloading {CLINVAR_URL}...")
    resp = requests.get(CLINVAR_URL, stream=True, timeout=120)
    resp.raise_for_status()
    with open(cache_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    print("  Downloaded.")
else:
    print("  Using cached variant_summary.txt.gz")

# Parse dates: ClinVar ID -> year (match by both AlleleID and VariationID)
id_to_year = {}
with gzip.open(cache_path, "rt", encoding="utf-8", errors="replace") as f:
    header = f.readline().strip().split("\t")
    col_idx = {name: i for i, name in enumerate(header)}
    allele_col = col_idx.get("#AlleleID", 0)
    varid_col = col_idx.get("VariationID", 30)
    date_col = col_idx.get("LastEvaluated", 8)
    gene_col = col_idx.get("GeneSymbol", 4)

    for line in f:
        fields = line.strip().split("\t")
        if len(fields) <= max(varid_col, date_col, gene_col):
            continue
        gene = fields[gene_col].strip()
        if gene not in TARGET_GENES:
            continue

        aid = fields[allele_col].strip()
        vid = fields[varid_col].strip() if varid_col < len(fields) else ""

        # Match by either ID column (our clinvar_id could be either)
        matched_id = None
        if aid in allele_ids:
            matched_id = aid
        elif vid in allele_ids:
            matched_id = vid

        if matched_id is None:
            continue

        date_str = fields[date_col].strip()
        if not date_str or len(date_str) < 4:
            continue
        parts = date_str.replace(",", "").split()
        if len(parts) >= 3:
            try:
                year = int(parts[-1])
                if matched_id not in id_to_year:
                    id_to_year[matched_id] = year
            except ValueError:
                pass

print(f"  Matched dates for {len(id_to_year)} variants")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Assign dates to master dataset rows
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/8] Assigning dates to master dataset rows...")

# The master dataset was built by concatenating per-gene unified datasets
# plus gnomAD proxy-benign. We need to map master_df rows to clinvar_ids.
# Strategy: match by Gene + row order within each gene group

master_df["_date_year"] = np.nan
master_df["_clinvar_id"] = ""

for gene_upper in sorted(TARGET_GENES):
    gene_lower = gene_upper.lower()
    unified_gene = unified_df[unified_df["Gene"] == gene_upper].reset_index(drop=True)
    master_gene_mask = master_df["Gene"] == gene_upper
    master_gene_idx = master_df[master_gene_mask].index

    # Match by position within gene group
    n_unified = len(unified_gene)
    n_master = len(master_gene_idx)

    matched = 0
    for i in range(min(n_unified, n_master)):
        cid = str(unified_gene.loc[i, "clinvar_id"]).strip()
        if cid and cid != "nan":
            master_row = master_gene_idx[i]
            master_df.loc[master_row, "_clinvar_id"] = cid
            if cid in id_to_year:
                master_df.loc[master_row, "_date_year"] = id_to_year[cid]
                matched += 1

    print(f"  {gene_upper}: {matched}/{n_master} master rows dated")

total_dated = master_df["_date_year"].notna().sum()
total_undated = master_df["_date_year"].isna().sum()
print(f"\n  Total dated: {total_dated}, undated: {total_undated}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Temporal split
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[4/8] Splitting by temporal cutoff (before {CUTOFF_YEAR} / {CUTOFF_YEAR}+)...")

# Variants with dates before cutoff → train
# Variants with dates >= cutoff → test
# Undated variants → include in training (conservative choice)
train_mask = (master_df["_date_year"] < CUTOFF_YEAR) | master_df["_date_year"].isna()
test_mask = master_df["_date_year"] >= CUTOFF_YEAR

X_all = master_df[feature_names].values
y_all = master_df["Label"].values
genes_all = master_df["Gene"].values

X_train_temporal = X_all[train_mask]
y_train_temporal = y_all[train_mask]
genes_train_temporal = genes_all[train_mask]

X_test_temporal = X_all[test_mask]
y_test_temporal = y_all[test_mask]
genes_test_temporal = genes_all[test_mask]

print(f"  Temporal Train (before {CUTOFF_YEAR} + undated): {len(X_train_temporal)}")
print(f"  Temporal Test ({CUTOFF_YEAR}+): {len(X_test_temporal)}")
print(f"\n  Train class distribution: {(y_train_temporal==1).sum()}P / {(y_train_temporal==0).sum()}B")
print(f"  Test class distribution:  {(y_test_temporal==1).sum()}P / {(y_test_temporal==0).sum()}B")

print(f"\n  Per-gene temporal test set:")
for gene in sorted(TARGET_GENES):
    g_mask = genes_test_temporal == gene
    n = g_mask.sum()
    n_p = (y_test_temporal[g_mask] == 1).sum()
    n_b = (y_test_temporal[g_mask] == 0).sum()
    print(f"    {gene:8s}: {n:5d} ({n_p}P / {n_b}B)")

if len(X_test_temporal) < 50:
    print(f"\n  [ERROR] Only {len(X_test_temporal)} test variants. Too few for meaningful analysis.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Train temporal model
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[5/8] Training temporal model (XGBoost + MLP on pre-{CUTOFF_YEAR} data)...")

from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, matthews_corrcoef, balanced_accuracy_score,
    accuracy_score,
)
from imblearn.over_sampling import SMOTE

# Split temporal train into train + calibration
strata_temp = np.array([f"{g}_{l}" for g, l in zip(genes_train_temporal, y_train_temporal)])
# Handle edge case: some strata might have only 1 sample
from collections import Counter
strata_counts = Counter(strata_temp)
min_count = min(strata_counts.values())
if min_count < 2:
    # Remove strata with <2 samples from stratification
    valid_mask = np.array([strata_counts[s] >= 2 for s in strata_temp])
    print(f"  Dropping {(~valid_mask).sum()} samples with singleton strata")
    X_train_temporal = X_train_temporal[valid_mask]
    y_train_temporal = y_train_temporal[valid_mask]
    genes_train_temporal = genes_train_temporal[valid_mask]
    strata_temp = strata_temp[valid_mask]

X_tt, X_tc, y_tt, y_tc = train_test_split(
    X_train_temporal, y_train_temporal,
    test_size=0.2, random_state=RANDOM_STATE, stratify=strata_temp,
)

# Scale
scaler_t = StandardScaler()
X_tt_s = scaler_t.fit_transform(X_tt)
X_tc_s = scaler_t.transform(X_tc)
X_test_s = scaler_t.transform(X_test_temporal)

# SMOTE
smote = SMOTE(random_state=RANDOM_STATE)
X_tt_sm, y_tt_sm = smote.fit_resample(X_tt_s, y_tt)
print(f"  Temporal train: {len(X_tt)} -> {len(X_tt_sm)} after SMOTE")

# XGBoost
cw = float(np.sum(y_tt_sm == 0)) / np.sum(y_tt_sm == 1)
xgb_clf = xgb.XGBClassifier(
    n_estimators=400, max_depth=7, learning_rate=0.05,
    scale_pos_weight=cw, random_state=RANDOM_STATE,
    eval_metric='logloss', n_jobs=-1,
)
xgb_clf.fit(X_tt_sm, y_tt_sm, verbose=False)
print("  XGBoost trained.")

# MLP
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

nn = Sequential([
    Dense(128, activation='relu', input_dim=X_tt_sm.shape[1]),
    BatchNormalization(), Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(), Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])
nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
nn.fit(X_tt_sm, y_tt_sm, epochs=100, batch_size=32,
       validation_split=0.15, callbacks=[es], verbose=0)
print("  MLP trained.")

# Calibrate
nn_cal = nn.predict(X_tc_s, verbose=0).flatten()
xgb_cal = xgb_clf.predict_proba(X_tc_s)[:, 1]
blended_cal = XGB_WEIGHT * xgb_cal + NN_WEIGHT * nn_cal
calibrator_t = IsotonicRegression(out_of_bounds='clip')
calibrator_t.fit(blended_cal, y_tc)
print("  Calibrator fitted.")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Evaluate on temporal test set
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[6/8] Evaluating on temporal test set ({CUTOFF_YEAR}+ variants)...")

nn_test = nn.predict(X_test_s, verbose=0).flatten()
xgb_test = xgb_clf.predict_proba(X_test_s)[:, 1]
blended_test = XGB_WEIGHT * xgb_test + NN_WEIGHT * nn_test
y_probs = calibrator_t.predict(blended_test)

# Threshold from calibration set
precisions, recalls, thresholds = precision_recall_curve(y_tc, calibrator_t.predict(blended_cal))
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_threshold = float(thresholds[np.argmax(f1_scores[:-1])])
y_pred = (y_probs >= best_threshold).astype(int)

print(f"\n{'='*70}")
print(f"  TEMPORAL VALIDATION RESULTS")
print(f"  Train: variants classified before {CUTOFF_YEAR} ({len(X_train_temporal)})")
print(f"  Test:  variants classified {CUTOFF_YEAR}+ ({len(X_test_temporal)})")
print(f"{'='*70}")

metrics = {"cutoff_year": CUTOFF_YEAR, "n_train": len(X_train_temporal), "n_test": len(X_test_temporal)}

if len(set(y_test_temporal)) >= 2:
    tn, fp, fn, tp = confusion_matrix(y_test_temporal, y_pred).ravel()
    metrics.update({
        "roc_auc": round(roc_auc_score(y_test_temporal, y_probs), 4),
        "pr_auc": round(average_precision_score(y_test_temporal, y_probs), 4),
        "mcc": round(matthews_corrcoef(y_test_temporal, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_test_temporal, y_pred), 4),
        "accuracy": round(accuracy_score(y_test_temporal, y_pred), 4),
        "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0,
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0,
        "threshold": round(best_threshold, 4),
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    })

    print(f"  ROC-AUC:            {metrics['roc_auc']}")
    print(f"  PR-AUC:             {metrics['pr_auc']}")
    print(f"  MCC:                {metrics['mcc']}")
    print(f"  Balanced Accuracy:  {metrics['balanced_accuracy']}")
    print(f"  Accuracy:           {metrics['accuracy']}")
    print(f"  Sensitivity:        {metrics['sensitivity']}")
    print(f"  Specificity:        {metrics['specificity']}")
    print(f"  Threshold:          {metrics['threshold']}")
    print(f"  Confusion Matrix:   TP={tp} FP={fp} FN={fn} TN={tn}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Per-gene temporal performance
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n  Per-gene temporal performance:")
print(f"  {'Gene':8s} {'n':>5s}  {'ROC-AUC':>8s}  {'MCC':>6s}  {'BalAcc':>7s}  {'Sens':>6s}  {'Spec':>6s}")
print(f"  {'-'*52}")

metrics["per_gene"] = {}
for gene in sorted(TARGET_GENES):
    mask = genes_test_temporal == gene
    n = mask.sum()
    y_g = y_test_temporal[mask]
    p_g = y_probs[mask]

    if n < 2 or len(set(y_g)) < 2:
        print(f"  {gene:8s} {n:5d}  (insufficient)")
        metrics["per_gene"][gene] = {"n": int(n), "insufficient": True}
        continue

    gene_auc = roc_auc_score(y_g, p_g)
    gene_pred = (p_g >= best_threshold).astype(int)
    gene_mcc = matthews_corrcoef(y_g, gene_pred)
    gene_bal = balanced_accuracy_score(y_g, gene_pred)
    tn_g, fp_g, fn_g, tp_g = confusion_matrix(y_g, gene_pred).ravel()
    gene_sens = tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 0
    gene_spec = tn_g / (tn_g + fp_g) if (tn_g + fp_g) > 0 else 0

    print(f"  {gene:8s} {n:5d}  {gene_auc:8.4f}  {gene_mcc:6.3f}  {gene_bal:7.4f}  {gene_sens:6.3f}  {gene_spec:6.3f}")
    metrics["per_gene"][gene] = {
        "n": int(n), "roc_auc": round(gene_auc, 4), "mcc": round(gene_mcc, 4),
        "balanced_accuracy": round(gene_bal, 4), "sensitivity": round(gene_sens, 4),
        "specificity": round(gene_spec, 4),
    }

# ─────────────────────────────────────────────────────────────────────────────
# 8. Generate figure
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8/8] Generating temporal validation figure...")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    BRAND = "#6260FF"
    RED = "#FF3B30"
    GREEN = "#34C759"
    GRAY = "#8E8E93"

    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Panel A: Probability distribution by true label
    ax1 = fig.add_subplot(gs[0])
    path_probs = y_probs[y_test_temporal == 1]
    ben_probs = y_probs[y_test_temporal == 0]
    ax1.hist(ben_probs, bins=30, alpha=0.7, color=GREEN, label=f"True Benign (n={len(ben_probs)})", density=True)
    ax1.hist(path_probs, bins=30, alpha=0.7, color=RED, label=f"True Pathogenic (n={len(path_probs)})", density=True)
    ax1.axvline(best_threshold, color="black", ls="--", lw=1.5, label=f"Threshold ({best_threshold:.2f})")
    ax1.set_xlabel("Predicted Probability", fontweight="bold")
    ax1.set_ylabel("Density", fontweight="bold")
    ax1.set_title(f"A. Temporal Test Distribution\n({len(y_probs)} variants from {CUTOFF_YEAR}+)", fontweight="bold")
    ax1.legend(fontsize=8)

    # Panel B: Per-gene AUC comparison
    ax2 = fig.add_subplot(gs[1])
    gene_names = []
    temporal_aucs = []
    for gene in sorted(TARGET_GENES):
        gm = metrics["per_gene"].get(gene, {})
        if "roc_auc" in gm:
            gene_names.append(gene)
            temporal_aucs.append(gm["roc_auc"])
    if temporal_aucs:
        bars = ax2.barh(gene_names, temporal_aucs, color=BRAND, alpha=0.85, edgecolor="white")
        ax2.set_xlim(0, 1)
        ax2.set_xlabel("ROC-AUC", fontweight="bold")
        ax2.set_title(f"B. Per-Gene Temporal AUC\n(trained before {CUTOFF_YEAR})", fontweight="bold")
        for bar, val in zip(bars, temporal_aucs):
            ax2.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", fontweight="bold", fontsize=10)
        if "roc_auc" in metrics:
            ax2.axvline(metrics["roc_auc"], color=RED, ls="--", lw=1.5,
                        label=f"Overall: {metrics['roc_auc']:.3f}")
            ax2.legend(fontsize=9)

    # Panel C: ROC curve
    ax3 = fig.add_subplot(gs[2])
    if len(set(y_test_temporal)) >= 2:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test_temporal, y_probs)
        ax3.plot(fpr, tpr, color=BRAND, lw=2,
                 label=f"Temporal (AUC={metrics['roc_auc']:.3f})")
        ax3.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
        ax3.set_xlabel("False Positive Rate", fontweight="bold")
        ax3.set_ylabel("True Positive Rate", fontweight="bold")
        ax3.set_title(f"C. Temporal ROC Curve\n(future predictions)", fontweight="bold")
        ax3.legend(fontsize=9)
        ax3.set_xlim(-0.02, 1.02)
        ax3.set_ylim(-0.02, 1.02)

    auc_str = metrics.get("roc_auc", "N/A")
    plt.suptitle(
        f"SteppeDNA Temporal Validation: Trained before {CUTOFF_YEAR}, tested on {CUTOFF_YEAR}+ classifications\n"
        f"Train: {metrics['n_train']} variants | Test: {metrics['n_test']} variants | "
        f"AUC: {auc_str} | MCC: {metrics.get('mcc', 'N/A')}",
        fontsize=11, fontweight="bold", y=1.02,
    )

    fig_path = os.path.join(VISUAL_DIR, "temporal_validation.png")
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.savefig(fig_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {fig_path}")
except Exception as e:
    print(f"  [WARN] Figure generation failed: {e}")

# Save metrics
with open(os.path.join(DATA_DIR, "temporal_validation_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# Save per-variant results
results = pd.DataFrame({
    "gene": genes_test_temporal,
    "true_label": y_test_temporal,
    "pred_prob": y_probs,
    "pred_label": y_pred,
    "correct": (y_pred == y_test_temporal).astype(int),
})
results.to_csv(os.path.join(DATA_DIR, "temporal_validation_results.csv"), index=False)

print(f"\n  Results: {DATA_DIR}/temporal_validation_results.csv")
print(f"  Metrics: {DATA_DIR}/temporal_validation_metrics.json")
print("\n  [TEMPORAL VALIDATION COMPLETE]")
