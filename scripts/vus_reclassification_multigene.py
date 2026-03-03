"""
SteppeDNA: Multi-Gene VUS Reclassification Analysis
=====================================================
Fetches all missense VUS from ClinVar for ALL 5 HR genes (BRCA1, BRCA2,
PALB2, RAD51C, RAD51D) and runs them through the universal ensemble model.

Uses the proper engineer_features() pipeline for consistent feature generation.

Outputs:
  data/vus_predictions_multigene.csv       (variant-level results)
  visual_proofs/12_VUS_Multigene.pdf       (per-gene breakdown figure)

Run:  python scripts/vus_reclassification_multigene.py
"""

import os
import sys
import re
import time
import json
import pickle
import requests
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.feature_engineering import engineer_features

DATA_DIR = "data"
RANDOM_STATE = 42
os.makedirs("visual_proofs", exist_ok=True)

HIGH_PATH_THRESH   = 0.80
LIKELY_PATH_THRESH = 0.60
LIKELY_BEN_THRESH  = 0.20
HIGH_BEN_THRESH    = 0.10

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

GENES = ["BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D"]

AA3_TO_1 = {
    "Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E",
    "Gly":"G","His":"H","Ile":"I","Leu":"L","Lys":"K","Met":"M","Phe":"F",
    "Pro":"P","Ser":"S","Thr":"T","Trp":"W","Tyr":"Y","Val":"V","Ter":"*",
}

GENE_CDS_LENGTH = {
    "BRCA1": 5592, "BRCA2": 10257, "PALB2": 3561,
    "RAD51C": 1131, "RAD51D": 987,
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load Model
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65, flush=True)
print("  SteppeDNA: Multi-Gene VUS Reclassification", flush=True)
print("=" * 65, flush=True)

print("\n[1] Loading universal ensemble model...", flush=True)

with open(f"{DATA_DIR}/universal_feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)
with open(f"{DATA_DIR}/universal_scaler_ensemble.pkl", "rb") as f:
    scaler = pickle.load(f)
with open(f"{DATA_DIR}/universal_calibrator_ensemble.pkl", "rb") as f:
    calibrator = pickle.load(f)
with open(f"{DATA_DIR}/universal_threshold_ensemble.pkl", "rb") as f:
    threshold = pickle.load(f)

xgb_model = xgb.Booster()
xgb_model.load_model(f"{DATA_DIR}/universal_xgboost_final.json")

nn_model = None
try:
    import tensorflow as tf
    nn_model = tf.keras.models.load_model(f"{DATA_DIR}/universal_nn.h5", compile=False)
    print(f"  Ensemble loaded: XGBoost + MLP, {len(feature_names)} features", flush=True)
except Exception:
    print(f"  XGBoost-only mode (TensorFlow not available)", flush=True)

# Load per-gene lookup data
def load_gene_data(gene):
    """Load lookup tables for a gene."""
    data = {}

    def _load_pkl(fname):
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return {}

    # Try gene-specific files first, fall back to global
    for key, gene_file, global_file in [
        ("phylop", f"{gene.lower()}_phylop_scores.pkl", "phylop_scores.pkl"),
        ("mave", f"{gene.lower()}_mave_scores.pkl", "mave_scores.pkl"),
        ("am", f"{gene.lower()}_alphamissense_scores.pkl", "alphamissense_scores.pkl"),
        ("structural", f"{gene.lower()}_structural_features.pkl", "structural_features.pkl"),
        ("gnomad", f"{gene.lower()}_gnomad_frequencies.pkl", "gnomad_frequencies.pkl"),
        ("spliceai", f"{gene.lower()}_spliceai_scores.pkl", "spliceai_scores.pkl"),
        ("esm2", f"{gene.lower()}_esm2_embeddings.pkl", "esm2_embeddings.pkl"),
    ]:
        gene_path = os.path.join(DATA_DIR, gene_file)
        global_path = os.path.join(DATA_DIR, global_file)
        if os.path.exists(gene_path):
            raw = _load_pkl(gene_file)
        elif os.path.exists(global_path):
            raw = _load_pkl(global_file)
        else:
            raw = {}

        # ESM-2 pickle has {embeddings: {...}, pca_model: ...}
        # engineer_features expects just the embeddings sub-dict
        if key == "esm2" and isinstance(raw, dict) and "embeddings" in raw:
            raw = raw["embeddings"]

        data[key] = raw

    return data


def predict_ensemble(X_scaled):
    """Run ensemble prediction (XGBoost 60% + MLP 40%)."""
    dmat = xgb.DMatrix(X_scaled, feature_names=feature_names)
    xgb_prob = xgb_model.predict(dmat)

    if nn_model is not None:
        nn_prob = nn_model.predict(X_scaled, verbose=0).flatten()
        raw_prob = 0.6 * xgb_prob + 0.4 * nn_prob
    else:
        raw_prob = xgb_prob

    return raw_prob


# ─────────────────────────────────────────────────────────────────────────────
# 2. ClinVar API
# ─────────────────────────────────────────────────────────────────────────────
def esearch(query, db="clinvar", retmax=10000):
    url = f"{NCBI_BASE}/esearch.fcgi"
    params = {"db": db, "term": query, "retmax": retmax, "retmode": "json"}
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json().get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        print(f"  [WARN] esearch failed: {e}", flush=True)
        return []


def esummary_batch(ids, batch_size=50):
    summaries = {}
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i+batch_size]
        url = f"{NCBI_BASE}/esummary.fcgi"
        params = {"db": "clinvar", "id": ",".join(batch), "retmode": "json"}
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            result = resp.json().get("result", {})
            for uid in batch:
                if uid in result:
                    summaries[uid] = result[uid]
        except Exception as e:
            print(f"  [WARN] esummary batch failed: {e}", flush=True)
        time.sleep(0.35)
        if (i // batch_size + 1) % 20 == 0:
            n_batches = (len(ids) // batch_size) + 1
            print(f"    Summary batch {i//batch_size+1}/{n_batches}...", flush=True)
    return summaries


def parse_hgvs_pro(s):
    m = re.search(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|Ter|\*)", s or "")
    if not m:
        return None, None, None
    ref3 = m.group(1)
    pos = int(m.group(2))
    alt3 = m.group(3).replace("*", "Ter")
    return ref3, pos, alt3


# ─────────────────────────────────────────────────────────────────────────────
# 3. Load Training Keys for Deduplication
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Loading training set for deduplication...", flush=True)

training_keys_by_gene = {}
for gene in GENES:
    # Try unified dataset
    unified_path = os.path.join(DATA_DIR, gene.lower(), f"{gene.lower()}_missense_dataset_unified.csv")
    if not os.path.exists(unified_path):
        # Try older format
        unified_path = os.path.join(DATA_DIR, f"{gene.lower()}_missense_dataset_unified.csv")
    if not os.path.exists(unified_path):
        unified_path = os.path.join(f"{gene.lower()}_missense_dataset_2.csv")

    keys = set()
    if os.path.exists(unified_path):
        try:
            tdf = pd.read_csv(unified_path)
            for _, row in tdf.iterrows():
                aa_ref = str(row.get("AA_ref", ""))
                aa_pos = row.get("AA_pos", 0)
                aa_alt = str(row.get("AA_alt", ""))
                if aa_ref and aa_pos and aa_alt:
                    keys.add(f"{aa_ref}{int(aa_pos)}{aa_alt}")
        except Exception:
            pass

    training_keys_by_gene[gene] = keys
    print(f"  {gene}: {len(keys)} training variants loaded", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Fetch & Score VUS for Each Gene
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Fetching VUS from ClinVar and scoring...", flush=True)

all_predictions = []

for gene in GENES:
    print(f"\n{'-' * 50}", flush=True)
    print(f"  {gene}", flush=True)
    print(f"{'-' * 50}", flush=True)

    # Fetch VUS IDs
    query = (
        f'{gene}[gene] AND "uncertain significance"[clinical significance] '
        f'AND "missense variant"[molecular consequence] AND "homo sapiens"[organism]'
    )
    vus_ids = esearch(query)
    print(f"  ClinVar VUS IDs: {len(vus_ids)}", flush=True)

    if not vus_ids:
        print(f"  [SKIP] No VUS found", flush=True)
        continue

    # Get summaries
    summaries = esummary_batch(vus_ids)
    print(f"  Summaries retrieved: {len(summaries)}", flush=True)

    # Parse HGVS
    parsed = []
    for uid, rec in summaries.items():
        title = rec.get("title", "")
        aa_ref, aa_pos, aa_alt = parse_hgvs_pro(title)
        if aa_ref and aa_pos and aa_alt and aa_alt != "Ter" and aa_ref != aa_alt:
            parsed.append({
                "clinvar_id": uid,
                "title": title,
                "aa_ref": aa_ref,
                "aa_pos": aa_pos,
                "aa_alt": aa_alt,
                "variant_key": f"{aa_ref}{aa_pos}{aa_alt}",
                "gene": gene,
            })
    print(f"  Parsed missense VUS: {len(parsed)}", flush=True)

    if not parsed:
        continue

    # Load gene-specific lookup data
    gene_data = load_gene_data(gene)
    cds_length = GENE_CDS_LENGTH.get(gene, 10000)
    training_keys = training_keys_by_gene.get(gene, set())

    # Build ALL mutations into a single DataFrame for batch processing
    print(f"  Building feature matrix for {len(parsed)} variants...", flush=True)
    rows = []
    for vus in parsed:
        cdna_pos_est = vus["aa_pos"] * 3 - 1
        rows.append({
            "cDNA_pos": cdna_pos_est,
            "AA_pos": vus["aa_pos"],
            "AA_ref": vus["aa_ref"],
            "AA_alt": vus["aa_alt"],
            "Ref_nt": "N",
            "Alt_nt": "N",
            "Mutation": "N>N",
        })

    mut_df = pd.DataFrame(rows)

    try:
        features_df = engineer_features(
            mut_df,
            phylop_scores=gene_data.get("phylop"),
            mave_data=gene_data.get("mave"),
            am_data=gene_data.get("am"),
            structural_data=gene_data.get("structural"),
            gnomad_data=gene_data.get("gnomad"),
            spliceai_data=gene_data.get("spliceai"),
            esm2_data=gene_data.get("esm2"),
            gene_name=gene,
        )
    except Exception as e:
        print(f"  [ERROR] engineer_features failed: {e}", flush=True)
        continue

    # Align to model's feature order
    feature_matrix = np.zeros((len(features_df), len(feature_names)))
    for j, f in enumerate(feature_names):
        if f in features_df.columns:
            feature_matrix[:, j] = features_df[f].values

    # Scale
    scaled_matrix = scaler.transform(feature_matrix)

    # Predict (batch)
    print(f"  Running ensemble prediction...", flush=True)
    dmat = xgb.DMatrix(scaled_matrix, feature_names=feature_names)
    xgb_probs = xgb_model.predict(dmat)

    if nn_model is not None:
        nn_probs = nn_model.predict(scaled_matrix, verbose=0).flatten()
        raw_probs = 0.6 * xgb_probs + 0.4 * nn_probs
    else:
        raw_probs = xgb_probs

    # Calibrate (batch)
    cal_probs = calibrator.predict(raw_probs)

    # Classify
    scored = 0
    for i, vus in enumerate(parsed):
        prob = float(cal_probs[i])

        if prob >= HIGH_PATH_THRESH:
            classification = "Likely Pathogenic"
        elif prob >= LIKELY_PATH_THRESH:
            classification = "Possibly Pathogenic"
        elif prob < HIGH_BEN_THRESH:
            classification = "Likely Benign"
        elif prob < LIKELY_BEN_THRESH:
            classification = "Possibly Benign"
        else:
            classification = "Uncertain (Remains VUS)"

        all_predictions.append({
            "gene": gene,
            "clinvar_id": vus["clinvar_id"],
            "variant": f"{gene}:p.{vus['aa_ref']}{vus['aa_pos']}{vus['aa_alt']}",
            "variant_key": vus["variant_key"],
            "aa_ref": vus["aa_ref"],
            "aa_pos": vus["aa_pos"],
            "aa_alt": vus["aa_alt"],
            "steppedna_prob": prob,
            "classification": classification,
            "novel": vus["variant_key"] not in training_keys,
        })
        scored += 1

    print(f"  Scored: {scored}/{len(parsed)} ({100*scored/len(parsed):.1f}%)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Results
# ─────────────────────────────────────────────────────────────────────────────
pred_df = pd.DataFrame(all_predictions)
if len(pred_df) == 0:
    print("\n[ERROR] No VUS scored. Check internet connection.", flush=True)
    sys.exit(1)

pred_df.sort_values(["gene", "steppedna_prob"], ascending=[True, False], inplace=True)
pred_df.to_csv(os.path.join(DATA_DIR, "vus_predictions_multigene.csv"), index=False)

print(f"\n{'=' * 65}", flush=True)
print("  MULTI-GENE VUS RECLASSIFICATION REPORT", flush=True)
print(f"{'=' * 65}", flush=True)

n_total = len(pred_df)
n_novel = pred_df["novel"].sum()
counts = pred_df["classification"].value_counts()
n_reclassified = n_total - counts.get("Uncertain (Remains VUS)", 0)

print(f"\n  Total VUS analyzed:    {n_total}", flush=True)
print(f"  Novel (not in train):  {n_novel} ({100*n_novel/n_total:.1f}%)", flush=True)
print(f"  Reclassified:          {n_reclassified} ({100*n_reclassified/n_total:.1f}%)", flush=True)

print(f"\n  Classification breakdown:", flush=True)
for cls in ["Likely Pathogenic", "Possibly Pathogenic",
            "Uncertain (Remains VUS)", "Possibly Benign", "Likely Benign"]:
    c = counts.get(cls, 0)
    print(f"    {cls:<30}: {c:4d}  ({100*c/n_total:.1f}%)", flush=True)

print(f"\n  Per-Gene Summary:", flush=True)
print(f"  {'Gene':<8} {'VUS':>5} {'Novel':>6} {'Reclass':>8} {'%':>6} {'Med.Prob':>9}", flush=True)
print(f"  {'-'*45}", flush=True)
for gene in GENES:
    gdf = pred_df[pred_df["gene"] == gene]
    if len(gdf) == 0:
        continue
    g_novel = gdf["novel"].sum()
    g_reclass = len(gdf) - (gdf["classification"] == "Uncertain (Remains VUS)").sum()
    g_pct = 100 * g_reclass / len(gdf)
    g_med = gdf["steppedna_prob"].median()
    print(f"  {gene:<8} {len(gdf):>5} {g_novel:>6} {g_reclass:>8} {g_pct:>5.1f}% {g_med:>8.3f}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Figure
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Generating figure...", flush=True)

BRAND  = "#6260FF"
RED    = "#FF3B30"
GREEN  = "#34C759"
GRAY   = "#8E8E93"
ORANGE = "#FF9500"
BLUE   = "#007AFF"

gene_colors = {"BRCA1": "#FF6B6B", "BRCA2": BRAND, "PALB2": "#4ECDC4",
               "RAD51C": ORANGE, "RAD51D": "#A78BFA"}

fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

# Panel A: Overall probability histogram
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(pred_df["steppedna_prob"], bins=40, color=BRAND, alpha=0.8, edgecolor="white")
ax1.axvline(HIGH_PATH_THRESH, color=RED, ls="--", lw=1.5, label=f"Likely Path ({HIGH_PATH_THRESH})")
ax1.axvline(HIGH_BEN_THRESH, color=GREEN, ls="--", lw=1.5, label=f"Likely Ben ({HIGH_BEN_THRESH})")
ax1.set_xlabel("SteppeDNA Probability", fontweight="bold")
ax1.set_ylabel("Count", fontweight="bold")
ax1.set_title(f"A. Probability Distribution\n(n={n_total} VUS across 5 genes)", fontweight="bold")
ax1.legend(fontsize=7)

# Panel B: Per-gene VUS counts
ax2 = fig.add_subplot(gs[0, 1])
gene_counts = pred_df.groupby("gene").size()
reclass_counts = pred_df[pred_df["classification"] != "Uncertain (Remains VUS)"].groupby("gene").size()
x_pos = range(len(GENES))
bars_total = [gene_counts.get(g, 0) for g in GENES]
bars_reclass = [reclass_counts.get(g, 0) for g in GENES]
ax2.bar(x_pos, bars_total, color=[gene_colors.get(g, GRAY) for g in GENES], alpha=0.4, label="Total VUS")
ax2.bar(x_pos, bars_reclass, color=[gene_colors.get(g, GRAY) for g in GENES], alpha=0.9, label="Reclassified")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(GENES, rotation=15)
ax2.set_ylabel("Number of VUS", fontweight="bold")
ax2.set_title("B. VUS per Gene\n(reclassified vs total)", fontweight="bold")
ax2.legend(fontsize=8)

# Panel C: Classification pie chart
ax3 = fig.add_subplot(gs[0, 2])
cls_order = ["Likely Pathogenic", "Possibly Pathogenic", "Uncertain (Remains VUS)",
             "Possibly Benign", "Likely Benign"]
cls_colors = [RED, ORANGE, GRAY, "#5AC8FA", GREEN]
pie_sizes = [counts.get(c, 0) for c in cls_order]
pie_labels = [f"{c}\n({counts.get(c,0)})" for c in cls_order]
non_zero = [(s, l, col) for s, l, col in zip(pie_sizes, pie_labels, cls_colors) if s > 0]
if non_zero:
    ax3.pie([x[0] for x in non_zero], labels=[x[1] for x in non_zero],
            colors=[x[2] for x in non_zero], startangle=90,
            wedgeprops=dict(width=0.5, edgecolor="white"), textprops={"fontsize": 7})
ax3.set_title(f"C. Classification Outcome\n({n_reclassified}/{n_total} actionable)", fontweight="bold")

# Panel D: Per-gene probability distributions
ax4 = fig.add_subplot(gs[1, 0])
for gene in GENES:
    gdf = pred_df[pred_df["gene"] == gene]
    if len(gdf) > 0:
        ax4.hist(gdf["steppedna_prob"], bins=30, alpha=0.5,
                color=gene_colors.get(gene, GRAY), label=f"{gene} (n={len(gdf)})", density=True)
ax4.set_xlabel("SteppeDNA Probability", fontweight="bold")
ax4.set_ylabel("Density", fontweight="bold")
ax4.set_title("D. Score Distribution by Gene", fontweight="bold")
ax4.legend(fontsize=7)

# Panel E: Novel vs Seen
ax5 = fig.add_subplot(gs[1, 1])
novel_probs = pred_df[pred_df["novel"] == True]["steppedna_prob"]
seen_probs = pred_df[pred_df["novel"] == False]["steppedna_prob"]
if len(seen_probs) > 0:
    ax5.hist(seen_probs, bins=30, alpha=0.5, color=GRAY, label=f"In training (n={len(seen_probs)})", density=True)
if len(novel_probs) > 0:
    ax5.hist(novel_probs, bins=30, alpha=0.7, color=BRAND, label=f"Novel (n={len(novel_probs)})", density=True)
ax5.set_xlabel("SteppeDNA Probability", fontweight="bold")
ax5.set_ylabel("Density", fontweight="bold")
ax5.set_title("E. Novel vs Seen VUS", fontweight="bold")
ax5.legend(fontsize=8)

# Panel F: Top reclassified variants
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")
top_path = pred_df[pred_df["classification"] == "Likely Pathogenic"].head(8)
top_ben = pred_df[pred_df["classification"] == "Likely Benign"].tail(5)

text_lines = ["Top Likely Pathogenic VUS:"]
for _, row in top_path.iterrows():
    text_lines.append(f"  {row['variant']}  p={row['steppedna_prob']:.3f}")
text_lines.append("")
text_lines.append("Top Likely Benign VUS:")
for _, row in top_ben.iterrows():
    text_lines.append(f"  {row['variant']}  p={row['steppedna_prob']:.3f}")

ax6.text(0.05, 0.95, "\n".join(text_lines), transform=ax6.transAxes,
         fontsize=8, verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0"))
ax6.set_title("F. Top Reclassified Variants", fontweight="bold")

plt.suptitle(
    f"SteppeDNA: Multi-Gene VUS Reclassification  |  {n_total} ClinVar VUS across 5 HR genes\n"
    f"{n_reclassified} ({100*n_reclassified/n_total:.1f}%) reclassified  |  "
    f"{n_novel} ({100*n_novel/n_total:.1f}%) novel variants",
    fontsize=12, fontweight="bold", y=1.02
)

plt.savefig("visual_proofs/12_VUS_Multigene.pdf", bbox_inches="tight")
plt.savefig("visual_proofs/12_VUS_Multigene.png", bbox_inches="tight", dpi=150)
plt.close()

print("  Saved -> visual_proofs/12_VUS_Multigene.pdf", flush=True)
print("  Saved -> visual_proofs/12_VUS_Multigene.png", flush=True)
print("\nDone!", flush=True)
