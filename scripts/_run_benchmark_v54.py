# v5.4 benchmark: predict ALL variants via API TestClient, compare with v5.3.
import os, sys, json, time

os.environ["RATE_LIMIT"] = "999999"
os.environ["RATE_WINDOW"] = "60"

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, matthews_corrcoef, balanced_accuracy_score

benchmark_df = pd.read_csv("data/benchmark/gold_standard_benchmark.csv")
print(f"Benchmark loaded: {len(benchmark_df)} variants")

old_results = None
if os.path.exists("data/benchmark/benchmark_results.json"):
    with open("data/benchmark/benchmark_results.json") as f:
        old_results = json.load(f)

from fastapi.testclient import TestClient
from backend.main import app
client = TestClient(app)

AA_1TO3 = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
    "Q": "Gln", "E": "Glu", "G": "Gly", "H": "His", "I": "Ile",
    "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
    "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
}

predictions = []
errors = []
t0 = time.time()
for i, row in benchmark_df.iterrows():
    if i % 300 == 0:
        print(f"  Processing {i}/{len(benchmark_df)} ...", flush=True)
    gene = row["gene"]
    aa_pos = int(row["aa_pos"])
    aa_ref = str(row["aa_ref"])
    aa_alt = str(row["aa_alt"])
    ref3 = AA_1TO3.get(aa_ref, aa_ref)
    alt3 = AA_1TO3.get(aa_alt, aa_alt)
    cdna_pos = (aa_pos - 1) * 3 + 1
    payload = {"gene_name": gene, "cDNA_pos": cdna_pos, "AA_ref": ref3, "AA_alt": alt3, "AA_pos": aa_pos, "Mutation": "Unknown"}
    try:
        resp = client.post("/predict", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            prob = data.get("probability")
            if prob is not None:
                predictions.append({"gene": gene, "aa_pos": aa_pos, "aa_ref": aa_ref, "aa_alt": aa_alt, "source": row["source"], "label": row["label"], "dms_score": row.get("dms_score"), "dms_score_bin": row.get("dms_score_bin"), "in_training": bool(row.get("in_training", False)), "in_test": bool(row.get("in_test", False)), "steppedna_prob": prob, "risk_tier": data.get("risk_tier")})
            else:
                errors.append({"idx": i, "gene": gene, "reason": "no prob"})
        else:
            errors.append({"idx": i, "gene": gene, "reason": f"status {resp.status_code}"})
    except Exception as e:
        errors.append({"idx": i, "gene": gene, "reason": str(e)[:100]})

elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s. Predicted: {len(predictions)}, Errors: {len(errors)}")
if errors:
    for e in errors[:5]:
        print(f"  ERR: {e}")
if not predictions:
    sys.exit(1)

pred_df = pd.DataFrame(predictions)
pred_df.to_csv("data/benchmark/benchmark_v54_predictions.csv", index=False)
results = {"total_predicted": len(predictions), "errors": len(errors)}

# DMS evaluation
dms_df = pred_df[pred_df["source"] == "ProteinGym_DMS"].copy()
print("=== DMS Evaluation ===")
results["dms"] = {}
for gene in sorted(dms_df["gene"].unique()):
    gdf = dms_df[dms_df["gene"] == gene].copy()
    gdf = gdf[gdf["dms_score"].notna()]
    if len(gdf) < 10:
        continue
    r, p = stats.spearmanr(gdf["steppedna_prob"], gdf["dms_score"])
    gdf_b = gdf[gdf["dms_score_bin"].notna()].copy()
    auc_v = mcc_v = None
    if len(gdf_b) > 10 and gdf_b["dms_score_bin"].nunique() > 1:
        dl = (1 - gdf_b["dms_score_bin"]).astype(int)
        if dl.nunique() > 1:
            auc_v = roc_auc_score(dl, gdf_b["steppedna_prob"])
            mcc_v = matthews_corrcoef(dl, (gdf_b["steppedna_prob"] >= 0.5).astype(int))
    results["dms"][gene] = {"n": len(gdf), "spearman_r": round(float(r), 4), "roc_auc": round(float(auc_v), 4) if auc_v else None, "mcc": round(float(mcc_v), 4) if mcc_v else None}
    auc_str = f", AUC={auc_v:.4f}" if auc_v else ""
    mcc_str = f", MCC={mcc_v:.4f}" if mcc_v else ""
    print(f"  {gene}: n={len(gdf)}, r={r:.4f}" + auc_str + mcc_str)

# Expert panel
exp_df = pred_df[pred_df["source"] == "ClinVar_ExpertPanel"].copy()
exp_df = exp_df[exp_df["label"].notna()]
print("=== Expert Panel ===")
results["expert"] = {}
if len(exp_df) > 5 and exp_df["label"].nunique() > 1:
    lb = exp_df["label"].astype(int)
    av = roc_auc_score(lb, exp_df["steppedna_prob"])
    pb = (exp_df["steppedna_prob"] >= 0.5).astype(int)
    mv = matthews_corrcoef(lb, pb)
    bv = balanced_accuracy_score(lb, pb)
    results["expert"]["overall"] = {"n": len(exp_df), "roc_auc": round(float(av), 4), "mcc": round(float(mv), 4), "bal_acc": round(float(bv), 4)}
    print(f"  Overall: n={len(exp_df)}, AUC={av:.4f}, MCC={mv:.4f}, BalAcc={bv:.4f}")
    for gene in sorted(exp_df["gene"].unique()):
        gdf = exp_df[exp_df["gene"] == gene]
        if len(gdf) < 5 or gdf["label"].nunique() < 2:
            continue
        gl = gdf["label"].astype(int)
        ga = roc_auc_score(gl, gdf["steppedna_prob"])
        gm = matthews_corrcoef(gl, (gdf["steppedna_prob"] >= 0.5).astype(int))
        results["expert"][gene] = {"n": len(gdf), "roc_auc": round(float(ga), 4), "mcc": round(float(gm), 4)}
        print(f"  {gene}: n={len(gdf)}, AUC={ga:.4f}, MCC={gm:.4f}")

# Overlap
n_tr = int(pred_df["in_training"].sum())
n_te = int(pred_df["in_test"].sum())
n_nv = len(pred_df) - n_tr - n_te
results["overlap"] = {"training": n_tr, "test": n_te, "novel": n_nv}
print(f"Overlap: training={n_tr}, test={n_te}, novel={n_nv}")

# Novel-only
novel_df = pred_df[(~pred_df["in_training"]) & (~pred_df["in_test"])].copy()
nl = novel_df[novel_df["label"].notna()]
if len(nl) > 5 and nl["label"].nunique() > 1:
    na = roc_auc_score(nl["label"].astype(int), nl["steppedna_prob"])
    nm = matthews_corrcoef(nl["label"].astype(int), (nl["steppedna_prob"] >= 0.5).astype(int))
    results["novel_only"] = {"n": len(nl), "roc_auc": round(float(na), 4), "mcc": round(float(nm), 4)}
    print(f"  Novel-only: n={len(nl)}, AUC={na:.4f}, MCC={nm:.4f}")

# v5.3 comparison
if old_results:
    print("=== v5.3 vs v5.4 ===")
    od = old_results.get("dms", {}).get("BRCA1", {})
    nd = results.get("dms", {}).get("BRCA1", {})
    if od.get("roc_auc") and nd.get("roc_auc"):
        d = nd["roc_auc"] - od["roc_auc"]
        print(f"  DMS BRCA1 AUC: {od['roc_auc']:.4f} -> {nd['roc_auc']:.4f} ({d:+.4f})")
    oe = old_results.get("expert", {}).get("overall", {})
    ne = results.get("expert", {}).get("overall", {})
    if oe.get("roc_auc") and ne.get("roc_auc"):
        d = ne["roc_auc"] - oe["roc_auc"]
        print(f"  Expert Overall AUC: {oe['roc_auc']:.4f} -> {ne['roc_auc']:.4f} ({d:+.4f})")
    for gene in ["BRCA1", "BRCA2"]:
        og = old_results.get("expert", {}).get(gene, {})
        ng = results.get("expert", {}).get(gene, {})
        if og.get("roc_auc") and ng.get("roc_auc"):
            d = ng["roc_auc"] - og["roc_auc"]
            print(f"  Expert {gene} AUC: {og['roc_auc']:.4f} -> {ng['roc_auc']:.4f} ({d:+.4f})")

with open("data/benchmark/benchmark_v54_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("DONE - saved to data/benchmark/benchmark_v54_results.json")
