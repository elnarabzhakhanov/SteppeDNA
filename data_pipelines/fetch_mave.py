"""
Fetch MAVE (Multiplexed Assay of Variant Effect) functional scores for BRCA2.
Source: MaveDB — Hu C et al. 2024 (URN: urn:mavedb:00001224-a-1)
  "Functional analysis and clinical classification of 462 germline BRCA2
   missense variants affecting the DNA binding domain"
   Am J Hum Genet. 2024; 111:584-593.

Assay: Homology-directed repair (HDR) in brca2-deficient V-C8 cells.
Score scale: 1 (pathogenic standard, Asp2723His) to 5 (wild-type).
  Functionally abnormal: score < 1.49
  Functionally normal:  score > 2.50
  Intermediate:         1.49 – 2.50

Outputs: data/mave_scores.pkl  — dict {cDNA_pos: {'score': float, 'hgvs_p': str}}
         data/mave_scores.csv  — human-readable CSV
"""

import json, pickle, re, sys, io, csv
import urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

URN = "urn:mavedb:00001224-a-1"
API_URL = f"https://api.mavedb.org/api/v1/score-sets/{URN}/scores"

print("=" * 60)
print("  FETCHING MAVE HDR SCORES FOR BRCA2 (Hu C et al. 2024)")
print("=" * 60)

# ---------- 1) Download scores from MaveDB API ----------
print(f"\n  Fetching scores from: {API_URL}")
resp = urllib.request.urlopen(API_URL, timeout=60)
raw = resp.read().decode("utf-8")

# API returns CSV: accession,hgvs_nt,hgvs_splice,hgvs_pro,score
reader = csv.DictReader(io.StringIO(raw))
variants = list(reader)
print(f"  Downloaded {len(variants)} variant records")

# ---------- 2) Parse into usable format ----------
# Each record has: hgvs_nt (e.g. "c.7522G>C"), hgvs_pro (e.g. "p.Gly2508Arg"),
#                  score (float)

records = []
cdna_pattern = re.compile(r"c\.(\d+)")

for v in variants:
    hgvs_nt = v.get("hgvs_nt", "")
    hgvs_pro = v.get("hgvs_pro", "")
    score = v.get("score")

    if score is None or not hgvs_nt:
        continue

    m = cdna_pattern.search(hgvs_nt)
    if not m:
        continue

    cdna_pos = int(m.group(1))
    records.append({
        "cdna_pos": cdna_pos,
        "hgvs_nt": hgvs_nt,
        "hgvs_pro": hgvs_pro,
        "mave_score": float(score),
    })

print(f"  Parsed {len(records)} records with valid scores")

# ---------- 3) Build lookup by cDNA position ----------
# Some positions may have multiple variants (different AA changes).
# We store the AVERAGE score per cDNA position for use as a feature,
# but also keep all individual records for reference.

from collections import defaultdict
pos_scores = defaultdict(list)
for r in records:
    pos_scores[r["cdna_pos"]].append(r["mave_score"])

mave_by_pos = {}
for pos, scores in pos_scores.items():
    mave_by_pos[pos] = sum(scores) / len(scores)

print(f"  Unique cDNA positions with scores: {len(mave_by_pos)}")

# ---------- 4) Also build a lookup by variant (ref_aa, pos, alt_aa) ----------
# This gives us exact variant-level scores for the prediction API
aa_pattern = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})")
mave_by_variant = {}
for r in records:
    m = aa_pattern.search(r["hgvs_pro"])
    if m:
        ref_aa_3 = m.group(1)
        aa_pos = int(m.group(2))
        alt_aa_3 = m.group(3)
        key = f"{ref_aa_3}{aa_pos}{alt_aa_3}"
        mave_by_variant[key] = r["mave_score"]

print(f"  Unique variant-level lookups: {len(mave_by_variant)}")

# ---------- 5) Stats ----------
import numpy as np
scores_arr = np.array([r["mave_score"] for r in records])
print(f"\n  Score Distribution:")
print(f"    Min:    {scores_arr.min():.3f}")
print(f"    Max:    {scores_arr.max():.3f}")
print(f"    Mean:   {scores_arr.mean():.3f}")
print(f"    Median: {np.median(scores_arr):.3f}")
print(f"    Std:    {scores_arr.std():.3f}")

n_abnormal = (scores_arr < 1.49).sum()
n_normal = (scores_arr > 2.50).sum()
n_intermediate = len(scores_arr) - n_abnormal - n_normal
print(f"\n  Functional Classification:")
print(f"    Abnormal (score < 1.49): {n_abnormal}")
print(f"    Intermediate (1.49-2.50): {n_intermediate}")
print(f"    Normal (score > 2.50):   {n_normal}")

# ---------- 6) Save ----------
# Save position-average lookup (for feature engineering in training)
with open("data/mave_scores.pkl", "wb") as f:
    pickle.dump({"by_position": mave_by_pos, "by_variant": mave_by_variant}, f)

# Save CSV for human inspection
records.sort(key=lambda r: r["cdna_pos"])
with open("data/mave_scores.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["cdna_pos", "hgvs_nt", "hgvs_pro", "mave_score"])
    w.writeheader()
    w.writerows(records)

print(f"\n  Saved: data/mave_scores.pkl")
print(f"  Saved: data/mave_scores.csv ({len(records)} rows)")
print(f"\n{'='*60}")
print(f"  DONE")
print(f"{'='*60}")
