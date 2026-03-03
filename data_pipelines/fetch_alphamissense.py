"""
Fetch AlphaMissense pathogenicity scores for BRCA2 missense variants.
Source: Google DeepMind AlphaMissense (Cheng et al. 2023, Science)
  "Accurate proteome-wide missense variant effect prediction with AlphaMissense"

Data: AlphaMissense_aa_substitutions.tsv.gz from Google Cloud Storage
      Streams the 1.2GB compressed file and filters only BRCA2 (P51587) lines.

Score: 0.0 (benign) to 1.0 (pathogenic)
  Benign:     score < 0.340
  Ambiguous:  0.340 <= score <= 0.564
  Pathogenic: score > 0.564

Outputs: data/alphamissense_scores.pkl — dict {variant_key: score}
         data/alphamissense_scores.csv — human-readable CSV
"""

import gzip, io, pickle, csv, sys, os, json, argparse
import urllib.request
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

URL = "https://storage.googleapis.com/dm_alphamissense/AlphaMissense_aa_substitutions.tsv.gz"

# ---------- Setup Arguments & Config ----------
parser = argparse.ArgumentParser(description="Fetch AlphaMissense scores for a given gene.")
parser.add_argument("--gene", type=str, default="BRCA2", help="Target gene name (e.g. BRCA2, PALB2)")
args = parser.parse_args()

gene_name = args.gene.upper()
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend", "gene_configs", f"{gene_name.lower()}.json")

if not os.path.exists(config_path):
    print(f"ERROR: Configuration file not found at {config_path}")
    sys.exit(1)

with open(config_path, "r") as f:
    gene_config = json.load(f)

TARGET_UNIPROT = gene_config.get("uniprot_id")
if not TARGET_UNIPROT:
    print(f"ERROR: 'uniprot_id' not found in {config_path}")
    sys.exit(1)


# 3-letter to 1-letter AA mapping
AA_1TO3 = {
    'A':'Ala','R':'Arg','N':'Asn','D':'Asp','C':'Cys',
    'E':'Glu','Q':'Gln','G':'Gly','H':'His','I':'Ile',
    'L':'Leu','K':'Lys','M':'Met','F':'Phe','P':'Pro',
    'S':'Ser','T':'Thr','W':'Trp','Y':'Tyr','V':'Val',
}

print("=" * 60)
print(f"  FETCHING ALPHAMISSENSE SCORES FOR {gene_name} ({TARGET_UNIPROT})")
print("=" * 60)

# ---------- Stream and filter ----------
print(f"\n  Streaming from: {URL}")
print(f"  Filtering for: {TARGET_UNIPROT}")
print(f"  (This streams ~1.2GB compressed, filtering in memory...)")

resp = urllib.request.urlopen(URL, timeout=120)

# Read in chunks and decompress
compressed_data = io.BytesIO()
chunk_count = 0
total_bytes = 0
while True:
    chunk = resp.read(1024 * 1024)  # 1MB chunks
    if not chunk:
        break
    compressed_data.write(chunk)
    total_bytes += len(chunk)
    chunk_count += 1
    if chunk_count % 100 == 0:
        print(f"    Downloaded {total_bytes / 1024 / 1024:.0f} MB...")

print(f"  Download complete: {total_bytes / 1024 / 1024:.1f} MB")

compressed_data.seek(0)

# Decompress and filter
print(f"  Decompressing and filtering for {gene_name}...")
records = []
with gzip.open(compressed_data, 'rt') as f:
    header = None
    for line in f:
        if line.startswith('#'):
            continue
        if header is None:
            header = line.strip().split('\t')
            continue
        if TARGET_UNIPROT not in line:
            continue
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            # Format: uniprot_id  protein_variant  am_pathogenicity  am_class
            uniprot = parts[0]
            variant_str = parts[1]  # e.g. "A1V" (1-letter codes)
            score = float(parts[2])
            am_class = parts[3]

            # Parse variant: e.g. "M1V" -> ref_1=M, pos=1, alt_1=V
            ref_1 = variant_str[0]
            alt_1 = variant_str[-1]
            pos = int(variant_str[1:-1])

            ref_3 = AA_1TO3.get(ref_1, ref_1)
            alt_3 = AA_1TO3.get(alt_1, alt_1)

            records.append({
                "variant": variant_str,
                "aa_pos": pos,
                "ref_aa": ref_3,
                "alt_aa": alt_3,
                "am_score": score,
                "am_class": am_class,
            })

print(f"  Found {len(records)} {gene_name} variants")

# ---------- Build lookup ----------
am_by_variant = {}
for r in records:
    key = f"{r['ref_aa']}{r['aa_pos']}{r['alt_aa']}"
    am_by_variant[key] = r["am_score"]

# Position-level average
from collections import defaultdict
pos_scores = defaultdict(list)
for r in records:
    pos_scores[r["aa_pos"]].append(r["am_score"])
am_by_position = {pos: sum(s)/len(s) for pos, s in pos_scores.items()}

print(f"  Unique variant-level lookups: {len(am_by_variant)}")
print(f"  Unique position-level lookups: {len(am_by_position)}")

# ---------- Stats ----------
scores_arr = np.array([r["am_score"] for r in records])
print(f"\n  Score Distribution:")
print(f"    Min:    {scores_arr.min():.3f}")
print(f"    Max:    {scores_arr.max():.3f}")
print(f"    Mean:   {scores_arr.mean():.3f}")
print(f"    Median: {np.median(scores_arr):.3f}")
print(f"    Std:    {scores_arr.std():.3f}")

n_benign = (scores_arr < 0.340).sum()
n_pathogenic = (scores_arr > 0.564).sum()
n_ambiguous = len(scores_arr) - n_benign - n_pathogenic
print(f"\n  AlphaMissense Classification:")
print(f"    Benign (score < 0.340):     {n_benign}")
print(f"    Ambiguous (0.340-0.564):    {n_ambiguous}")
print(f"    Pathogenic (score > 0.564): {n_pathogenic}")

# ---------- Save ----------
os.makedirs("data", exist_ok=True)
pkl_path = f"data/{gene_name.lower()}_alphamissense_scores.pkl"
csv_path = f"data/{gene_name.lower()}_alphamissense_scores.csv"

with open(pkl_path, "wb") as f:
    pickle.dump({"by_variant": am_by_variant, "by_position": am_by_position}, f)

records.sort(key=lambda r: (r["aa_pos"], r["variant"]))
with open(csv_path, "w", newline="", encoding="utf-8") as fout:
    w = csv.DictWriter(fout, fieldnames=["aa_pos", "variant", "ref_aa", "alt_aa", "am_score", "am_class"])
    w.writeheader()
    w.writerows(records)

print(f"\n  Saved: {pkl_path}")
print(f"  Saved: {csv_path} ({len(records)} rows)")
print(f"\n{'='*60}")
print(f"  DONE")
print(f"{'='*60}")
