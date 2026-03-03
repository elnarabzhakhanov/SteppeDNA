"""
fetch_eve_scores.py — Fetch EVE (Evolutionary model of Variant Effect) scores
from myvariant.info for all training variants.

EVE uses deep generative models (Bayesian VAE) trained on evolutionary sequence
alignments to predict variant pathogenicity — entirely orthogonal to our
supervised approach. Adding EVE as a feature captures evolutionary constraint
signals that complement our structural and functional features.

API: myvariant.info (dbNSFP field: dbnsfp.eve)
Output: data/eve_scores.pkl  {by_variant: {key: score}, by_position: {pos: score}}

Run: python data_pipelines/fetch_eve_scores.py
"""

import os
import sys
import json
import time
import pickle
import requests
import numpy as np
import pandas as pd

RANDOM_STATE = 42
DATA_DIR = "data"
CACHE_FILE = os.path.join(DATA_DIR, "eve_scores_cache.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "eve_scores.pkl")
MYVARIANT_API = "https://myvariant.info/v1/query"
HEADERS = {
    "User-Agent": "SteppeDNA-Research/1.0 (academic variant pathogenicity prediction)",
    "Accept": "application/json",
}

AA_3TO1 = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
    'Ter': '*',
}
AA_1TO3 = {v: k for k, v in AA_3TO1.items() if k != "Ter"}

FIELDS = ",".join([
    "dbnsfp.aa",
    "dbnsfp.eve.score",
    "dbnsfp.eve.rankscore",
    "dbnsfp.eve.class30_pred",
    "dbnsfp.eve.class50_pred",
    "dbnsfp.hgvsp",
])


def safe_float(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, list):
        for v in val:
            if v is not None:
                try:
                    return float(v)
                except (ValueError, TypeError):
                    continue
        return None
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return None
    return None


def fetch_batch(gene, positions_batch, retries=3):
    pos_str = " OR ".join(str(int(p)) for p in positions_batch)
    q = f'dbnsfp.genename:{gene} AND dbnsfp.aa.pos:({pos_str})'

    all_hits = []
    offset = 0

    while True:
        params = {
            "q": q,
            "fields": FIELDS,
            "size": 1000,
            "from": offset,
        }

        for attempt in range(retries):
            try:
                resp = requests.get(
                    MYVARIANT_API, params=params,
                    headers=HEADERS, timeout=30
                )
                if resp.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    print(f"    Rate limited, waiting {wait}s...", flush=True)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"    Failed after {retries} retries: {e}", flush=True)
                    return all_hits

        hits = data.get("hits", [])
        total = data.get("total", 0)
        all_hits.extend(hits)

        offset += 1000
        if offset >= total or offset >= 10000 or not hits:
            break
        time.sleep(0.3)

    return all_hits


def parse_hit(hit):
    dbnsfp = hit.get("dbnsfp", {})
    if isinstance(dbnsfp, list):
        dbnsfp = dbnsfp[0] if dbnsfp else {}

    aa = dbnsfp.get("aa", {})
    if isinstance(aa, list):
        aa = aa[0] if aa else {}

    pos_raw = aa.get("pos") if isinstance(aa, dict) else None
    if isinstance(pos_raw, list):
        pos = int(pos_raw[0]) if pos_raw else None
    elif pos_raw is not None:
        pos = int(pos_raw)
    else:
        pos = None

    aa_ref = aa.get("ref")
    aa_alt = aa.get("alt")

    eve_raw = dbnsfp.get("eve", {})
    if isinstance(eve_raw, dict):
        eve_score = safe_float(eve_raw.get("score"))
        eve_rankscore = safe_float(eve_raw.get("rankscore"))
        eve_class30 = eve_raw.get("class30_pred")
        eve_class50 = eve_raw.get("class50_pred")
    elif isinstance(eve_raw, list):
        eve_raw = eve_raw[0] if eve_raw else {}
        eve_score = safe_float(eve_raw.get("score") if isinstance(eve_raw, dict) else eve_raw)
        eve_rankscore = None
        eve_class30 = None
        eve_class50 = None
    else:
        eve_score = safe_float(eve_raw)
        eve_rankscore = None
        eve_class30 = None
        eve_class50 = None

    return {
        "pos": pos,
        "aa_ref": aa_ref,
        "aa_alt": aa_alt,
        "eve_score": eve_score,
        "eve_rankscore": eve_rankscore,
        "eve_class30": eve_class30,
        "eve_class50": eve_class50,
    }


def main():
    print("=" * 65, flush=True)
    print("  Fetching EVE Scores from myvariant.info (dbNSFP)", flush=True)
    print("=" * 65, flush=True)

    # Load master dataset
    df = pd.read_csv(os.path.join(DATA_DIR, "master_training_dataset.csv"))
    print(f"\nLoaded {len(df)} variants from master dataset", flush=True)

    # Decode amino acids from one-hot columns
    aa_names_3 = ['Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly', 'His', 'Ile',
                  'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser', 'Ter', 'Thr', 'Trp', 'Tyr', 'Val']

    def decode_onehot(row, prefix):
        for aa in aa_names_3:
            col = f"{prefix}_{aa}"
            if col in row.index and row[col] == 1:
                return AA_3TO1.get(aa)
        return None

    df["aa_ref_1"] = df.apply(lambda r: decode_onehot(r, "AA_ref"), axis=1)
    df["aa_alt_1"] = df.apply(lambda r: decode_onehot(r, "AA_alt"), axis=1)

    GENE_AA_LENGTH = {'BRCA1': 1863, 'BRCA2': 3418, 'PALB2': 1186, 'RAD51C': 376, 'RAD51D': 328}
    df["AA_pos"] = df.apply(
        lambda r: round(r["relative_aa_pos"] * GENE_AA_LENGTH.get(r["Gene"], 1000)),
        axis=1
    ).astype(int)

    n_decoded = df["aa_ref_1"].notna().sum()
    print(f"Decoded {n_decoded}/{len(df)} AA ref/alt from one-hot columns", flush=True)

    # Load cache
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
        print(f"Loaded cache ({sum(len(v) for v in cache.values())} records)", flush=True)

    # Fetch EVE scores for each gene
    by_variant = {}  # "Lys123Glu" -> eve_score
    by_position = {}  # aa_pos -> eve_score (max across all variants at that position)
    total_matched = 0

    for gene in ["BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D"]:
        gene_mask = df["Gene"] == gene
        gene_df = df[gene_mask]
        if len(gene_df) == 0:
            continue

        unique_positions = sorted(gene_df["AA_pos"].dropna().astype(int).unique())
        cache_key = f"{gene}_eve_hits"

        if cache_key in cache:
            print(f"\n{gene}: Using cached data ({len(cache[cache_key])} records)", flush=True)
            raw_hits = cache[cache_key]
        else:
            print(f"\n{gene}: Fetching EVE scores for {len(unique_positions)} positions...", flush=True)
            raw_hits = []
            batch_size = 100
            n_batches = (len(unique_positions) - 1) // batch_size + 1

            for i in range(0, len(unique_positions), batch_size):
                batch = unique_positions[i : i + batch_size]
                batch_num = i // batch_size + 1
                if batch_num % 5 == 1 or batch_num == n_batches:
                    print(f"  Batch {batch_num}/{n_batches} ({len(batch)} positions)...", flush=True)
                hits = fetch_batch(gene, batch)
                raw_hits.extend(hits)
                if batch_num < n_batches:
                    time.sleep(0.5)

            print(f"  Retrieved {len(raw_hits)} variant records", flush=True)

            cache[cache_key] = raw_hits
            with open(CACHE_FILE, "w") as f:
                json.dump(cache, f)

        # Parse hits
        gene_matched = 0
        for hit in raw_hits:
            parsed = parse_hit(hit)
            if parsed["eve_score"] is not None and parsed["pos"] is not None:
                aa_ref_1 = parsed["aa_ref"]
                aa_alt_1 = parsed["aa_alt"]
                pos = parsed["pos"]

                # Build 3-letter key for compatibility with training data
                ref_3 = AA_1TO3.get(aa_ref_1, aa_ref_1) if aa_ref_1 else None
                alt_3 = AA_1TO3.get(aa_alt_1, aa_alt_1) if aa_alt_1 else None

                if ref_3 and alt_3 and pos:
                    vkey = f"{ref_3}{pos}{alt_3}"
                    by_variant[vkey] = parsed["eve_score"]

                    if pos not in by_position or parsed["eve_score"] > by_position[pos]:
                        by_position[pos] = parsed["eve_score"]

                    gene_matched += 1

        total_matched += gene_matched
        pct = gene_matched / len(gene_df) * 100 if len(gene_df) > 0 else 0
        print(f"  EVE scores found: {gene_matched} variants ({pct:.1f}% coverage)", flush=True)

    # Save
    output = {
        "by_variant": by_variant,
        "by_position": by_position,
    }
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(output, f)

    print(f"\n{'=' * 65}", flush=True)
    print(f"EVE SCORES SUMMARY", flush=True)
    print(f"{'=' * 65}", flush=True)
    print(f"  Total variants with EVE scores: {len(by_variant)}", flush=True)
    print(f"  Total positions with EVE scores: {len(by_position)}", flush=True)

    if by_variant:
        scores = list(by_variant.values())
        print(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}]", flush=True)
        print(f"  Mean: {np.mean(scores):.4f}, Median: {np.median(scores):.4f}", flush=True)
    else:
        print("  NOTE: No EVE scores found in myvariant.info (dbNSFP).", flush=True)
        print("  EVE scores may not be available in this dbNSFP version.", flush=True)
        print("  Alternative: Download directly from https://evemodel.org/", flush=True)

    print(f"\nSaved -> {OUTPUT_FILE}", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
