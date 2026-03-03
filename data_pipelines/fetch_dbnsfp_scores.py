"""
fetch_dbnsfp_scores.py — Fetch real SOTA predictor scores from myvariant.info

Retrieves REVEL, CADD, and BayesDel scores for SteppeDNA test set variants
using the myvariant.info REST API (public academic resource, Scripps Research).

These are INDEPENDENT predictors not used as input features in SteppeDNA,
enabling a fair head-to-head SOTA comparison.

API docs: https://docs.myvariant.info
"""

import pandas as pd
import numpy as np
import requests
import time
import json
import os
import sys
import pickle

RANDOM_STATE = 42
CACHE_FILE = "data/dbnsfp_scores_cache.json"
OUTPUT_FILE = "data/dbnsfp_sota_scores.csv"
MYVARIANT_API = "https://myvariant.info/v1/query"
HEADERS = {
    "User-Agent": "SteppeDNA-Research/1.0 (academic variant pathogenicity prediction)",
    "Accept": "application/json",
}

# 3-letter to 1-letter amino acid codes
AA_3TO1 = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
    'Ter': '*', 'Del': '-',
}

# Fields to retrieve from myvariant.info
FIELDS = ",".join([
    "dbnsfp.aa",
    "dbnsfp.revel.score",
    "dbnsfp.bayesdel.add_af.score",
    "dbnsfp.hgvsp",
    "cadd.phred",
])


def safe_float(val):
    """Extract a float from a field that might be a list or nested."""
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
    """Fetch scores for a batch of AA positions for a gene."""
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
    """Parse a myvariant.info hit into a score record."""
    dbnsfp = hit.get("dbnsfp", {})
    if isinstance(dbnsfp, list):
        dbnsfp = dbnsfp[0] if dbnsfp else {}
    aa = dbnsfp.get("aa", {})
    # aa can be a list (multiple transcripts) — take first dict
    if isinstance(aa, list):
        aa = aa[0] if aa else {}

    # AA position (list or int)
    pos_raw = aa.get("pos") if isinstance(aa, dict) else None
    if isinstance(pos_raw, list):
        pos = int(pos_raw[0]) if pos_raw else None
    elif pos_raw is not None:
        pos = int(pos_raw)
    else:
        pos = None

    # AA ref and alt (single letter)
    aa_ref = aa.get("ref")
    aa_alt = aa.get("alt")

    # REVEL score (list or float)
    revel_raw = dbnsfp.get("revel", {})
    if isinstance(revel_raw, dict):
        revel = safe_float(revel_raw.get("score"))
    else:
        revel = safe_float(revel_raw)

    # BayesDel score
    bayesdel_raw = dbnsfp.get("bayesdel", {})
    if isinstance(bayesdel_raw, dict):
        add_af = bayesdel_raw.get("add_af", {})
        if isinstance(add_af, dict):
            bayesdel = safe_float(add_af.get("score"))
        else:
            bayesdel = safe_float(add_af)
    else:
        bayesdel = safe_float(bayesdel_raw)

    # CADD phred (top-level field)
    cadd_raw = hit.get("cadd", {})
    if isinstance(cadd_raw, dict):
        cadd = safe_float(cadd_raw.get("phred"))
    else:
        cadd = safe_float(cadd_raw)

    return {
        "pos": pos,
        "aa_ref": aa_ref,
        "aa_alt": aa_alt,
        "revel": revel,
        "cadd_phred": cadd,
        "bayesdel": bayesdel,
    }


def main():
    print("=" * 65, flush=True)
    print("Fetching REVEL / CADD / BayesDel from myvariant.info", flush=True)
    print("(Public academic API — Scripps Research, no download needed)", flush=True)
    print("=" * 65, flush=True)

    # Load dataset
    df = pd.read_csv("data/master_training_dataset.csv")
    print(f"\nLoaded {len(df)} variants from master dataset", flush=True)

    # Load feature names to reproduce exact split
    with open("data/universal_feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    X = df[feature_names].values
    y = df["Label"].values
    strata = df["Gene"].astype(str) + "_" + df["Label"].astype(str)

    # Reproduce the 80/20 train+cal / test split
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(df))
    idx_traincal, idx_test = train_test_split(
        indices, test_size=0.20, random_state=RANDOM_STATE, stratify=strata
    )

    test_df = df.iloc[idx_test].copy().reset_index(drop=True)
    print(f"Test set: {len(test_df)} variants", flush=True)
    print(f"  Per gene: {dict(test_df['Gene'].value_counts())}", flush=True)

    # Decode amino acids from one-hot columns (master dataset doesn't have raw AA_ref/AA_alt)
    aa_names_3 = ['Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly', 'His', 'Ile',
                  'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser', 'Ter', 'Thr', 'Trp', 'Tyr', 'Val']

    def decode_onehot(row, prefix):
        for aa in aa_names_3:
            col = f"{prefix}_{aa}"
            if col in row.index and row[col] == 1:
                return AA_3TO1.get(aa)
        return None

    test_df["aa_ref_1"] = test_df.apply(lambda r: decode_onehot(r, "AA_ref"), axis=1)
    test_df["aa_alt_1"] = test_df.apply(lambda r: decode_onehot(r, "AA_alt"), axis=1)

    # Reconstruct AA position from relative_aa_pos and gene protein lengths
    GENE_AA_LENGTH = {'BRCA1': 1863, 'BRCA2': 3418, 'PALB2': 1186, 'RAD51C': 376, 'RAD51D': 328}
    test_df["AA_pos"] = test_df.apply(
        lambda r: round(r["relative_aa_pos"] * GENE_AA_LENGTH.get(r["Gene"], 1000)),
        axis=1
    ).astype(int)

    n_decoded = test_df["aa_ref_1"].notna().sum()
    print(f"  Decoded {n_decoded}/{len(test_df)} AA ref/alt from one-hot columns", flush=True)

    # Load cache
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
        print(f"Loaded cache ({sum(len(v) for v in cache.values())} records)", flush=True)

    # Process each gene
    all_scores = {
        "revel_score": [None] * len(test_df),
        "cadd_phred": [None] * len(test_df),
        "bayesdel_score": [None] * len(test_df),
    }

    for gene in ["BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D"]:
        gene_mask = test_df["Gene"] == gene
        gene_df = test_df[gene_mask]
        if len(gene_df) == 0:
            continue

        unique_positions = sorted(gene_df["AA_pos"].dropna().astype(int).unique())
        cache_key = f"{gene}_hits"

        if cache_key in cache:
            print(f"\n{gene}: Using cached data ({len(cache[cache_key])} records)", flush=True)
            raw_hits = cache[cache_key]
        else:
            print(f"\n{gene}: Fetching {len(unique_positions)} unique positions...", flush=True)
            raw_hits = []
            batch_size = 100
            n_batches = (len(unique_positions) - 1) // batch_size + 1

            for i in range(0, len(unique_positions), batch_size):
                batch = unique_positions[i : i + batch_size]
                batch_num = i // batch_size + 1
                print(f"  Batch {batch_num}/{n_batches} ({len(batch)} positions)...", flush=True)
                hits = fetch_batch(gene, batch)
                raw_hits.extend(hits)
                if batch_num < n_batches:
                    time.sleep(0.5)

            print(f"  Retrieved {len(raw_hits)} variant records", flush=True)

            # Cache raw hits
            cache[cache_key] = raw_hits
            with open(CACHE_FILE, "w") as f:
                json.dump(cache, f)

        # Build lookup: (pos, aa_ref, aa_alt) -> scores
        score_lookup = {}
        for hit in raw_hits:
            parsed = parse_hit(hit)
            if parsed["pos"] is not None and parsed["aa_ref"] and parsed["aa_alt"]:
                key = (parsed["pos"], parsed["aa_ref"], parsed["aa_alt"])
                score_lookup[key] = parsed

        # Match to test variants
        matched = 0
        for idx in gene_df.index:
            row = test_df.loc[idx]
            key = (int(row["AA_pos"]), row["aa_ref_1"], row["aa_alt_1"])
            if key in score_lookup:
                scores = score_lookup[key]
                all_scores["revel_score"][idx] = scores["revel"]
                all_scores["cadd_phred"][idx] = scores["cadd_phred"]
                all_scores["bayesdel_score"][idx] = scores["bayesdel"]
                matched += 1

        pct = matched / len(gene_df) * 100 if len(gene_df) > 0 else 0
        print(f"  Matched: {matched}/{len(gene_df)} ({pct:.1f}%)", flush=True)

    # Add scores to test dataframe
    for col, vals in all_scores.items():
        test_df[col] = vals

    # Save
    out_cols = ["Gene", "AA_pos", "aa_ref_1", "aa_alt_1", "Label",
                "revel_score", "cadd_phred", "bayesdel_score"]
    test_df[out_cols].to_csv(OUTPUT_FILE, index=False)

    # Coverage summary
    print(f"\n{'=' * 65}", flush=True)
    print("COVERAGE SUMMARY", flush=True)
    print(f"{'=' * 65}", flush=True)
    for col, name in [
        ("revel_score", "REVEL"),
        ("cadd_phred", "CADD"),
        ("bayesdel_score", "BayesDel"),
    ]:
        n = test_df[col].notna().sum()
        total = len(test_df)
        print(f"  {name:12s}: {n:>5}/{total} ({n / total * 100:.1f}%)", flush=True)

    print(f"\nScores saved to {OUTPUT_FILE}", flush=True)
    print("Now run: python scripts/sota_comparison.py  (to update SOTA figures)", flush=True)


if __name__ == "__main__":
    main()
