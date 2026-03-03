"""
build_gold_standard_benchmark.py — Curate a gold-standard benchmark dataset
=============================================================================

Sources:
  1. ProteinGym DMS (Deep Mutational Scanning) — experimental variant fitness
     - BRCA1: Findlay et al. 2018 (saturation genome editing, 1,837 variants)
     - BRCA2: Erwood et al. 2022 (CRISPR prime editing, 265 variants)
  2. ClinVar Expert-Reviewed — ClinGen Variant Curation Expert Panel
     classifications (review_status = "reviewed by expert panel")
  3. ENIGMA/ClinGen consensus variants for BRCA1/BRCA2

Output: data/benchmark/gold_standard_benchmark.csv
        data/benchmark/benchmark_metadata.json

Run from project root:
  python data_pipelines/build_gold_standard_benchmark.py
"""

import pandas as pd
import numpy as np
import requests
import time
import json
import os
import sys
import io
import zipfile

# ─── Configuration ─────────────────────────────────────────────────────
BENCHMARK_DIR = "data/benchmark"
OUTPUT_FILE = os.path.join(BENCHMARK_DIR, "gold_standard_benchmark.csv")
METADATA_FILE = os.path.join(BENCHMARK_DIR, "benchmark_metadata.json")
MASTER_DATASET = "data/master_training_dataset.csv"

HEADERS = {
    "User-Agent": "SteppeDNA-Research/1.0 (academic variant pathogenicity prediction)",
    "Accept": "application/json",
}

# ProteinGym DMS datasets (HuggingFace direct CSV links)
PROTEINGYM_BASE = "https://huggingface.co/datasets/OATML-Markslab/ProteinGym/resolve/main/ProteinGym_substitutions"
PROTEINGYM_ASSAYS = {
    "BRCA1": {
        "file": "BRCA1_HUMAN_Findlay_2018",
        "citation": "Findlay et al., Nature 2018",
        "selection": "Growth (functional complementation)",
        "n_expected": 1837,
    },
    # Note: BRCA2 (Erwood 2022) is not available in ProteinGym v1.3
    # Only BRCA1 Findlay 2018 DMS data available via HuggingFace
}

# Gene protein lengths (for position mapping)
GENE_AA_LENGTH = {
    "BRCA1": 1863, "BRCA2": 3418, "PALB2": 1186,
    "RAD51C": 376, "RAD51D": 328,
}

# ClinVar E-Utils API
CLINVAR_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
CLINVAR_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
CLINVAR_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

# Amino acid codes
AA_3TO1 = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
    'Ter': '*', 'Del': '-',
}
AA_1TO3 = {v: k for k, v in AA_3TO1.items()}


# ═══════════════════════════════════════════════════════════════════════
# 1. ProteinGym DMS Data
# ═══════════════════════════════════════════════════════════════════════
def fetch_proteingym_dms():
    """Download ProteinGym DMS CSV files for BRCA1 and BRCA2."""
    all_records = []

    for gene, info in PROTEINGYM_ASSAYS.items():
        csv_name = info["file"] + ".csv"
        url = f"{PROTEINGYM_BASE}/{csv_name}"
        cache_file = os.path.join(BENCHMARK_DIR, csv_name)

        if os.path.exists(cache_file):
            print(f"  {gene}: Loading cached {csv_name}", flush=True)
            dms_df = pd.read_csv(cache_file)
        else:
            print(f"  {gene}: Downloading {csv_name}...", flush=True)
            for attempt in range(3):
                try:
                    resp = requests.get(url, headers=HEADERS, timeout=60)
                    if resp.status_code == 200:
                        dms_df = pd.read_csv(io.StringIO(resp.text))
                        dms_df.to_csv(cache_file, index=False)
                        break
                    elif resp.status_code == 404:
                        print(f"    404 — trying alternate URL...", flush=True)
                        # Try alternate URL pattern
                        alt_url = f"https://marks.hms.harvard.edu/proteingym/DMS_ProteinGym_substitutions/{csv_name}"
                        resp2 = requests.get(alt_url, headers=HEADERS, timeout=60)
                        if resp2.status_code == 200:
                            dms_df = pd.read_csv(io.StringIO(resp2.text))
                            dms_df.to_csv(cache_file, index=False)
                            break
                        else:
                            print(f"    Alternate URL also failed ({resp2.status_code})", flush=True)
                    else:
                        print(f"    HTTP {resp.status_code}, retry {attempt+1}/3", flush=True)
                        time.sleep(2 ** attempt)
                except Exception as e:
                    print(f"    Error: {e}, retry {attempt+1}/3", flush=True)
                    time.sleep(2 ** attempt)
            else:
                print(f"    FAILED to download {gene} DMS data", flush=True)
                continue

        print(f"    Raw records: {len(dms_df)}", flush=True)
        print(f"    Columns: {list(dms_df.columns)}", flush=True)

        # Parse variants — ProteinGym format: "A123V" or "A123V:B456C" (multi-mutant)
        # We only want single-amino-acid substitutions
        n_single = 0
        for _, row in dms_df.iterrows():
            mutant = str(row.get("mutant", ""))
            if ":" in mutant:
                continue  # Skip multi-mutants

            # Parse "A123V" → ref=A, pos=123, alt=V
            if len(mutant) < 3:
                continue
            aa_ref = mutant[0]
            aa_alt = mutant[-1]
            try:
                aa_pos = int(mutant[1:-1])
            except ValueError:
                continue

            # Skip synonymous
            if aa_ref == aa_alt:
                continue

            record = {
                "gene": gene,
                "aa_pos": aa_pos,
                "aa_ref": aa_ref,
                "aa_alt": aa_alt,
                "source": "ProteinGym_DMS",
                "source_detail": info["file"],
                "citation": info["citation"],
                "dms_score": row.get("DMS_score"),
                "dms_score_bin": row.get("DMS_score_bin"),
                "label": int(row.get("DMS_score_bin", -1)) if pd.notna(row.get("DMS_score_bin")) else None,
                "label_source": "DMS_binary",
            }
            all_records.append(record)
            n_single += 1

        print(f"    Single-AA substitutions: {n_single}", flush=True)

    return all_records


# ═══════════════════════════════════════════════════════════════════════
# 2. ClinVar Expert-Reviewed Variants
# ═══════════════════════════════════════════════════════════════════════
def fetch_clinvar_expert_variants():
    """Fetch ClinVar variants with 'reviewed by expert panel' status for HR genes."""
    all_records = []

    for gene in ["BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D"]:
        print(f"  {gene}: Querying ClinVar for expert-reviewed variants...", flush=True)

        # Search ClinVar for missense variants with expert panel review
        query = (
            f'{gene}[gene] AND "reviewed by expert panel"[review_status] '
            f'AND missense_variant[molecular_consequence]'
        )

        try:
            # Step 1: Search for IDs
            resp = requests.get(CLINVAR_ESEARCH, params={
                "db": "clinvar",
                "term": query,
                "retmax": 5000,
                "retmode": "json",
            }, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            search_data = resp.json()

            id_list = search_data.get("esearchresult", {}).get("idlist", [])
            total = int(search_data.get("esearchresult", {}).get("count", 0))
            print(f"    Found {total} expert-reviewed missense variants", flush=True)

            if not id_list:
                continue

            time.sleep(1.0)

            # Step 2: Fetch summaries in batches
            batch_size = 100
            for batch_start in range(0, len(id_list), batch_size):
                batch_ids = id_list[batch_start:batch_start + batch_size]

                for attempt in range(3):
                    try:
                        resp2 = requests.get(CLINVAR_ESUMMARY, params={
                            "db": "clinvar",
                            "id": ",".join(batch_ids),
                            "retmode": "json",
                        }, headers=HEADERS, timeout=30)
                        if resp2.status_code == 429:
                            wait = 3 * (attempt + 1)
                            print(f"    Rate limited, waiting {wait}s...", flush=True)
                            time.sleep(wait)
                            continue
                        resp2.raise_for_status()
                        break
                    except Exception as e:
                        if attempt < 2:
                            time.sleep(2 ** attempt)
                        else:
                            raise
                summary_data = resp2.json()

                results = summary_data.get("result", {})
                for uid in batch_ids:
                    entry = results.get(uid, {})
                    if not isinstance(entry, dict):
                        continue

                    # Extract clinical significance from germline_classification
                    gc = entry.get("germline_classification", {})
                    if isinstance(gc, dict):
                        description = gc.get("description", "").lower()
                    else:
                        description = str(gc).lower()

                    # Map to binary label
                    if "pathogenic" in description and "benign" not in description:
                        label = 1
                        label_text = "pathogenic"
                    elif "benign" in description and "pathogenic" not in description:
                        label = 0
                        label_text = "benign"
                    else:
                        continue  # Skip VUS, conflicting, etc.

                    # Extract protein change — ClinVar returns e.g. "M1361L, M1314L, ..."
                    # Take the first one (canonical transcript)
                    protein_change = entry.get("protein_change", "")
                    aa_pos = None
                    aa_ref = None
                    aa_alt = None

                    if protein_change:
                        # Take first protein change (canonical transcript)
                        first_change = protein_change.split(",")[0].strip()
                        aa_pos, aa_ref, aa_alt = parse_protein_change_1letter(first_change)

                    # Fallback: parse from title like "(p.Arg1835Ter)"
                    if aa_pos is None:
                        title = entry.get("title", "")
                        if "(p." in title:
                            prot_part = title.split("(p.")[1].split(")")[0]
                            aa_pos, aa_ref, aa_alt = parse_hgvsp_3letter(prot_part)

                    variant_id = entry.get("uid", uid)

                    record = {
                        "gene": gene,
                        "aa_pos": aa_pos,
                        "aa_ref": aa_ref,
                        "aa_alt": aa_alt,
                        "source": "ClinVar_ExpertPanel",
                        "source_detail": f"ClinVar:{variant_id}",
                        "citation": "ClinGen VCEP / ENIGMA",
                        "dms_score": None,
                        "dms_score_bin": None,
                        "label": label,
                        "label_source": f"expert_panel_{label_text}",
                    }
                    all_records.append(record)

                time.sleep(1.0)

        except Exception as e:
            print(f"    Error fetching ClinVar data: {e}", flush=True)

    return all_records


def parse_protein_change_1letter(change):
    """Parse 1-letter protein change like 'M1361L' into (pos, ref, alt)."""
    import re
    match = re.match(r'^([A-Z])(\d+)([A-Z])$', change.strip())
    if match:
        return int(match.group(2)), match.group(1), match.group(3)
    return None, None, None


def parse_hgvsp_3letter(prot_change):
    """Parse 3-letter HGVS protein notation like 'Arg1835Ter' into (pos, ref_1letter, alt_1letter)."""
    import re
    # Pattern: 3-letter AA + digits + 3-letter AA
    match = re.match(r'([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', prot_change)
    if match:
        ref_3 = match.group(1)
        pos = int(match.group(2))
        alt_3 = match.group(3)
        ref_1 = AA_3TO1.get(ref_3)
        alt_1 = AA_3TO1.get(alt_3)
        if ref_1 and alt_1:
            return pos, ref_1, alt_1
    return None, None, None


# ═══════════════════════════════════════════════════════════════════════
# 3. Deduplicate Against Training Data
# ═══════════════════════════════════════════════════════════════════════
def deduplicate_against_training(benchmark_df, master_df):
    """
    Mark benchmark variants that overlap with training data.
    Returns benchmark_df with 'in_training' and 'in_test' columns.
    """
    import pickle
    from sklearn.model_selection import train_test_split

    RANDOM_STATE = 42

    # Reconstruct the train/test split
    with open("data/universal_feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    strata = master_df["Gene"].astype(str) + "_" + master_df["Label"].astype(str)
    indices = np.arange(len(master_df))
    idx_traincal, idx_test = train_test_split(
        indices, test_size=0.20, random_state=RANDOM_STATE, stratify=strata
    )

    # Decode AA from master dataset one-hot
    aa_names_3 = ['Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly', 'His', 'Ile',
                  'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser', 'Thr', 'Trp', 'Tyr', 'Val']

    def decode_onehot(row, prefix):
        for aa in aa_names_3:
            col = f"{prefix}_{aa}"
            if col in row.index and row[col] == 1:
                return AA_3TO1.get(aa)
        return None

    master_df = master_df.copy()
    master_df["aa_ref_1"] = master_df.apply(lambda r: decode_onehot(r, "AA_ref"), axis=1)
    master_df["aa_alt_1"] = master_df.apply(lambda r: decode_onehot(r, "AA_alt"), axis=1)
    master_df["AA_pos_approx"] = master_df.apply(
        lambda r: round(r["relative_aa_pos"] * GENE_AA_LENGTH.get(r["Gene"], 1000)),
        axis=1
    ).astype(int)

    # Build lookup sets
    train_set = set()
    test_set = set()
    for idx in idx_traincal:
        row = master_df.iloc[idx]
        key = (row["Gene"], row["AA_pos_approx"], row["aa_ref_1"], row["aa_alt_1"])
        train_set.add(key)
    for idx in idx_test:
        row = master_df.iloc[idx]
        key = (row["Gene"], row["AA_pos_approx"], row["aa_ref_1"], row["aa_alt_1"])
        test_set.add(key)

    # Mark overlaps
    in_training = []
    in_test = []
    for _, row in benchmark_df.iterrows():
        key = (row["gene"], row.get("aa_pos"), row.get("aa_ref"), row.get("aa_alt"))
        in_training.append(key in train_set)
        in_test.append(key in test_set)

    benchmark_df["in_training"] = in_training
    benchmark_df["in_test"] = in_test

    return benchmark_df


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65, flush=True)
    print("Building Gold-Standard Benchmark Dataset", flush=True)
    print("=" * 65, flush=True)

    os.makedirs(BENCHMARK_DIR, exist_ok=True)

    # ─── Source 1: ProteinGym DMS ──────────────────────────────────────
    print("\n[1] ProteinGym DMS (Deep Mutational Scanning)", flush=True)
    print("-" * 50, flush=True)
    dms_records = fetch_proteingym_dms()
    print(f"\n  Total DMS records: {len(dms_records)}", flush=True)

    # ─── Source 2: ClinVar Expert Panel ────────────────────────────────
    print("\n[2] ClinVar Expert-Reviewed Variants", flush=True)
    print("-" * 50, flush=True)
    expert_records = fetch_clinvar_expert_variants()
    print(f"\n  Total expert-reviewed records: {len(expert_records)}", flush=True)

    # ─── Combine ───────────────────────────────────────────────────────
    all_records = dms_records + expert_records
    if not all_records:
        print("\nERROR: No benchmark records collected from any source!", flush=True)
        sys.exit(1)

    benchmark_df = pd.DataFrame(all_records)
    print(f"\n[3] Combined benchmark: {len(benchmark_df)} total records", flush=True)

    # Remove records missing position info
    n_before = len(benchmark_df)
    benchmark_df = benchmark_df.dropna(subset=["aa_pos", "aa_ref", "aa_alt"]).copy()
    benchmark_df["aa_pos"] = benchmark_df["aa_pos"].astype(int)
    n_after = len(benchmark_df)
    if n_before != n_after:
        print(f"  Dropped {n_before - n_after} records with missing position info", flush=True)

    # Remove exact duplicates (same gene + pos + ref + alt + source)
    n_before = len(benchmark_df)
    benchmark_df = benchmark_df.drop_duplicates(
        subset=["gene", "aa_pos", "aa_ref", "aa_alt", "source"], keep="first"
    ).reset_index(drop=True)
    n_after = len(benchmark_df)
    if n_before != n_after:
        print(f"  Removed {n_before - n_after} exact duplicates", flush=True)

    # ─── Deduplicate Against Training Data ─────────────────────────────
    print("\n[4] Deduplication against training/test data", flush=True)
    print("-" * 50, flush=True)
    master_df = pd.read_csv(MASTER_DATASET)
    benchmark_df = deduplicate_against_training(benchmark_df, master_df)

    n_in_train = benchmark_df["in_training"].sum()
    n_in_test = benchmark_df["in_test"].sum()
    n_novel = (~benchmark_df["in_training"] & ~benchmark_df["in_test"]).sum()
    print(f"  In training set:  {n_in_train}", flush=True)
    print(f"  In test set:      {n_in_test}", flush=True)
    print(f"  Novel (unseen):   {n_novel}", flush=True)

    # ─── Save ──────────────────────────────────────────────────────────
    benchmark_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[5] Saved to {OUTPUT_FILE}", flush=True)

    # ─── Metadata ──────────────────────────────────────────────────────
    metadata = {
        "created": pd.Timestamp.now().isoformat(),
        "total_variants": len(benchmark_df),
        "sources": {},
        "per_gene": {},
        "overlap": {
            "in_training": int(n_in_train),
            "in_test": int(n_in_test),
            "novel": int(n_novel),
        },
    }

    for source in benchmark_df["source"].unique():
        src_df = benchmark_df[benchmark_df["source"] == source]
        metadata["sources"][source] = {
            "n_variants": len(src_df),
            "genes": sorted(src_df["gene"].unique().tolist()),
            "citations": sorted(src_df["citation"].unique().tolist()),
            "label_distribution": {
                "pathogenic": int((src_df["label"] == 1).sum()),
                "benign": int((src_df["label"] == 0).sum()),
                "unlabeled": int(src_df["label"].isna().sum()),
            },
        }

    for gene in sorted(benchmark_df["gene"].unique()):
        gene_df = benchmark_df[benchmark_df["gene"] == gene]
        metadata["per_gene"][gene] = {
            "total": len(gene_df),
            "sources": {
                src: int((gene_df["source"] == src).sum())
                for src in gene_df["source"].unique()
            },
            "novel_variants": int(
                (~gene_df["in_training"] & ~gene_df["in_test"]).sum()
            ),
        }

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    # ─── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 65}", flush=True)
    print("GOLD-STANDARD BENCHMARK SUMMARY", flush=True)
    print(f"{'=' * 65}", flush=True)
    print(f"{'Source':<30s} {'n':>6s}  {'Path':>5s}  {'Benign':>6s}  {'Novel':>6s}", flush=True)
    print("-" * 60, flush=True)

    for source in benchmark_df["source"].unique():
        src_df = benchmark_df[benchmark_df["source"] == source]
        n_path = (src_df["label"] == 1).sum()
        n_ben = (src_df["label"] == 0).sum()
        n_nov = (~src_df["in_training"] & ~src_df["in_test"]).sum()
        print(f"  {source:<28s} {len(src_df):>6d}  {n_path:>5d}  {n_ben:>6d}  {n_nov:>6d}", flush=True)

    print("-" * 60, flush=True)
    total_path = (benchmark_df["label"] == 1).sum()
    total_ben = (benchmark_df["label"] == 0).sum()
    print(f"  {'TOTAL':<28s} {len(benchmark_df):>6d}  {total_path:>5d}  {total_ben:>6d}  {n_novel:>6d}", flush=True)

    print(f"\nPer-gene breakdown:", flush=True)
    for gene in sorted(benchmark_df["gene"].unique()):
        gene_df = benchmark_df[benchmark_df["gene"] == gene]
        n_nov = (~gene_df["in_training"] & ~gene_df["in_test"]).sum()
        print(f"  {gene}: {len(gene_df)} variants ({n_nov} novel)", flush=True)

    print(f"\nMetadata saved to {METADATA_FILE}", flush=True)
    print(f"Now run: python scripts/evaluate_benchmark.py", flush=True)


if __name__ == "__main__":
    main()
