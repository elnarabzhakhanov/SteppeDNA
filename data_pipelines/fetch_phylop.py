"""
Fetch PhyloP 100-way Conservation Scores for BRCA2 Variants
=============================================================
Maps cDNA positions to hg38 genomic coordinates using BRCA2's
canonical transcript (ENST00000380152) exon structure, then fetches
per-nucleotide PhyloP scores from the UCSC Genome Browser REST API.

Outputs: data/phylop_scores.pkl — dict {cDNA_pos: phyloP_score}
"""

import json
import pickle
import time
import urllib.request
import pandas as pd
import numpy as np
import argparse
import os
import sys

# ---------- Setup Arguments & Config ----------
parser = argparse.ArgumentParser(description="Fetch PhyloP scores for a given gene.")
parser.add_argument("--gene", type=str, default="BRCA2", help="Target gene name (e.g. BRCA2, PALB2)")
args = parser.parse_args()

gene_name = args.gene.upper()
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend", "gene_configs", f"{gene_name.lower()}.json")

if not os.path.exists(config_path):
    print(f"ERROR: Configuration file not found at {config_path}")
    sys.exit(1)

with open(config_path, "r") as f:
    gene_config = json.load(f)

TRANSCRIPT_ID = gene_config.get("transcript_id", "").split(".")[0]
CHROMOSOME = gene_config.get("chromosome")
STRAND = gene_config.get("strand")

if not all([TRANSCRIPT_ID, CHROMOSOME, STRAND]):
    print(f"ERROR: Missing sequence metadata in {config_path}")
    sys.exit(1)
def fetch_transcript_structure():
    """Fetch exon and CDS structure from Ensembl REST API."""
    url = f"https://rest.ensembl.org/lookup/id/{TRANSCRIPT_ID}?expand=1;content-type=application/json"
    print(f"  Fetching transcript structure: {TRANSCRIPT_ID}")
    
    req = urllib.request.urlopen(url, timeout=30)
    data = json.loads(req.read().decode())
    
    exons = [(e["start"], e["end"]) for e in data.get("Exon", [])]
    
    translation = data.get("Translation", {})
    cds_start = translation.get("start", min(e[0] for e in exons))
    cds_end = translation.get("end", max(e[1] for e in exons))
    
    # Genomic coordinates are always returned start < end regardless of strand
    return exons, cds_start, cds_end

def build_cdna_to_genomic_map(exons, cds_start, cds_end):
    """Build mapping from cDNA position (1-based) to genomic coordinate."""
    cdna_to_genomic = {}
    cdna_pos = 1

    # Process exons from 5' to 3' based on strand
    sorted_exons = sorted(exons, key=lambda x: x[0], reverse=(STRAND == "-"))

    for exon_start, exon_end in sorted_exons:
        coding_start = max(exon_start, cds_start)
        coding_end = min(exon_end, cds_end)

        if coding_start > coding_end:
            continue

        if STRAND == "+":
            for genomic_pos in range(coding_start, coding_end + 1):
                cdna_to_genomic[cdna_pos] = genomic_pos
                cdna_pos += 1
        else:
            for genomic_pos in range(coding_end, coding_start - 1, -1):
                cdna_to_genomic[cdna_pos] = genomic_pos
                cdna_pos += 1

    return cdna_to_genomic


def fetch_phylop_batch(chrom, start, end, retries=3):
    """
    Fetch PhyloP 100-way scores from UCSC REST API for a genomic region.
    Returns dict {genomic_pos: score}.
    API uses 0-based half-open coordinates.
    """
    url = (
        f"https://api.genome.ucsc.edu/getData/track?"
        f"genome=hg38&track=phyloP100way&chrom={chrom}"
        f"&start={start}&end={end}"
    )

    for attempt in range(retries):
        try:
            req = urllib.request.urlopen(url, timeout=30)
            data = json.loads(req.read().decode())
            scores = {}
            for item in data.get(chrom, []):
                # API returns 0-based start, we convert to 1-based
                pos = item["start"] + 1
                scores[pos] = item["value"]
            return scores
        except Exception as e:
            if attempt < retries - 1:
                print(f"  Retry {attempt+1} for {start}-{end}: {e}")
                time.sleep(2)
            else:
                print(f"  FAILED for {start}-{end}: {e}")
                return {}


def main():
    print("=" * 60)
    print(f"  FETCHING PhyloP 100-WAY CONSERVATION SCORES FOR {gene_name}")
    print("=" * 60)

    # 1. Build cDNA → genomic mapping
    exons, cds_start, cds_end = fetch_transcript_structure()
    cdna_to_genomic = build_cdna_to_genomic_map(exons, cds_start, cds_end)
    print(f"\n  cDNA positions mapped: {len(cdna_to_genomic)}")
    print(f"  cDNA range: 1 - {max(cdna_to_genomic.keys())}")
    print(f"  Genomic range: {min(cdna_to_genomic.values())} - {max(cdna_to_genomic.values())}")

    # 2. Load dataset to find which cDNA positions we actually need
    dataset_path = f"{gene_name.lower()}_missense_dataset_2.csv"
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        needed_cdna = sorted(df["cDNA_pos"].dropna().astype(int).unique())
        print(f"  Unique cDNA positions in dataset: {len(needed_cdna)}")
    else:
        print(f"  [WARN] {dataset_path} not found. Fetching ALL cdna positions.")
        needed_cdna = sorted(cdna_to_genomic.keys())

    # 3. Get the genomic positions we need
    needed_genomic = set()
    for cdna in needed_cdna:
        if cdna in cdna_to_genomic:
            needed_genomic.add(cdna_to_genomic[cdna])

    print(f"  Genomic positions to fetch: {len(needed_genomic)}")

    # 4. Fetch PhyloP scores in batches (covering full BRCA2 coding region)
    #    We fetch by exon to be efficient with API calls
    print(f"\n  Fetching PhyloP scores from UCSC (by exon)...")

    all_phylop = {}  # genomic_pos -> score
    for i, (exon_start, exon_end) in enumerate(sorted(exons)):
        coding_start = max(exon_start, cds_start)
        coding_end = min(exon_end, cds_end)
        if coding_start > coding_end:
            continue

        # UCSC API uses 0-based half-open: [start, end)
        scores = fetch_phylop_batch(f"chr{CHROMOSOME}", coding_start - 1, coding_end)
        all_phylop.update(scores)
        print(f"    Exon {i+1:2d}: fetched {len(scores):5d} scores "
              f"(chr{CHROMOSOME}:{coding_start}-{coding_end})")
        time.sleep(0.3)  # Rate limiting

    print(f"\n  Total PhyloP scores fetched: {len(all_phylop)}")

    # 5. Map back to cDNA positions
    cdna_phylop = {}
    missing = 0
    for cdna in needed_cdna:
        if cdna in cdna_to_genomic:
            genomic = cdna_to_genomic[cdna]
            if genomic in all_phylop:
                cdna_phylop[int(cdna)] = float(all_phylop[genomic])
            else:
                cdna_phylop[int(cdna)] = 0.0  # fallback
                missing += 1
        else:
            cdna_phylop[int(cdna)] = 0.0
            missing += 1

    print(f"  Successfully mapped: {len(cdna_phylop) - missing}/{len(needed_cdna)}")
    if missing > 0:
        print(f"  Missing (set to 0.0): {missing}")

    # 6. Show some stats
    scores_array = np.array(list(cdna_phylop.values()))
    print(f"\n  PhyloP Score Distribution:")
    print(f"    Min:    {scores_array.min():.3f}")
    print(f"    Max:    {scores_array.max():.3f}")
    print(f"    Mean:   {scores_array.mean():.3f}")
    print(f"    Median: {np.median(scores_array):.3f}")
    print(f"    Std:    {scores_array.std():.3f}")

    # 7. Save
    os.makedirs("data", exist_ok=True)
    phylop_pkl_path = f"data/{gene_name.lower()}_phylop_scores.pkl"
    cdna_pkl_path = f"data/{gene_name.lower()}_cdna_to_genomic.pkl"
    
    with open(phylop_pkl_path, "wb") as f:
        pickle.dump(cdna_phylop, f)

    # Also save the full cDNA→genomic mapping for reference
    with open(cdna_pkl_path, "wb") as f:
        pickle.dump(cdna_to_genomic, f)

    print(f"\n  Saved: {phylop_pkl_path} ({len(cdna_phylop)} entries)")
    print(f"  Saved: {cdna_pkl_path} ({len(cdna_to_genomic)} entries)")
    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
