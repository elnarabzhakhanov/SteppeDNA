"""
Fetch Published Predictor Scores from dbNSFP for BRCA2 Variants
================================================================
Downloads and extracts REVEL, BayesDel, CADD, PolyPhen-2, and SIFT
scores from the dbNSFP database (v4.9a academic) for all BRCA2
missense variants in the training set.

dbNSFP (database for nonsynonymous SNPs' functional predictions)
is maintained by Dr. Xiaoming Liu at the University of South Florida.
Citation: Liu et al., Genome Medicine (2020) 12:103

Strategy:
  1. Download the chr13 file from dbNSFP4.9a (academic version)
  2. Filter to BRCA2 genomic region (hg38: chr13:32315474-32400268)
  3. Match variants to our training set via genomic coordinates
  4. Save as pickle dict keyed by (AA_ref, AA_pos, AA_alt) triplet

Output: data/dbnsfp_brca2_scores.pkl
  Format: {
      "by_variant": {
          "Ala100Val": {"revel": 0.5, "bayesdel": 0.1, "cadd": 25.3, ...},
          ...
      },
      "by_position": {
          100: {"revel": 0.5, ...},  # position-level average (fallback)
          ...
      }
  }

Prerequisite: data/cdna_to_genomic.pkl must exist (run fetch_phylop.py first).

Usage:
  If you already have dbNSFP4.9a_variant.chr13.gz downloaded:
    python data_pipelines/fetch_dbnsfp.py --chr13-file /path/to/dbNSFP4.9a_variant.chr13.gz

  To download automatically (requires ~36GB disk space for full zip):
    python data_pipelines/fetch_dbnsfp.py --download
"""

import os
import sys
import gzip
import pickle
import argparse
import urllib.request
import zipfile
import numpy as np
import pandas as pd

OUTPUT_DIR = "data"

# dbNSFP4.9a download (academic version — includes REVEL, CADD, PolyPhen-2)
DBNSFP_URL = "https://dbnsfp.s3.amazonaws.com/dbNSFP4.9a.zip"

parser = argparse.ArgumentParser(
    description="Fetch SOTA predictor scores from dbNSFP for a given gene"
)
parser.add_argument("--gene", type=str, default="BRCA2", help="Target gene name (e.g. BRCA2, PALB2)")
parser.add_argument(
    "--chr-file",
    help="Path to pre-downloaded dbNSFP4.9a_variant.chrN.gz file",
    default=None
)
parser.add_argument(
    "--download",
    action="store_true",
    help="Download dbNSFP4.9a.zip from Amazon S3 (WARNING: ~36GB)"
)
args, _ = parser.parse_known_args()

gene_name = args.gene.upper()
import json
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend", "gene_configs", f"{gene_name.lower()}.json")

if not os.path.exists(config_path):
    print(f"ERROR: Configuration file not found at {config_path}")
    sys.exit(1)

with open(config_path, "r") as f:
    gene_config = json.load(f)

CHROMOSOME = gene_config.get("chromosome")
GENOMIC_START = gene_config.get("genomic_start")
GENOMIC_END = gene_config.get("genomic_end")

if not all([CHROMOSOME, GENOMIC_START, GENOMIC_END]):
    print(f"ERROR: Missing genomic coordinate metadata in {config_path}")
    sys.exit(1)

DBNSFP_CHR_FILENAME = f"dbNSFP4.9a_variant.chr{CHROMOSOME}.gz"

# The specific columns we want to extract from dbNSFP
# Column names based on the dbNSFP4.9a README
SCORE_COLUMNS = {
    "REVEL_score":             "revel",
    "BayesDel_addAF_score":    "bayesdel",
    "CADD_phred":              "cadd_phred",
    "Polyphen2_HDIV_score":    "polyphen2_hdiv",
    "Polyphen2_HVAR_score":    "polyphen2_hvar",
    "SIFT_score":              "sift",
    "SIFT4G_score":            "sift4g",
}

# Columns we need for coordinate matching
MATCH_COLUMNS = [
    "#chr",           # chromosome (e.g., "13")
    "pos(1-based)",   # hg38 genomic position
    "ref",            # reference allele
    "alt",            # alternate allele
    "aaref",          # reference amino acid (single-letter)
    "aaalt",          # alternate amino acid (single-letter)
    "aapos",          # amino acid position (can be multiple transcripts)
    "genename",       # gene name
    "Ensembl_transcriptid",  # for filtering to canonical BRCA2 transcript
]

# Single-letter to three-letter amino acid mapping
AA_1TO3 = {
    'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys',
    'Q': 'Gln', 'E': 'Glu', 'G': 'Gly', 'H': 'His', 'I': 'Ile',
    'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe', 'P': 'Pro',
    'S': 'Ser', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val',
    '*': 'Ter',
}

# Reverse mapping for matching
AA_3TO1 = {v: k for k, v in AA_1TO3.items()}

COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}


def download_chr13_from_zip(output_path):
    """
    Download dbNSFP4.9a.zip and extract only the relevant chromosome file.
    This avoids downloading the entire ~36GB and only extracts ~1.5GB.
    """
    zip_path = os.path.join(OUTPUT_DIR, "dbNSFP4.9a.zip")

    if os.path.exists(output_path):
        print(f"  [SKIP] {output_path} already exists")
        return True

    print(f"  Downloading dbNSFP4.9a.zip from Amazon S3...")
    print(f"  URL: {DBNSFP_URL}")
    print(f"  WARNING: This is a large file (~36GB). The download may take a while.")
    print(f"  TIP: If you have slow internet, download manually and use --chr13-file flag.")

    try:
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 / total_size)
                mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                sys.stdout.write(f"\r  Progress: {mb:.0f}/{total_mb:.0f} MB ({pct:.1f}%)")
                sys.stdout.flush()

        urllib.request.urlretrieve(DBNSFP_URL, zip_path, reporthook=report_progress)
        print()
    except Exception as e:
        print(f"\n  [ERROR] Download failed: {e}")
        print(f"  Please download manually from: {DBNSFP_URL}")
        print(f"  Then extract {DBNSFP_CHR13_FILENAME} and use --chr13-file flag")
        return False

    # Extract only the target chromosome from the zip
    print(f"  Extracting {DBNSFP_CHR_FILENAME} from zip...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Find the chr file in the archive
            chr_names = [n for n in zf.namelist() if f"chr{CHROMOSOME}" in n and "variant" in n]
            if not chr_names:
                print(f"  [ERROR] Could not find chr{CHROMOSOME} variant file in archive")
                print(f"  Archive contents: {zf.namelist()[:10]}...")
                return False

            chr_name = chr_names[0]
            print(f"  Found: {chr_name}")
            zf.extract(chr_name, OUTPUT_DIR)

            extracted_path = os.path.join(OUTPUT_DIR, chr_name)
            if extracted_path != output_path:
                os.rename(extracted_path, output_path)

        print(f"  [OK] Extracted to {output_path}")

        # Clean up the massive zip file
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print(f"  [OK] Cleaned up zip file")

        return True

    except Exception as e:
        print(f"  [ERROR] Extraction failed: {e}")
        return False


def parse_score(value):
    """Safely parse a dbNSFP score value. Returns float or NaN."""
    if value is None or value == '.' or value == '' or value == 'NA':
        return np.nan
    # dbNSFP can have multiple transcript scores separated by ";"
    # Take the first valid one (usually canonical)
    if isinstance(value, str) and ';' in value:
        for part in value.split(';'):
            part = part.strip()
            if part and part != '.' and part != 'NA':
                try:
                    return float(part)
                except ValueError:
                    continue
        return np.nan
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def parse_aapos(value):
    """Parse amino acid position from dbNSFP (may have multiple transcripts)."""
    if value is None or value == '.' or value == '' or value == 'NA':
        return None
    if isinstance(value, str) and ';' in value:
        for part in value.split(';'):
            part = part.strip()
            if part and part != '.' and part != '-1':
                try:
                    pos = int(part)
                    if pos > 0:
                        return pos
                except ValueError:
                    continue
        return None
    try:
        pos = int(float(value))
        return pos if pos > 0 else None
    except (ValueError, TypeError):
        return None


def extract_gene_scores(chr_path):
    """
    Parse the dbNSFP chr file and extract pathogenicity scores
    for all variants in the gene's genomic region.

    Returns a list of dicts, one per variant, with scores and coordinates.
    """
    print(f"\n  Parsing {chr_path}...")
    print(f"  Filtering to {gene_name} region: chr{CHROMOSOME}:{GENOMIC_START}-{GENOMIC_END}")

    all_needed_cols = MATCH_COLUMNS + list(SCORE_COLUMNS.keys())
    variants = []
    total_lines = 0
    brca2_lines = 0

    opener = gzip.open if chr_path.endswith('.gz') else open
    mode = 'rt' if chr_path.endswith('.gz') else 'r'

    with opener(chr_path, mode, encoding='utf-8', errors='replace') as f:
        # Read header
        header_line = f.readline().strip()
        headers = header_line.split('\t')

        # Find column indices
        col_indices = {}
        for col_name in all_needed_cols:
            if col_name in headers:
                col_indices[col_name] = headers.index(col_name)
            else:
                # Try alternate names
                alt_names = {
                    "#chr": ["chr", "CHROM"],
                    "pos(1-based)": ["hg38_pos(1-based)", "pos"],
                    "aapos": ["aapos_ENST00000380152", "HGVSp_VEP"],
                }
                found = False
                for alt in alt_names.get(col_name, []):
                    if alt in headers:
                        col_indices[col_name] = headers.index(alt)
                        found = True
                        break
                if not found:
                    print(f"  [WARN] Column '{col_name}' not found in dbNSFP header")

        print(f"  Columns found: {len(col_indices)}/{len(all_needed_cols)}")

        # Check we have the essential columns
        if "pos(1-based)" not in col_indices:
            # Try to find position column by searching
            for i, h in enumerate(headers):
                if 'pos' in h.lower() and '1-based' in h.lower():
                    col_indices["pos(1-based)"] = i
                    print(f"  [FIX] Using '{h}' as position column")
                    break

        if "pos(1-based)" not in col_indices:
            print(f"  [ERROR] Cannot find genomic position column")
            print(f"  Available columns (first 30): {headers[:30]}")
            return []

        # Parse line by line (memory efficient for large files)
        for line in f:
            total_lines += 1

            if total_lines % 500000 == 0:
                print(f"    Processed {total_lines:,} lines, found {brca2_lines:,} BRCA2 variants...")

            fields = line.strip().split('\t')

            # Quick position filter first (most efficient rejection)
            pos_idx = col_indices["pos(1-based)"]
            if pos_idx >= len(fields):
                continue

            try:
                genomic_pos = int(fields[pos_idx])
            except ValueError:
                continue

            # Skip if outside region
            if genomic_pos < GENOMIC_START or genomic_pos > GENOMIC_END:
                continue

            brca2_lines += 1

            # Extract all fields
            variant = {"genomic_pos": genomic_pos}

            for col_name, idx in col_indices.items():
                if idx < len(fields):
                    variant[col_name] = fields[idx]
                else:
                    variant[col_name] = None

            # Parse scores
            scores = {}
            for dbnsfp_col, our_name in SCORE_COLUMNS.items():
                raw = variant.get(dbnsfp_col)
                scores[our_name] = parse_score(raw)

            variant["scores"] = scores
            variant["aa_pos_parsed"] = parse_aapos(variant.get("aapos"))

            variants.append(variant)

    print(f"\n  Total lines scanned: {total_lines:,}")
    print(f"  {gene_name} region variants found: {brca2_lines:,}")

    return variants


def map_dbnsfp_scores(dbnsfp_variants):
    """
    Map dbNSFP variants to dictionaries using AA position and dbNSFP's own annotations.
    """
    by_variant = {}   # "Ala100Val" -> {scores}
    by_position = {}  # AA_pos -> {scores} (average across variants at position)
    position_scores = {}  # AA_pos -> list of score dicts

    for dv in dbnsfp_variants:
        scores = dv["scores"]

        # Skip if no useful scores
        if all(np.isnan(v) for v in scores.values()):
            continue

        # Match by AA change from dbNSFP directly
        dbnsfp_aaref = dv.get("aaref", "")
        dbnsfp_aaalt = dv.get("aaalt", "")
        dbnsfp_aapos = dv.get("aa_pos_parsed")

        if dbnsfp_aaref and dbnsfp_aaalt and dbnsfp_aapos:
            # Convert single-letter to 3-letter
            aa_ref_3 = AA_1TO3.get(dbnsfp_aaref)
            aa_alt_3 = AA_1TO3.get(dbnsfp_aaalt)

            if aa_ref_3 and aa_alt_3:
                variant_key = f"{aa_ref_3}{dbnsfp_aapos}{aa_alt_3}"
                if variant_key not in by_variant:
                    by_variant[variant_key] = scores

                    if dbnsfp_aapos not in position_scores:
                        position_scores[dbnsfp_aapos] = []
                    position_scores[dbnsfp_aapos].append(scores)

    # Compute position-level averages (fallback for unmatched variants)
    for aa_pos, score_list in position_scores.items():
        avg_scores = {}
        for score_name in SCORE_COLUMNS.values():
            vals = [s[score_name] for s in score_list if not np.isnan(s[score_name])]
            avg_scores[score_name] = float(np.mean(vals)) if vals else np.nan
        by_position[aa_pos] = avg_scores

    print(f"  Unique variant keys compiled: {len(by_variant)}")
    print(f"  Position-level entries compiled: {len(by_position)}")

    return by_variant, by_position


def print_coverage_report(by_variant):
    """Print how many training variants have each score type."""
    df = pd.read_csv(DATASET_PATH)
    total = len(df)

    print(f"\n  Score Coverage Report ({total} training variants):")
    print(f"  {'Score':<20} {'Available':>10} {'Coverage':>10}")
    print(f"  {'-'*42}")

    for score_name in SCORE_COLUMNS.values():
        count = sum(
            1 for v in by_variant.values()
            if not np.isnan(v.get(score_name, np.nan))
        )
        pct = 100 * count / total if total > 0 else 0
        print(f"  {score_name:<20} {count:>10,} {pct:>9.1f}%")


def main():
    print("=" * 60)
    print(f"  FETCHING SOTA PREDICTOR SCORES FROM dbNSFP FOR {gene_name}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Determine chr file path
    if args.chr_file:
        chr_path = args.chr_file
        if not os.path.exists(chr_path):
            print(f"  [ERROR] File not found: {chr_path}")
            sys.exit(1)
    else:
        chr_path = os.path.join(OUTPUT_DIR, DBNSFP_CHR_FILENAME)

        if not os.path.exists(chr_path):
            if args.download:
                success = download_chr13_from_zip(chr_path)
                if not success:
                    sys.exit(1)
            else:
                print(f"\n  [ERROR] dbNSFP file not found at: {chr_path}")
                print(f"\n  To download automatically (~36GB zip, extracts chr only):")
                print(f"    python data_pipelines/fetch_dbnsfp.py --gene {gene_name} --download")
                print(f"\n  Or download manually:")
                print(f"    1. Download: {DBNSFP_URL}")
                print(f"    2. Extract {DBNSFP_CHR_FILENAME} from the zip")
                print(f"    3. Place it at: {chr_path}")
                print(f"    4. Re-run: python data_pipelines/fetch_dbnsfp.py --gene {gene_name} --chr-file {chr_path}")
                sys.exit(1)

    print(f"\n  Using: {chr_path}")
    file_size_mb = os.path.getsize(chr_path) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")

    # Step 1: Extract gene variants from dbNSFP
    dbnsfp_variants = extract_gene_scores(chr_path)

    if not dbnsfp_variants:
        print(f"  [ERROR] No {gene_name} variants found in dbNSFP file")
        sys.exit(1)

    # Step 2: Map to dicts
    by_variant, by_position = map_dbnsfp_scores(dbnsfp_variants)

    # Step 4: Save
    output = {
        "by_variant": by_variant,
        "by_position": by_position,
    }
    output_path = os.path.join(OUTPUT_DIR, f"{gene_name.lower()}_dbnsfp_scores.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(output, f)

    print(f"\n  Saved: {output_path}")
    print(f"  Variant entries: {len(by_variant)}")
    print(f"  Position entries: {len(by_position)}")

    # Show sample
    if by_variant:
        sample_key = list(by_variant.keys())[0]
        sample = by_variant[sample_key]
        print(f"\n  Sample entry: {sample_key}")
        for k, v in sample.items():
            print(f"    {k}: {v}")

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
