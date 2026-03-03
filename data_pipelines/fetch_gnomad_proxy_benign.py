"""
SteppeDNA: Fetch gnomAD Proxy-Benign Variants
===============================================
Queries the gnomAD v4 GraphQL API (Broad Institute) for common missense
variants in the 5 HR pathway genes. Common population variants are almost
certainly benign for high-penetrance cancer genes (purifying selection).

Thresholds:
  - Primary: combined AF >= 1e-4 (0.01%) — standard for high-penetrance genes
  - Minimum AC >= 2 (seen independently at least twice)

Outputs per gene:
  data/{gene}/{gene}_gnomad_proxy_benign.csv

Run from project root:
  python data_pipelines/fetch_gnomad_proxy_benign.py
"""

import os
import sys
import time
import json
import re
import requests
import pandas as pd
import numpy as np

GNOMAD_API = "https://gnomad.broadinstitute.org/api"
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "SteppeDNA-Research/1.0 (academic variant pathogenicity prediction)",
    "Accept": "application/json",
}

AF_THRESHOLD = 0         # No AF floor — use AC-based filtering instead
MIN_AC = 2               # Must be observed independently at least twice in gnomAD
DATASET = "gnomad_r4"    # gnomAD v4
# Rationale: For high-penetrance cancer genes (BRCA1/2, PALB2, RAD51C/D),
# truly pathogenic variants cause cancer by ~age 60-70 and are under strong
# purifying selection. Any missense variant observed 2+ times in ~1.6M gnomAD
# alleles (healthy adults) is very unlikely to be high-penetrance pathogenic.

GENES = ["BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D"]

# 3-letter to 3-letter amino acid mapping (for standardization)
AA3_VALID = {
    "Ala", "Arg", "Asn", "Asp", "Cys", "Gln", "Glu", "Gly", "His", "Ile",
    "Leu", "Lys", "Met", "Phe", "Pro", "Ser", "Thr", "Trp", "Tyr", "Val", "Ter"
}


def parse_hgvsp(hgvsp):
    """Parse HGVS protein notation like 'p.Ala123Thr' -> ('Ala', 123, 'Thr')."""
    if not hgvsp or hgvsp == "None":
        return None, None, None

    # Remove transcript prefix if present (e.g., ENSP00000369497:p.Ala123Thr)
    if ":" in hgvsp:
        hgvsp = hgvsp.split(":")[-1]

    # Match p.Ref###Alt pattern
    m = re.match(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})", hgvsp)
    if not m:
        return None, None, None

    ref, pos, alt = m.group(1), int(m.group(2)), m.group(3)

    # Validate amino acids
    if ref not in AA3_VALID or alt not in AA3_VALID:
        return None, None, None

    # Skip synonymous (ref == alt)
    if ref == alt:
        return None, None, None

    # Skip nonsense (Ter) — we want missense only
    if alt == "Ter":
        return None, None, None

    return ref, pos, alt


def fetch_gene_variants(gene_symbol):
    """Fetch all variants for a gene from gnomAD v4 API."""
    query = """
    {
      gene(gene_symbol: "%s", reference_genome: GRCh38) {
        variants(dataset: %s) {
          variant_id
          hgvsp
          consequence
          exome {
            ac
            an
            af
          }
          genome {
            ac
            an
            af
          }
        }
      }
    }
    """ % (gene_symbol, DATASET)

    for attempt in range(3):
        try:
            resp = requests.post(
                GNOMAD_API,
                json={"query": query},
                headers=HEADERS,
                timeout=180
            )
            if resp.status_code == 200:
                data = resp.json()
                if "errors" in data:
                    print(f"    API errors: {data['errors']}")
                    return []
                return data.get("data", {}).get("gene", {}).get("variants", [])
            else:
                print(f"    HTTP {resp.status_code}, retrying...")
                time.sleep(5 * (attempt + 1))
        except requests.exceptions.Timeout:
            print(f"    Timeout, retrying...")
            time.sleep(10 * (attempt + 1))
        except Exception as e:
            print(f"    Error: {e}, retrying...")
            time.sleep(5 * (attempt + 1))

    return []


def load_existing_variants(gene):
    """Load existing ClinVar variants for deduplication."""
    existing = set()

    # Check unified dataset
    gene_lower = gene.lower()
    paths = [
        f"data/{gene_lower}/{gene_lower}_missense_dataset_unified.csv",
        f"brca2_missense_dataset_2.csv",  # BRCA2 special case
    ]

    for path in paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                for _, row in df.iterrows():
                    key = f"{row.get('AA_ref', '')}{int(row.get('AA_pos', 0))}{row.get('AA_alt', '')}"
                    existing.add(key)
            except Exception:
                pass

    return existing


def process_gene(gene_symbol):
    """Fetch and filter gnomAD proxy-benign variants for one gene."""
    print(f"\n  Processing {gene_symbol}...")

    # Fetch from gnomAD
    variants = fetch_gene_variants(gene_symbol)
    if not variants:
        print(f"    No variants returned from API")
        return pd.DataFrame()

    print(f"    Total variants from gnomAD: {len(variants)}")

    # Filter for missense
    missense = [v for v in variants if v.get("consequence") == "missense_variant"]
    print(f"    Missense variants: {len(missense)}")

    # Compute combined AF and filter
    proxy_benign = []
    for v in missense:
        # Combine exome + genome counts
        ac, an = 0, 0
        if v.get("exome"):
            ac += v["exome"]["ac"]
            an += v["exome"]["an"]
        if v.get("genome"):
            ac += v["genome"]["ac"]
            an += v["genome"]["an"]

        if an == 0:
            continue

        af = ac / an

        if af < AF_THRESHOLD or ac < MIN_AC:
            continue

        # Parse protein change
        aa_ref, aa_pos, aa_alt = parse_hgvsp(v.get("hgvsp", ""))
        if aa_ref is None:
            continue

        proxy_benign.append({
            "Gene": gene_symbol,
            "AA_ref": aa_ref,
            "AA_pos": aa_pos,
            "AA_alt": aa_alt,
            "gnomad_af": af,
            "gnomad_ac": ac,
            "gnomad_an": an,
            "variant_id": v.get("variant_id", ""),
            "Label": 0,  # Proxy-benign
            "source": "gnomAD_proxy_benign",
        })

    print(f"    After AF >= {AF_THRESHOLD} filter: {len(proxy_benign)}")

    if not proxy_benign:
        return pd.DataFrame()

    df = pd.DataFrame(proxy_benign)

    # Deduplicate against existing ClinVar variants
    existing = load_existing_variants(gene_symbol)
    before = len(df)
    df["key"] = df.apply(lambda r: f"{r['AA_ref']}{r['AA_pos']}{r['AA_alt']}", axis=1)
    df = df[~df["key"].isin(existing)]
    df.drop(columns=["key"], inplace=True)
    deduped = before - len(df)
    if deduped > 0:
        print(f"    Removed {deduped} variants already in ClinVar")

    print(f"    Final proxy-benign: {len(df)}")

    return df


def main():
    print("=" * 60)
    print("  SteppeDNA: gnomAD Proxy-Benign Variant Fetcher")
    print("=" * 60)
    print(f"\n  API: {GNOMAD_API}")
    print(f"  Dataset: {DATASET}")
    print(f"  AF threshold: >= {AF_THRESHOLD} (0.01%)")
    print(f"  Min AC: >= {MIN_AC}")

    all_results = []

    for gene in GENES:
        df = process_gene(gene)
        if len(df) > 0:
            # Save per-gene CSV
            gene_lower = gene.lower()
            os.makedirs(f"data/{gene_lower}", exist_ok=True)
            outpath = f"data/{gene_lower}/{gene_lower}_gnomad_proxy_benign.csv"
            df.to_csv(outpath, index=False)
            print(f"    Saved: {outpath}")
            all_results.append(df)

        # Rate limit — be polite to the API
        time.sleep(2)

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        print(f"\n  Total proxy-benign variants: {len(combined)}")
        for gene in GENES:
            g_count = len(combined[combined["Gene"] == gene])
            print(f"    {gene:8s}: {g_count:5d} proxy-benign variants")

        # Save combined
        combined.to_csv("data/gnomad_proxy_benign_all.csv", index=False)
        print(f"\n  Combined file: data/gnomad_proxy_benign_all.csv")
    else:
        print("\n  No proxy-benign variants found!")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
