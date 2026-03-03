"""
Fetch gnomAD Allele Frequencies for BRCA2 Variants (via Ensembl Overlap)
========================================================================
Uses the Ensembl REST API 'overlap/region' endpoint to retrieve
population allele frequencies for all known variants in the BRCA2
coding region. This data mitigates ascertainment bias by allowing
the model to learn molecular mechanisms, not just variant rarity.

Strategy: Query the BRCA2 genomic region in small chunks,
collect all SNV variation data with MAF annotations.

Output: data/gnomad_frequencies.pkl
  Format: {"by_variant": {"Ref{pos}Alt": AF}, "by_position": {cDNA: AF}}
"""

import os
import pickle
import time
import requests
import numpy as np
import pandas as pd
import argparse
import json
import sys

CODON_TABLE = {
    'TTT':'Phe','TTC':'Phe','TTA':'Leu','TTG':'Leu',
    'CTT':'Leu','CTC':'Leu','CTA':'Leu','CTG':'Leu',
    'ATT':'Ile','ATC':'Ile','ATA':'Ile','ATG':'Met',
    'GTT':'Val','GTC':'Val','GTA':'Val','GTG':'Val',
    'TCT':'Ser','TCC':'Ser','TCA':'Ser','TCG':'Ser',
    'CCT':'Pro','CCC':'Pro','CCA':'Pro','CCG':'Pro',
    'ACT':'Thr','ACC':'Thr','ACA':'Thr','ACG':'Thr',
    'GCT':'Ala','GCC':'Ala','GCA':'Ala','GCG':'Ala',
    'TAT':'Tyr','TAC':'Tyr','TAA':'Ter','TAG':'Ter',
    'CAT':'His','CAC':'His','CAA':'Gln','CAG':'Gln',
    'AAT':'Asn','AAC':'Asn','AAA':'Lys','AAG':'Lys',
    'GAT':'Asp','GAC':'Asp','GAA':'Glu','GAG':'Glu',
    'TGT':'Cys','TGC':'Cys','TGA':'Ter','TGG':'Trp',
    'CGT':'Arg','CGC':'Arg','CGA':'Arg','CGG':'Arg',
    'AGT':'Ser','AGC':'Ser','AGA':'Arg','AGG':'Arg',
    'GGT':'Gly','GGC':'Gly','GGA':'Gly','GGG':'Gly',
}

COMPLEMENT = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}

# ---------- Setup Arguments & Config ----------
parser = argparse.ArgumentParser(description="Fetch gnomAD Allele Frequencies for a given gene.")
parser.add_argument("--gene", type=str, default="BRCA2", help="Target gene name (e.g. BRCA2, PALB2)")
args = parser.parse_args()

gene_name = args.gene.upper()
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend", "gene_configs", f"{gene_name.lower()}.json")

if not os.path.exists(config_path):
    print(f"ERROR: Configuration file not found at {config_path}")
    sys.exit(1)

with open(config_path, "r") as f:
    gene_config = json.load(f)

CHROMOSOME = gene_config.get("chromosome")
GENOMIC_START = gene_config.get("genomic_start")
GENOMIC_END = gene_config.get("genomic_end")
STRAND = gene_config.get("strand")

if not all([CHROMOSOME, GENOMIC_START, GENOMIC_END, STRAND]):
    print(f"ERROR: Missing genomic coordinate metadata in {config_path}")
    sys.exit(1)

OUTPUT_DIR = "data"
DATASET_PATH = f"{gene_name.lower()}_missense_dataset_2.csv"
ENSEMBL_API = "https://rest.ensembl.org"

CHUNK_SIZE = 5000  # Query in 5kb chunks


def fetch_region_variants(chrom, start, end):
    """Fetch all known variants in a genomic region via Ensembl."""
    url = f"{ENSEMBL_API}/overlap/region/human/{chrom}:{start}-{end}"
    params = {"feature": "variation", "content-type": "application/json"}

    try:
        resp = requests.get(url, params=params, timeout=30)

        if resp.status_code == 429:
            retry = int(resp.headers.get("Retry-After", 3))
            time.sleep(retry)
            resp = requests.get(url, params=params, timeout=30)

        if resp.status_code != 200:
            return []

        return resp.json()
    except Exception:
        return []


def extract_allele_frequencies():
    """
    Scan the entire BRCA2 region and extract allele frequencies
    for all SNVs from Ensembl/gnomAD.
    """
    # Genomic position -> {allele: MAF}
    freq_by_genomic = {}
    total_variants = 0

    n_chunks = (GENOMIC_END - GENOMIC_START) // CHUNK_SIZE + 1
    print(f"  Scanning {gene_name} region in {n_chunks} chunks of {CHUNK_SIZE}bp...")

    for i, start in enumerate(range(GENOMIC_START, GENOMIC_END, CHUNK_SIZE)):
        end = min(start + CHUNK_SIZE - 1, GENOMIC_END)
        chunk_num = i + 1

        if chunk_num % 5 == 1 or chunk_num == n_chunks:
            print(f"    Chunk {chunk_num}/{n_chunks}: {start}-{end}")

        variants = fetch_region_variants(CHROMOSOME, start, end)

        for v in variants:
            # Only process SNVs
            if not isinstance(v, dict):
                continue

            alleles = v.get("alleles", "")
            if isinstance(alleles, str):
                alleles = alleles.split("/")
            if len(alleles) != 2 or any(len(a) != 1 for a in alleles):
                continue  # Skip non-SNVs

            pos = v.get("start")
            af = v.get("minor_allele_freq")
            
            # Ensembl also returns an 'assemblies' or 'populations' field in some endpoints, 
            # but for the `overlap/region` feature=variation we need to extract from `colocated_variants`
            # or `frequencies` if available. Ensembl overlap/region with feature=variation usually only
            # has basic MAF. Let's build a structure that defaults to `af` for all if granular is missing,
            # but is ready for structural dicts.
            if pos and af:
                try:
                    af = float(af)
                except (ValueError, TypeError):
                    af = 0.0

                ref = alleles[0]
                alt = alleles[1]
                key = f"{pos}:{ref}>{alt}"
                
                if key not in freq_by_genomic:
                    freq_by_genomic[key] = {"af": 0.0, "popmax": 0.0, "afr": 0.0, "amr": 0.0, "eas": 0.0, "nfe": 0.0}
                
                freq_by_genomic[key]["af"] = max(freq_by_genomic[key]["af"], af)
                freq_by_genomic[key]["popmax"] = max(freq_by_genomic[key]["popmax"], af) # Popmax defaults to AF if only one is returned in this endpoint
                total_variants += 1

        # Rate limiting
        time.sleep(0.3)

    print(f"  Total SNVs with MAF: {total_variants}")
    return freq_by_genomic


def map_frequencies(freq_by_genomic):
    """Map genomic frequencies to cDNA positions."""
    # Load coordinate mapping
    mapping_path = os.path.join(OUTPUT_DIR, f"{gene_name.lower()}_cdna_to_genomic.pkl")
    # Fallback to pure global file if explicit one missing
    if not os.path.exists(mapping_path):
        print(f"  [WARN] {mapping_path} not found. Cannot map frequencies to dataset.")
        return {}, {}

    with open(mapping_path, "rb") as f:
        cdna_to_genomic = pickle.load(f)

    # Reverse mapping
    genomic_to_cdna = {v: k for k, v in cdna_to_genomic.items()}
    
    cds_path = os.path.join(OUTPUT_DIR, f"{gene_name.lower()}_cds.txt")
    if not os.path.exists(cds_path):
        print(f"  [WARN] {cds_path} not found. By-variant dictionary will not be built.")
        
    cds_seq = None
    if os.path.exists(cds_path):
        with open(cds_path, "r") as f:
            cds_seq = f.read().strip()
            
    by_variant = {}
    by_position = {}

    matched = 0
    for key, af_dict in freq_by_genomic.items():
        # key format: "pos:ref>alt"
        if not af_dict or "af" not in af_dict:
            continue
            
        try:
            genomic_pos = int(key.split(":")[0])
        except (ValueError, IndexError):
            continue
            
        cdna_pos = genomic_to_cdna.get(genomic_pos)
        if cdna_pos is None:
            continue
            
        if cdna_pos not in by_position:
            by_position[cdna_pos] = {"af": 0.0, "popmax": 0.0, "afr": 0.0, "amr": 0.0, "eas": 0.0, "nfe": 0.0}
            
        by_position[cdna_pos]["af"] = max(by_position[cdna_pos]["af"], af_dict["af"])
        by_position[cdna_pos]["popmax"] = max(by_position[cdna_pos]["popmax"], af_dict["popmax"])
        by_position[cdna_pos]["afr"] = max(by_position[cdna_pos]["afr"], af_dict.get("afr", 0.0))
        by_position[cdna_pos]["amr"] = max(by_position[cdna_pos]["amr"], af_dict.get("amr", 0.0))
        by_position[cdna_pos]["eas"] = max(by_position[cdna_pos]["eas"], af_dict.get("eas", 0.0))
        by_position[cdna_pos]["nfe"] = max(by_position[cdna_pos]["nfe"], af_dict.get("nfe", 0.0))
        
        # Translate to build by_variant if CDS is available
        if cds_seq:
            try:
                ref_allele = key.split(":")[1].split(">")[0]
                alt_allele = key.split(":")[1].split(">")[1]
                
                if STRAND == "-":
                    cds_ref = COMPLEMENT.get(ref_allele, ref_allele)
                    cds_alt = COMPLEMENT.get(alt_allele, alt_allele)
                else:
                    cds_ref = ref_allele
                    cds_alt = alt_allele
                    
                codon_index = (cdna_pos - 1) // 3
                pos_in_codon = (cdna_pos - 1) % 3
                codon_start = codon_index * 3
                
                if codon_start + 3 <= len(cds_seq):
                    ref_codon = cds_seq[codon_start:codon_start+3]
                    # verify ref matches
                    if ref_codon[pos_in_codon] == cds_ref:
                        alt_codon = list(ref_codon)
                        alt_codon[pos_in_codon] = cds_alt
                        alt_codon = "".join(alt_codon)
                        
                        ref_aa = CODON_TABLE.get(ref_codon.upper(), "Unknown")
                        alt_aa = CODON_TABLE.get(alt_codon.upper(), "Unknown")
                        aa_pos = codon_index + 1
                        
                        if ref_aa != "Unknown" and alt_aa != "Unknown" and ref_aa != alt_aa:
                            variant_key = f"{ref_aa}{aa_pos}{alt_aa}"
                            if variant_key not in by_variant:
                                by_variant[variant_key] = {"af": 0.0, "popmax": 0.0, "afr": 0.0, "amr": 0.0, "eas": 0.0, "nfe": 0.0}
                            by_variant[variant_key]["af"] = max(by_variant[variant_key]["af"], af_dict["af"])
                            by_variant[variant_key]["popmax"] = max(by_variant[variant_key]["popmax"], af_dict["popmax"])
                            by_variant[variant_key]["afr"] = max(by_variant[variant_key]["afr"], af_dict.get("afr", 0.0))
                            by_variant[variant_key]["amr"] = max(by_variant[variant_key]["amr"], af_dict.get("amr", 0.0))
                            by_variant[variant_key]["eas"] = max(by_variant[variant_key]["eas"], af_dict.get("eas", 0.0))
                            by_variant[variant_key]["nfe"] = max(by_variant[variant_key]["nfe"], af_dict.get("nfe", 0.0))
            except Exception:
                pass
        
        if af_dict["af"] > 0:
            matched += 1

    print(f"  Mapped {matched} genomic frequencies to {len(by_position)} distinct cDNA positions")
    return by_variant, by_position


if __name__ == "__main__":
    print("=" * 60)
    print(f"  {gene_name} gnomAD Frequency Fetcher (Ensembl Overlap)")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Fetch all BRCA2 region variants
    freq_data = extract_allele_frequencies()

    # Step 2: Map to cDNA positions
    by_variant, by_position = map_frequencies(freq_data)

    # Step 3: Save
    output = {"by_variant": by_variant, "by_position": by_position, "raw_genomic": freq_data}
    output_path = os.path.join(OUTPUT_DIR, f"{gene_name.lower()}_gnomad_frequencies.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(output, f)

    print(f"\n  Saved {len(by_variant)} variant frequencies -> {output_path}")

    # Stats
    afs = [v["af"] for v in by_variant.values() if isinstance(v, dict) and v.get("af", 0) > 0]
    if afs:
        print(f"\n  Non-zero AFs: {len(afs)} ({100*len(afs)/len(by_variant):.1f}%)")
        print(f"  Median AF: {np.median(afs):.6f}")
        print(f"  Max AF:    {max(afs):.6f}")
    else:
        print("  No variants matched gnomAD (all novel)")

    print("\n  Done!")
