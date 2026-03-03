"""
SteppeDNA: Unified Multi-Gene Dataset Builder
=============================================
This script fetches ClinVar missense variants (Benign and Pathogenic)
for all target HR pathway genes: BRCA1, BRCA2, PALB2, RAD51C, RAD51D.
It saves individual CSV datasets for each gene.
"""

import os
import sys
import time
import requests
import json
import pandas as pd
import re

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

TARGET_GENES = ["brca1", "brca2", "palb2", "rad51c", "rad51d"]

def _load_gene_config(gene_prefix):
    # Path is relative to data_pipelines directory run
    config_path = os.path.join("backend", "gene_configs", f"{gene_prefix}.json")
    if not os.path.exists(config_path):
        print(f"[ERROR] Config for {gene_prefix} not found at {config_path}")
        return None
    with open(config_path, "r") as f:
        return json.load(f)

def esearch(query, retmax=20000):
    url = f"{NCBI_BASE}/esearch.fcgi"
    params = {"db": "clinvar", "term": query, "retmax": retmax, "retmode": "json"}
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json().get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        print(f"  [WARN] esearch failed: {e}")
        return []

def esummary(ids, batch_size=100):
    summaries = {}
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i+batch_size]
        url = f"{NCBI_BASE}/esummary.fcgi"
        params = {"db": "clinvar", "id": ",".join(batch), "retmode": "json"}
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            result = resp.json().get("result", {})
            for uid in batch:
                if uid in result:
                    summaries[uid] = result[uid]
        except Exception as e:
            print(f"  [WARN] esummary batch failed: {e}")
        time.sleep(0.35)
    return summaries

def parse_protein_change(title):
    m = re.search(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|Ter|\*)", title or "")
    if not m: return None, None, None
    return m.group(1), int(m.group(2)), m.group(3).replace("*", "Ter")

def parse_cdna_change(title):
    m = re.search(r"c\.(\d+)", title or "")
    return int(m.group(1)) if m else None

def fetch_gene_dataset(gene_prefix):
    print(f"\n{'='*60}")
    print(f" Processing {gene_prefix.upper()}")
    print(f"{'='*60}")
    
    config = _load_gene_config(gene_prefix)
    if not config: return
    
    gene_symbol = config.get("gene", gene_prefix.upper())
    
    path_query = f'{gene_symbol}[gene] AND "pathogenic"[clinical significance] AND "missense variant"[molecular consequence] AND "homo sapiens"[organism]'
    path_ids = esearch(path_query)
    print(f"  Pathogenic IDs: {len(path_ids)}")

    ben_query = f'{gene_symbol}[gene] AND ("benign"[clinical significance] OR "likely benign"[clinical significance]) AND "missense variant"[molecular consequence] AND "homo sapiens"[organism]'
    ben_ids = esearch(ben_query)
    print(f"  Benign/Likely Benign IDs: {len(ben_ids)}")

    all_ids = list(set(path_ids + ben_ids))
    print(f"  Total unique IDs fetching: {len(all_ids)}")
    if not all_ids:
        print("  Skipping, no variants found.")
        return

    summaries = esummary(all_ids)
    
    path_set = set(path_ids)
    ben_set  = set(ben_ids)
    
    rows = []
    seen = set()

    for uid, rec in summaries.items():
        title = rec.get("title", "")
        aa_ref, aa_pos, aa_alt = parse_protein_change(title)
        if not aa_ref or not aa_alt or aa_alt == "Ter" or aa_ref == aa_alt:
            continue
            
        cdna_pos = parse_cdna_change(title)
        if cdna_pos is None:
            cdna_pos = aa_pos * 3 - 1
            
        if uid in path_set: label = 1
        elif uid in ben_set: label = 0
        else: continue
            
        vkey = f"{aa_ref}{aa_pos}{aa_alt}"
        if vkey in seen: continue
        seen.add(vkey)
        
        rows.append({
            "cDNA_pos": cdna_pos,
            "Ref_nt": "N",
            "Alt_nt": "N",
            "Mutation": f"c.{cdna_pos}N>N",
            "AA_pos": aa_pos,
            "AA_ref": aa_ref,
            "AA_alt": aa_alt,
            "Label": label,
            "clinvar_id": uid,
            "Gene": gene_symbol
        })
        
    if not rows:
        print("  Skipping, no valid missense variants found after parsing.")
        return

    df = pd.DataFrame(rows)
    out_dir = os.path.join("data", gene_prefix)
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, f"{gene_prefix}_missense_dataset_unified.csv")
    df.to_csv(out_path, index=False)
    
    n_path = (df["Label"] == 1).sum()
    n_ben  = (df["Label"] == 0).sum()
    print(f"\n  [SAVED: {out_path}]")
    print(f"   => Variants: {len(df)} (Pathogenic: {n_path}, Benign: {n_ben})")

if __name__ == "__main__":
    for gene in TARGET_GENES:
        fetch_gene_dataset(gene)
    print("\n[ALL GENES PROCESSED]")
