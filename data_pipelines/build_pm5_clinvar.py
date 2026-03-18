#!/usr/bin/env python3
"""
Build PM5 pathogenic positions from ClinVar via myvariant.info API.

PM5 criterion: "Novel missense amino acid change at a position where a
different pathogenic missense has been seen before."

This script fetches ALL pathogenic/likely-pathogenic missense variants
from ClinVar for our 5 HR genes, extracts AA positions, and builds a
comprehensive PM5 lookup table.

Replaces the previous training-data-only PM5 table.
"""

import json
import os
import re
import time
import requests
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

GENES = {
    "BRCA1": {"entrez": "672"},
    "BRCA2": {"entrez": "675"},
    "PALB2": {"entrez": "79728"},
    "RAD51C": {"entrez": "5889"},
    "RAD51D": {"entrez": "5892"},
}


def fetch_clinvar_pathogenic(gene_name, entrez_id):
    """Fetch pathogenic/likely-pathogenic missense from ClinVar via NCBI."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    search_url = f"{base_url}/esearch.fcgi"
    params = {
        "db": "clinvar",
        "term": f"{gene_name}[gene] AND (pathogenic[clinsig] OR likely_pathogenic[clinsig]) AND missense_variant[molcons]",
        "retmax": 10000,
        "retmode": "json",
    }
    print(f"  Searching ClinVar for {gene_name} pathogenic missense...")
    resp = requests.get(search_url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    id_list = data.get("esearchresult", {}).get("idlist", [])
    print(f"  Found {len(id_list)} ClinVar records")
    if not id_list:
        return set()

    positions = set()
    batch_size = 200
    for i in range(0, len(id_list), batch_size):
        batch = id_list[i:i + batch_size]
        summary_url = f"{base_url}/esummary.fcgi"
        p = {"db": "clinvar", "id": ",".join(batch), "retmode": "json"}
        resp = requests.post(summary_url, data=p, timeout=60)
        resp.raise_for_status()
        summaries = resp.json().get("result", {})
        for uid in batch:
            entry = summaries.get(uid, {})
            title = entry.get("title", "")
            m = re.search(r"p\.([A-Z][a-z]{2})(\d+)", title)
            if m:
                positions.add(int(m.group(2)))
        time.sleep(0.4)
    return positions


def fetch_via_myvariant(gene_name):
    """Supplement from myvariant.info."""
    url = "https://myvariant.info/v1/query"
    positions = set()
    for sig in ["Pathogenic", "Likely_pathogenic"]:
        query = f"clinvar.gene.symbol:{gene_name} AND clinvar.rcv.clinical_significance:{sig}"
        params = {"q": query, "fields": "clinvar.hgvs.protein", "size": 1000}
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            for hit in resp.json().get("hits", []):
                protein = hit.get("clinvar", {}).get("hgvs", {}).get("protein", "")
                proteins = [protein] if isinstance(protein, str) else protein
                for p in proteins:
                    m = re.search(r"p\.[A-Z][a-z]{2}(\d+)", str(p))
                    if m:
                        positions.add(int(m.group(1)))
        except Exception as e:
            print(f"  myvariant.info {sig} failed: {e}")
    return positions


def load_existing_pm5():
    path = os.path.join(DATA_DIR, "pathogenic_positions.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return {k: set(v) for k, v in json.load(f).items()}
    return {}


def main():
    print("=" * 60)
    print("Building PM5 pathogenic positions from full ClinVar")
    print("=" * 60)
    all_positions = load_existing_pm5()
    print(f"Existing PM5: {', '.join(f'{k}={len(v)}' for k, v in all_positions.items())}")

    for gene_name, info in GENES.items():
        print(f"\n--- {gene_name} ---")
        if gene_name not in all_positions:
            all_positions[gene_name] = set()
        try:
            ncbi_pos = fetch_clinvar_pathogenic(gene_name, info["entrez"])
            print(f"  NCBI: {len(ncbi_pos)} positions")
            all_positions[gene_name].update(ncbi_pos)
        except Exception as e:
            print(f"  NCBI failed: {e}")
        time.sleep(1)
        try:
            mv_pos = fetch_via_myvariant(gene_name)
            print(f"  myvariant.info: {len(mv_pos)} positions")
            all_positions[gene_name].update(mv_pos)
        except Exception as e:
            print(f"  myvariant.info failed: {e}")
        time.sleep(1)

        print("\n" + "=" * 60)
    for gene, pos in sorted(all_positions.items()):
        print(f"  {gene}: {len(pos)} positions")

    output = {g: sorted(list(ps)) for g, ps in all_positions.items()}
    out_path = os.path.join(DATA_DIR, "pathogenic_positions.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {out_path}")

    meta = {
        "source": "ClinVar (NCBI + myvariant.info) + training dataset",
        "genes": list(GENES.keys()),
        "counts": {g: len(ps) for g, ps in all_positions.items()},
        "note": "PM5 positions from full ClinVar, not limited to training variants",
    }
    meta_path = os.path.join(DATA_DIR, "pathogenic_positions_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
