"""
fetch_alphafold_all_genes.py — Fetch AlphaFold predicted structures for ALL 5 HR genes
and extract per-residue structural features.

This equalizes the structural feature gap: previously only BRCA2 had structural data
(RSA, bfactor, secondary structure, contact distances). After running this script,
all 5 genes will have real structural features instead of default zeros.

AlphaFold DB UniProt IDs:
  BRCA1: P38398 (1863 AA)
  BRCA2: P51587 (3418 AA)
  PALB2: Q86YC2 (1186 AA)
  RAD51C: O43502 (376 AA)
  RAD51D: O75771 (328 AA)

Output: data/{gene}_structural_features.pkl for each gene
  Format: { aa_pos: {rsa, bfactor, ss_helix, ss_sheet, dist_dna, dist_partner, is_dna_contact, plddt} }

Dependencies: biopython, requests, numpy
Run: python data_pipelines/fetch_alphafold_all_genes.py
"""

import os
import sys
import pickle
import requests
import numpy as np

try:
    from Bio.PDB.DSSP import dssp_dict_from_pdb_file
except ImportError:
    pass  # DSSP optional, will fallback to pLDDT-based estimation

DATA_DIR = "data"
ALPHAFOLD_BASE = "https://alphafold.ebi.ac.uk/files"

GENE_UNIPROT = {
    "BRCA1": "P38398",
    "BRCA2": "P51587",
    "PALB2": "Q86YC2",
    "RAD51C": "O43502",
    "RAD51D": "O75771",
}

# DNA-binding domain residue ranges per gene
GENE_DNA_BINDING = {
    "BRCA1": [(452, 1079)],
    "BRCA2": [(2402, 3190)],
    "PALB2": [(395, 442)],
    "RAD51C": [(80, 376)],
    "RAD51D": [(1, 83)],
}

# Primary interaction partner interface residues
GENE_PARTNER_INTERFACE = {
    "BRCA1": [(8, 96)],
    "BRCA2": [(10, 40)],
    "PALB2": [(1085, 1186)],
    "RAD51C": [(1, 376)],
    "RAD51D": [(84, 328)],
}


def download_alphafold_pdb(uniprot_id, gene_name):
    """Download AlphaFold PDB file from EBI."""
    cache_path = os.path.join(DATA_DIR, f"alphafold_{gene_name.lower()}.pdb")
    if os.path.exists(cache_path):
        print(f"  Using cached PDB: {cache_path}")
        return cache_path

    url = f"{ALPHAFOLD_BASE}/AF-{uniprot_id}-F1-model_v4.pdb"
    print(f"  Downloading AlphaFold for {gene_name} ({uniprot_id})...")
    resp = requests.get(url, timeout=120)

    if resp.status_code != 200:
        print(f"  WARNING: HTTP {resp.status_code} for {gene_name}")
        return None

    with open(cache_path, "w") as f:
        f.write(resp.text)
    print(f"  Saved: {cache_path} ({len(resp.text) // 1024} KB)")
    return cache_path


def extract_plddt(pdb_path):
    """Extract pLDDT scores from B-factor column of AlphaFold PDB."""
    plddt = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                resnum = int(line[22:26].strip())
                bfactor = float(line[60:66].strip())
                plddt[resnum] = bfactor
    return plddt


def extract_ca_coords(pdb_path):
    """Extract Calpha coordinates."""
    coords = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                resnum = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                coords[resnum] = np.array([x, y, z])
    return coords


def compute_rsa_ss(pdb_path, gene_name):
    """Compute RSA and secondary structure using DSSP or pLDDT fallback."""
    rsa_dict, ss_dict = {}, {}

    try:
        dssp_result = dssp_dict_from_pdb_file(pdb_path)
        dssp_data = dssp_result[0]
        for key, values in dssp_data.items():
            resnum = key[1][1]
            rsa_dict[resnum] = float(values[2])
            ss_dict[resnum] = values[1]
        print(f"  DSSP: {len(rsa_dict)} residues for {gene_name}")
        return rsa_dict, ss_dict
    except Exception:
        pass

    # Fallback: pLDDT -> approximate RSA
    plddt = extract_plddt(pdb_path)
    for resnum, score in plddt.items():
        if score >= 90:
            rsa_dict[resnum] = 0.10
        elif score >= 70:
            rsa_dict[resnum] = 0.15 + (90 - score) * 0.01
        elif score >= 50:
            rsa_dict[resnum] = 0.35 + (70 - score) * 0.015
        else:
            rsa_dict[resnum] = 0.65 + (50 - score) * 0.01
        ss_dict[resnum] = "H" if score >= 80 else ("C" if score >= 60 else "-")

    print(f"  pLDDT-estimated RSA: {len(rsa_dict)} residues for {gene_name}")
    return rsa_dict, ss_dict


def min_dist_to_region(pos, coords, regions):
    """Min Ca distance from pos to any residue in functional regions."""
    if pos not in coords or not regions:
        return 999.0
    pos_coord = coords[pos]
    min_d = 999.0
    for start, end in regions:
        for t in range(start, end + 1):
            if t in coords and t != pos:
                d = np.linalg.norm(pos_coord - coords[t])
                if d < min_d:
                    min_d = d
    return float(min_d)


def process_gene(gene_name, uniprot_id):
    """Process one gene: download, extract structural features."""
    print(f"\n--- {gene_name} ({uniprot_id}) ---")
    pdb_path = download_alphafold_pdb(uniprot_id, gene_name)
    if pdb_path is None:
        return None

    plddt = extract_plddt(pdb_path)
    print(f"  pLDDT: {len(plddt)} residues, range {min(plddt.values()):.0f}-{max(plddt.values()):.0f}")

    rsa_dict, ss_dict = compute_rsa_ss(pdb_path, gene_name)
    ca_coords = extract_ca_coords(pdb_path)

    dna_regions = GENE_DNA_BINDING.get(gene_name, [])
    partner_regions = GENE_PARTNER_INTERFACE.get(gene_name, [])

    features = {}
    for pos in sorted(plddt.keys()):
        rsa = rsa_dict.get(pos, 0.4)
        ss = ss_dict.get(pos, "-")
        dist_dna = min_dist_to_region(pos, ca_coords, dna_regions)
        dist_partner = min_dist_to_region(pos, ca_coords, partner_regions)

        features[pos] = {
            "rsa": round(rsa, 4),
            "bfactor": round(plddt[pos], 1),
            "ss_helix": 1 if ss in ("H", "G", "I") else 0,
            "ss_sheet": 1 if ss in ("E", "B") else 0,
            "dist_dna": round(dist_dna, 2),
            "dist_palb2": round(dist_partner, 2),
            "is_dna_contact": 1 if dist_dna < 8.0 else 0,
            "is_buried": 1 if rsa < 0.2 else 0,
            "plddt": round(plddt[pos], 1),
        }
    return features


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    results = {}

    for gene_name, uniprot_id in GENE_UNIPROT.items():
        features = process_gene(gene_name, uniprot_id)
        if features is not None:
            results[gene_name] = features
            out_path = os.path.join(DATA_DIR, f"{gene_name.lower()}_structural_features.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(features, f)
            print(f"  Saved: {out_path} ({len(features)} residues)")

    # Backward compat: update combined structural_features.pkl
    if "BRCA2" in results:
        combined_path = os.path.join(DATA_DIR, "structural_features.pkl")
        with open(combined_path, "wb") as f:
            pickle.dump(results["BRCA2"], f)
        print(f"\n  Updated structural_features.pkl (BRCA2 backward compat)")

    print(f"\n{'='*60}")
    print("STRUCTURAL FEATURE EXTRACTION COMPLETE")
    print(f"{'='*60}")
    for gene in GENE_UNIPROT:
        if gene in results:
            n = len(results[gene])
            avg_plddt = np.mean([v["plddt"] for v in results[gene].values()])
            buried = sum(1 for v in results[gene].values() if v["is_buried"])
            print(f"  {gene}: {n} residues, pLDDT {avg_plddt:.0f} avg, {buried} buried ({100*buried/n:.0f}%)")
        else:
            print(f"  {gene}: FAILED")


if __name__ == "__main__":
    main()
