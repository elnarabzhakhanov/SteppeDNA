"""
SteppeDNA: AlphaFold Full-Length Structural Enhancement
========================================================
Downloads AlphaFold's full-length BRCA2 structure prediction and enriches
the existing structural features with per-residue pLDDT confidence and
refined burial/RSA estimates.

This supplements the existing fetch_alphafold.py (ESMFold + PDB 1MIU)
by filling the mid-protein gap (~residues 400-2400) that lacks
experimental coverage.

Three-tier merge strategy:
  Tier 1: Experimental PDB 1MIU data (residues with real 3D coords)
  Tier 2: AlphaFold high-conf pLDDT > 70 (fills structural gap)
  Tier 3: Curated estimates + defaults (low-confidence regions)

Usage:
  python data_pipelines/fetch_alphafold_fullength.py

Output:
  data/structural_features_enhanced.pkl
"""

import os
import sys
import pickle
import urllib.request
import numpy as np

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# BRCA2 UniProt: P51587
UNIPROT_ID = "P51587"
BRCA2_LEN  = 3418

# AlphaFold DB now provides full-length structures for all human proteins
# Fragments are used for proteins >2700 AA. BRCA2 has multiple fragments.
AF_BASE = f"https://alphafold.ebi.ac.uk/files/AF-{UNIPROT_ID}-F"
AF_SUFFIX = "-model_v4.pdb"

PLDDT_HIGH = 70.0
PLDDT_LOW  = 50.0

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  SteppeDNA: AlphaFold Full-Length Enhancement")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Download AlphaFold fragment PDBs
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Downloading AlphaFold fragments for BRCA2...")

def parse_pdb_ca(pdb_path):
    """Parse CA atoms from a PDB file. Returns dict: resid -> {plddt, x, y, z}"""
    residues = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            try:
                resid   = int(line[22:26].strip())
                x       = float(line[30:38])
                y       = float(line[38:46])
                z       = float(line[46:54])
                bfactor = float(line[60:66])   # pLDDT in AlphaFold
            except (ValueError, IndexError):
                continue
            residues[resid] = {"plddt": bfactor, "x": x, "y": y, "z": z}
    return residues

all_af_residues = {}

# Try up to 6 fragments (F1 through F6, typical for 3418-AA protein)
for frag in range(1, 7):
    url = f"{AF_BASE}{frag}{AF_SUFFIX}"
    local = os.path.join(DATA_DIR, f"AF-{UNIPROT_ID}-F{frag}-model_v4.pdb")

    if os.path.exists(local):
        print(f"  Fragment F{frag}: cached ({local})")
    else:
        print(f"  Fragment F{frag}: downloading from {url} ...")
        try:
            urllib.request.urlretrieve(url, local)
            size_kb = os.path.getsize(local) / 1024
            print(f"    -> {size_kb:.0f} KB")
        except Exception as e:
            print(f"    -> Not found or failed ({e}). Stopping at F{frag-1}.")
            break

    parsed = parse_pdb_ca(local)
    # Merge (later fragments override earlier if overlapping)
    all_af_residues.update(parsed)
    print(f"    -> {len(parsed)} CA atoms (cumulative: {len(all_af_residues)})")

print(f"\n  Total AlphaFold residues: {len(all_af_residues)}/{BRCA2_LEN}")

if len(all_af_residues) == 0:
    print("\n  [ERROR] No AlphaFold data fetched.")
    print("  AlphaFold may fragment BRCA2 differently. Check:")
    print(f"    https://alphafold.ebi.ac.uk/entry/{UNIPROT_ID}")
    sys.exit(1)

# pLDDT stats
plddts = [r["plddt"] for r in all_af_residues.values()]
print(f"  pLDDT: mean={np.mean(plddts):.1f}, median={np.median(plddts):.1f}")
n_hi = sum(1 for p in plddts if p >= PLDDT_HIGH)
print(f"  High confidence (>={PLDDT_HIGH}): {n_hi} ({100*n_hi/len(plddts):.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Compute neighbor density for burial estimate
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Computing per-residue burial from AlphaFold coordinates...")

sorted_ids = sorted(all_af_residues.keys())
coord_arr  = np.array([[all_af_residues[r]["x"],
                         all_af_residues[r]["y"],
                         all_af_residues[r]["z"]] for r in sorted_ids])

RADIUS = 10.0
density = {}
for i, ri in enumerate(sorted_ids):
    dists = np.sqrt(np.sum((coord_arr - coord_arr[i])**2, axis=1))
    density[ri] = int(np.sum(dists < RADIUS)) - 1

max_d = max(density.values()) if density else 1
min_d = min(density.values()) if density else 0

# ─────────────────────────────────────────────────────────────────────────────
# 3. Three-tier merge with existing features
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Merging with existing structural features...")

existing_path = os.path.join(DATA_DIR, "structural_features.pkl")
if not os.path.exists(existing_path):
    print(f"  [ERROR] {existing_path} not found. Run fetch_alphafold.py first.")
    sys.exit(1)

with open(existing_path, "rb") as f:
    existing = pickle.load(f)
print(f"  Loaded {len(existing)} positions from structural_features.pkl")

enhanced = {}
tier_counts = {"pdb": 0, "alphafold": 0, "curated": 0}

for pos in range(1, BRCA2_LEN + 1):
    base    = existing.get(pos, {})
    af_data = all_af_residues.get(pos)

    # Determine if base has real PDB data (not just curated estimates)
    pdb_real = (
        base.get("dist_dna", 999.0) != 999.0 or
        base.get("is_dna_contact", False) or
        base.get("is_dss1_contact", False) or
        base.get("is_buried", False)
    )

    entry = dict(base)  # Start from existing data

    if pdb_real:
        # Tier 1: Experimental data — keep as-is, just add pLDDT info
        entry["af_plddt"]  = af_data["plddt"] if af_data else 0.0
        entry["data_tier"] = "PDB_experimental"
        tier_counts["pdb"] += 1

    elif af_data and af_data["plddt"] >= PLDDT_HIGH:
        # Tier 2: AlphaFold high-confidence fills the structural gap
        plddt = af_data["plddt"]
        dens  = density.get(pos, 0)

        # Refined RSA from neighbor density
        if max_d > min_d:
            af_rsa = round(1.0 - (dens - min_d) / (max_d - min_d), 3)
        else:
            af_rsa = 0.5

        entry["rsa"]       = af_rsa
        entry["is_buried"]  = af_rsa < 0.25
        entry["bfactor"]   = round(100.0 - plddt, 2)
        entry["af_plddt"]  = plddt
        entry["data_tier"] = "AlphaFold_v4"

        # Refine secondary structure from pLDDT pattern
        if plddt > 90:
            entry["ss"] = "H"
        elif plddt > 80:
            entry["ss"] = "E"
        # else keep the curated SS from fetch_alphafold.py

        tier_counts["alphafold"] += 1

    else:
        # Tier 3: Curated estimates (from fetch_alphafold.py) as-is
        entry["af_plddt"]  = af_data["plddt"] if af_data else 0.0
        entry["data_tier"] = "curated_estimate"
        tier_counts["curated"] += 1

    # Derived features for the model
    entry["ss_helix"] = int(entry.get("ss", "C") == "H")
    entry["ss_sheet"] = int(entry.get("ss", "C") == "E")

    enhanced[pos] = entry

# ─────────────────────────────────────────────────────────────────────────────
# 4. Save
# ─────────────────────────────────────────────────────────────────────────────
out_path = os.path.join(DATA_DIR, "structural_features_enhanced.pkl")
with open(out_path, "wb") as f:
    pickle.dump(enhanced, f)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Report
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STRUCTURAL ENHANCEMENT REPORT")
print("=" * 65)
total = BRCA2_LEN
print(f"\n  Tier 1 (PDB 1MIU experimental):  {tier_counts['pdb']:5d}  ({100*tier_counts['pdb']/total:.1f}%)")
print(f"  Tier 2 (AlphaFold high-conf):    {tier_counts['alphafold']:5d}  ({100*tier_counts['alphafold']/total:.1f}%)")
print(f"  Tier 3 (Curated/default):        {tier_counts['curated']:5d}  ({100*tier_counts['curated']/total:.1f}%)")
real_cov = tier_counts["pdb"] + tier_counts["alphafold"]
print(f"  Real structural data:            {real_cov:5d}  ({100*real_cov/total:.1f}%)")
print(f"\n  Output: {out_path}")
print(f"\n  To activate: replace structural_features.pkl with this file:")
print(f"    copy data/structural_features_enhanced.pkl data/structural_features.pkl")
print(f"\n  Done.")
