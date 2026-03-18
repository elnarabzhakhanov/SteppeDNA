#!/usr/bin/env python3
"""
build_pm5_from_clinvar.py - Build PM5 pathogenic-position lookup.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
TARGET_GENES = {"BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D"}
CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
CLINVAR_CACHE = DATA_DIR / "variant_summary.txt.gz"
OUTPUT_PATH = DATA_DIR / "pathogenic_positions_clinvar.json"

STANDARD_AA_3 = {
    "Ala", "Arg", "Asn", "Asp", "Cys", "Gln", "Glu", "Gly", "His", "Ile",
    "Leu", "Lys", "Met", "Phe", "Pro", "Ser", "Thr", "Trp", "Tyr", "Val",
}

PROTEIN_CHANGE_RE = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})")
PATHOGENIC_SIGS = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}


def extract_from_existing_json():
    """Load positions from existing pathogenic_positions.json as baseline."""
    positions = defaultdict(set)
    existing = DATA_DIR / "pathogenic_positions.json"
    if not existing.exists():
        print(f"  [WARN] Existing positions file not found: {existing}")
        return positions
    with open(existing, "r", encoding="utf-8") as f:
        data = json.load(f)
    for gene in TARGET_GENES:
        if gene in data:
            positions[gene] = set(data[gene])
    return positions


def download_clinvar(force=False):
    if CLINVAR_CACHE.exists() and not force:
        print(f"  Using cached {CLINVAR_CACHE.name}")
        return True
    try:
        import requests
    except ImportError:
        print("  [WARN] requests not installed")
        return False
    print("  Downloading ClinVar variant_summary...")
    try:
        resp = requests.get(CLINVAR_URL, stream=True, timeout=300)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        dl = 0
        with open(CLINVAR_CACHE, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024*1024):
                f.write(chunk)
                dl += len(chunk)
                if total > 0:
                    print(f"\r  {dl/1048576:.1f} MB ({dl*100/total:.0f}%)", end="", flush=True)
        print("\n  Done.")
        return True
    except Exception as e:
        print(f"\n  [ERROR] {e}")
        if CLINVAR_CACHE.exists():
            CLINVAR_CACHE.unlink()
        return False


def extract_from_clinvar():
    positions = defaultdict(set)
    if not CLINVAR_CACHE.exists():
        return positions
    print(f"  Parsing {CLINVAR_CACHE.name}...")
    total_rows = matched = skipped_ns = skipped_syn = 0
    with gzip.open(CLINVAR_CACHE, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split("\t")
        ci = {n: i for i, n in enumerate(header)}
        tc, nc, gc, cc = ci.get("Type",1), ci.get("Name",2), ci.get("GeneSymbol",4), ci.get("ClinicalSignificance",6)
        for line in f:
            flds = line.strip().split("\t")
            if len(flds) <= max(tc, nc, gc, cc):
                continue
            gene = flds[gc].strip()
            if gene not in TARGET_GENES:
                continue
            total_rows += 1
            if flds[tc].strip() != "single nucleotide variant":
                continue
            if flds[cc].strip() not in PATHOGENIC_SIGS:
                continue
            m = PROTEIN_CHANGE_RE.search(flds[nc])
            if not m:
                continue
            ref, pos, alt = m.group(1), int(m.group(2)), m.group(3)
            if alt == "Ter":
                skipped_ns += 1; continue
            if ref == alt:
                skipped_syn += 1; continue
            if ref not in STANDARD_AA_3 or alt not in STANDARD_AA_3:
                continue
            if pos > 0:
                positions[gene].add(pos); matched += 1
    print(f"  Rows: {total_rows}, Matched: {matched}, Nonsense: {skipped_ns}, Syn: {skipped_syn}")
    return positions


def merge_positions(*sources):
    merged = defaultdict(set)
    for src in sources:
        for g, ps in src.items():
            merged[g].update(ps)
    for g in TARGET_GENES:
        if g not in merged:
            merged[g] = set()
    return {g: sorted(merged[g]) for g in sorted(merged)}


def write_output(positions, path):
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(positions, f, indent=2)
    print(f"\n  Written to {path}")
    for g in sorted(positions):
        print(f"    {g:8s}: {len(positions[g]):5d} positions")
    print(f"    TOTAL   : {sum(len(v) for v in positions.values()):5d} positions")


def compare_with_existing(new_pos):
    ep = DATA_DIR / "pathogenic_positions.json"
    if not ep.exists():
        return
    with open(ep) as f:
        old = json.load(f)
    print("\n  Comparison with existing pathogenic_positions.json:")
    print(f"  {'Gene':8s}  {'Old':>6s}  {'New':>6s}  {'Added':>6s}  {'Same':>6s}")
    for g in sorted(TARGET_GENES):
        o, n = set(old.get(g, [])), set(new_pos.get(g, []))
        print(f"  {g:8s}  {len(o):6d}  {len(n):6d}  {len(n-o):6d}  {len(n&o):6d}")


def main():
    ap = argparse.ArgumentParser(description="Build PM5 lookup from ClinVar + local data.")
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()
    out = Path(args.output) if args.output else OUTPUT_PATH

    print("=" * 60)
    print("  SteppeDNA: Build PM5 Pathogenic Positions Lookup")
    print("=" * 60)

    print("\n[1/3] Existing pathogenic_positions.json...")
    local = extract_from_existing_json()
    for g in sorted(local):
        print(f"  {g:8s}: {len(local[g]):5d} (local)")

    cv = defaultdict(set)
    if not args.local_only:
        print("\n[2/3] ClinVar...")
        if download_clinvar(force=args.force_download):
            cv = extract_from_clinvar()
            for g in sorted(cv):
                print(f"  {g:8s}: {len(cv[g]):5d} (ClinVar)")
        else:
            print("  [WARN] ClinVar failed; local only.")
    else:
        print("\n[2/3] Skipping ClinVar (--local-only).")

    print("\n[3/3] Merging...")
    merged = merge_positions(local, cv)
    write_output(merged, out)
    compare_with_existing(merged)
    print("\nDone.")


if __name__ == "__main__":
    main()
