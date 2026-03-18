"""
fetch_gnomad_myvariant.py - Fix gnomAD allele frequencies using myvariant.info API.

CRITICAL FIX: All gnomAD AFs in the current training data are zeros because the
original fetch_gnomad.py used the wrong API. This script uses myvariant.info
batch POST with ClinVar variant IDs to retrieve real gnomAD v4 allele frequencies.

Strategy:
  - Non-BRCA2 genes: query by ClinVar variant ID (clinvar_id column in raw data)
  - BRCA2: query by HGVS cDNA notation (has real nucleotide data)
  - gnomAD proxy-benign: already have AF from gnomAD (by definition)

API: myvariant.info POST /v1/query (scopes: clinvar.variant_id)
Output: data/{gene}_gnomad_frequencies.pkl per gene
"""

import os, sys, json, time, pickle, requests
import pandas as pd
import numpy as np

DATA_DIR = "data"
MYVARIANT_API = "https://myvariant.info/v1/query"
FIELDS = ",".join([
    "gnomad_exome.af.af",
    "gnomad_exome.af.af_afr", "gnomad_exome.af.af_amr",
    "gnomad_exome.af.af_asj", "gnomad_exome.af.af_eas",
    "gnomad_exome.af.af_fin", "gnomad_exome.af.af_nfe",
    "gnomad_exome.af.af_sas",
    "gnomad_genome.af.af",
    "gnomad_genome.af.af_afr", "gnomad_genome.af.af_amr",
    "gnomad_genome.af.af_eas", "gnomad_genome.af.af_nfe",
    "gnomad_genome.af.af_sas",
])

GENE_NM = {
    "BRCA1": "NM_007294.4", "BRCA2": "NM_000059.4",
    "PALB2": "NM_024675.4", "RAD51C": "NM_058216.3",
    "RAD51D": "NM_002878.3",
}


def safe_float(val, default=0.0):
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, list):
        for v in val:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def extract_gnomad(hit):
    result = {"af": 0.0, "afr": 0.0, "amr": 0.0, "eas": 0.0, "nfe": 0.0, "sas": 0.0, "popmax_af": 0.0}
    for source in ["gnomad_exome", "gnomad_genome"]:
        data = hit.get(source, {})
        if not data:
            continue
        af_data = data.get("af", {})
        if not af_data:
            continue
        af = safe_float(af_data.get("af"))
        if af > result["af"]:
            result["af"] = af
            result["afr"] = safe_float(af_data.get("af_afr"))
            result["amr"] = safe_float(af_data.get("af_amr"))
            result["eas"] = safe_float(af_data.get("af_eas"))
            result["nfe"] = safe_float(af_data.get("af_nfe"))
            result["sas"] = safe_float(af_data.get("af_sas"))
            pops = [result["afr"], result["amr"], result["eas"], result["nfe"], result["sas"]]
            result["popmax_af"] = max(pops) if pops else 0.0
    return result


def query_by_clinvar_ids(ids, batch_size=200):
    """Query myvariant.info by ClinVar variant IDs."""
    results = {}
    id_strs = [str(i) for i in ids]
    for i in range(0, len(id_strs), batch_size):
        batch = id_strs[i:i + batch_size]
        try:
            resp = requests.post(MYVARIANT_API, data={
                "q": ",".join(batch),
                "scopes": "clinvar.variant_id",
                "fields": FIELDS,
                "size": len(batch),
            }, timeout=60)
            if resp.status_code == 200:
                hits = resp.json()
                if isinstance(hits, list):
                    for hit in hits:
                        qid = str(hit.get("query", ""))
                        if not hit.get("notfound"):
                            results[qid] = extract_gnomad(hit)
            else:
                print(f"  WARNING: HTTP {resp.status_code} for batch {i // batch_size}")
        except Exception as e:
            print(f"  ERROR: {e} for batch {i // batch_size}")
        if i + batch_size < len(id_strs):
            time.sleep(0.5)
        if (i // batch_size) % 5 == 0:
            print(f"  Progress: {min(i + batch_size, len(id_strs))}/{len(id_strs)}")
    return results


def query_by_hgvs(gene, df, batch_size=200):
    """Query myvariant.info by HGVS cDNA for BRCA2 (which has real nucleotide data)."""
    nm = GENE_NM.get(gene, "")
    results = {}
    queries = []
    query_to_idx = {}
    for idx, row in df.iterrows():
        ref = str(row.get("Ref_nt", ""))
        alt = str(row.get("Alt_nt", ""))
        cpos = int(row.get("cDNA_pos", 0))
        if ref != "N" and alt != "N" and ref and alt and cpos > 0:
            hgvs = f"{nm}:c.{cpos}{ref}>{alt}"
            queries.append(hgvs)
            query_to_idx[hgvs] = idx

    print(f"  HGVS queries: {len(queries)}")
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        try:
            resp = requests.post(MYVARIANT_API, data={
                "q": ",".join(batch),
                "scopes": "clinvar.hgvs.coding,dbnsfp.hgvsc",
                "fields": FIELDS,
                "size": len(batch),
            }, timeout=60)
            if resp.status_code == 200:
                hits = resp.json()
                if isinstance(hits, list):
                    for hit in hits:
                        q = hit.get("query", "")
                        if not hit.get("notfound") and q in query_to_idx:
                            results[query_to_idx[q]] = extract_gnomad(hit)
            else:
                print(f"  WARNING: HTTP {resp.status_code} for batch {i // batch_size}")
        except Exception as e:
            print(f"  ERROR: {e} for batch {i // batch_size}")
        if i + batch_size < len(queries):
            time.sleep(0.5)
        if (i // batch_size) % 5 == 0:
            print(f"  Progress: {min(i + batch_size, len(queries))}/{len(queries)}")
    return results


def main():
    print("=" * 60)
    print("gnomAD Allele Frequency Fix via myvariant.info")
    print("=" * 60)

    genes = ["brca1", "brca2", "palb2", "rad51c", "rad51d"]

    for g in genes:
        print(f"\n--- {g.upper()} ---")

        if g == "brca2":
            path = "brca2_missense_dataset_2.csv"
        else:
            path = os.path.join(DATA_DIR, g, f"{g}_missense_dataset_unified.csv")

        if not os.path.exists(path):
            print(f"  SKIP: {path} not found")
            continue

        df = pd.read_csv(path)
        print(f"  Loaded {len(df)} variants")

        by_variant = {}
        by_position = {}

        if g == "brca2":
            idx_results = query_by_hgvs(g.upper(), df)
            for idx, gnomad_data in idx_results.items():
                row = df.iloc[idx]
                aa_key = f"{row['AA_ref']}{int(row['AA_pos'])}{row['AA_alt']}"
                by_variant[aa_key] = gnomad_data
                by_position[int(row["AA_pos"])] = gnomad_data
        else:
            if "clinvar_id" not in df.columns:
                print(f"  SKIP: no clinvar_id column")
                continue
            valid = df[df["clinvar_id"].notna()]
            ids = valid["clinvar_id"].astype(int).tolist()
            print(f"  Querying {len(ids)} ClinVar IDs...")
            id_results = query_by_clinvar_ids(ids)

            for _, row in valid.iterrows():
                cid = str(int(row["clinvar_id"]))
                if cid in id_results:
                    aa_key = f"{row['AA_ref']}{int(row['AA_pos'])}{row['AA_alt']}"
                    by_variant[aa_key] = id_results[cid]
                    by_position[int(row["AA_pos"])] = id_results[cid]

        non_zero = sum(1 for v in by_variant.values() if v.get("af", 0) > 0)
        print(f"  Results: {len(by_variant)} variants, {non_zero} with AF>0")

        output = {"by_variant": by_variant, "by_position": by_position}
        out_path = os.path.join(DATA_DIR, f"{g}_gnomad_frequencies.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(output, f)
        print(f"  Saved: {out_path}")

    print("\n" + "=" * 60)
    print("Done! Per-gene gnomAD frequency files created.")


if __name__ == "__main__":
    main()
