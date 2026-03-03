"""
Unified script to build CDS strings and cDNA-to-Genomic coordinate mappings
for all genes defined in backend/gene_configs/*.json.

Outputs:
  data/{gene}_cds.txt
  data/{gene}_cdna_to_genomic.pkl
"""
import os
import sys
import glob
import json
import time
import pickle
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(ROOT, "backend", "gene_configs")
DATA_DIR = os.path.join(ROOT, "data")

def fetch_cds(gene, transcript_id):
    url = f"https://rest.ensembl.org/sequence/id/{transcript_id}?type=cds"
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read().decode())
        cds = data["seq"]
        print(f"  [OK] Fetched {gene} CDS: {len(cds)} bp")
        out_path = os.path.join(DATA_DIR, f"{gene.lower()}_cds.txt")
        with open(out_path, "w") as f:
            f.write(cds)
    except Exception as e:
        print(f"  [ERROR] Failed to fetch CDS for {gene}: {e}")

def fetch_transcript_structure(gene, transcript_id, strand):
    url = f"https://rest.ensembl.org/lookup/id/{transcript_id}?expand=1;content-type=application/json"
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read().decode())
        
        exons = [(e["start"], e["end"]) for e in data.get("Exon", [])]
        translation = data.get("Translation", {})
        cds_start = translation.get("start", min(e[0] for e in exons))
        cds_end = translation.get("end", max(e[1] for e in exons))
        
        # Build mapping
        cdna_to_genomic = {}
        cdna_pos = 1
        
        sorted_exons = sorted(exons, key=lambda x: x[0], reverse=(strand == "-"))
        for exon_start, exon_end in sorted_exons:
            coding_start = max(exon_start, cds_start)
            coding_end = min(exon_end, cds_end)
            if coding_start > coding_end:
                continue
                
            if strand == "+":
                for genomic_pos in range(coding_start, coding_end + 1):
                    cdna_to_genomic[cdna_pos] = genomic_pos
                    cdna_pos += 1
            else:
                for genomic_pos in range(coding_end, coding_start - 1, -1):
                    cdna_to_genomic[cdna_pos] = genomic_pos
                    cdna_pos += 1

        print(f"  [OK] Built mapping for {gene}: {len(cdna_to_genomic)} positions")
        out_path = os.path.join(DATA_DIR, f"{gene.lower()}_cdna_to_genomic.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(cdna_to_genomic, f)
            
    except Exception as e:
        print(f"  [ERROR] Failed to build mapping for {gene}: {e}")

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    configs = glob.glob(os.path.join(CONFIG_DIR, "*.json"))
    
    print("=" * 60)
    print("  SteppeDNA: Building Universal Gene References")
    print("=" * 60)
    
    for cfg_path in configs:
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
            
        gene = cfg.get("gene_name", os.path.basename(cfg_path).split('.')[0].upper())
        transcript_id = cfg.get("transcript_id", "").split(".")[0]
        strand = cfg.get("strand")
        
        if not transcript_id or not strand:
            print(f"Skipping {gene}: Missing transcript_id or strand in config.")
            continue
            
        print(f"\nProcessing {gene} ({transcript_id}, Strand: {strand})...")
        time.sleep(0.5) # respect Ensembl API rate limits
        fetch_cds(gene, transcript_id)
        time.sleep(0.5)
        fetch_transcript_structure(gene, transcript_id, strand)

if __name__ == "__main__":
    main()
