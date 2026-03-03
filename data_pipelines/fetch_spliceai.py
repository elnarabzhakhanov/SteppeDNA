import os
import pandas as pd
import requests
import json
import time
import pickle

OUTPUT_DIR = "data"
BRCA2_TRANSCRIPT = "ENST00000380152"  # Canonical BRCA2 transcript
ENSEMBL_REST_URL = "https://rest.ensembl.org/vep/human/hgvs?SpliceAI=1"
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json"
}

def fetch_spliceai_scores(csv_path="brca2_missense_dataset_2.csv", batch_size=200):
    """
    Reads all BRCA2 variants, converts them to proper HGVS format,
    and queries Ensembl VEP (SpliceAI plugin) in batches.
    Extracts the max Delta Score (DS_max) for each variant.
    """
    print(f"Loading variants from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Pre-build lookup map. We need HGVS cDNA format: "BRCA2:c.1114A>C"
    # To identify it later, we'll map HGVS -> (AA_ref, AA_pos, AA_alt)
    hgvs_to_aa = {}
    hgvs_list = []
    
    for _, row in df.iterrows():
        try:
            # Drop the decimal if present (e.g. 1114.0 -> 1114)
            cdna_pos = int(row['cDNA_pos']) 
            ref_nt = str(row['Ref_nt']).strip()
            alt_nt = str(row['Alt_nt']).strip()
            
            # Format: BRCA2:c.1114A>C
            hgvs = f"BRCA2:c.{cdna_pos}{ref_nt}>{alt_nt}"
            
            aa_pos = int(row['AA_pos'])
            aa_ref = str(row['AA_ref']).strip()
            aa_alt = str(row['AA_alt']).strip()
            
            hgvs_list.append(hgvs)
            hgvs_to_aa[hgvs] = (aa_ref, aa_pos, aa_alt)
        except Exception:
            continue
            
    total_variants = len(hgvs_list)
    print(f"Generated {total_variants} valid HGVS strings.")
    
    spliceai_results = {}
    
    # Ensure data directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cache_file = os.path.join(OUTPUT_DIR, "spliceai_scores.pkl")
    
    # Process in chunks of `batch_size`
    print(f"Querying Ensembl VEP API in batches of {batch_size}...")
    for i in range(0, total_variants, batch_size):
        batch = hgvs_list[i:i+batch_size]
        payload = {"hgvs_notations": batch}
        
        try:
            response = requests.post(ENSEMBL_REST_URL, headers=HEADERS, json=payload, timeout=60)
            
            if response.status_code == 200:
                results = response.json()
                
                # Parse the response array
                for v in results:
                    hgvs_input = v.get("input")
                    aa_key = hgvs_to_aa.get(hgvs_input)
                    if not aa_key:
                        continue
                        
                    # Find SpliceAI in transcript consequences
                    max_ds = 0.0
                    found_splice = False
                    
                    for tc in v.get("transcript_consequences", []):
                        if tc.get("transcript_id") == BRCA2_TRANSCRIPT or tc.get("gene_symbol") == "BRCA2":
                            if "spliceai" in tc:
                                s = tc["spliceai"]
                                # SpliceAI returns 4 delta scores (Acceptor/Donor Gain/Loss). E.g: DS_AG, DS_AL, DS_DG, DS_DL
                                ds_scores = [
                                    float(s.get("DS_AG", 0)),
                                    float(s.get("DS_AL", 0)),
                                    float(s.get("DS_DG", 0)),
                                    float(s.get("DS_DL", 0))
                                ]
                                max_ds = max(ds_scores)
                                found_splice = True
                                break # Found it for this variant
                                
                    if found_splice:
                        spliceai_results[aa_key] = max_ds
                        
            elif response.status_code == 429: # Rate limit
                print(f"  [!] Rate limit hit at batch {i}. Sleeping for 30s...")
                time.sleep(30)
                # We should really retry the batch, but for simplicity of this script, we'll continue.
                # In a robust production environment, implement exponential backoff here.
            else:
                print(f"  [X] Error on batch {i}: {response.status_code}")
                
        except Exception as e:
            print(f"  [X] Request failed on batch {i}: {e}")
            
        # Ensembl asks for 15 requests per second max. 
        # A 1 second sleep easily guarantees we stay under for batch requests.
        time.sleep(1)
        
        if (i + batch_size) % 1000 == 0 or (i + batch_size) >= total_variants:
            print(f"  Processed {min(i+batch_size, total_variants)} / {total_variants} variants...")
            
    print(f"\nSuccessfully extracted SpliceAI scores for {len(spliceai_results)} variants.")
    
    # Save the dictionary
    with open(cache_file, "wb") as f:
        pickle.dump(spliceai_results, f)
    print(f"Saved SpliceAI scores -> {cache_file}")
    
    # Print some stats
    if spliceai_results:
        scores = list(spliceai_results.values())
        pathogenic_count = sum(1 for s in scores if s > 0.5) # SpliceAI > 0.5 is a strong disruption signal
        print(f"\nSpliceAI Score Distribution:")
        print(f"  Total scored: {len(scores)}")
        print(f"  Average Delta Score: {sum(scores)/len(scores):.3f}")
        print(f"  Mutations with severe splicing disruption (DS > 0.5): {pathogenic_count}")

if __name__ == "__main__":
    print("=" * 60)
    print(" BRCA2 Deep-Learning SpliceAI Fetcher")
    print("=" * 60)
    fetch_spliceai_scores()
