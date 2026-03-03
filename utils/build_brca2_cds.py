"""Fetch BRCA2 coding DNA sequence (CDS) from Ensembl and save locally."""
import urllib.request
import json

TRANSCRIPT_ID = "ENST00000380152"

def fetch_cds():
    url = f"https://rest.ensembl.org/sequence/id/{TRANSCRIPT_ID}?type=cds"
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req)
    data = json.loads(resp.read())
    cds = data["seq"]
    print(f"CDS length: {len(cds)} bp ({len(cds)//3} codons)")
    print(f"First 30bp: {cds[:30]}")
    print(f"Last 30bp:  {cds[-30:]}")
    
    with open("data/brca2_cds.txt", "w") as f:
        f.write(cds)
    print("Saved to data/brca2_cds.txt")
    return cds

if __name__ == "__main__":
    fetch_cds()
