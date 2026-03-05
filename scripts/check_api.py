# quick sanity check — hit the predict endpoint and dump what comes back.
# mostly used after restarting uvicorn to make sure nothing broke.
import requests
import json

url = "http://localhost:8000/predict"
data = {
    "cDNA_pos": 8165,
    "AA_ref": "Thr",
    "AA_alt": "Arg",
    "Mutation": "c.8165A>G"
}

try:
    resp = requests.post(url, json=data)
    print("Status:", resp.status_code)
    try:
        j = resp.json()
        print(json.dumps(j, indent=2))
        
        ds = j.get("data_sources", {})
        print("\nDATA SOURCES CHECK:")
        print(f"AM: {ds.get('alphamissense')}")
        print(f"MAVE: {ds.get('mave')}")
        print(f"PhyloP: {ds.get('phylop')}")
    except Exception as e:
        print("JSON Error:", e)
        print("Raw:", resp.text)
except Exception as e:
    print("Request Failed:", e)
