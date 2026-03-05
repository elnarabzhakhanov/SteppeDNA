import requests
import time

payload = {
    "cDNA_pos": 1500,
    "AA_ref": "Ala",
    "AA_alt": "Ter",
    "Mutation": "Unknown",
    "gene_name": "BRCA2"
}

print(f"Testing Nonsense mutation (Ala -> Ter) via /predict...")

max_retries = 15
for i in range(max_retries):
    try:
        resp = requests.post("http://localhost:8000/predict", json=payload, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print("\nSUCCESS! Tier-1 Rule Intercepted.")
            print(f"Prediction: {data.get('prediction')}")
            print(f"Probability: {data.get('probability')}")
            print(f"ACMG Evidence: {data.get('acmg_evidence')}")
            print(f"SHAP Reason: {data.get('shap_explanation')[0] if data.get('shap_explanation') else None}")
            break
        else:
            print(f"Failed with status: {resp.status_code}")
            print(resp.text)
            break
    except Exception as e:
        print(f"[{i+1}/{max_retries}] Server not ready, waiting 3s...")
        time.sleep(3)
