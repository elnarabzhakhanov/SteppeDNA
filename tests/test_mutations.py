# Regression tests: run known pathogenic/benign BRCA2 mutations through the API
# and verify prediction stability across restarts.
import requests
import json

URL = "http://localhost:8000/predict"

mutations = [
    # Classic pathogenic missense mutations
    ("D2723H (pathogenic)", {"cDNA_pos": 8167, "AA_ref": "Asp", "AA_alt": "His", "Mutation": "G>C"}),
    ("R2842C (pathogenic)", {"cDNA_pos": 8524, "AA_ref": "Arg", "AA_alt": "Cys", "Mutation": "C>T"}),
    ("T2722R (pathogenic)", {"cDNA_pos": 8165, "AA_ref": "Thr", "AA_alt": "Arg", "Mutation": "C>G"}),
    ("A2770T (pathogenic)", {"cDNA_pos": 8308, "AA_ref": "Ala", "AA_alt": "Thr", "Mutation": "G>A"}),
    # Classic benign missense mutations
    ("N372H (benign)", {"cDNA_pos": 1114, "AA_ref": "Asn", "AA_alt": "His", "Mutation": "A>C"}),
    ("V2466A (benign)", {"cDNA_pos": 7397, "AA_ref": "Val", "AA_alt": "Ala", "Mutation": "T>C"}),
    # Nonsense (current placeholder)
    ("Ser2058Ter (nonsense)", {"cDNA_pos": 6174, "AA_ref": "Ser", "AA_alt": "Ter", "Mutation": "C>A"}),
    # Some more clear-cut missense mutations
    ("Y3035S (pathogenic?)", {"cDNA_pos": 9103, "AA_ref": "Tyr", "AA_alt": "Ser", "Mutation": "A>C"}),
    ("D3095E (pathogenic?)", {"cDNA_pos": 9283, "AA_ref": "Asp", "AA_alt": "Glu", "Mutation": "T>A"}),
    ("I2627F (DNA-binding)", {"cDNA_pos": 7879, "AA_ref": "Ile", "AA_alt": "Phe", "Mutation": "A>T"}),
]

print("=" * 70)
print(f"{'Mutation':<28} {'Pred':<12} {'Prob':>6}  {'Confidence'}")
print("=" * 70)

for name, body in mutations:
    try:
        r = requests.post(URL, json=body, timeout=120)
        d = r.json()
        ci = d.get("confidence", {})
        print(f"{name:<28} {d['prediction']:<12} {d['probability']:>6.4f}  {ci.get('label', '?')}")
    except Exception as e:
        print(f"{name:<28} ERROR: {e}")

print("=" * 90)
print("\nRun the same test TWICE to check for fluctuations...")
print("\n--- Second run (same mutations, checking stability) ---\n")

# Run 3 key ones again to check stability
stability_tests = [
    ("D2723H (pathogenic)", {"cDNA_pos": 8167, "AA_ref": "Asp", "AA_alt": "His", "Mutation": "G>C"}),
    ("N372H (benign)", {"cDNA_pos": 1114, "AA_ref": "Asn", "AA_alt": "His", "Mutation": "A>C"}),
    ("Ser2058Ter (nonsense)", {"cDNA_pos": 6174, "AA_ref": "Ser", "AA_alt": "Ter", "Mutation": "C>A"}),
]

for name, body in stability_tests:
    results = []
    for i in range(3):
        r = requests.post(URL, json=body, timeout=120)
        d = r.json()
        ci = d.get("confidence", {})
        results.append((d["probability"], ci.get("mean", 0), ci.get("std", 0)))
    
    probs = [r[0] for r in results]
    means = [r[1] for r in results]
    print(f"{name}: probs={[round(p,4) for p in probs]} mc_means={[round(m,4) for m in means]}")
