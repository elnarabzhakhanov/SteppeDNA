# checks that the MAVE functional scores are actually making it into
# the SHAP explanations. kept getting zeros here until I fixed the lookup key format.
import requests

API = "http://localhost:8000/predict"

tests = [
    {
        "name": "Asp2723His (MAVE pathogenic standard, score~1.0)",
        "data": {"cDNA_pos": 8167, "AA_ref": "Asp", "AA_alt": "His", "Mutation": "G>C", "AA_pos": 2723},
    },
    {
        "name": "Ser2483Asn (MAVE normal, score~5.09)",
        "data": {"cDNA_pos": 7448, "AA_ref": "Ser", "AA_alt": "Asn", "Mutation": "G>A", "AA_pos": 2483},
    },
    {
        "name": "Ala34Val (no MAVE data, outside DBD)",
        "data": {"cDNA_pos": 100, "AA_ref": "Ala", "AA_alt": "Val", "Mutation": "C>T", "AA_pos": 34},
    },
]

for t in tests:
    r = requests.post(API, json=t["data"])
    d = r.json()
    pred = d.get("prediction", "ERROR")
    prob = d.get("probability", -1)
    tier = d.get("risk_tier", "?")
    print(f"  {t['name']}")
    print(f"    Prediction: {pred} | Probability: {prob:.3f} | Risk: {tier}")
    mave_feats = [e for e in d.get("shap_explanation", []) if "MAVE" in e.get("name", "") or "mave" in e.get("name", "")]
    if mave_feats:
        for mf in mave_feats:
            print(f"    SHAP: {mf['name']} = {mf['value']:.4f} (impact: {mf['impact']:.4f})")
    print()
