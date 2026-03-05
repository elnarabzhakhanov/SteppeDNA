# grabs known pathogenic/benign variants straight from the training CSV and checks
# if the live API gets them right. basically a regression test so I don't break things.
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
import json
import urllib.request

df = pd.read_csv("brca2_missense_dataset_2.csv")

def predict(row):
    data = json.dumps({
        "cDNA_pos": int(row["cDNA_pos"]),
        "AA_ref": row["AA_ref"],
        "AA_alt": row["AA_alt"],
        "Mutation": row["Mutation"],
        "AA_pos": int(row["AA_pos"])
    }).encode()
    req = urllib.request.Request(
        "http://localhost:8000/predict", data=data,
        headers={"Content-Type": "application/json"}
    )
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())

# Test known pathogenic MISSENSE variants
path_missense = df[(df["Label"] == 1) & (df["AA_alt"] != "Ter")].head(5)
print("=== Known PATHOGENIC missense variants ===")
correct_path = 0
for _, row in path_missense.iterrows():
    r = predict(row)
    name = f"{row['AA_ref']}{int(row['AA_pos'])}{row['AA_alt']}"
    phylop_feats = [f for f in r["shap_explanation"] if "PhyloP" in f["feature"] or "Conservation" in f["feature"]]
    ok = "CORRECT" if r["prediction"] == "Pathogenic" else "WRONG"
    if r["prediction"] == "Pathogenic":
        correct_path += 1
    print(f"  {name:20s} => {r['prediction']:12s} (p={r['probability']:.4f}) [{ok}]  PhyloP SHAP: {phylop_feats}")

print()

# Test known benign MISSENSE variants
ben_missense = df[(df["Label"] == 0) & (df["AA_alt"] != "Ter")].head(5)
print("=== Known BENIGN missense variants ===")
correct_ben = 0
for _, row in ben_missense.iterrows():
    r = predict(row)
    name = f"{row['AA_ref']}{int(row['AA_pos'])}{row['AA_alt']}"
    ok = "CORRECT" if r["prediction"] == "Benign" else "WRONG"
    if r["prediction"] == "Benign":
        correct_ben += 1
    print(f"  {name:20s} => {r['prediction']:12s} (p={r['probability']:.4f}) [{ok}]")

print()
print(f"Pathogenic accuracy: {correct_path}/{len(path_missense)}")
print(f"Benign accuracy:     {correct_ben}/{len(ben_missense)}")
print(f"Overall:             {correct_path + correct_ben}/{len(path_missense) + len(ben_missense)}")
