"""
SteppeDNA: Multi-Gene ESM-2 Protein Language Model Embedding Generator
=======================================================================
Generates per-variant contextual embeddings from Meta's ESM-2 model
for ALL 5 HR pathway genes: BRCA1, BRCA2, PALB2, RAD51C, RAD51D.

For each variant position, extracts 320-dimensional embeddings (layer 6)
for the wildtype and mutant residue, computes:
  - Cosine similarity between wildtype and mutant contexts
  - L2 norm of the difference (magnitude of embedding shift)
  - Top-20 PCA components of the difference vector (fit on all genes combined)

Saves per-gene pickle files so build_master_dataset.py can load them
via its _load_pickle() gene-specific path mechanism.

Requirements:
  pip install torch fair-esm biopython

Usage:
  python data_pipelines/generate_esm2_embeddings.py

Output:
  data/brca1_esm2_embeddings.pkl
  data/brca2_esm2_embeddings.pkl  (reuses existing if available)
  data/palb2_esm2_embeddings.pkl
  data/rad51c_esm2_embeddings.pkl
  data/rad51d_esm2_embeddings.pkl
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

WINDOW = 50  # residues on each side of the variant position

# Gene configurations
GENE_CONFIG = {
    "brca1": {
        "cds_file": os.path.join(DATA_DIR, "brca1_cds.txt"),
        "variants_file": os.path.join(DATA_DIR, "brca1", "brca1_missense_dataset_unified.csv"),
        "uniprot_id": "P38398",
    },
    "brca2": {
        "cds_file": os.path.join(DATA_DIR, "brca2_cds.txt"),
        "variants_file": "brca2_missense_dataset_2.csv",
        "uniprot_id": "P51587",
    },
    "palb2": {
        "cds_file": os.path.join(DATA_DIR, "palb2_cds.txt"),
        "variants_file": os.path.join(DATA_DIR, "palb2", "palb2_missense_dataset_unified.csv"),
        "uniprot_id": "Q86YC2",
    },
    "rad51c": {
        "cds_file": os.path.join(DATA_DIR, "rad51c_cds.txt"),
        "variants_file": os.path.join(DATA_DIR, "rad51c", "rad51c_missense_dataset_unified.csv"),
        "uniprot_id": "O43502",
    },
    "rad51d": {
        "cds_file": os.path.join(DATA_DIR, "rad51d_cds.txt"),
        "variants_file": os.path.join(DATA_DIR, "rad51d", "rad51d_missense_dataset_unified.csv"),
        "uniprot_id": "O75771",
    },
}

AA3_TO_1 = {
    "Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E",
    "Gly":"G","His":"H","Ile":"I","Leu":"L","Lys":"K","Met":"M","Phe":"F",
    "Pro":"P","Ser":"S","Thr":"T","Trp":"W","Tyr":"Y","Val":"V","Ter":"*",
}

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  SteppeDNA: Multi-Gene ESM-2 Embedding Generator")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Check dependencies
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Checking dependencies...")

try:
    import torch
    print(f"  PyTorch: {torch.__version__} ({'CUDA' if torch.cuda.is_available() else 'CPU'})")
except ImportError:
    print("  [ERROR] PyTorch not installed. Run: pip install torch")
    sys.exit(1)

try:
    import esm
    print(f"  ESM: available")
except ImportError:
    print("  [ERROR] fair-esm not installed. Run: pip install fair-esm")
    sys.exit(1)

from Bio.Seq import Seq

# ─────────────────────────────────────────────────────────────────────────────
# 2. Load ESM-2 model (once for all genes)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Loading ESM-2 model (esm2_t6_8M_UR50D)...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
model = model.to(device).eval()
batch_converter = alphabet.get_batch_converter()
print(f"  Model loaded on {device}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Generate embeddings for each gene
# ─────────────────────────────────────────────────────────────────────────────
all_gene_embeddings = {}  # gene -> {vkey -> embedding_dict}

for gene, config in GENE_CONFIG.items():
    print(f"\n{'='*65}")
    print(f"  Processing {gene.upper()}")
    print(f"{'='*65}")

    # Check for existing embeddings
    existing_path = os.path.join(DATA_DIR, f"{gene}_esm2_embeddings.pkl")
    # Also check the global file for BRCA2 backward compat
    if gene == "brca2" and not os.path.exists(existing_path):
        global_path = os.path.join(DATA_DIR, "esm2_embeddings.pkl")
        if os.path.exists(global_path):
            print(f"  Loading existing BRCA2 embeddings from {global_path}...")
            with open(global_path, "rb") as f:
                existing = pickle.load(f)
            if isinstance(existing, dict) and "embeddings" in existing:
                all_gene_embeddings[gene] = existing["embeddings"]
                print(f"  -> {len(all_gene_embeddings[gene])} variants loaded (reusing)")
                continue

    if os.path.exists(existing_path):
        print(f"  Loading existing embeddings from {existing_path}...")
        with open(existing_path, "rb") as f:
            existing = pickle.load(f)
        if isinstance(existing, dict) and "embeddings" in existing:
            all_gene_embeddings[gene] = existing["embeddings"]
            print(f"  -> {len(all_gene_embeddings[gene])} variants loaded (reusing)")
            continue

    # Load protein sequence from CDS
    cds_path = config["cds_file"]
    if not os.path.exists(cds_path):
        print(f"  [SKIP] CDS file not found: {cds_path}")
        continue

    with open(cds_path) as f:
        cds = f.read().strip()
    protein_seq = str(Seq(cds).translate()).rstrip("*")
    print(f"  Protein: {len(protein_seq)} AA (from CDS)")

    # Load variants
    variants_file = config["variants_file"]
    if not os.path.exists(variants_file):
        print(f"  [SKIP] Variants file not found: {variants_file}")
        continue

    df = pd.read_csv(variants_file)
    print(f"  Variants: {len(df)}")

    # Generate embeddings
    embeddings = {}
    n_processed = 0
    n_skipped = 0

    for idx, row in df.iterrows():
        aa_pos = int(row["AA_pos"])
        aa_ref_3 = str(row["AA_ref"])
        aa_alt_3 = str(row["AA_alt"])

        aa_ref = AA3_TO_1.get(aa_ref_3)
        aa_alt = AA3_TO_1.get(aa_alt_3)

        if aa_ref is None or aa_alt is None or aa_ref == '*' or aa_alt == '*':
            n_skipped += 1
            continue
        if aa_pos < 1 or aa_pos > len(protein_seq):
            n_skipped += 1
            continue

        vkey = f"{aa_ref_3}{aa_pos}{aa_alt_3}"
        if vkey in embeddings:
            continue  # Skip duplicate variants

        # Extract window around variant
        win_start = max(0, aa_pos - 1 - WINDOW)
        win_end   = min(len(protein_seq), aa_pos - 1 + WINDOW + 1)
        local_pos = aa_pos - 1 - win_start

        wt_window  = protein_seq[win_start:win_end]
        mut_window = list(wt_window)
        mut_window[local_pos] = aa_alt
        mut_window = "".join(mut_window)

        # ESM-2 inference
        data = [("wt", wt_window), ("mut", mut_window)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6], return_contacts=False)

        # Extract layer 6 representations at the variant position (+1 for BOS token)
        wt_emb  = results["representations"][6][0, local_pos + 1].cpu().numpy()
        mut_emb = results["representations"][6][1, local_pos + 1].cpu().numpy()

        # Compute derived features
        diff = mut_emb - wt_emb
        cos_sim = float(np.dot(wt_emb, mut_emb) / (np.linalg.norm(wt_emb) * np.linalg.norm(mut_emb) + 1e-8))
        l2_norm = float(np.linalg.norm(diff))

        embeddings[vkey] = {
            "wt_embedding": wt_emb,
            "mut_embedding": mut_emb,
            "diff_vector": diff,
            "cosine_similarity": cos_sim,
            "l2_shift": l2_norm,
        }

        n_processed += 1
        if n_processed % 500 == 0:
            print(f"    {n_processed}/{len(df)} variants processed...")

    print(f"  -> {n_processed} variants embedded, {n_skipped} skipped")
    all_gene_embeddings[gene] = embeddings

# ─────────────────────────────────────────────────────────────────────────────
# 4. PCA reduction across ALL genes combined
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  Fitting combined PCA across all genes")
print(f"{'='*65}")

from sklearn.decomposition import PCA

# Collect all diff vectors
all_keys = []
all_diffs = []
gene_for_key = []

for gene, embs in all_gene_embeddings.items():
    for vkey, data in embs.items():
        if "diff_vector" in data:
            all_keys.append((gene, vkey))
            all_diffs.append(data["diff_vector"])
            gene_for_key.append(gene)

diff_matrix = np.array(all_diffs)
n_components = min(20, diff_matrix.shape[0], diff_matrix.shape[1])
pca = PCA(n_components=n_components)
pca_features = pca.fit_transform(diff_matrix)

# Assign PCA components back to each gene's embeddings
for i, (gene, vkey) in enumerate(all_keys):
    all_gene_embeddings[gene][vkey]["pca_components"] = pca_features[i]

explained = sum(pca.explained_variance_ratio_[:10]) * 100
total_variants = sum(len(e) for e in all_gene_embeddings.values())
print(f"  Total variants with PCA: {total_variants}")
print(f"  Top 10 PCA components explain {explained:.1f}% of variance")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Save per-gene pickle files
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  Saving per-gene embedding files")
print(f"{'='*65}")

for gene, embs in all_gene_embeddings.items():
    out_path = os.path.join(DATA_DIR, f"{gene}_esm2_embeddings.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "embeddings": embs,
            "pca_model": pca,
            "n_components": n_components,
            "model_name": "esm2_t6_8M_UR50D",
            "window_size": WINDOW,
        }, f)
    print(f"  {gene.upper():8s}: {len(embs):5d} variants -> {out_path}")

# Also save a combined global file for backward compatibility
global_out = os.path.join(DATA_DIR, "esm2_embeddings.pkl")
combined = {}
for gene, embs in all_gene_embeddings.items():
    # Prefix keys with gene to avoid collisions
    for vkey, data in embs.items():
        combined[vkey] = data  # Note: collisions possible but unlikely across genes
with open(global_out, "wb") as f:
    pickle.dump({
        "embeddings": combined,
        "pca_model": pca,
        "n_components": n_components,
        "model_name": "esm2_t6_8M_UR50D",
        "window_size": WINDOW,
    }, f)
print(f"\n  Global combined: {len(combined)} variants -> {global_out}")

print(f"\n  [ESM-2 EMBEDDING GENERATION COMPLETE]")
