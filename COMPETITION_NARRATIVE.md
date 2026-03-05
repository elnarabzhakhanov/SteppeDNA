# SteppeDNA: Competition Narrative & Framing Guide

**For: Infomatrix Asia 2026 — Applied Science Category**
**Author: Elnar Abzhakhanov**

---

## The Core Story (30-Second Pitch)

Hereditary cancer risk assessment depends on correctly classifying genetic variants — but current tools were built primarily for well-studied genes with large European datasets. **SteppeDNA is a multi-gene pathogenicity predictor that demonstrates both what AI can achieve when data is abundant (BRCA2: AUC 0.983) and, critically, where it fails when data is scarce** — exposing a structural equity gap in genomic medicine that disproportionately affects underserved populations, including Central Asia.

---

## Why This Matters: The Equity Gap

### The Problem SteppeDNA Exposes

1. **Data concentration:** 52% of our 19,223 training variants come from a single gene (BRCA2). The remaining four genes — BRCA1, PALB2, RAD51C, RAD51D — have progressively fewer classified variants.

2. **Performance mirrors data availability:**
   | Gene | Training variants | Test AUC | Reliability |
   |------|------------------|----------|-------------|
   | BRCA2 | 10,085 | **0.983** | HIGH |
   | RAD51D | 410 | 0.804 | MODERATE |
   | RAD51C | 675 | 0.743 | MODERATE |
   | BRCA1 | 5,432 | 0.706 | LOW |
   | PALB2 | 2,621 | 0.641 | LOW |

3. **Population bias is structural:** Our gnomAD population frequency analysis found that **0% of training variants have East Asian allele frequency data.** The model has no population-specific frequency signals for Central Asian, East Asian, or African populations — it was trained entirely on data biased toward European ancestry.

4. **Temporal validation confirms the gap:** When we retrained on pre-2024 ClinVar data and tested on 2024+ classifications, BRCA2 maintained AUC 0.983 — but non-BRCA2 genes collapsed to 0.51-0.61, because most non-BRCA2 classifications only happened recently.

### Why Kazakhstan / Central Asia

- Kazakhstan has **no population-specific variant frequency data** in global databases (gnomAD, ExAC, 1000 Genomes)
- BRCA1/2 founder mutations documented in Central Asian populations are underrepresented in ClinVar
- Hereditary breast/ovarian cancer genetic testing is emerging in Kazakhstan — but the tools being deployed were calibrated for European populations
- SteppeDNA's transparent per-gene reliability tiers would prevent a clinician from over-trusting a prediction for an underserved gene

---

## Technical Contributions (What Makes This More Than "Just a Classifier")

### 1. Honest Performance Reporting
Unlike most ML papers that report a single headline AUC, SteppeDNA reports:
- **Per-gene performance** with explicit reliability tiers (HIGH/MODERATE/LOW)
- **Gene-specific warnings** shown to users in the UI
- **Known limitations** documented: BRCA2 dominance, European ancestry bias, temporal bias, compound heterozygosity limitation

### 2. Temporal Validation (Prospective Simulation)
- Train on pre-2024 ClinVar, test on 2024+ classifications
- Overall AUC: **0.964** (demonstrates temporal generalization)
- BRCA2: **0.983** temporal AUC (robust across time)
- Non-BRCA2 degradation quantified and explained

### 3. Multi-Source Feature Integration (103 Features)
- **ESM-2 protein language model** embeddings (per-gene, 20 PCA components)
- **AlphaMissense** deep learning scores
- **MAVE** functional assay data (where available)
- **AlphaFold 3D structure** features (RSA, B-factor, DNA/protein contact distances)
- **PhyloP** cross-species conservation
- **SpliceAI** splice prediction
- Gene-agnostic design — no gene-identifying features

### 4. GPU-Enhanced Embeddings (Ablation Study)
- **ESM-2 650M** (upgraded from 8M) + **LoRA fine-tuned** embeddings
- **GNN structural features** from AlphaFold 3D structure
- Result: Overall AUC improved to **0.984** (+0.005) but PALB2/RAD51C/D degraded
- **Key insight:** More features don't help when data is scarce — they overfit to the majority gene
- This finding itself is a contribution: demonstrates the data-scarcity bottleneck

### 5. ACMG Rule Approximation
- Computational approximation of 8 ACMG clinical criteria (PVS1, PS1, PM1, PM2, PM4, PM5, PP3, BA1, BS1, BP4, BP7)
- Gene-specific thresholds (BS1: BRCA1/2: 0.001, PALB2: 0.002, RAD51C/D: 0.005)
- Clearly labeled as "computational approximation" — not replacing expert review

### 6. Variant Type Coverage
- Missense, nonsense, frameshift, splice-site (canonical ±1-2bp and near ±3-8bp)
- In-frame indels with domain-awareness
- Synonymous variants with splice-proximity check
- Compound heterozygosity detection from VCF

---

## SOTA Comparison

| Predictor | Test Set AUC | Independent Benchmark AUC |
|-----------|-------------|--------------------------|
| **SteppeDNA** | **0.978** | **0.719-0.793** |
| REVEL | 0.725 | — |
| BayesDel | 0.721 | — |
| CADD | 0.539 | — |

**Context note:** SteppeDNA's 0.978 is evaluated on its own test set (potential optimistic bias). Independent benchmark on ProteinGym DMS + ClinVar Expert Panel gives 0.719-0.793. REVEL/BayesDel/CADD scores from myvariant.info on the same test set.

---

## The Narrative Arc (For Judges)

### Opening: The Promise
"AI can predict whether a genetic variant causes cancer — but only if it's trained on enough data."

### Middle: The Gap
"SteppeDNA achieves 0.983 AUC on BRCA2 (the most studied gene). But when we look at PALB2 — a gene that matters equally for hereditary cancer risk — performance drops to 0.641. Not because the method is wrong, but because the data doesn't exist."

### Turning Point: The Equity Question
"For a patient in Almaty or Nur-Sultan undergoing genetic testing for hereditary breast cancer, the variant might be in PALB2 or RAD51C. SteppeDNA would flag this with a red warning: 'Low reliability — limited training data.' Most commercial tools don't tell you this."

### Resolution: What SteppeDNA Contributes
"This project doesn't just predict pathogenicity — it maps where prediction fails, and for whom. The per-gene reliability system, population bias analysis, and temporal validation together create a framework for responsible deployment of genomic AI in regions like Central Asia where data is scarce."

---

## Key Figures to Show Judges

1. **Per-gene AUC bar chart** — dramatic visual showing BRCA2 dominance vs underserved genes
2. **Temporal validation ROC** — proves model generalizes across time
3. **Population analysis** — 0% East Asian AF data, structural gap
4. **GPU ablation** — shows that more features != better for small genes
5. **VUS reclassification** — real-world clinical utility demonstration
6. **ACMG rule badges** — computational approximation of clinical standards

---

## Potential Judge Questions & Answers

**Q: Isn't this just another variant classifier?**
A: The classifier is the vehicle. The contribution is the systematic analysis of where AI-based genomic prediction fails — per-gene, per-population, across time — and an honest reporting framework that prevents misuse.

**Q: Why is BRCA2 so much better?**
A: Data volume. BRCA2 has 10,085 training variants because it's been the focus of clinical research for 25+ years. PALB2 was only recognized as a high-risk gene in 2014. This is a scientific resource allocation problem, not a modeling problem.

**Q: How is this relevant to Kazakhstan?**
A: Kazakhstan has zero population-specific variant frequency data in gnomAD. When a genetic test returns a VUS (variant of uncertain significance) for a Kazakh patient, current tools provide no population context. SteppeDNA's reliability warnings and population bias documentation directly address this gap.

**Q: What would improve the model?**
A: More data for underserved genes (clinical partnerships needed), population-specific variant databases for Central Asia (BioBank studies), and functional assay data (MAVE experiments) for PALB2/RAD51C/RAD51D.

**Q: Is this clinically deployable?**
A: Not yet — it's research-use-only (RUO). Clinical deployment requires prospective validation, regulatory certification, and clinical expert review. The software clearly labels this in every output.

---

## Competition Strategy

1. **Lead with the equity narrative** — judges remember the "why" more than the "how"
2. **Show the per-gene gap visually** — this is your most compelling figure
3. **Demonstrate temporal validation** — proves scientific rigor
4. **VUS reclassification demo** — shows real-world utility (live if possible)
5. **Acknowledge limitations first** — builds credibility ("we know what we don't know")
6. **Close with the Kazakhstan angle** — personal connection + regional relevance
