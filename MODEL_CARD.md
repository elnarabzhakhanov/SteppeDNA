# SteppeDNA Model Card

Following the framework of Mitchell et al. (2019), "Model Cards for Model Reporting."

---

## Model Details

- **Model name:** SteppeDNA v5.3
- **Model type:** Ensemble classifier (XGBoost 60% + MLP Neural Network 40%) with isotonic calibration
- **Version:** 5.3.0 (March 2026)
- **Developed by:** Elnar Abzhakhanov
- **License:** Research Use Only (RUO). Code license: MIT. Model artifacts and predictions: Research Use Only (RUO).
- **Input:** Missense variant features (cDNA position, amino acid change, nucleotide mutation, gene name)
- **Output:** Calibrated pathogenicity probability (0.0-1.0), classification (Pathogenic/Benign), ACMG evidence codes, SHAP feature attributions
- **Features:** 103 engineered features (gene-identifying features removed)

---

## Intended Use

- **Primary use:** Research exploration tool for missense variant classification in 5 HR genes (BRCA1, BRCA2, PALB2, RAD51C, RAD51D). Not validated for clinical diagnosis. ML model handles missense variants only; nonsense, frameshift, and splice variants use rule-based classification. Germline variants only.
- **Intended users:** Genetics researchers, bioinformaticians, genetic counseling trainees
- **Out-of-scope uses:**
  - Clinical diagnostic decision-making without expert review
  - Standalone basis for medical treatment decisions
  - Variants outside the 5 supported HR genes
  - Non-missense variants (nonsense/frameshift handled by Tier 1 rules, not ML)
  - Population screening without clinical geneticist oversight
  - Somatic variant interpretation

---

## Training Data

- **Source:** ClinVar (P/LP and B/LB missense variants) + gnomAD v4 proxy-benign (AC >= 2)
- **Snapshot:** February 2026
- **Total:** 19,223 variants across 5 genes
- **Split:** 60% train / 20% calibration / 20% test (gene x label stratified)
- **Gene distribution:**
  | Gene | Total variants | % of dataset |
  |------|---------------|-------------|
  | BRCA2 | ~10,000 | ~52% |
  | BRCA1 | ~5,400 | ~28% |
  | PALB2 | ~2,600 | ~14% |
  | RAD51C | ~670 | ~3.5% |
  | RAD51D | ~410 | ~2.1% |

---

## Evaluation Metrics

### Overall Performance (Held-Out Test Set, n=3,845)

| Metric | Value |
|--------|-------|
| ROC-AUC (overall, weighted) | 0.978* |
| Macro-Averaged AUC (equal-weight per-gene) | 0.775 |
| PR-AUC | 0.965 |
| MCC | 0.881 |
| Balanced Accuracy | 94.1% |
| Sensitivity | 95.7% |
| Specificity | 92.6% |
| 10-Fold CV AUC | 0.9797 +/- 0.0031 |

*Overall AUC is weighted by test set composition; BRCA2 comprises 52% of test variants (n=2,017 of 3,845). The macro-averaged per-gene AUC of 0.775 provides a more representative picture of cross-gene performance.

### Per-Gene Performance

| Gene | Test n | ROC-AUC | MCC | Balanced Acc | Reliability Tier |
|------|--------|---------|-----|-------------|-----------------|
| BRCA2 | 2,017 | **0.983** | 0.891 | 96.9% | High |
| RAD51D | 82 | 0.804 | 0.437 | 72.0% | Moderate |
| RAD51C | 135 | 0.743 | 0.403 | 70.6% | Moderate |
| BRCA1 | 1,087 | 0.706 | 0.312 | 66.0% | Low |
| PALB2 | 524 | 0.641 | 0.346 | 65.1% | Low |

### Gold-Standard Benchmark

| Benchmark | AUC | n |
|-----------|-----|---|
| ProteinGym DMS (BRCA1) | 0.719 | 386 |
| ClinVar Expert Panel (overall) | 0.793 | 74 |
| ClinVar Expert Panel (BRCA2) | 0.918 | 37 |

### SOTA Comparison (Independent Predictors)

| Predictor | ROC-AUC |
|-----------|---------|
| SteppeDNA | **0.978** |
| REVEL | 0.725 |
| BayesDel | 0.721 |
| CADD | 0.539 |

**Methodological caveat:** SteppeDNA is evaluated on its own held-out test set. Competitor tools (REVEL, BayesDel, CADD) were not trained on this distribution, giving SteppeDNA a methodological advantage. On independent benchmarks (ProteinGym DMS + ClinVar Expert Panel), SteppeDNA achieves AUC 0.719–0.793.

### Temporal Validation (Prospective Simulation)

| Metric | Value |
|--------|-------|
| Training (pre-2024) | 9,048 variants |
| Test (2024+) | 10,175 variants |
| Overall AUC | 0.964 |
| BRCA2 AUC | 0.983 (robust) |
| BRCA1 AUC | 0.527 (near-random) |
| PALB2 AUC | 0.513 (near-random) |
| RAD51C AUC | 0.561 |
| RAD51D AUC | 0.608 |

**Critical finding:** Only BRCA2 demonstrates robust temporal generalization. Non-BRCA2 temporal AUCs (0.51–0.61) indicate the model does not reliably generalize to newly classified variants for data-scarce genes. This reflects insufficient pre-2024 training data for these genes, not a modeling failure per se, but the practical consequence is the same: non-BRCA2 predictions for newly discovered variants should be interpreted with caution.

---

## Known Limitations

1. **Non-BRCA2 generalization:** Model predictions for non-BRCA2 genes (AUC 0.64-0.80) are below clinical-grade thresholds
2. **ClinVar bias:** Training labels reflect ClinVar consensus, which has known ascertainment biases
3. **ESM-2 model size:** Using 8M-parameter ESM-2; larger models (650M+) may improve non-BRCA2
4. **Population bias:** Training data predominantly from European-descent populations (ClinVar submission bias). ClinVar submitters skew heavily toward the United States and Europe. Performance on Central Asian, African, and East Asian populations is unknown and likely lower. Kazakh founder mutations in BRCA1/2 are not represented in training data. Population-stratified allele frequencies (gnomAD AFR, AMR, EAS, NFE) are unavailable due to an Ensembl API limitation, so the model has no population-specific frequency signals
5. **Temporal bias:** ClinVar classifications evolve; training labels are a February 2026 snapshot
6. **Compound heterozygosity:** Each variant evaluated independently; compound het effects not modeled
7. **ACMG approximations:** Generated ACMG codes are computational approximations, not reviewed by clinical geneticists
8. **No prospective validation:** All validation is retrospective on existing classified variants
9. **No clinical expert review:** No geneticist or genetic counselor has validated the tool
10. **MAVE leakage:** MAVE functional scores overlap with ClinVar labels (ablation shows minimal impact, deltaAUC=-0.002)
11. **BRCA2 dominance:** Overall AUC is weighted by BRCA2 prevalence (52% of test set)
12. **No population-specific AFs:** gnomAD population-stratified data is all zeros due to Ensembl API limitation
13. **GPU feature regression:** ESM-2 650M + LoRA improved overall but degraded small genes (PALB2: 0.641 to 0.521)
14. **Feature concentration:** Most predictive power from ~15 of 103 features
15. **Kazakh translations:** Medical terminology not expert-verified
16. **EVE coverage gap:** EVE scores available for BRCA1/PALB2/RAD51C/D but not BRCA2
17. **AlphaMissense indirect label leakage:** AM was partially trained on ClinVar, creating a potential circular dependency. Ablation not yet performed.
18. **PVS1 evidence code overwrite bug corrected in v5.3.**
19. **ACMG rule engine implements approximately 10 of 28+ standard criteria.**

---

## Ethical Considerations

- **Health equity:** The model is trained predominantly on European-ancestry data (ClinVar submission bias). Predictions for underrepresented populations — including Central Asian, African, and East Asian ancestries — may be less accurate. Kazakh founder mutations are not represented. Users must consider patient ancestry when interpreting results. The degree of population bias cannot be quantified because ClinVar lacks ancestry metadata.
- **Misuse potential:** Computational predictions could be misinterpreted as clinical diagnoses. All outputs include RUO (Research Use Only) disclaimers.
- **Transparency:** SHAP feature attributions and per-gene reliability tiers are provided to help users understand prediction basis and confidence.
- **Informed consent:** The model should not be used to inform patients without proper genetic counseling framework.
- **Regulatory status:** Not approved by any regulatory body. No CE-IVD, FDA clearance, or equivalent certification.

---

## Quantitative Analyses

### Fairness / Subgroup Performance

Performance varies significantly by gene (see per-gene metrics above). This reflects training data availability rather than inherent algorithmic bias, but the practical effect is unequal prediction quality across genes.

### Calibration

Isotonic calibration on held-out real data (20% calibration set). Calibration reliability diagram available in `visual_proofs/`.

### Feature Importance

Top features by SHAP importance (aggregated):
- `blosum62_score` (amino acid substitution severity)
- `phylop_score` (evolutionary conservation)
- `am_score` (AlphaMissense pathogenicity)
- `relative_aa_pos` (protein position)
- `volume_diff` (amino acid volume change)

---

## Citation

If using SteppeDNA in research, please cite:

> Abzhakhanov, E. (2026). SteppeDNA: Multi-Gene HR Variant Pathogenicity Classifier for Homologous Recombination DNA Repair Genes. Research Use Only.

---

## References

- Mitchell et al. (2019). "Model Cards for Model Reporting." *FAT\* Conference*.
- Richards et al. (2015). "Standards and guidelines for the interpretation of sequence variants." *Genet Med* 17:405-424.
- Ioannidis et al. (2016). "REVEL: An ensemble method for predicting the pathogenicity of rare missense variants." *Am J Hum Genet* 99:877-885.
- Cheng et al. (2023). "Accurate proteome-wide missense variant effect prediction with AlphaMissense." *Science* 381:eadg7492.
- Lin et al. (2023). "Evolutionary-scale prediction of atomic-level protein structure with a language model." *Science* 379:1123-1130.
- Findlay et al. (2018). "Accurate classification of BRCA1 variants with saturation genome editing." *Nature* 562:217-222.
- Frazer et al. (2021). "Disease variant prediction with deep generative models of evolutionary data." *Nature* 599:91-95.
- Pollard et al. (2010). "Detection of nonneutral substitution rates on mammalian phylogenies." *Genome Res* 20:110-121.
- Tavtigian et al. (2018). "Modeling the ACMG/AMP variant classification guidelines." *Hum Mutat* 39:1485-1492.
- Rentzsch et al. (2019). "CADD: predicting the deleteriousness of variants throughout the human genome." *Nucleic Acids Res* 47:D886-D894.
