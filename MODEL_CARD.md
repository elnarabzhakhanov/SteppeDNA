# SteppeDNA Model Card

Following the framework of Mitchell et al. (2019), "Model Cards for Model Reporting."

---

## Model Details

- **Model name:** SteppeDNA v5.4
- **Model type:** Ensemble classifier (XGBoost + MLP Neural Network (per-gene weights)) with isotonic calibration
- **Version:** 5.4.0 (March 2026)
- **Developed by:** Elnar Abzhakhanov
- **License:** Research Use Only (RUO). Code license: MIT. Model artifacts and predictions: Research Use Only (RUO).
- **Input:** Missense variant features (cDNA position, amino acid change, nucleotide mutation, gene name)
- **Output:** Calibrated pathogenicity probability (0.0-1.0), classification (Pathogenic/Benign), ACMG evidence codes, SHAP feature attributions
- **Features:** 120 engineered features (gene-identifying + AlphaMissense features removed)

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
| ROC-AUC (overall, weighted) | 0.985* |
| Macro-Averaged AUC (equal-weight per-gene) | 0.791 |
| PR-AUC | 0.965 |
| MCC | 0.881 |
| Balanced Accuracy | 94.1% |
| Sensitivity | 95.7% |
| Specificity | 92.6% |
| 10-Fold CV AUC | 0.9797 +/- 0.0031 |

*Overall AUC is weighted by test set composition; BRCA2 comprises 52% of test variants (n=2,017 of 3,845). The macro-averaged per-gene AUC of 0.791 provides a more representative picture of cross-gene performance.

### Per-Gene Performance

| Gene | Test n | ROC-AUC | MCC | Balanced Acc | Reliability Tier |
|------|--------|---------|-----|-------------|-----------------|
| BRCA2 | 2,017 | **0.994** | 0.891 | 96.9% | High |
| RAD51D | 82 | 0.824 | 0.437 | 72.0% | Moderate |
| RAD51C | 135 | 0.785 | 0.403 | 70.6% | Moderate |
| BRCA1 | 1,087 | 0.747 | 0.312 | 66.0% | Low |
| PALB2 | 524 | 0.605 | 0.346 | 65.1% | Low |

### Gold-Standard Benchmark

| Benchmark | AUC | n |
|-----------|-----|---|
| ProteinGym DMS (BRCA1) | 0.719 | 386 |
| ClinVar Expert Panel (overall) | 0.793 | 74 |
| ClinVar Expert Panel (BRCA2) | 0.918 | 37 |

### SOTA Comparison (DeLong Test, n≈2,700 variants with scores from all tools)

| Predictor | ROC-AUC | DeLong p-value | Significant? |
|-----------|---------|---------------|-------------|
| SteppeDNA | **0.994** | — | — |
| REVEL | 0.717 | 1.1×10⁻¹²⁸ | Yes |
| BayesDel | 0.713 | 6.8×10⁻¹³⁸ | Yes |
| CADD | 0.538 | ≈0 | Yes |

**Methodological caveat:** SteppeDNA is evaluated on its own held-out test set. Competitor tools (REVEL, BayesDel, CADD) were not trained on this distribution, giving SteppeDNA a methodological advantage. DeLong test confirms statistical significance (all p < 10⁻¹²⁸). On independent benchmarks (ProteinGym DMS + ClinVar Expert Panel), SteppeDNA achieves AUC 0.719–0.793.

### ACMG Rule Engine Comparison (Test Set, n=3,845)

| Metric | Value |
|--------|-------|
| ACMG codes implemented | 13 |
| Variants resolved (P/LP/B/LB) | 792 (20.6%) |
| Variants classified as VUS | 3,053 (79.4%) |
| Agreement rate (resolved vs ClinVar) | 71.2% |
| Cohen's kappa | 0.154 |
| BRCA1 agreement | 95.0% |
| BRCA2 agreement | 53.4% |

### Population Equity Analysis

| Metric | EAS | NFE | Gap |
|--------|-----|-----|-----|
| PM2 rate (AF=0) | 97.6% | 91.2% | +6.4pp |
| Excess PM2 variants | +245 more than NFE | baseline | 1,260 across full dataset |
| AF recalibration: variants gaining PM2 | 2,981 | 1,753 | +70% more for EAS |

**Key finding:** East Asian/Central Asian patients receive significantly more PM2 flags (variant absent from population databases) than European patients. This is a data representation problem, not a model problem, and affects all variant classifiers.

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

1. **Non-BRCA2 generalization:** Model predictions for non-BRCA2 genes (AUC 0.605-0.824) are below clinical-grade thresholds
2. **ClinVar bias:** Training labels reflect ClinVar consensus, which has known ascertainment biases
3. **ESM-2 model size:** Using 8M-parameter ESM-2; larger models (650M+) may improve non-BRCA2
4. **Population bias:** Training data predominantly from European-descent populations (ClinVar submission bias). ClinVar submitters skew heavily toward the United States and Europe. Performance on Central Asian, African, and East Asian populations is unknown and likely lower. Kazakh founder mutations in BRCA1/2 are not represented in training data. Population-stratified allele frequencies from gnomAD are now integrated, and Kazakh founder mutations are included in the database
5. **Temporal bias:** ClinVar classifications evolve; training labels are a February 2026 snapshot
6. **Compound heterozygosity:** Each variant evaluated independently; compound het effects not modeled
7. **ACMG approximations:** Generated ACMG codes are computational approximations, not reviewed by clinical geneticists
8. **No prospective validation:** All validation is retrospective on existing classified variants
9. **No clinical expert review:** No geneticist or genetic counselor has validated the tool
10. **MAVE leakage:** MAVE functional scores overlap with ClinVar labels (ablation shows minimal impact, deltaAUC=-0.002)
11. **BRCA2 dominance:** Overall AUC is weighted by BRCA2 prevalence (52% of test set)
12. **Population-aware AFs:** gnomAD population-stratified AFs now integrated via myvariant.info (3,508 variants with AF>0)
13. **GPU feature regression:** ESM-2 650M + LoRA improved overall but degraded small genes (PALB2: 0.641 to 0.521)
14. **Feature concentration:** Most predictive power from ~15 of 120 features
15. **Kazakh translations:** Medical terminology not expert-verified
16. **EVE coverage gap:** EVE scores available for BRCA1/PALB2/RAD51C/D but not BRCA2
17. **AlphaMissense removed in v5.4:** AM was partially trained on ClinVar labels. Removed entirely; ablation showed +0.02 AUC improvement for BRCA1/PALB2/RAD51C without AM.
18. **PVS1 evidence code overwrite bug corrected in v5.3.**
19. **ACMG rule engine implements 13 of 28+ standard criteria** (PVS1, PS1, PM1, PM2, PM4, PM5, PP3, PP3_splice, BA1, BS1, BP4, BP7, plus PVS1_moderate). Agreement rate with ClinVar labels is 71.2% on resolved variants (20.6% resolution rate).
20. **Population disparity:** EAS/Central Asian populations receive 6.4 percentage points more PM2 flags than NFE populations (97.6% vs 91.2%). AF recalibration shows 2,981 EAS variants gain PM2 vs 1,753 NFE.

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
- `dist_nearest_domain` (distance to nearest functional domain)
- `relative_aa_pos` (protein position)
- `volume_diff` (amino acid volume change)

*Note: AlphaMissense was removed in v5.4 due to label leakage.*

---

## Citation

If using SteppeDNA in research, please cite:

> Abzhakhanov, E. (2026). SteppeDNA: Multi-Gene HR Variant Pathogenicity Classifier for Homologous Recombination DNA Repair Genes. Research Use Only.

---

## References

- Mitchell et al. (2019). "Model Cards for Model Reporting." FAT* Conference.
- Ioannidis et al. (2016). REVEL. Am J Hum Genet.
- Cheng et al. (2023). AlphaMissense. Nature.
- Lin et al. (2023). ESM-2. Science.
- Richards et al. (2015). ACMG/AMP Standards and Guidelines.
