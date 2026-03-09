# SteppeDNA: Experimental Validation Report

**Universal Multi-Gene Model v5.3 (gnomAD-augmented)**
**Date:** March 2026
**Dataset:** ClinVar + gnomAD proxy-benign missense variants (19,223 across 5 HR genes)

---

## 1. Model Architecture

- **Ensemble:** XGBoost (60%) + MLP Neural Network (40%)
- **Calibration:** Isotonic regression on held-out calibration set
- **Features:** 103 engineered features (gene-identifying features removed)
- **Genes:** BRCA1, BRCA2, PALB2, RAD51C, RAD51D
- **Training data:** ClinVar P/LP and B/LB missense variants + gnomAD v4 proxy-benign (AC >= 2)

### Feature Categories
| Category | Count | Examples |
|----------|-------|---------|
| Positional | 2 | relative_cdna_pos, relative_aa_pos |
| Biochemical | 13 | blosum62_score, hydro_diff, volume_diff, charge_change |
| Conservation | 4 | phylop_score, high_conservation, ultra_conservation |
| Functional (MAVE) | 4 | mave_score, has_mave, mave_abnormal |
| AlphaMissense | 3 | am_score, am_pathogenic, am_x_phylop |
| Structural | 10 | rsa, is_buried, bfactor, dist_dna, ss_helix |
| Protein LM (ESM-2) | 22 | esm2_cosine_sim, esm2_l2_shift, esm2_pca_0..19 |
| Splice prediction | 2 | spliceai_score, splice_pathogenic |
| Amino acid encoding | 42 | AA_ref_*, AA_alt_* one-hot |
| Interaction terms | 4 | conserv_x_blosum, mave_x_blosum, etc. |

---

## 2. Held-Out Test Set Performance

**Split:** 60% train / 20% calibration / 20% test (gene x label stratified)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9781 |
| PR-AUC | 0.9651 |
| MCC | 0.8814 |
| Balanced Accuracy | 94.1% |
| Sensitivity | 95.7% |
| Specificity | 92.6% |
| Optimal Threshold | 0.4545 |

**Confusion Matrix (n=3,845):**
| | Predicted Benign | Predicted Pathogenic |
|---|---|---|
| True Benign | TN=1,888 | FP=152 |
| True Pathogenic | FN=77 | TP=1,728 |

---

## 3. Per-Gene Performance (Critical Transparency)

| Gene | n (test) | ROC-AUC | MCC | Balanced Acc |
|------|----------|---------|-----|-------------|
| BRCA2 | 2,017 | **0.983** | **0.891** | 96.9% |
| RAD51D | 82 | 0.804 | 0.437 | 72.0% |
| RAD51C | 135 | 0.743 | 0.403 | 70.6% |
| BRCA1 | 1,087 | 0.706 | 0.312 | 66.0% |
| PALB2 | 524 | 0.641 | 0.346 | 65.1% |

**Key Finding (v4.1 gnomAD augmentation):** Adding 485 gnomAD proxy-benign variants (AC >= 2 in gnomAD v4) dramatically improved non-BRCA2 per-gene metrics compared to v4 (pre-gnomAD):

| Gene | ROC-AUC (v4) | ROC-AUC (v4.1) | Change |
|------|-------------|---------------|--------|
| BRCA2 | 0.984 | 0.983 | -0.001 |
| BRCA1 | 0.563 | 0.706 | +0.143 |
| PALB2 | 0.546 | 0.641 | +0.095 |
| RAD51C | 0.509 | 0.743 | +0.234 |
| RAD51D | 0.586 | 0.804 | +0.218 |

The gnomAD augmentation addressed the extreme ClinVar class imbalance by providing population-common benign variants for under-represented genes.

**Remaining limitation:** Non-BRCA2 genes still underperform BRCA2. Root causes:
1. ClinVar still has far fewer classified variants for these genes
2. Feature effect sizes remain smaller (Cohen's d < 1.0 vs d > 2.0 for BRCA2)
3. Some residual class imbalance persists even after gnomAD augmentation

**Recommendation:** SteppeDNA predictions are clinically informative for **BRCA2 variants** (AUC 0.98). Non-BRCA2 predictions are now meaningful (AUC 0.64-0.80) but should be treated as supportive evidence, not standalone diagnostic tools.

### Gene-Specific Reliability Tiers

Based on test set performance, gene predictions are assigned reliability tiers:
- **BRCA2** (AUC 0.983): High reliability — sufficient training data and balanced classes
- **RAD51D** (AUC 0.804): Moderate reliability — limited test variants (82)
- **RAD51C** (AUC 0.743): Moderate reliability — limited test variants (135)
- **BRCA1** (AUC 0.706): Low reliability — extreme class imbalance (96.6% pathogenic in ClinVar)
- **PALB2** (AUC 0.641): Low reliability — smallest training set, near-random performance

The overall 0.978 AUC is weighted by gene prevalence in the test set (BRCA2 comprises 52.4%). Users should consult per-gene metrics rather than the overall AUC when interpreting predictions.

---

## 4. Cross-Validation (10-Fold, Gene-Stratified)

| Metric | Mean +/- Std | 95% CI |
|--------|-------------|--------|
| ROC-AUC | 0.9797 +/- 0.0031 | [0.9774, 0.9818] |
| PR-AUC | 0.9706 +/- 0.0020 | [0.9674, 0.9739] |
| MCC | 0.885 +/- 0.007 | - |
| Balanced Accuracy | 0.943 +/- 0.004 | - |
| Sensitivity | 0.959 +/- 0.011 | - |
| Specificity | 0.926 +/- 0.013 | - |

**Bootstrap:** 1,000 resamples of pooled out-of-fold predictions.

The tight confidence intervals (AUC range 0.003) confirm the model is highly stable across data splits.

---

## 5. Comparison Against SOTA Predictors

### 5a. Independent SOTA Predictors (Fair Head-to-Head)

These predictors are **not** used as input features in SteppeDNA, enabling a fair comparison.
Scores retrieved from myvariant.info/dbNSFP for each test-set variant.

| Predictor | ROC-AUC | PR-AUC | MCC | n scored | Citation |
|-----------|---------|--------|-----|----------|----------|
| **SteppeDNA (Ensemble)** | **0.978** | **0.969** | **0.881** | 3,845 | This work |
| REVEL | 0.725 | 0.533 | 0.027 | 2,784 | Ioannidis et al. 2016 |
| BayesDel | 0.721 | 0.555 | 0.097 | 2,802 | Feng 2017 |
| CADD | 0.539 | 0.324 | 0.026 | 2,784 | Rentzsch et al. 2019 |

**Coverage:** REVEL 72.4%, CADD 72.4%, BayesDel 72.9% of test set variants scored via myvariant.info API.

SteppeDNA outperforms all three independent SOTA predictors by substantial margins (ΔΔ AUC > 0.25).

### 5b. Input Features as Standalone Predictors

These are individual features used within SteppeDNA, tested as standalone classifiers for transparency:

| Predictor | ROC-AUC | PR-AUC | MCC | n scored | Citation |
|-----------|---------|--------|-----|----------|----------|
| MAVE | 0.672 | 0.332 | 0.195 | 134 | Findlay et al. 2018 |
| AlphaMissense | 0.660 | 0.600 | 0.278 | 3,845 | Cheng et al., Nature 2023 |
| BLOSUM62 | 0.584 | 0.609 | 0.108 | 3,845 | Henikoff & Henikoff 1992 |
| ESM-2 (cosine sim) | 0.554 | 0.553 | 0.078 | 3,845 | Lin et al. 2023 |
| PhyloP | 0.514 | 0.465 | 0.023 | 3,845 | Pollard et al. 2010 |
| SpliceAI | 0.470 | 0.323 | 0.000 | 3,845 | Jaganathan et al. 2019 |

**Note:** MAVE has only 3.5% coverage (134/3,845 variants scored).

### 5c. Per-Gene SOTA Comparison

The overall AUC comparison in Section 5a is dominated by BRCA2 (52.4% of the test set). The table below breaks down performance by gene, which provides a more granular and honest comparison.

| Gene | SteppeDNA | REVEL | BayesDel | CADD | Best Non-SteppeDNA |
|------|-----------|-------|----------|------|-------------------|
| BRCA2 | **0.983** | 0.891 | 0.949 | 0.908 | BayesDel |
| RAD51D | **0.804** | 0.461 | 0.448 | 0.457 | REVEL |
| RAD51C | **0.743** | 0.651 | 0.634 | 0.703 | CADD |
| BRCA1 | **0.706** | 0.595 | 0.646 | 0.527 | BayesDel |
| PALB2 | 0.641 | **0.732** | 0.432 | 0.564 | REVEL |

**Key findings:**

- SteppeDNA outperforms all three SOTA predictors on 4 of 5 genes at the per-gene level.
- REVEL outperforms SteppeDNA on PALB2 (0.732 vs 0.641), most likely because REVEL was trained on a broader, multi-gene dataset that includes more PALB2 diversity.
- Poor non-BRCA2 performance is a field-wide problem, not unique to SteppeDNA. It is driven by extreme class imbalance (BRCA1: 94.8% pathogenic, PALB2: 93.1%) and data scarcity (RAD51C: 675 total variants, RAD51D: 410 total variants).
- Coverage differs across predictors: REVEL, BayesDel, and CADD scored only 72–73% of test variants via myvariant.info/dbNSFP, whereas SteppeDNA scores 100%. Per-gene counts for SOTA predictors are proportionally lower.

**Note:** SOTA scores are retrieved from dbnsfp_sota_scores.csv. Per-gene comparisons for REVEL/BayesDel/CADD are computed on the scored subset only, which introduces a mild selection bias (unscored variants tend to be less well-characterised).

### SOTA Comparison Context

SteppeDNA is evaluated against REVEL, BayesDel, and CADD on SteppeDNA's own held-out test set. SteppeDNA was trained on ClinVar labels from these 5 genes; REVEL, BayesDel, and CADD are general-purpose predictors trained on different datasets. This gives SteppeDNA a methodological advantage on its own test set. The independent gold-standard benchmark (ProteinGym DMS + ClinVar Expert Panel) provides a more balanced assessment, showing AUC of 0.719 (DMS, BRCA1) and 0.793 (Expert Panel, overall).

REVEL and BayesDel achieve MCC values of 0.027 and 0.097 respectively on this test set, which are near-random. This indicates the test set distribution is hostile to their decision boundaries, not that they are poor predictors. In published benchmarks, REVEL typically achieves AUC > 0.90.

**BRCA2-specific comparison:**
| Predictor | ROC-AUC (BRCA2 only) |
|-----------|---------------------|
| SteppeDNA | 0.984 |
| BLOSUM62 | 0.971 |
| ESM-2 | 0.951 |
| AlphaMissense | 0.831 |

---

## 6. Gold-Standard Benchmark Evaluation

### 6a. Benchmark Dataset (n=2,234)

Curated from two independent sources:

| Source | n variants | Genes | Citation |
|--------|-----------|-------|----------|
| ProteinGym DMS | 1,837 | BRCA1 | Findlay et al., Nature 2018 |
| ClinVar Expert Panel | 397 | BRCA1, BRCA2, PALB2 | ClinGen VCEP / ENIGMA |

**Overlap:** 460 benchmark variants matched to held-out test set; 66 are completely novel (unseen during training or testing).

### 6b. DMS Experimental Correlation (BRCA1)

| Metric | Value | n |
|--------|-------|---|
| Spearman r | -0.319 | 386 |
| p-value | 1.45e-10 | - |
| ROC-AUC | 0.719 | 386 |

Negative Spearman correlation confirms SteppeDNA P(pathogenic) increases as DMS fitness decreases (loss-of-function). The AUC=0.719 on the Findlay DMS benchmark is consistent with BRCA1 per-gene AUC of 0.706 on the ClinVar test set.

### 6c. ClinGen Expert-Panel Classification

| Subset | ROC-AUC | MCC | n |
|--------|---------|-----|---|
| Overall | 0.793 | 0.508 | 74 |
| BRCA2 only | 0.918 | 0.841 | 37 |
| BRCA1 only | 0.791 | 0.227 | 36 |

SteppeDNA predictions validate well against ClinGen/ENIGMA expert-classified variants, particularly for BRCA2 (AUC=0.918).

---

## 7. Design Decisions and Mitigations

### gnomAD Proxy-Benign Augmentation (v4.1)
485 common population variants from gnomAD v4 (AC >= 2, deduplicated against ClinVar) were added as proxy-benign training data. For high-penetrance cancer genes, variants observed 2+ times in ~1.6M alleles are very unlikely to be pathogenic. Distribution: BRCA1 +97, BRCA2 +114, PALB2 +60, RAD51C +86, RAD51D +128. This increased total dataset from 18,738 to 19,223 variants and dramatically improved non-BRCA2 gene performance.

### gnomAD Augmentation Impact (Ablation Note)

The gnomAD proxy-benign augmentation (v4.1) was the single largest factor in improving non-BRCA2 gene performance. Pre-augmentation non-BRCA2 AUCs were near random:

| Gene | Pre-gnomAD AUC | Post-gnomAD AUC | Change |
|------|---------------|----------------|--------|
| BRCA1 | ~0.56 | 0.706 | +0.14 |
| PALB2 | ~0.55 | 0.641 | +0.09 |
| RAD51C | ~0.51 | 0.743 | +0.23 |
| RAD51D | ~0.59 | 0.804 | +0.22 |

**Important caveat:** This was not tested as an isolated ablation experiment. Between v4 and v4.1, other changes were made simultaneously (gene-identity feature removal, ESM-2 embedding updates, hyperparameter tuning). The improvement is attributed primarily to gnomAD augmentation based on the magnitude and timing of changes, but the exact contribution of each factor cannot be disaggregated. This is a known limitation of the validation methodology.

### Gene-Identity Leakage (Fixed)
17 gene-identifying features were removed (cDNA_pos, AA_pos, Mutation_* one-hots, is_transition, is_transversion) that allowed the model to shortcut "BRCA1 = pathogenic" based on position alone.

### Gene-Specific ESM-2 Embeddings
ESM-2 (esm2_t6_8M_UR50D, 8M params) embeddings generated per-gene using +/-50 residue context windows. Total: 17,528 embeddings across 5 genes. Top 20 PCA components explain 72.8% variance.

### No Per-Gene Sample Weighting
Tested gene x class sample weighting but it degraded overall AUC from 0.98 to 0.94 without improving non-BRCA2 genes. Root cause: features genuinely lack discriminative signal for non-BRCA2 genes.

### MAVE Data Leakage Assessment (Addressed)
MAVE functional scores (Hu et al. 2024, BRCA2 HDR assay) are used as 4 input features (mave_score, has_mave, mave_abnormal, mave_x_blosum). Since many MAVE-assayed variants also appear in ClinVar training data, indirect label leakage is possible.

**Ablation test result:** Zeroing out all MAVE features has minimal impact on overall performance:

| Metric | With MAVE | Without MAVE | Delta |
|--------|-----------|-------------|-------|
| ROC-AUC | 0.9781 | 0.9763 | -0.0017 |
| MCC | 0.8814 | 0.8795 | -0.0019 |

MAVE only affects BRCA2 (AUC: 0.983 to 0.965) with zero impact on other genes (no MAVE data). Only 3.5% of test variants have MAVE scores. The `use_mave=False` flag in `backend/feature_engineering.py` disables MAVE features for leakage-free evaluation.

### Ensemble Weight Justification

The default 60% XGBoost / 40% MLP blend was selected based on calibration set performance. XGBoost provides stronger tabular feature learning; MLP captures non-linear feature interactions.

**Gene-Adaptive Weights (v5.3):** Per-gene weight optimization was performed via grid search (0.0-1.0 in 0.05 steps) on calibration set AUC:

| Gene | XGB Weight | MLP Weight | Cal AUC | Test AUC | Default Test AUC | Delta |
|------|-----------|-----------|---------|----------|-----------------|-------|
| BRCA1 | 0.70 | 0.30 | 0.761 | 0.699 | 0.700 | -0.001 |
| BRCA2 | 0.65 | 0.35 | 0.993 | 0.987 | 0.987 | +0.000 |
| PALB2 | 0.65 | 0.35 | 0.741 | 0.640 | 0.640 | -0.000 |
| RAD51C | 0.95 | 0.05 | 0.677 | 0.770 | 0.752 | **+0.018** |
| RAD51D | 1.00 | 0.00 | 0.866 | 0.805 | 0.807 | -0.002 |

RAD51C benefits most from heavier XGBoost weighting. Gene-adaptive weights are loaded at startup from `data/gene_ensemble_weights.json` and applied per-prediction. Script: `scripts/optimize_gene_weights.py`.

### Calibration Strategy
Isotonic calibration on held-out real data (20% of train+val), NOT on SMOTE-augmented data.

**BRCA1 Calibration Limitation:** The BRCA1 calibration subset has extreme class imbalance (~96.6% pathogenic). This means isotonic regression has very few benign calibration points for BRCA1, potentially making calibrated probabilities unreliable in the benign range. A predicted probability of 0.2 for a BRCA1 variant may not be well-calibrated because the isotonic function has limited data in this region. Planned improvement: per-gene calibrators that can be tuned for each gene's class distribution, rather than the current universal calibrator.

---

## 8. Reproducibility

All scripts are deterministic (RANDOM_STATE=42) and run from project root:

```bash
# Full pipeline
python data_pipelines/fetch_gnomad_proxy_benign.py  # Fetch gnomAD proxy-benign variants
python scripts/build_master_dataset.py    # Build 103-feature dataset (incl. gnomAD)
python scripts/train_universal_model.py   # Train XGB+MLP ensemble
python scripts/cross_validate.py          # 10-fold CV with bootstrap CIs
python data_pipelines/fetch_dbnsfp_scores.py  # Fetch REVEL/CADD/BayesDel from myvariant.info
python scripts/sota_comparison.py         # SOTA predictor comparison (real + input features)
python data_pipelines/build_gold_standard_benchmark.py  # Curate benchmark from ProteinGym + ClinVar expert panel
python scripts/evaluate_benchmark.py     # Evaluate against gold-standard benchmark
python scripts/generate_figures.py        # All visual proofs
```

### Model Artifacts (data/)
- `universal_xgboost_final.json` — XGBoost model
- `universal_nn.h5` — MLP neural network
- `universal_scaler_ensemble.pkl` — StandardScaler
- `universal_calibrator_ensemble.pkl` — Isotonic calibrator
- `universal_threshold_ensemble.pkl` — Optimal F1 threshold
- `universal_feature_names.pkl` — Feature list (103)
- `master_training_dataset.csv` — Full training data
- `model_metrics.json` — Test set metrics
- `sota_comparison.json` — SOTA comparison results (real + input features)
- `dbnsfp_sota_scores.csv` — Per-variant REVEL/CADD/BayesDel scores from myvariant.info
- `dbnsfp_scores_cache.json` — Cached API responses (for reproducibility)
- `cv_results.pkl` — Cross-validation results
- `benchmark/gold_standard_benchmark.csv` — Curated benchmark (2,234 variants)
- `benchmark/benchmark_metadata.json` — Benchmark provenance
- `benchmark/benchmark_results.json` — Benchmark evaluation results

### Visual Proofs (visual_proofs/)
1. Per-Gene ROC Curves
2. Per-Gene PR Curves
3. Calibration Reliability Diagram
4. SHAP Global Feature Importances
5. Per-Gene Performance Summary (bar charts)
6. Confusion Matrices (overall + per-gene)
7. SOTA Comparison (ROC + PR + summary)
8. Benchmark DMS Correlation + Classification
9. Cross-Validation Results

---

## 9. Known Limitations

1. **Non-BRCA2 generalization:** Model predictions for non-BRCA2 genes (AUC 0.64-0.80) are improved but still below clinical-grade thresholds
2. **ClinVar bias:** Training labels reflect ClinVar consensus, which has known ascertainment biases
3. **ESM-2 model size:** Using 8M-parameter ESM-2; larger models (650M+) may improve non-BRCA2 discrimination
4. **SOTA coverage:** REVEL/CADD/BayesDel comparison covers ~72% of test set via myvariant.info; remaining variants lack dbNSFP annotations
5. **MAVE validation:** External MAVE validation is BRCA2-specific only (Findlay et al. exon 13)
6. **Population bias:** Training data is predominantly from populations of European descent (ClinVar submission bias). Performance on Central Asian, African, East Asian, and other underrepresented populations is unknown and likely lower.
7. **Temporal bias:** ClinVar classifications evolve over time through reclassification. Training labels represent a snapshot and may not reflect the latest expert consensus.
8. **Compound heterozygosity:** The model evaluates each variant independently. Two individually benign variants in the same gene can be pathogenic together (compound heterozygosity). This is not assessed.
9. **ACMG codes are computational approximations:** Generated evidence codes (PP3, PM1, BS1, BP4) are approximate and have not been reviewed by clinical geneticists. They should not be mistaken for validated ACMG classifications.
10. **No prospective validation:** All validation is retrospective on existing classified variants. The model has not been tested on newly discovered variants before their ClinVar classification.
11. **No clinical expert review:** No geneticist, oncologist, or genetic counselor has reviewed or validated the tool's output.
12. **Feature concentration:** Most predictive power likely comes from ~15 of the 103 features. The remaining features may contribute marginal signal.
13. **GPU feature regression on small genes:** ESM-2 650M + LoRA embeddings improved overall AUC to 0.984 but degraded PALB2 (0.641→0.521) and RAD51C/D. GNN structural features act as gene identifiers (non-zero only for non-BRCA2). More features do not help when data is scarce.
14. **Kazakh translations not expert-verified:** Medical/scientific terminology in Kazakh translations has not been reviewed by a native-speaking domain expert.
15. **EVE score coverage gap:** EVE scores (Frazer et al., Nature 2021) are available for BRCA1, PALB2, RAD51C, RAD51D but NOT BRCA2 via dbNSFP/myvariant.info. Total 18,253 EVE variant scores fetched.
16. **No population-specific allele frequencies:** gnomAD population-stratified AF data (AFR, AMR, EAS, NFE) is all zeros in the training data. The model has no population-specific frequency signals.
17. **AlphaMissense indirect label leakage:** `am_score`, `am_pathogenic`, and `am_x_phylop` are derived from a model that was partially trained on ClinVar pathogenicity labels, creating a circular dependency with SteppeDNA's ClinVar-based training labels (see Section 14 above). Ablation without AlphaMissense features is planned future work to quantify the exact impact.
18. **Test suite historical bias:** Prior to v5.3, feature engineering tests exclusively covered BRCA2 inputs, providing no automated verification that the pipeline behaves correctly for the 4 underperforming genes. Unit test coverage for BRCA1, PALB2, RAD51C, and RAD51D is a gap in the current QA process.
19. **PVS1 evidence code overwrite:** Prior to v5.3, a logic error in the ACMG rule evaluator allowed canonical splice-site PVS1 evidence to silently overwrite nonsense or frameshift PVS1 when both flags were set simultaneously for the same variant. This has been corrected in v5.3.

---

## 10. Critical: Temporal Generalization

**Methodology:** Date-stratified split using ClinVar variant_summary.txt.gz classification dates. Variants classified before 2024 form the training set; 2024+ classifications form the test set. Model retrained from scratch on the temporal-train split.

| Metric | Value |
|--------|-------|
| Training set (pre-2024 + undated) | 9,048 variants |
| Test set (2024+) | 10,175 variants |
| ROC-AUC | **0.964** |
| PR-AUC | 0.947 |
| MCC | 0.817 |
| Balanced Accuracy | 90.9% |
| Sensitivity | 90.4% |
| Specificity | 91.3% |

### Per-Gene Temporal Performance

| Gene | n (test) | Temporal AUC | Notes |
|------|----------|-------------|-------|
| BRCA2 | 5,228 | **0.983** | Robust across time |
| RAD51D | 236 | 0.608 | Limited pre-2024 training data |
| RAD51C | 498 | 0.561 | Limited pre-2024 training data |
| BRCA1 | 2,514 | 0.527 | Extreme class imbalance (pre-2024) |
| PALB2 | 1,699 | 0.513 | Most classifications are recent |

**Key Finding:** BRCA2 demonstrates excellent temporal generalization (AUC 0.983). Non-BRCA2 genes degrade because most of their ClinVar classifications occurred after 2024, leaving insufficient pre-cutoff training data. This is a data availability problem, not a modeling failure.

> **Critical Disclosure:** Non-BRCA2 temporal AUCs (0.51–0.61) indicate that the model does not reliably generalize over time for data-scarce genes. Temporal performance for PALB2 (0.513), BRCA1 (0.527), RAD51C (0.561), and RAD51D (0.608) are near-random, meaning predictions for newly classified variants in these genes should be interpreted with significant caution. Only BRCA2 predictions demonstrate robust temporal stability.

**Visual proof:** `visual_proofs/temporal_validation.pdf`

---

## 11. GPU Feature Ablation Study

Three GPU-dependent feature sets were computed and integrated:

| Feature Set | Dimensions | Coverage | Source |
|------------|-----------|---------|--------|
| ESM-2 LoRA (fine-tuned) | 20 PCA | 100% (19,223) | Google Colab T4 |
| ESM-2 650M | 20 PCA + 2 metrics | 100% (19,223) | Google Colab T4 |
| GNN Structural | 32D + indicator | 47.5% (9,138) | Google Colab T4 |

### Results (178 features, full GPU)

| Metric | Original (103 feat.) | GPU-augmented (178 feat.) | Delta |
|--------|---------------------|--------------------------|-------|
| Overall AUC | 0.978 | **0.984** | +0.005 |
| BRCA2 AUC | 0.983 | **0.994** | +0.011 |
| BRCA1 AUC | 0.706 | **0.763** | +0.057 |
| RAD51D AUC | 0.804 | 0.776 | -0.028 |
| RAD51C AUC | 0.743 | 0.700 | -0.043 |
| PALB2 AUC | 0.641 | 0.521 | **-0.120** |

### Results (145 features, ESM-2 only, no GNN)

| Metric | Original | ESM-2 only | Delta |
|--------|----------|-----------|-------|
| Overall AUC | 0.978 | **0.983** | +0.005 |
| BRCA2 AUC | 0.983 | **0.994** | +0.011 |
| PALB2 AUC | 0.641 | 0.537 | -0.104 |

**Key Finding:** GNN structural features act as gene identifiers (`has_gnn_features` XGBoost gain: 647, next feature: 21) because they are non-zero only for non-BRCA2 genes. Even without GNN, ESM-2 upgrades degrade small gene groups. **The production model retains the original 103 features.** GPU features are documented as an ablation study demonstrating that more features do not help when training data is scarce.

---

## 12. EVE Score Analysis

EVE (Evolutionary model of Variant Effect, Frazer et al., Nature 2021) scores fetched from myvariant.info/dbNSFP:

| Gene | EVE scores found | Coverage |
|------|-----------------|----------|
| BRCA1 | 28,673 | Available |
| PALB2 | 10,925 | Available |
| RAD51C | 2,832 | Available |
| RAD51D | 2,778 | Available |
| BRCA2 | 0 | **Not available** |

EVE score range: [0.006, 0.921], mean: 0.472, median: 0.483.

**Key Finding:** EVE covers the underserved genes (BRCA1, PALB2, RAD51C/D) but not BRCA2.

### EVE Retrain Experiment (v5.3)

EVE score was added as the 104th feature and models were retrained (XGBoost + MLP) with identical hyperparameters and RANDOM_STATE=42:

| Gene | Prod AUC (103 feat.) | EVE AUC (104 feat.) | Delta |
|------|---------------------|---------------------|-------|
| BRCA1 | 0.706 | 0.756 | **+0.050** |
| BRCA2 | 0.983 | 0.982 | -0.001 |
| PALB2 | 0.641 | 0.617 | -0.024 |
| RAD51C | 0.743 | 0.767 | **+0.024** |
| RAD51D | 0.804 | 0.798 | -0.006 |
| **Overall** | **0.978** | **0.980** | **+0.002** |

EVE coverage in training data: BRCA1 65.6%, BRCA2 38.9% (position fallback), PALB2 99.8%, RAD51C 100%, RAD51D 100%.

EVE feature importance: XGBoost gain=23.8 (ranked 10th of 104 features).

**Conclusion:** EVE significantly improves BRCA1 (+0.050) and RAD51C (+0.024), but slightly hurts PALB2 (-0.024). EVE models are saved with `_eve` suffix for A/B comparison; production models remain unchanged. Script: `scripts/retrain_with_eve.py`.

---

## 13. Population Equity Analysis

gnomAD population-stratified allele frequency analysis across all 19,223 training variants:

| Population | Variants with AF > 0 | Percentage |
|-----------|---------------------|-----------|
| Overall (AF) | 0 | 0.0% |
| African (AFR) | 0 | 0.0% |
| Latino/Admixed American (AMR) | 0 | 0.0% |
| East Asian (EAS) | 0 | 0.0% |
| Non-Finnish European (NFE) | 0 | 0.0% |

**Root Cause:** The Ensembl `overlap/region` API endpoint only returns a single `minor_allele_freq` field, not population-stratified frequencies. The population-specific fields were initialized but never populated from the API response.

**Implication:** The model has **no population-specific frequency information**. All allele frequency features are derived from a single global MAF, which is predominantly calibrated on European-ancestry populations. This means:
1. BS1 (benign standalone — frequency above threshold) may not correctly account for variants that are common in specific populations
2. Predictions for patients of Central Asian, East Asian, or African ancestry lack population-specific context
3. Kazakh founder mutations in BRCA1/2 are not represented in the training data

### Population Representation Analysis

**ClinVar submission bias:** ClinVar does not record submitter ethnicity or patient ancestry. The geographic distribution of ClinVar submitters skews heavily toward the United States and Europe, suggesting that training labels predominantly reflect variants observed in populations of European descent. This cannot be directly quantified because ClinVar lacks ancestry metadata.

**gnomAD population-specific frequencies:** An attempt was made to retrieve population-stratified allele frequencies (AFR, AMR, EAS, NFE) from gnomAD v4 via the Ensembl REST API. All population-specific AF values returned 0.0 due to an API limitation (the Ensembl `overlap/region` endpoint returns only a single `minor_allele_freq` field). The correct approach would use the gnomAD GraphQL API, which was not implemented.

**Quantification gap:** Because (1) ClinVar does not record ancestry, (2) population-specific AFs could not be retrieved, and (3) no external ancestry-annotated benchmark is available for these 5 genes, the degree of population bias in SteppeDNA's training data cannot be quantified with available data. This is an honest acknowledgment that the model's performance on non-European-ancestry variants is unknown and potentially degraded.

**Impact on clinical use:** Variants that are common in specific populations (e.g., BRCA1 c.5266dupC in Ashkenazi Jewish populations, BRCA2 founder mutations in Kazakh populations) may be misinterpreted because the model lacks population-specific context. Users from non-European ancestries should exercise additional caution when interpreting predictions.

---

## 14. MAVE Data Leakage Assessment

### Background

MAVE (Multiplexed Assays of Variant Effect) functional scores from Hu et al. 2024 (BRCA2 HDR assay) are used as 4 input features: `mave_score`, `has_mave`, `mave_abnormal`, `mave_x_blosum`. Since many MAVE-assayed variants also have ClinVar classifications used as training labels, indirect data leakage is a concern.

### Leakage Mechanism

MAVE scores are experimentally determined functional readouts. If a variant's MAVE score correlates with its ClinVar label (which it should, since both reflect pathogenicity), using MAVE as a feature while training on ClinVar labels creates circular reasoning. The model may learn "MAVE abnormal = pathogenic" rather than learning the underlying biology.

### Ablation Results

| Metric | With MAVE | Without MAVE (use_mave=False) | Delta |
|--------|-----------|-------------------------------|-------|
| ROC-AUC | 0.9781 | 0.9763 | -0.0017 |
| MCC | 0.8814 | 0.8795 | -0.0019 |
| BRCA2 AUC | 0.983 | 0.965 | -0.018 |
| Non-BRCA2 AUC | unchanged | unchanged | 0.000 |

### Conclusions

1. **Minimal overall impact:** Removing MAVE features changes AUC by only 0.002
2. **BRCA2-specific:** MAVE data exists only for BRCA2; non-BRCA2 genes are unaffected
3. **Low coverage:** Only 3.5% (134/3,845) of test variants have MAVE scores
4. **Mitigation:** The `use_mave=False` flag in `feature_engineering.py` provides a leakage-free evaluation mode
5. **Production decision:** MAVE features are kept in production because the leakage risk is minimal and the information is genuinely useful for BRCA2 variant interpretation

### AlphaMissense Indirect Label Leakage Risk

AlphaMissense (Cheng et al., Science 2023) was partially trained on ClinVar pathogenicity labels as part of its supervised fine-tuning step. SteppeDNA uses three AlphaMissense-derived features: `am_score`, `am_pathogenic`, and `am_x_phylop`. This creates an indirect leakage pathway: ClinVar labels were used to train AlphaMissense, whose scores are then used as features in SteppeDNA, which is itself trained on ClinVar labels.

**Standalone AlphaMissense AUC on this test set:** 0.660 (see Section 5b). This is notably low relative to published benchmarks on independent datasets (typically AUC ~0.90+), which likely reflects the class imbalance in this test set rather than a flaw in AlphaMissense itself.

**Feature importance:** `am_score` ranks within the top 15 features by mean absolute SHAP value.

**Risk assessment:** Low-to-moderate. The circular dependency exists in principle, but AlphaMissense's low standalone AUC on this test set (0.660) indicates it is not a dominant driver of SteppeDNA's performance. The model's performance is not simply recapitulating AlphaMissense's predictions. Ablation (retraining without AM features) is recommended as future work to quantify the exact impact of the circular dependency on reported metrics.

**Mitigation:** No action taken in production at this time. Full ablation (zero-out or drop `am_score`, `am_pathogenic`, `am_x_phylop` from all training and calibration data) would quantify the leakage impact and is a planned future experiment.

---

## 15. gnomAD API Limitation

### Ensembl API Single-MAF Limitation

The gnomAD population-stratified allele frequencies (AFR, AMR, EAS, NFE) in the training data are all zeros. This is not because the variants are ultra-rare in all populations -- it is because the Ensembl `overlap/region` REST API endpoint only returns a single `minor_allele_freq` field rather than population-specific frequencies.

### Technical Details

- **API used:** Ensembl REST API `overlap/region/{species}/{region}?feature=variation`
- **Response format:** Returns `minor_allele_freq` (single float) but not population-stratified AFs
- **Correct API:** gnomAD GraphQL API or VEP REST endpoint would provide population-stratified data
- **Impact:** Features `gnomad_af_afr`, `gnomad_af_amr`, `gnomad_af_eas`, `gnomad_af_nfe` are all 0.0 in training data
- **Model effect:** The model has no population-specific frequency signals

### Recommendation

Re-fetch gnomAD data using the gnomAD GraphQL API (`https://gnomad.broadinstitute.org/api`) to populate population-specific allele frequencies. This would improve the BS1 benign evidence code for population-specific variant interpretation.

## 16. Bootstrap Confidence Intervals

### Methodology

To provide empirically-grounded confidence intervals rather than parametric approximations, we train 50 bootstrap XGBoost models by resampling the training set (60% split) with replacement. Each bootstrap model uses identical hyperparameters to the production model (max_depth=7, learning_rate=0.05, n_estimators=400) but trains on a different bootstrap sample.

At prediction time, each input variant is scored by all 50 bootstrap models. The 5th and 95th percentiles of the resulting prediction distribution form a 90% confidence interval. The CI width directly quantifies prediction uncertainty.

### Equity Thesis Validation

Bootstrap CIs directly demonstrate the equity gap between data-rich and data-scarce genes:
- **BRCA2** (10,085 training variants): expected narrow CIs, indicating reliable predictions
- **PALB2** (2,456 training variants): expected wider CIs, reflecting less training data
- **RAD51D** (785 training variants): expected widest CIs, reflecting the smallest training set

This provides a quantitative measure of when users should exercise additional caution, beyond the qualitative gene reliability tiers.

### Implementation

- **Script:** `scripts/generate_bootstrap_models.py`
- **Models:** `data/bootstrap_models/bootstrap_0.json` through `bootstrap_49.json`
- **Integration:** `compute_bootstrap_ci()` in `backend/main.py`
- **Frontend:** CI displayed inline with probability and as a visual error bar

## 17. Active Learning Priority Ranker

### Methodology

To guide future experimental validation efforts, we implement an active learning priority ranking system that identifies VUS variants most likely to improve the model when experimentally characterized.

The priority score combines three factors:

1. **Query-by-Committee (QBC) Score:** The absolute difference between XGBoost and MLP predictions for each VUS. High disagreement indicates the models are uncertain, and resolving these variants would provide the most informative training signal.

2. **Gene Scarcity Weight:** Inversely proportional to the square root of gene training set size (1/sqrt(N)). Variants from data-scarce genes (RAD51D, RAD51C) are weighted higher because each new training example has proportionally greater impact.

3. **Positional Novelty Weight:** Inversely proportional to the number of nearby training variants within +/-50 amino acid positions (1/(1+nearby_count)). Variants in under-represented protein regions receive higher priority.

**Combined Priority = QBC_score x scarcity_weight x novelty_weight**

### Output

The top 50 variants per gene are saved to `data/active_learning_priorities.json` with the following fields per variant:
- `variant`: gene and protein notation
- `priority_score`: combined ranking metric
- `qbc_score`: model disagreement magnitude
- `current_prediction`: calibrated ensemble probability
- `nearby_training`: count of training variants within 50 AA positions
- `reason`: human-readable explanation of priority factors

### Clinical Utility

This feature provides actionable guidance for wet-lab validation:
- **High QBC + data-scarce gene:** The highest-value experiments for model improvement
- **High QBC + novel position:** Filling positional gaps in training coverage
- **Low nearby training:** Expanding the model's ability to generalize to uncharacterized protein regions

### Implementation

- **Script:** `scripts/active_learning_ranker.py`
- **Data:** `data/active_learning_priorities.json`
- **API:** `GET /research/priorities?gene=BRCA1&limit=10`
- **Frontend:** Research Priorities section with per-gene filtering

## 18. Contrastive Explanation Pairs

### Methodology

For each prediction, we identify the nearest training variant with the **opposite** classification (nearest benign if the query is predicted pathogenic, and vice versa). The nearest neighbor is found via a KD-tree built on scaled feature vectors per gene.

Feature-wise differences between the query and its contrastive pair are ranked by magnitude, and the top 5 are reported. This answers: "What would need to change for this variant to flip its classification?"

### Implementation

- **Index Building:** On startup, the master training dataset is split by gene and class (pathogenic/benign). Feature vectors are scaled using the production StandardScaler and indexed into `scipy.spatial.KDTree` structures for efficient nearest-neighbor lookup.
- **Query:** At prediction time, the query's scaled feature vector is compared against the opposite-class tree using Euclidean distance.
- **Output:** The response includes `contrastive_explanation` with:
  - `contrast_variant`: protein notation of the nearest opposite-class training variant (e.g., "Arg2660Gly")
  - `contrast_class`: "Benign" or "Pathogenic"
  - `contrast_distance`: Euclidean distance in scaled feature space
  - `key_differences`: top 5 features by absolute difference, with human-readable names and importance levels (high >2.0, moderate >1.0, low <1.0 in scaled units)

### Clinical Utility

Contrastive explanations complement SHAP by providing a concrete reference point: rather than explaining feature contributions in abstract, they show *which specific training variant* is most similar but classified differently, and *what features* distinguish them. This helps clinicians understand the model's decision boundary in the local feature neighborhood.

### Limitations

- Distance is measured in scaled feature space (StandardScaler), which normalizes all features to unit variance. This means Euclidean distance treats all features equally regardless of their predictive importance.
- The contrastive variant is the nearest *training* example, not necessarily the most clinically informative comparison.

## 19. Cross-Gene Feature Importance Transfer

### Motivation

BRCA2 dominates the training set (52%) with high AUC (0.983), while smaller genes (PALB2 0.641, RAD51C 0.743) suffer from data scarcity. This analysis tests whether BRCA2-derived SHAP feature importances can improve small-gene predictions by guiding feature weighting.

### Methodology

1. **BRCA2 SHAP Importance:** Computed mean absolute SHAP values across all 10,085 BRCA2 training variants using the production XGBoost model. Top features: `in_critical_repeat_region` (1.32), `am_score` (0.92), `relative_cdna_pos` (0.79), `dist_dna` (0.69), `bfactor` (0.67).

2. **Three Transfer Approaches:**
   - **Importance-Weighted Features:** Scale each feature by sqrt(BRCA2_importance + 0.1), amplifying features BRCA2 found predictive.
   - **Feature Selection:** Keep only features with above-median BRCA2 importance (52 of 103 features).
   - **Regularization-Guided:** Stronger L1 regularization (alpha=0.5) + reduced colsample_bytree (0.6) to force the model toward sparse, important features.

3. **Evaluation:** Each approach trained on 60% of gene-specific data, evaluated on 20% test set. Compared against both the universal model and a gene-specific baseline XGBoost.

### Results

| Gene   | Universal | Gene-Specific | Weighted | Feature Sel. | Reg-Guided | Best Approach |
|--------|-----------|---------------|----------|-------------|------------|---------------|
| BRCA1  | 0.6942    | 0.7541        | 0.7541   | 0.7550      | **0.7591** | Reg-Guided (+0.065 vs univ.) |
| PALB2  | **0.9079** | 0.6722       | 0.6722   | 0.6722      | 0.6601     | Universal best |
| RAD51C | **0.8951** | 0.6691       | 0.6691   | 0.6691      | 0.6521     | Universal best |
| RAD51D | **0.9547** | 0.8056       | 0.8056   | 0.8056      | 0.8031     | Universal best |

### Key Findings

1. **BRCA1:** Regularization-guided transfer provides a modest +0.065 AUC improvement over the universal model. This is the only gene where transfer shows clear benefit, likely because BRCA1 has sufficient data (5,432 variants) for a gene-specific model to outperform the universal model, and BRCA2-informed regularization further refines feature selection.

2. **PALB2, RAD51C, RAD51D:** The **universal model significantly outperforms** all gene-specific models (including transfer variants). These genes have too few training samples for gene-specific training to generalize well; the universal model's broader training data provides superior generalization.

3. **Feature importance transfer approaches (weighted features, feature selection) showed no improvement** over gene-specific baselines for any gene. The BRCA2-derived feature importances may be too BRCA2-specific (e.g., `in_critical_repeat_region` is a BRCA2-specific domain feature) to transfer effectively.

### Recommendation

**Analysis-only.** No transfer approach should be deployed to production:
- For PALB2/RAD51C/RAD51D, the universal model already provides the best performance.
- For BRCA1, the +0.065 improvement from regularization-guided training is interesting but would require further validation (cross-validation, temporal validation) before production use.
- The gene-adaptive ensemble weights (Item 38) already capture gene-specific optimization at the ensemble blending level.

### Script

- **Script:** `scripts/cross_gene_transfer.py`
- **Data:** `data/cross_gene_transfer_results.json`
