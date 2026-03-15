# SteppeDNA v5.3 — Complete Project Roadmap & Suggested Improvements

> **Document Date:** March 2026
> **Current Version:** 5.3.0
> **Scope:** Every identified improvement, enhancement, and fix — organized by priority and domain.

---

## Table of Contents

1. [Critical Data & Model Improvements](#1-critical-data--model-improvements)
2. [Feature Engineering Upgrades](#2-feature-engineering-upgrades)
3. [Model Architecture Improvements](#3-model-architecture-improvements)
4. [Calibration & Uncertainty Quantification](#4-calibration--uncertainty-quantification)
5. [ACMG Rule Engine Expansion](#5-acmg-rule-engine-expansion)
6. [Validation & Benchmarking](#6-validation--benchmarking)
7. [Population Equity & Bias Mitigation](#7-population-equity--bias-mitigation)
8. [Backend Infrastructure](#8-backend-infrastructure)
9. [Frontend & UX Improvements](#9-frontend--ux-improvements)
10. [Testing & Quality Assurance](#10-testing--quality-assurance)
11. [Deployment & Operations](#11-deployment--operations)
12. [Documentation & Compliance](#12-documentation--compliance)
13. [Research & Experimental](#13-research--experimental)

---

## 1. Critical Data & Model Improvements

These address the biggest performance gaps in SteppeDNA — non-BRCA2 generalization failure and data scarcity.

### 1.1 Expand Non-BRCA2 Training Data

**Problem:** PALB2 has only 2,456 training variants, RAD51C has 675, RAD51D has 346. Per-gene AUCs range from 0.641 (PALB2) to 0.804 (RAD51D), far below clinical-grade thresholds. Temporal AUCs for non-BRCA2 genes are 0.51–0.61 (near-random).

**Improvements:**
- Fetch additional proxy-benign variants from gnomAD v4.1 with relaxed AC threshold (AC ≥ 1 instead of AC ≥ 2), with appropriate quality filters
- Integrate LOVD (Leiden Open Variation Database) variants for PALB2, RAD51C, RAD51D — LOVD has curated submissions not always in ClinVar
- Incorporate InSiGHT database variants for mismatch repair genes that overlap HR pathway
- Add BRCAExchange (BRCA Challenge) variants — comprehensive BRCA1/BRCA2 expert-curated classifications
- Use ClinGen Evidence Repository curated variant interpretations as additional high-confidence training labels
- Scrape UniProt/Swiss-Prot reviewed variant annotations for functional evidence
- Integrate COSMIC somatic data (with appropriate labels) as weak supervision signal for rare genes
- Consider synthetic data augmentation (SMOTE or conditional GANs) specifically for RAD51C/RAD51D benign class, with careful evaluation for data leakage

### 1.2 Fix BRCA1 Class Imbalance

**Problem:** BRCA1 is 96.6% pathogenic in ClinVar training data. Isotonic calibration has almost no benign calibration points, making low probabilities (p < 0.3) unreliable.

**Improvements:**
- Aggressively mine gnomAD for BRCA1 common variants (AF > 0.001) as proxy-benign — there should be hundreds available
- Apply class-weighted loss during XGBoost training: `scale_pos_weight = n_benign / n_pathogenic` per gene
- Train a BRCA1-specific sub-model with SMOTE oversampling of benign class, validated on held-out real benign variants
- Use Platt scaling instead of isotonic regression for BRCA1 (parametric calibration handles sparse calibration data better)
- Consider focal loss for MLP training (down-weights easy pathogenic examples, focuses on hard benign examples)
- Implement cost-sensitive learning with per-gene misclassification costs

### 1.3 Address Temporal Generalization Failure

**Problem:** Non-BRCA2 temporal AUCs (0.51–0.61) indicate the model does not generalize to newly classified variants over time. Only BRCA2 (AUC 0.983) is temporally robust.

**Improvements:**
- Implement rolling-window retraining: retrain monthly on latest ClinVar snapshot, track per-gene AUC drift
- Add temporal features: `years_since_first_clinvar_submission`, `clinvar_review_stars`, `n_submitters` — these capture evolving classification confidence
- Apply domain adaptation techniques: treat pre-2024 as source domain, 2024+ as target domain, use adversarial training to learn time-invariant features
- Implement concept drift detection: monitor prediction distribution shift on incoming variants vs training distribution
- Create a temporal validation dashboard that automatically evaluates model performance on each quarterly ClinVar release
- Add online learning capability: incrementally update model with newly classified variants (with appropriate hold-out validation)

### 1.4 Resolve MAVE Data Leakage

**Problem:** MAVE functional assay data overlaps with ClinVar training labels, creating circular reasoning risk. Current ablation shows minimal impact (delta AUC = -0.0017), but leakage is methodologically concerning.

**Improvements:**
- Split MAVE-assayed variants into separate validation fold — never train on variants that have both MAVE scores and ClinVar labels simultaneously
- Create a `mave_holdout` flag in `master_training_dataset.csv` for variants where MAVE data was used to inform ClinVar classification
- Implement leave-MAVE-out cross-validation: for each fold, remove MAVE features from variants whose ClinVar labels were informed by MAVE
- Replace current MAVE features with MAVE residuals (MAVE score minus expected score from other features) to capture only independent MAVE signal
- Document the exact MAVE→ClinVar submission pathway for transparency

### 1.5 Resolve AlphaMissense Indirect Label Leakage

**Problem:** AlphaMissense was partially trained on ClinVar pathogenicity labels. SteppeDNA uses `am_score`, `am_pathogenic`, `am_x_phylop` as features — creating an indirect leakage pathway. Ablation has not been performed yet.

**Improvements:**
- Perform full AlphaMissense ablation: retrain model with `am_score`, `am_pathogenic`, `am_x_phylop` zeroed out, report per-gene delta AUC
- If delta AUC > 0.01 for any gene, implement one of:
  - Use AlphaMissense residuals (AM score minus ClinVar-predicted AM score)
  - Replace AM with EVE scores (which don't use ClinVar labels) where available
  - Train a debiased AM score using only non-ClinVar training signal
- Add `use_alphamissense=False` flag (analogous to existing `use_mave=False`) for leakage-free evaluation mode
- Document the AM→ClinVar label dependency chain in MODEL_CARD.md

---

## 2. Feature Engineering Upgrades

### 2.1 Upgrade ESM-2 Model (8M → 650M)

**Problem:** Current ESM-2 model has only 8M parameters. The 650M model significantly improves BRCA2 (+0.011 AUC) and BRCA1 (+0.057) but catastrophically degrades PALB2 (0.641 → 0.521) when naively integrated.

**Improvements:**
- Use ESM-2 650M embeddings but apply per-gene PCA dimensionality reduction with gene-specific component counts (more components for data-rich genes, fewer for data-scarce)
- Apply feature selection (e.g., mutual information) on ESM-2 650M features to keep only those with positive per-gene predictive signal
- Use ESM-2 650M for BRCA1/BRCA2 only (where it helps) and keep 8M for PALB2/RAD51C/RAD51D (where 650M hurts)
- Fine-tune ESM-2 with LoRA on HR gene family sequences specifically, then extract embeddings
- Implement an ESM-2 feature gating mechanism: learn per-gene gates that selectively use 650M features
- Try ESM-2 3B or ESMFold for even larger context window and structural awareness

### 2.2 Fix GNN Structural Feature Leakage

**Problem:** GNN AlphaFold structural features act as gene identifiers (non-zero only for non-BRCA2 genes, XGBoost gain = 647 vs next feature gain = 21). This creates a shortcut that masks true biological signal.

**Improvements:**
- Normalize GNN features per-gene (z-score within each gene) to remove gene-identifying signal
- Apply adversarial debiasing: add a gene-prediction head during training and penalize XGBoost for using gene-identifying GNN features
- Use only gene-agnostic structural features: RSA, B-factor, secondary structure — not gene-specific 3D coordinates
- Train GNN on all 5 genes jointly with gene-masking augmentation (randomly mask gene identity during training)

### 2.3 Add New Feature Categories

**Improvements:**
- **Population-stratified allele frequencies:** Switch from Ensembl REST API to gnomAD GraphQL API to get per-population AFs (AFR, AMR, EAS, NFE, SAS) — current implementation returns all zeros due to API limitation
- **EVE scores (Evolutionary model of Variant Effect):** Add as 104th feature — improves BRCA1 (+0.050) and RAD51C (+0.024). EVE available for BRCA1/PALB2/RAD51C/RAD51D but NOT BRCA2
- **REVEL/BayesDel/CADD as meta-features:** Use competitor predictor scores as input features (creates circular dependency but dramatically improves small-gene predictions)
- **Protein-protein interaction (PPI) features:** Distance to known PPI interface residues (BRCA1-PALB2, BRCA2-RAD51, PALB2-BRCA2)
- **Post-translational modification (PTM) proximity:** Distance to known phosphorylation, ubiquitination, SUMOylation sites
- **Codon usage bias:** Rare codon at mutation site may affect translation efficiency
- **mRNA secondary structure change:** RNAfold predicted MFE change at mutation site
- **Protein stability predictions:** DynaMut2/FoldX predicted ddG (free energy change)
- **Evolutionary rate features:** dN/dS ratio at mutation site
- **CpG dinucleotide context:** CpG sites have 10x higher mutation rate — captures mutational mechanism

### 2.4 Improve Existing Feature Quality

**Improvements:**
- Replace single PhyloP score with multi-resolution conservation: PhyloP (20-way, 100-way, 470-way primates/mammals/vertebrates)
- Add GERP++ rejected substitution scores alongside PhyloP
- Compute SpliceAI with larger context window (10,000bp instead of default 50bp) for better cryptic splice detection
- Add per-gene domain-specific features: BRCA2 BRC repeat number, BRCA1 RING finger zinc coordination residues
- Compute amino acid physicochemical property vectors (Grantham distance, Miyata distance, Sneath index) in addition to BLOSUM62
- Replace binary `in_critical_repeat_region` with continuous domain importance scores (weighted by pathogenic variant density in each domain)

---

## 3. Model Architecture Improvements

### 3.1 Per-Gene Sub-Models

**Problem:** Universal model trained on all 5 genes simultaneously. BRCA2 dominates (52% of data) and its patterns may not transfer to smaller genes.

**Improvements:**
- Train separate per-gene XGBoost models for BRCA1 and BRCA2 (sufficient data for gene-specific learning)
- Keep universal model for PALB2/RAD51C/RAD51D (too little data for standalone models)
- Implement a meta-learner that selects between gene-specific and universal model per prediction
- Use multi-task learning: shared lower layers with gene-specific output heads
- Implement mixture-of-experts: route variants to gene-specific expert networks

### 3.2 Ensemble Architecture Expansion

**Improvements:**
- Add LightGBM as a third ensemble member (different boosting algorithm may capture complementary patterns)
- Add a Random Forest for calibration diversity (less prone to overfitting than boosting on small datasets)
- Replace fixed ensemble weights with learned stacking: train a logistic regression meta-learner on out-of-fold predictions from XGBoost + MLP + LightGBM
- Implement Bayesian model averaging: weight models by their posterior probability given the calibration data
- Add a CatBoost model (handles categorical features natively, good for one-hot AA encoding)
- Try neural architecture search (NAS) for MLP topology per gene

### 3.3 Hyperparameter Optimization

**Improvements:**
- Run Optuna/Hyperopt Bayesian optimization instead of manual tuning
- Optimize per-gene hyperparameters (not just ensemble weights)
- Search space:
  - XGBoost: max_depth (3–10), learning_rate (0.01–0.3), n_estimators (100–1000), min_child_weight (1–10), gamma (0–5)
  - MLP: layer sizes (64–512), dropout rates (0.1–0.5), learning rate (1e-4–1e-2), batch size (16–128)
- Use nested cross-validation for unbiased hyperparameter selection
- Implement early stopping on per-gene validation AUC (not overall AUC) to prevent BRCA2 domination

### 3.4 Advanced ML Techniques

**Improvements:**
- Implement attention-based feature weighting: let the model learn which features matter per-gene dynamically
- Try TabNet (attention-based tabular model) as ensemble member — designed specifically for tabular data
- Implement contrastive learning on feature vectors: learn a representation space where same-class variants are close
- Use graph neural networks on protein contact maps (not as features, but as direct model input)
- Try ordinal regression instead of binary classification: Benign < Likely Benign < VUS < Likely Pathogenic < Pathogenic
- Implement conformal prediction with adaptive coverage per gene (wider prediction sets for data-scarce genes)

---

## 4. Calibration & Uncertainty Quantification

### 4.1 Improve Per-Gene Calibration

**Problem:** BRCA1 isotonic calibration has ~96.6% pathogenic calibration points → unreliable probabilities in benign range.

**Improvements:**
- Switch BRCA1 to Platt scaling (sigmoid calibration) — parametric method handles sparse data better than non-parametric isotonic
- Implement Beta calibration (Kull et al. 2017) — more flexible than Platt, handles both over- and under-confidence
- Use Venn-Abers prediction (multi-probability calibration) for per-gene uncertainty-aware calibration
- Apply temperature scaling (Guo et al. 2017) on MLP output before ensemble blending
- Cross-validate calibration: use 5-fold calibration within the 20% calibration set to reduce calibration variance
- Report calibration error (ECE, MCE) per gene in model metrics

### 4.2 Enhance Bootstrap Confidence Intervals

**Problem:** Current 50-model bootstrap may underestimate true uncertainty for data-scarce genes.

**Improvements:**
- Increase bootstrap count from 50 to 200 (more stable CI estimates)
- Use stratified bootstrap resampling (maintain per-gene class ratios in each resample)
- Implement both XGBoost and MLP bootstrap (currently only XGBoost) — combine for ensemble CI
- Add bias-corrected and accelerated (BCa) bootstrap intervals instead of simple percentile method
- Report CI width as an explicit uncertainty metric alongside probability
- Implement prediction intervals (not just confidence intervals) to capture both model and data uncertainty

### 4.3 Improve Conformal Prediction

**Improvements:**
- Switch from split conformal to cross-conformal prediction (uses all calibration data more efficiently)
- Implement adaptive conformal inference (ACI) for time-varying data distributions
- Per-gene conformal thresholds with gene-specific alpha levels (stricter for data-scarce genes)
- Add Mondrian conformal prediction: separate conformal predictors for each gene class
- Report conformal prediction set sizes as quality metric (smaller sets = more informative)

### 4.4 Add Epistemic vs Aleatoric Uncertainty Decomposition

**Improvements:**
- Implement MC Dropout in MLP: run 100 forward passes with dropout enabled, decompose variance into epistemic (model uncertainty) and aleatoric (data noise)
- Use Deep Ensembles (Lakshminarayanan et al. 2017): train 5 independent MLP networks with different random seeds
- Report `epistemic_uncertainty` and `aleatoric_uncertainty` separately in API response
- High epistemic uncertainty → "model unsure, more data needed"
- High aleatoric uncertainty → "inherently ambiguous variant"

---

## 5. ACMG Rule Engine Expansion

### 5.1 Add Missing ACMG/AMP Criteria

**Problem:** Only ~10 of 28+ established ACMG criteria implemented. Missing criteria reduce classification accuracy and clinical utility.

**Currently implemented:** PVS1, PS1, PM1, PM2, PM4, PM5, PP3, BA1, BS1, BP4, BP7

**Improvements — add these missing criteria:**

**Pathogenic:**
- **PS2 (De novo):** Accept parental genotype data via API to confirm de novo status
- **PS3 (Functional studies):** Integrate MAVE functional data as PS3 evidence (with well-established assay designation)
- **PS4 (Prevalence in affected):** Compare variant frequency in case cohorts vs controls
- **PM3 (Trans with pathogenic):** Accept compound heterozygosity data to assess trans configuration
- **PM6 (Assumed de novo):** Without parental confirmation
- **PP1 (Cosegregation):** Accept family segregation data (LOD score input)
- **PP2 (Missense in constrained gene):** Use o/e ratio from gnomAD constraint metrics
- **PP4 (Patient phenotype):** Accept HPO phenotype terms to match expected gene-disease association
- **PP5 (Reputable source):** Already partially implemented via ClinVar lookup — formalize as PP5

**Benign:**
- **BS2 (Observed in healthy adult):** Use gnomAD age-stratified data to confirm observation in healthy adults
- **BS3 (Functional studies benign):** Integrate MAVE benign functional data
- **BS4 (Non-segregation):** Accept family segregation data showing non-cosegregation
- **BP1 (Missense in truncation gene):** Flag if gene's primary mechanism is truncation (relevant for BRCA1)
- **BP2 (Observed in trans with pathogenic):** Use zygosity data to confirm compound het with known pathogenic
- **BP3 (In-frame in non-functional region):** Check if in-frame indel is outside critical domains
- **BP5 (Found in case with alternate cause):** Accept diagnostic information
- **BP6 (Reputable source benign):** ClinVar benign classifications with review stars ≥ 2

### 5.2 Gene-Specific ACMG Modifications

**Improvements:**
- Implement ClinGen BRCA1/BRCA2 VCEP-specific rule modifications (2018 Biesecker et al.)
- Use gene-specific PVS1 decision trees from ClinGen SVI
- Calibrate PP3/BP4 thresholds per gene using posterior probability (Tavtigian et al. 2018 Bayesian framework)
- Implement ACMG point-based system (Tavtigian et al. 2020): assign numerical points to each evidence code, sum for classification
- Add PVS1 strength modifiers: last-exon NMD escape (already in v5.3), biologically relevant alternate transcripts, rescue by read-through

### 5.3 ACMG Evidence Combination Improvements

**Improvements:**
- Replace current rule-based combination with Bayesian framework (Tavtigian et al. 2018): prior probability × likelihood ratios → posterior probability
- Implement evidence code conflicts detection: flag when pathogenic and benign evidence codes coexist
- Add evidence strength modification: allow PP3 upgrade to PM or PS based on calibrated odds of pathogenicity
- Report ACMG classification confidence: how many evidence codes triggered, how strong the combined evidence
- Add ACMG classification explanation: natural language description of why variant received its classification

---

## 6. Validation & Benchmarking

### 6.1 Prospective Validation

**Problem:** All current validation is retrospective on existing ClinVar classifications. No prospective validation has been performed.

**Improvements:**
- Partner with clinical genetics labs for prospective variant classification comparison
- Submit SteppeDNA classifications to ClinVar as computational evidence (with appropriate review star level)
- Implement a prospective tracking dashboard: log predictions for VUS, check back quarterly against updated ClinVar
- Collect wet-lab validation data for active learning-prioritized VUS
- Compare SteppeDNA predictions with ENIGMA consortium expert classifications prospectively

### 6.2 Expand Benchmark Suite

**Improvements:**
- Add ProteinGym DMS benchmarks for BRCA2 (currently only BRCA1 Findlay et al.)
- Integrate MAVE-DB benchmarks for PALB2 functional assays
- Add ClinVar expert panel variants for RAD51C/RAD51D (currently only BRCA1/2 and PALB2)
- Benchmark against VariBench curated variant datasets
- Compare with VARITY, VEST4, MetaRNN, and other recent predictors
- Add benchmarks from CAGI (Critical Assessment of Genome Interpretation) challenges
- Evaluate on dbscSNV splice prediction benchmark for splice variant classification

### 6.3 Clinical Expert Review

**Problem:** No geneticist, oncologist, or genetic counselor has validated SteppeDNA outputs.

**Improvements:**
- Recruit 3+ clinical geneticists for blinded review of 100 SteppeDNA predictions
- Measure concordance between SteppeDNA and expert classifications (Cohen's kappa)
- Identify systematic disagreements and incorporate expert feedback into model updates
- Have a genetic counselor review ACMG evidence code accuracy on 50 test variants
- Document expert review results in VALIDATION_REPORT.md

### 6.4 Continuous Validation Pipeline

**Improvements:**
- Automate quarterly re-evaluation on latest ClinVar snapshot
- Track per-gene AUC drift over time (alert if delta AUC > 0.02)
- Implement A/B testing infrastructure for model updates
- Create regression test suite: known-correct predictions that must not change between model versions
- Add ClinVar reclassification tracking: detect when training labels change and flag affected predictions

---

## 7. Population Equity & Bias Mitigation

### 7.1 Fix gnomAD Population-Stratified Allele Frequencies

**Problem:** Current implementation uses Ensembl `overlap/region` API which only returns single `minor_allele_freq`, not population-stratified AFs. All population-specific features (`gnomad_af_afr`, `gnomad_af_amr`, `gnomad_af_eas`, `gnomad_af_nfe`) are zeros.

**Improvements:**
- Switch to gnomAD GraphQL API (`https://gnomad.broadinstitute.org/api`) which provides per-population AFs (AFR, AMR, ASJ, EAS, FIN, NFE, SAS, OTH)
- Alternatively, use Ensembl VEP REST endpoint (`/vep/human/hgvs/`) which returns gnomAD population data
- Download gnomAD sites VCF for offline lookup (most reliable, no API rate limits)
- Add population-specific BS1/BA1 thresholds (different frequency thresholds for different populations)
- Implement population-maximum AF (popmax) feature for ACMG filtering

### 7.2 Address Central Asian / Kazakh Population Gap

**Problem:** Kazakh founder mutations not represented in training data. ClinVar submissions skew heavily toward US/European institutions. Central Asian populations underrepresented in gnomAD.

**Improvements:**
- Integrate Kazakhstan Genome Project data (if available) for Kazakh-specific variant frequencies
- Add known Kazakh BRCA1/BRCA2 founder mutations (e.g., BRCA1 c.5382insC, BRCA2 c.5946delT) to rule-based Tier 1 interceptor
- Partner with Kazakh National Center for Biotechnology for population-specific variant data
- Flag variants with zero representation in gnomAD Central Asian populations: "No population data available for this ancestry"
- Add ancestry-aware disclaimers in API response when patient ancestry is non-European

### 7.3 Broader Population Bias Mitigation

**Improvements:**
- Report per-population prediction performance (if population-labeled test data becomes available)
- Add ancestry as an input parameter and adjust predictions for population-specific variant frequencies
- Implement fairness metrics: equal opportunity (TPR equal across populations), demographic parity
- Use adversarial debiasing to remove ancestry-correlated signal from feature space
- Create a population bias dashboard showing variant coverage per gnomAD population
- Add population-specific calibrators when sufficient data is available

### 7.4 Kazakh Language Translation Improvements

**Problem:** Kazakh medical/scientific terminology not expert-verified (Limitation #15).

**Improvements:**
- Engage Kazakh-speaking clinical geneticist for medical term review
- Verify translations of ACMG criteria, risk tier names, feature descriptions
- Add Kazakh-specific clinical context (local healthcare system, genetic counseling practices)
- Review SHAP feature explanations in Kazakh for accuracy
- Add Kazakh language disclaimer if not expert-reviewed

---

## 8. Backend Infrastructure

### 8.1 Performance Optimizations

**Improvements:**
- Implement async feature computation: compute BLOSUM62, PhyloP, ESM-2 lookups concurrently (currently sequential)
- Add Redis caching layer for frequently queried variants (replace in-memory LRU)
- Pre-compute feature vectors for all known ClinVar variants at startup (warm cache)
- Batch prediction API: accept multiple variants in single request (reduce HTTP overhead)
- Implement connection pooling for external API calls (ClinVar, gnomAD)
- Profile and optimize `build_feature_vector()` — currently ~50ms per variant, target <10ms
- Use ONNX runtime for MLP inference (faster than TensorFlow for single predictions)
- Lazy-load bootstrap models (don't load all 50 at startup if CI not always requested)

### 8.2 API Enhancements

**Improvements:**
- Add API versioning: `/v1/predict`, `/v2/predict` for backward compatibility
- Implement WebSocket endpoint for real-time prediction streaming (batch VCF progress)
- Add GraphQL endpoint as alternative to REST (flexible field selection)
- Implement pagination for `/history` and `/research/priorities` endpoints
- Add variant normalization endpoint: accept various notation formats (HGVS, VCF, protein) and normalize
- Support HGVS notation input directly (e.g., `NM_000059.4:c.6174delT`)
- Add bulk lookup endpoint: `/predict/batch` accepting JSON array of variants
- Implement response compression (gzip/brotli) for large payloads

### 8.3 Database Improvements

**Improvements:**
- Migrate from SQLite to PostgreSQL for production (better concurrency, full-text search, JSON support)
- Add database migrations (Alembic) for schema versioning
- Implement data retention policies: auto-archive analyses older than 1 year
- Add indexes on frequently queried columns (gene, prediction, timestamp)
- Implement database connection pooling (SQLAlchemy or asyncpg)
- Add full-text search on variant descriptions
- Store SHAP values in database for retrospective analysis
- Add user accounts and analysis ownership (optional, for institutional deployments)

### 8.4 Security Enhancements

**Improvements:**
- Implement JWT authentication (replace simple API key)
- Add role-based access control (RBAC): admin, researcher, viewer
- Implement API rate limiting per user (not just per IP)
- Add input sanitization for VCF file uploads (prevent path traversal, zip bombs)
- Implement CORS with explicit origin whitelist (currently allows configured origins)
- Add CSP (Content Security Policy) headers
- Implement audit log tamper detection (hash chain on log entries)
- Add GDPR compliance features: data export, data deletion, consent tracking
- Encrypt database at rest (SQLCipher for SQLite, or PostgreSQL TDE)
- Add IP allowlisting for admin endpoints

### 8.5 Logging & Monitoring

**Improvements:**
- Implement structured logging with correlation IDs across request lifecycle
- Add Prometheus metrics exporter for Grafana dashboards
- Implement distributed tracing (OpenTelemetry) for multi-service deployments
- Add alerting: email/Slack notifications for model performance drift, high error rates
- Implement log rotation and archival (currently logs grow unbounded)
- Add latency histograms per endpoint per gene
- Track prediction distribution drift (KL divergence from training distribution)
- Add health check for external API connectivity (ClinVar, gnomAD reachability)

---

## 9. Frontend & UX Improvements

### 9.1 Visualization Enhancements

**Improvements:**
- Add interactive UMAP variant landscape with zoom, pan, and variant selection (current UMAP is precomputed static)
- Implement SHAP waterfall plots (not just bar charts) showing cumulative feature contributions
- Add SHAP beeswarm plot for global feature importance across all predictions in session
- Implement protein 3D structure viewer with mutation highlighting (NGL.js or Mol* viewer)
- Add domain architecture visualization (lollipop plot) with interactive domain tooltips
- Implement calibration reliability diagram in frontend (currently only in visual proofs)
- Add per-gene ROC curve visualization with interactive threshold adjustment
- Implement variant distribution heatmap along protein sequence
- Add confidence interval visualization (error bars or violin plots)

### 9.2 UX Workflow Improvements

**Improvements:**
- Add multi-variant queue: analyze multiple variants sequentially without page refresh
- Implement variant bookmarking: save interesting variants for later comparison
- Add variant sharing: generate shareable URL for specific predictions
- Implement batch result filtering and sorting (by probability, gene, risk tier)
- Add keyboard shortcuts for power users (Ctrl+Enter to analyze, Tab through fields)
- Implement auto-complete for amino acid names (type "Ser" → suggest "Serine (S)")
- Add variant input validation with real-time feedback (before submission)
- Implement undo/redo for form inputs
- Add quick-fill buttons for common test variants
- Implement session persistence (resume analysis after browser close)

### 9.3 Report Generation Improvements

**Improvements:**
- Add clinical report template (structured for genetic counseling sessions)
- Implement multi-variant PDF report (all variants from VCF in single document)
- Add institution logo and customization options for reports
- Include full ACMG evidence summary in PDF
- Add QR code linking to online result viewer
- Implement report versioning (track when model version changes)
- Add HGVS notation in reports alongside protein change notation
- Include population frequency context in reports

### 9.4 Accessibility Improvements

**Improvements:**
- Add screen reader testing (NVDA, VoiceOver) and fix any issues
- Implement full keyboard navigation for all interactive elements
- Add high contrast mode (separate from dark mode)
- Ensure all charts have text alternatives (alt text for SVG/canvas elements)
- Add aria-live announcements for prediction results
- Implement reduced motion mode (respect `prefers-reduced-motion`)
- Add text resizing support (up to 200% without horizontal scroll)
- Test with WAVE accessibility evaluation tool and fix all issues

### 9.5 Mobile Experience

**Improvements:**
- Optimize VCF upload for mobile (camera capture for paper lab results)
- Add bottom navigation bar for mobile (currently relies on scrolling)
- Implement swipe gestures for variant comparison
- Add haptic feedback for prediction results on supported devices
- Optimize touch targets (minimum 48x48px per WCAG)
- Add pull-to-refresh on analysis history

---

## 10. Testing & Quality Assurance

### 10.1 Expand Test Coverage

**Problem:** Current suite has 200+ tests across 10 modules, but coverage gaps exist for non-BRCA2 genes and edge cases.

**Improvements:**
- Add clinical correctness tests for PALB2, RAD51C, RAD51D known pathogenic/benign variants
- Add integration tests: full end-to-end from API request → feature engineering → prediction → response
- Add VCF parsing edge case tests: multi-allelic, multi-sample, missing genotype, structural variants
- Add concurrency stress tests: 100 simultaneous predictions
- Add database failure resilience tests: verify predictions work when SQLite is locked/corrupted
- Add external API timeout tests: verify graceful degradation when ClinVar/gnomAD are unreachable
- Add ACMG evidence combination exhaustive tests: all valid combinations of evidence codes
- Add per-gene calibration tests: verify calibrated probabilities are well-ordered
- Add bootstrap CI tests: verify CI width correlates with data scarcity
- Measure and report test coverage percentage (target >90% line coverage)

### 10.2 Property-Based Testing Expansion

**Improvements:**
- Add Hypothesis strategies for generating random valid variants across all 5 genes
- Test monotonicity properties: increasing PhyloP → increasing pathogenicity probability
- Test symmetry: BLOSUM62(ref, alt) == BLOSUM62(alt, ref)
- Test boundary conditions: probability always in [0.005, 0.995], CI lower ≤ probability ≤ CI upper
- Test feature vector invariants: exactly 103 features, no NaN values, no infinity
- Test ACMG evidence consistency: pathogenic evidence codes should correlate with high probability
- Test contrastive explanation properties: contrastive variant always has opposite class

### 10.3 Performance & Load Testing

**Improvements:**
- Implement locust or k6 load testing scripts
- Benchmark: target <100ms p95 latency for single predictions
- Benchmark: target <30s for 1000-variant VCF processing
- Memory profiling: ensure no memory leaks under sustained load
- Add startup time benchmark: model loading should complete in <30s
- Test under memory pressure: verify graceful degradation when OOM

### 10.4 Regression Testing

**Improvements:**
- Create golden test set: 100 variants with locked-in expected predictions
- Version pin all predictions: any model update must not change >5% of golden test predictions
- Add visual regression testing for frontend (Playwright or Cypress screenshots)
- Implement canary deployments: test new model on 1% traffic before full rollout

---

## 11. Deployment & Operations

### 11.1 Container & Orchestration

**Improvements:**
- Add multi-stage Docker build (reduce image size from ~2GB to <500MB)
- Implement Kubernetes deployment manifests (Deployment, Service, Ingress, HPA)
- Add horizontal pod autoscaler: scale workers based on prediction latency
- Implement health check liveness and readiness probes (separate from current `/health`)
- Add resource limits and requests (CPU, memory) in container specs
- Implement init containers for model download and checksum verification
- Add sidecar container for log collection (Fluentd/Filebeat)
- Use distroless base image for security (no shell, no package manager)

### 11.2 CI/CD Pipeline Enhancement

**Improvements:**
- Add model validation step: run benchmark suite on every model artifact change
- Implement staged deployments: dev → staging → production with promotion gates
- Add code quality gates: minimum test coverage, no linting errors, no type errors
- Implement infrastructure-as-code (Terraform/Pulumi) for cloud deployment
- Add dependency vulnerability scanning (Dependabot, Snyk)
- Implement semantic versioning automation
- Add performance regression detection in CI (compare latency benchmarks across commits)
- Implement rollback automation: auto-revert if health check fails after deployment

### 11.3 Monitoring & Alerting

**Improvements:**
- Implement model performance monitoring dashboard (per-gene AUC over time)
- Add prediction distribution monitoring (detect concept drift)
- Implement error budget tracking (SLO: 99.9% availability, p95 latency <200ms)
- Add business metrics tracking: predictions per day, unique users, VCF uploads
- Implement on-call alerting (PagerDuty/OpsGenie) for critical failures
- Add synthetic monitoring: scheduled test predictions to verify end-to-end functionality
- Implement log-based alerting for unusual patterns (spike in pathogenic predictions, unusual genes)

### 11.4 Backup & Disaster Recovery

**Improvements:**
- Implement automated database backups (daily, retained 30 days)
- Add model artifact versioning (S3/GCS with version tags)
- Implement point-in-time recovery for analysis database
- Add cross-region deployment for high availability
- Document recovery time objective (RTO) and recovery point objective (RPO)
- Implement blue-green deployment for zero-downtime updates

---

## 12. Documentation & Compliance

### 12.1 Technical Documentation

**Improvements:**
- Add API documentation with request/response examples for every endpoint
- Create architecture decision records (ADRs) for major design choices
- Add data dictionary: every feature name, type, range, source, and interpretation
- Document model update procedures: retraining, validation, deployment checklist
- Add developer onboarding guide: environment setup, local development, testing
- Create troubleshooting guide for common deployment issues

### 12.2 Clinical Documentation

**Improvements:**
- Write clinical user guide: interpretation of predictions for genetic counselors
- Add case studies: example variant analyses with clinical context
- Create training materials: webinar slides for clinical genetics programs
- Write limitations disclosure for clinical settings (informed consent language)
- Add bibliography of validation studies and relevant literature

### 12.3 Regulatory Compliance Preparation

**Improvements:**
- Begin CE-IVD (In Vitro Diagnostic) documentation preparation (EU MDR Class C)
- Prepare FDA 510(k) submission materials (if pursuing US market)
- Document Quality Management System (QMS) per ISO 13485
- Implement risk analysis per ISO 14971 (medical device risk management)
- Prepare clinical evaluation report (CER) per MEDDEV 2.7/1 Rev 4
- Document intended use statement, contraindications, warnings
- Implement change control procedures for model updates
- Add traceability matrix: requirements → design → implementation → testing

### 12.4 Ethical & Responsible AI Documentation

**Improvements:**
- Expand Model Card with Datasheets for Datasets (Gebru et al. 2021)
- Add fairness impact assessment: per-population, per-gene, per-variant-type
- Document stakeholder engagement (patients, clinicians, researchers)
- Create responsible AI checklist for model updates
- Add transparency report: model limitations, known failure modes, bias sources
- Implement model interpretability documentation: how SHAP, ACMG, contrastive explanations work

---

## 13. Research & Experimental

### 13.1 Multi-Omics Integration

**Improvements:**
- Integrate RNA-seq expression data: tissue-specific expression levels for each gene
- Add methylation data: CpG methylation status at variant site
- Incorporate proteomics data: protein abundance changes associated with variants
- Integrate Hi-C chromatin interaction data: 3D genome context at variant site
- Add long-read sequencing data: detect structural variants and complex rearrangements

### 13.2 Transfer Learning from Related Genes

**Improvements:**
- Pre-train on all known missense variant classifiers (ClinVar-wide, not just HR genes)
- Fine-tune on HR gene family with adapter layers
- Explore cross-gene knowledge transfer using protein family embeddings (Pfam, InterPro)
- Implement few-shot learning for new gene addition: train on BRCA2, adapt to new gene with <100 variants
- Use protein language model fine-tuning (ESM-2 LoRA) on HR gene family specifically

### 13.3 Gene Panel Expansion

**Improvements:**
- Add ATM (Ataxia Telangiectasia Mutated) — key HR pathway gene
- Add CHEK2 (Checkpoint Kinase 2) — moderate-penetrance breast cancer gene
- Add NBN (Nibrin) — Nijmegen breakage syndrome gene
- Add BARD1 (BRCA1-associated RING domain 1) — BRCA1 interaction partner
- Add TP53 (Li-Fraumeni syndrome) — complementary cancer predisposition gene
- For each new gene: curate training data, compute features, validate independently
- Implement gene addition pipeline: standardized workflow for adding new genes without code changes

### 13.4 Variant Effect Prediction Beyond Pathogenicity

**Improvements:**
- Predict functional impact type: loss-of-function, gain-of-function, dominant-negative
- Predict drug response: variants affecting PARP inhibitor sensitivity (olaparib, rucaparib)
- Predict cancer risk quantification: variant-specific odds ratio estimation
- Predict protein stability change (ddG) alongside pathogenicity
- Implement variant-specific penetrance estimation (lifetime cancer risk)

### 13.5 Clinical Decision Support Integration

**Improvements:**
- Develop FHIR (Fast Healthcare Interoperability Resources) integration for EHR systems
- Implement HL7 v2 messaging for laboratory information systems (LIS)
- Create SMART on FHIR app for embedding SteppeDNA within EHR workflows
- Develop API integration with genetic testing platforms (Invitae, GeneDx, Ambry)
- Implement Clinical Decision Support (CDS) hooks for real-time variant classification during genetic testing

### 13.6 Explainability Research

**Improvements:**
- Implement LIME (Local Interpretable Model-agnostic Explanations) alongside SHAP for comparison
- Add counterfactual explanations: "What minimal change to this variant would flip the prediction?"
- Implement concept-based explanations: explain predictions in terms of biological concepts (charge change, conservation) rather than raw features
- Add causal inference: use do-calculus to distinguish correlation from causation in feature attributions
- Implement global interpretability: decision tree approximation of the ensemble for human understanding

---

## Priority Matrix

| Priority | Category | Impact | Effort |
|----------|----------|--------|--------|
| **P0 — Critical** | Fix gnomAD population AF (7.1) | High | Low |
| **P0 — Critical** | AlphaMissense ablation (1.5) | High | Low |
| **P0 — Critical** | Expand non-BRCA2 data (1.1) | Very High | Medium |
| **P0 — Critical** | Fix BRCA1 class imbalance (1.2) | High | Medium |
| **P1 — High** | ESM-2 650M per-gene integration (2.1) | High | Medium |
| **P1 — High** | Add EVE scores (2.3) | Medium | Low |
| **P1 — High** | Population equity (7.2, 7.3) | High | Medium |
| **P1 — High** | ACMG expansion (5.1) | High | High |
| **P1 — High** | Prospective validation (6.1) | Very High | High |
| **P1 — High** | Clinical expert review (6.3) | Very High | Medium |
| **P2 — Medium** | Per-gene sub-models (3.1) | Medium | Medium |
| **P2 — Medium** | Ensemble expansion (3.2) | Medium | Medium |
| **P2 — Medium** | Hyperparameter optimization (3.3) | Medium | Low |
| **P2 — Medium** | Bootstrap CI improvements (4.2) | Medium | Low |
| **P2 — Medium** | MAVE leakage fix (1.4) | Low | Medium |
| **P2 — Medium** | New feature categories (2.3) | Medium | High |
| **P2 — Medium** | Temporal generalization (1.3) | High | High |
| **P3 — Low** | Backend performance (8.1) | Low | Medium |
| **P3 — Low** | Frontend viz (9.1) | Medium | Medium |
| **P3 — Low** | Test expansion (10.1) | Medium | Medium |
| **P3 — Low** | Container/K8s (11.1) | Low | High |
| **P3 — Low** | Regulatory prep (12.3) | High | Very High |
| **P4 — Research** | Multi-omics (13.1) | Unknown | Very High |
| **P4 — Research** | Gene panel expansion (13.3) | High | Very High |
| **P4 — Research** | FHIR integration (13.5) | Medium | High |

---

## Summary

This roadmap contains **120+ specific improvements** across 13 categories. The highest-impact, lowest-effort wins are:

1. **Fix gnomAD population AFs** (switch API endpoint — immediate feature quality improvement)
2. **Run AlphaMissense ablation** (quantify leakage risk — methodological necessity)
3. **Expand non-BRCA2 training data** (LOVD, BRCAExchange, relaxed gnomAD thresholds)
4. **Fix BRCA1 class imbalance** (class-weighted loss, additional benign mining)
5. **Add EVE scores** (104th feature, improves BRCA1 +0.050 and RAD51C +0.024)

The fundamental challenge remains **data scarcity for non-BRCA2 genes** — no amount of model architecture improvement can substitute for more labeled training variants.
