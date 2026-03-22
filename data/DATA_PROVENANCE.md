# SteppeDNA Data Provenance

Records the exact versions and snapshot dates of all external data sources used in model training and inference.

## Training Data Sources

| Source | Version / Snapshot | Size | Access Date | Notes |
|--------|-------------------|------|-------------|-------|
| ClinVar | February 2026 snapshot | 18,738 variants | Feb 2026 | P/LP and B/LB missense variants for BRCA1, BRCA2, PALB2, RAD51C, RAD51D |
| gnomAD | v4 | 485 proxy-benign variants | Feb 2026 | AC >= 2, deduplicated against ClinVar |
| **Total dataset** | | **19,223 variants** | | Combined ClinVar + gnomAD across 5 HR genes |

## Feature Data Sources

| Source | Version | Coverage | Notes |
|--------|---------|----------|-------|
| ESM-2 | esm2_t6_8M_UR50D (8M params) | 17,528 embeddings | Per-gene, +/-50 residue context windows, 20 PCA components |
| PhyloP | 100-way vertebrate conservation | Per-cDNA position | Via UCSC Genome Browser |
| EVE | Frazer et al., Nature 2021 | BRCA1/PALB2/RAD51C/RAD51D (65-100% coverage) | Evolutionary coupling scores; not available for BRCA2 |
| MAVE | Findlay et al. 2018 (BRCA1 SGE) + Hu et al. 2024 (BRCA2 HDR) | BRCA1 + BRCA2 | Potential leakage, see VALIDATION_REPORT.md Section 14 |
| SpliceAI | Jaganathan et al. 2019 | Per-variant scores | Via myvariant.info/dbNSFP |
| AlphaFold | v6 | Per-residue structural features (all 5 genes) | RSA, B-factor, secondary structure, domain distances. URL format: AF-{uniprot}-F1-model_v6.pdb |
| gnomAD (allele freq) | v4 | 3,508 variants with AF>0 | Real population-stratified AFs via myvariant.info (v5.4 fix; was all zeros in v5.3) |
| Domain proximity | Gene-specific | All 5 genes | dist_nearest_domain, functional_zone_score, n_domains_hit, etc. (new in v5.4) |
| ~~AlphaMissense~~ | ~~Cheng et al., Nature 2023~~ | ~~Removed in v5.4~~ | Removed due to indirect ClinVar label leakage; ablation showed +0.02 AUC without AM |

## SOTA Comparison Sources

| Predictor | Source | Coverage |
|-----------|--------|----------|
| REVEL | Ioannidis et al. 2016, via dbNSFP | 72.4% of test set |
| BayesDel | Feng 2017, via dbNSFP | 72.9% of test set |
| CADD | Rentzsch et al. 2019, via dbNSFP | 72.4% of test set |

## Gold-Standard Benchmark Sources

| Source | Size | Citation |
|--------|------|----------|
| ProteinGym DMS | 1,837 BRCA1 variants | Findlay et al., Nature 2018 |
| ClinVar Expert Panel | 397 variants (BRCA1/2, PALB2) | ClinGen VCEP / ENIGMA |

## Gene Configuration Sources

| Gene | Transcript | UniProt | Chromosome |
|------|-----------|---------|------------|
| BRCA1 | ENST00000357654.9 | P38398 | chr17 |
| BRCA2 | ENST00000380152.8 | P51587 | chr13 |
| PALB2 | ENST00000261584.9 | Q86YC2 | chr16 |
| RAD51C | ENST00000337432.9 | O43502 | chr17 |
| RAD51D | ENST00000345365.10 | O75771 | chr17 |

## Reproducibility

- Random seed: `RANDOM_STATE=42` throughout all scripts
- Train/Cal/Test split: 60/20/20 with gene x label stratification
- All data fetching scripts in `data_pipelines/`
- All training scripts in `scripts/`
