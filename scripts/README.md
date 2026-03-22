# Scripts Directory

## Active Scripts (Current Pipeline)

| Script | Purpose |
|--------|---------|
| `build_master_dataset.py` | Build v4 dataset with gnomAD augmentation (19,223 variants) |
| `train_universal_model.py` | Production model training (XGBoost + MLP ensemble) |
| `sota_comparison.py` | SOTA comparison vs REVEL/BayesDel/CADD |
| `evaluate_benchmark.py` | Gold-standard benchmark evaluation |
| `mave_ablation.py` | MAVE feature ablation analysis |
| `cross_validate.py` | 10-fold cross-validation |
| `integrate_gpu_embeddings.py` | Merge GPU embeddings into training data |
| `train_gpu_model.py` | Retrain ensemble with GPU-augmented features (178 feat.) |
| `temporal_validation.py` | Date-stratified temporal validation (pre/post-2024) |
| `vus_reclassification_multigene.py` | Multi-gene VUS reclassification (all 5 HR genes) |
| `run_metrics.py` | Compute model evaluation metrics |
| `delong_test.py` | DeLong test for statistical significance vs REVEL/CADD/BayesDel |
| `fair_sota_benchmark.py` | Fair SOTA benchmark with deduplication |
| `kz_population_summary.py` | Kazakhstan population frequency summary |

## Legacy Scripts (Historical Reference)

Training experiments from earlier versions (v1-v3):
- `baseline_model.py`, `train_xgboost.py`, `tune_xgboost.py`, `tune_xgboost_blind.py`
- `train_ensemble_baseline.py`, `train_ensemble_blind.py`
- `train_mave_blind_model.py`, `train_multitask_blind.py`

External validation (superseded by evaluate_benchmark.py):
- `evaluate.py`, `externa_validation.py`, `external_validation_ensemble.py`, `external_validation_multitask.py`
- `eval_mave_holdout.py`

PDF/Report generation (various versions, may have outdated metrics):
- `generate_bw_pdf.py`, `generate_bw_pdf_ru.py`, `generate_simple_pdf.py`
- `generate_simplified_bw_pdf.py`, `generate_explanation_html.py`, `generate_explanation_pdf.py`
- `generate_comprehensive_guide.py`, `generate_comprehensive_guide_ru.py`
- `generate_final_unified_pdf.py`, `generate_ultimate_pdf.py`
- `generate_v3_deep_dive.py`, `generate_v3_purple_pdf.py`
- `generate_figures.py`

Utilities:
- `ablation_study.py`, `ablation_study_xgb.py` — Feature ablation experiments
- `check_leakage.py`, `check_leakage_brca2.py` — Data leakage verification
- `check_mave_api.py`, `check_mutations.py`, `check_vcf.py`, `check_known_variants.py`
- `precompute_umap.py` — UMAP coordinate precomputation
- `vus_reclassification.py` — VUS analysis
- `run_benchmark.py` — Benchmark runner
- `md_to_pdf.py` — Markdown to PDF converter
