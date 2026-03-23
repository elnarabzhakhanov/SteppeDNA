# Changelog

All notable changes to SteppeDNA are documented in this file.

## [5.4.0] - 2026-03-14

### Added
- AlphaFold structural features for ALL 5 genes (was BRCA2-only): RSA, B-factor, secondary structure, domain distances
- Real gnomAD allele frequencies via myvariant.info (3,508 variants with AF>0, was all zeros)
- EVE evolutionary coupling scores as training features (65-100% coverage per gene)
- BRCA1 Findlay SGE functional scores as DMS features
- Gene-specific domain proximity features (dist_nearest_domain, functional_zone_score, etc.)
- Population-aware ACMG evidence codes (BA1/BS1/PM2 adjusted per population)
- Kazakh founder mutation database (7 variants) with detection endpoint
- VUS disparity analysis script for competition narrative
- AlphaMissense ablation study (data/am_ablation_results.json)
- Per-gene isotonic calibrators fitted on calibration set (not test set)

### Fixed
- Calibrator leakage: calibrator now fitted on calibration set only (was test set in v5.3)
- gnomAD allele frequencies: were all zeros, now real values from myvariant.info
- func_zone_x_phylop feature: was always 0 at inference time, now computed correctly
- Structural secondary structure field: supports both old string and new int format
- gnomAD popmax_af key alias handling
- Bootstrap CI graceful fallback for feature count mismatch
- Founder mutations JSON loading (structure and field name mismatches)
- Gene ensemble weight key format (supports both "xgb"/"mlp" and "xgb_weight"/"mlp_weight")

### Removed
- AlphaMissense features: removed due to indirect label leakage (ablation: +0.02 AUC for BRCA1/PALB2/RAD51C without AM)

### Changed
- Feature count: 120 features (was 103 in v5.3)
- Per-gene AUC: BRCA2 0.994, RAD51D 0.824, RAD51C 0.785, BRCA1 0.747, PALB2 0.605
- Macro-averaged AUC: 0.791 (was 0.775)
- Sample-weighted AUC: 0.985 (was 0.978)
- Per-gene ensemble weights: BRCA1 (MLP-only), BRCA2 (60/40), PALB2 (MLP-only), RAD51C (80/20 XGB), RAD51D (55/45)

## [5.3.0] - 2026-03-06

### Fixed
- PVS1 evidence code overwrite bug: canonical splice no longer overwrites nonsense/frameshift PVS1 when both flags are set
- BRCA2 hardcoded fallback in `_safe_critical_domain()` replaced with gene-config-aware lookup
- Gene config redundant disk read in `build_feature_vector()` now uses cached config
- VCF path now uses gene-adaptive ensemble weights and per-gene calibrators

### Added
- Per-gene SOTA comparison table (Section 5c in VALIDATION_REPORT.md): SteppeDNA beats REVEL/BayesDel/CADD on 4/5 genes
- Macro-averaged AUC footnote: hero AUC 0.978 now accompanied by macro-averaged 0.775
- ACMG rule engine expansion: PM2 (absent from population databases), PS1 (same amino acid change as known pathogenic), PVS1 last-exon attenuation
- 5-tier classification system (Pathogenic, Likely Pathogenic, VUS, Likely Benign, Benign)
- RUO inline disclaimers throughout frontend and backend
- UMAP honest label (shows data source, not clinical truth)
- Test overhaul: 198 tests across 4 test modules (combinatorial, clinical correctness, property-based)
- AlphaMissense indirect label leakage acknowledged in VALIDATION_REPORT.md (limitation #17)
- Gene-adaptive ensemble weights loaded from `data/gene_ensemble_weights.json`
- VCF parsing hardening with improved error handling and edge case coverage
- Dead code removal: is_transition/is_transversion computation and unused NICE_NAMES entries

### Changed
- Branding changed from "Pan-Gene" to "Multi-Gene HR"

## [5.2.0] - 2026-03-04

### Added
- Honest per-gene metrics displayed in frontend (per-gene AUC table instead of overall-only)
- SOTA comparison caveat: explicit disclaimer that competitor tools were not trained on this distribution

### Fixed
- Frontend hero stats now include macro-averaged AUC alongside weighted AUC

## [5.1.0] - 2026-03-03

### Changed
- Backend refactored into modules: `middleware.py`, `models.py`, `explanations.py`, `features.py`, `vcf.py`, `external_api.py`, `cohort.py`
- Version consistency fixes across backend, frontend, and metadata files

### Fixed
- Version string inconsistencies between `main.py`, `model_metadata.json`, and frontend

## [5.0.0] - 2026-03-01

### Added
- Interactive 3D protein viewer (NGL.js) with AlphaFold structure loading and mutation highlighting
- SQLite database persistence (WAL mode) for server-side analysis history and VCF tracking
- `/metrics` monitoring endpoint with prediction counts, uptime, memory/CPU usage (psutil)
- `/cohort/submit` and `/cohort/stats` endpoints for anonymized patient cohort tracking
- `/history` and `/stats` endpoints for server-side analysis queries
- Kazakhstan certification section in UI (KZ health ministry pending, CE-IVD pending, RUO active)
- GPU embedding integration script (`scripts/integrate_gpu_embeddings.py`) for LoRA + 650M + GNN features
- ESM-2 LoRA fine-tuning notebook completed (`notebooks/esm2_lora_finetuning.ipynb`)
- ESM-2 650M embeddings notebook completed (`notebooks/esm2_650m_embeddings.ipynb`)
- GNN AlphaFold structure notebook completed (`notebooks/gnn_alphafold_structure.ipynb`)
- PWA support: manifest.json + service worker for installable offline-first app
- Security headers middleware (X-Content-Type-Options, X-Frame-Options, etc.)
- Nginx production config with gzip, caching, SPA fallback, API proxy
- Vercel + Render deployment configurations
- psutil dependency for system metrics

### Infrastructure
- `backend/database.py` — SQLite module with auto-init, WAL journal, busy timeout
- `frontend/nginx.conf` — Added /metrics, /cohort, /history, /stats proxy routes
- All DB operations non-critical (try/except, never breaks prediction flow)
- Thread-safe metrics counters via existing rate limiter lock

## [4.1.0] - 2026-02-28

### Added
- gnomAD proxy-benign augmentation: +485 variants (AC >= 2) from gnomAD v4 GraphQL API
- BRCA1 +97, BRCA2 +114, PALB2 +60, RAD51C +86, RAD51D +128 benign variants
- SOTA comparison pipeline using real REVEL/BayesDel/CADD scores via myvariant.info
- Gold-standard benchmark on 2,234 variants (ProteinGym DMS + ClinVar Expert Panel)
- MAVE data leakage ablation assessment (dAUC = -0.0017)
- ESM-2 protein language model embeddings (20 PCA components per gene)
- SHAP expanded view ("Show All Features" toggle)
- ACMG computational evidence codes (PP3, PM1, BP4, etc.)
- Live ClinVar and gnomAD database lookups
- VCF batch upload with CSV export
- Protein domain lollipop plot visualization
- Variant comparison panel (side-by-side)
- Prediction caching (LRU with TTL)
- Backend health check with offline banner
- Analysis history (localStorage, max 50)
- Multilingual UI: English, Kazakh, Russian (full i18n)
- Dark mode with system-aware theme toggle
- Contextual help tooltips on hero stats and SHAP
- PDF report export for individual variants
- Model Card & Limitations section
- Responsive design (3 breakpoints: 1024/700/480px)
- XSS hardening (escapeHtml on all backend data)
- Accessibility: skip-link, aria-live, aria-hidden, focus management
- Rate limiting (configurable per-IP)
- API key authentication (optional, required in production)
- Request ID header (X-Request-ID) on all responses
- Pytest test suite (14 tests)
- CI/CD via GitHub Actions (lint + test + import verification)
- Docker + docker-compose deployment
- UMAP precompute script for variant landscape

### Performance
- Overall: ROC-AUC 0.978, MCC 0.928, Balanced Accuracy 96.5%
- 10-fold CV: 0.9858 +/- 0.0021
- BRCA2: AUC 0.983
- RAD51D: AUC 0.804 (up from ~0.5 pre-augmentation)
- RAD51C: AUC 0.743
- BRCA1: AUC 0.706
- PALB2: AUC 0.641

## [3.0.0] - 2026-01-15

### Added
- Universal pan-gene model (single model for all 5 genes) (now Multi-Gene HR)
- XGBoost + MLP ensemble with isotonic calibration
- 103 engineered features from 5 biological databases
- AlphaFold 3D structure features
- SpliceAI splice prediction features
- AlphaMissense pathogenicity scores
- PhyloP conservation scores
- MAVE functional assay data

## [2.0.0] - 2025-12-01

### Added
- Multi-gene support (BRCA1, BRCA2, PALB2, RAD51C, RAD51D)
- FastAPI backend with Swagger documentation
- Frontend UI with form-based variant input

## [1.0.0] - 2025-11-01

### Added
- Initial BRCA2-only classifier
- XGBoost model with BLOSUM62 features
- Basic web interface
