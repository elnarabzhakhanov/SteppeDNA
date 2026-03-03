# Changelog

All notable changes to SteppeDNA are documented in this file.

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
- Overall: ROC-AUC 0.978, MCC 0.881, Balanced Accuracy 94.1%
- 10-fold CV: 0.9797 +/- 0.0031
- BRCA2: AUC 0.983
- RAD51D: AUC 0.804 (up from ~0.5 pre-augmentation)
- RAD51C: AUC 0.743
- BRCA1: AUC 0.706
- PALB2: AUC 0.641

## [3.0.0] - 2026-01-15

### Added
- Universal pan-gene model (single model for all 5 genes)
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
