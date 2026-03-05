# Future Improvements and "Wants" List

## Localization & Regional Integration
- [x] ~~**Multi-language Support**~~ (COMPLETED): Full UI localization for English, Kazakh, and Russian with i18n key system across all UI elements, error messages, and tooltips.
- [x] ~~**Regional Legal/Clinical Certification**~~ (COMPLETED): Certification section added to frontend with Kazakhstan health ministry (pending), CE-IVD (pending), and Research Use Only (active) status cards. i18n translations for en/kk/ru.

## User Interface & Experience (UX)
- [x] ~~**Contextual Help / Tooltips**~~ (COMPLETED): Added subtle grey `(?)` hover icons next to all technical terms (ROC-AUC, MCC, Balanced Accuracy, SHAP, Ensemble, HR Genes). Hovering displays a clean, concise tooltip explaining the metric.
- [x] ~~**Responsive Design**~~ (COMPLETED): Three breakpoints (1024px tablet, 700px mobile, 480px small phone) with form stacking, hero stat wrapping, and touch-friendly sizing.
- [x] ~~**Accessibility**~~ (COMPLETED): Skip-to-content link, aria-live on results, aria-hidden on decorative SVGs, focus management after predictions.
- [x] ~~**XSS Hardening**~~ (COMPLETED): escapeHtml() on all backend data interpolated via innerHTML.
- [x] ~~**Dark Mode**~~ (COMPLETED): System-aware theme toggle with smooth crossfade transitions.
- [x] ~~**Offline Mode / Error Boundary**~~ (COMPLETED): Backend health check on page load with i18n-aware offline banner (yellow warning / green connected). Auto-retry every 30s.
- [x] ~~**PWA Support**~~ (COMPLETED): manifest.json + service worker (sw.js) for installable app with offline-first caching of static assets.
- [x] ~~**Interactive 3D Protein Viewer**~~ (COMPLETED): NGL.js integration for real-time 3D mutation visualization. Loads AlphaFold structures via UniProt ID, highlights mutation site with ball+stick representation, auto-rotation. BRCA2 handled with graceful size warning (3,418 AA).

## Model Interpretability
- [x] ~~**Expanded SHAP Explanations UI**~~ (COMPLETED): Added "Show All Features" toggle button that expands the SHAP chart from top 8 to all non-zero features. Backend returns full shap_all array alongside the top-8 summary.
- [x] ~~**PDF Report Export**~~ (COMPLETED): Professional variant report with prediction, confidence interval, SHAP features, ACMG evidence, and data sources.
- [x] ~~**Live Database Integration (One-Click Fetch)**~~ (COMPLETED): Live ClinVar and gnomAD fetch buttons in the results panel, querying real-time variant status.
- [x] ~~**Gene Panel Expansion (Homologous Recombination Pathway)**~~ (COMPLETED): Expanded from BRCA2 to BRCA1, BRCA2, PALB2, RAD51C, RAD51D.
- [x] ~~**Deep Learning Explorations (Multi-Task Networks)**~~ (COMPLETED - EXPERIMENTAL ONLY): Investigated multi-task neural network. Result: Negative Transfer proved XGBoost+MLP ensemble is superior.
- [x] ~~**Batch VCF Export & Reporting**~~ (COMPLETED): CSV download button on VCF batch results, exporting all predictions with gene, HGVS, AA change, prediction, and probability.
- [x] ~~**Ensemble Confidence Calibration**~~ (COMPLETED): XGBoost (60%) + MLP (40%) with isotonic calibration on held-out data. Beta-distribution confidence intervals.
- [x] ~~**Variant Comparison Panel**~~ (COMPLETED): Side-by-side comparison of two variants with SHAP overlay.
- [x] ~~**Analysis History**~~ (COMPLETED): localStorage-based history (max 50) with language-aware timestamps and quick re-analysis.

## ML Architecture & Validation
- [x] ~~**Fix MAVE Data Leakage**~~ (COMPLETED): Added `use_mave=False` flag to `engineer_features()`. Ablation test shows minimal impact (dAUC=-0.0017). Only BRCA2 affected (no MAVE data for other genes).
- [x] ~~**Real SOTA Comparison**~~ (COMPLETED): Independent comparison against REVEL (0.725), BayesDel (0.721), CADD (0.539) via myvariant.info API. SteppeDNA (0.978) outperforms all.
- [x] ~~**Gold-Standard Benchmark**~~ (COMPLETED): 2,234 variants from ProteinGym DMS (Findlay 2018) + ClinVar Expert Panel. DMS Spearman r=-0.319 (p=1.45e-10), Expert Panel AUC=0.793.
- [x] ~~**gnomAD Proxy-Benign Augmentation**~~ (COMPLETED): 485 gnomAD v4 common variants added as proxy-benign. Dramatically improved non-BRCA2 performance (+0.095 to +0.234 AUC).
- [x] ~~**Indel & Frameshift Pipeline**~~ (COMPLETED): Automatic frameshift detection with 99.99% pathogenic flag.
- [x] ~~**Full-Length 3D Modeling via ESMFold**~~ (COMPLETED): Dual-scaffold AlphaFold+ESMFold architecture.
- [x] ~~**Ascertainment Bias Mitigation**~~ (COMPLETED): gnomAD popmax + granular ancestral frequencies.

## Backend & Infrastructure
- [x] ~~**API Rate Limiting**~~ (COMPLETED): Configurable per-IP rate limiting (default 60 req/min) with sliding window.
- [x] ~~**Prediction Caching**~~ (COMPLETED): LRU cache with TTL on /predict endpoint using composite cache key.
- [x] ~~**Security Headers**~~ (COMPLETED): X-Content-Type-Options, X-Frame-Options, Referrer-Policy, Permissions-Policy on all responses.
- [x] ~~**Structured Logging**~~ (COMPLETED): JSON logging mode (LOG_FORMAT=json) for production log aggregation.
- [x] ~~**API Key Authentication**~~ (COMPLETED): Optional X-API-Key header, mandatory in production mode.
- [x] ~~**Request ID Tracing**~~ (COMPLETED): X-Request-ID header on every response for debugging.
- [x] ~~**UMAP Precompute**~~ (COMPLETED): Script to generate variant landscape coordinates (scripts/precompute_umap.py).
- [x] ~~**Database Persistence**~~ (COMPLETED): SQLite with WAL mode for server-side analysis history (`data/steppedna.db`). Tracks all predictions and VCF uploads. `/history` and `/stats` API endpoints.
- [x] ~~**Monitoring Dashboard**~~ (COMPLETED): `/metrics` endpoint with prediction counts, VCF uploads, uptime, memory/CPU usage (via psutil). Error rate and latency tracking.

## DevOps & Deployment
- [x] ~~**Pytest Test Suite**~~ (COMPLETED): 14 tests covering health check, prediction endpoints, input validation, caching, and probability clipping.
- [x] ~~**CI/CD Pipeline**~~ (COMPLETED): GitHub Actions workflow with flake8 lint, pytest, and import verification.
- [x] ~~**Docker + Docker Compose**~~ (COMPLETED): Dockerfile with health checks + docker-compose.yml with nginx frontend + API backend.
- [x] ~~**Nginx Production Config**~~ (COMPLETED): Security headers, gzip compression, SPA fallback, API proxy routing.
- [x] ~~**Vercel Configuration**~~ (COMPLETED): vercel.json with API rewrites to Render backend.
- [x] ~~**Render Configuration**~~ (COMPLETED): render.yaml with auto-deploy for both frontend and backend.
- [x] ~~**README.md**~~ (COMPLETED): Comprehensive README with badges, metrics, architecture, quick start, and API docs.
- [x] ~~**CHANGELOG.md**~~ (COMPLETED): Version history from v1.0.0 to v4.1.0.
- [x] ~~**LICENSE**~~ (COMPLETED): MIT License.
- [x] ~~**.gitignore**~~ (COMPLETED): Python, IDE, OS, Docker, Jupyter exclusions.
- [x] ~~**.env.example**~~ (COMPLETED): Environment variable template.
- [ ] **Custom Domain**: Purchase and configure domain (optional, ~$10-15/year).
- [ ] **Live Deployment**: Push to GitHub, deploy frontend to Vercel, backend to Render. All configs ready (vercel.json, render.yaml, docker-compose.yml).

## GPU-Dependent (Colab Notebooks Ready)
- [x] ~~**ESM-2 LoRA Fine-Tuning**~~ (COMPLETED): `notebooks/esm2_lora_finetuning.ipynb` — Fine-tuned ESM-2 650M on HR pathway sequences with LoRA adapters. Embeddings saved for integration.
- [x] ~~**ESM-2 650M Embeddings**~~ (COMPLETED): `notebooks/esm2_650m_embeddings.ipynb` — Upgraded from 8M to 650M model. Per-gene PCA embeddings exported.
- [x] ~~**GNN on AlphaFold 3D Structure**~~ (COMPLETED): `notebooks/gnn_alphafold_structure.ipynb` — Graph Neural Network over protein contact graphs. 32-dim structural features for 4 genes (BRCA2 excluded due to AlphaFold size limit).
- [x] ~~**GPU Embedding Integration Script**~~ (COMPLETED): `scripts/integrate_gpu_embeddings.py` — Merges LoRA (20 PCA) + 650M (20 PCA) + GNN (32-dim) = 72 new features into training dataset.

## Nice-to-Have (Low Priority)
- [x] ~~**Interactive 3D Protein Viewer**~~ (COMPLETED): See UX section above.
- [x] ~~**Local Patient Cohort Tracking**~~ (COMPLETED): `POST /cohort/submit` for anonymized variant observations + `GET /cohort/stats` for aggregate statistics. CSV-based append-only storage.
- [x] ~~**Regional Legal/Clinical Certification**~~ (COMPLETED): See Localization section above.
