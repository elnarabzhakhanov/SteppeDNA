# SteppeDNA

**Multi-Gene HR (BRCA1/2, PALB2, RAD51C/D) Variant Pathogenicity Classifier for HR DNA Repair Genes**

[![CI](https://github.com/elnarabzhakhanov/SteppeDNA/actions/workflows/ci.yml/badge.svg)](https://github.com/elnarabzhakhanov/SteppeDNA/actions)
[![License: MIT (code) / RUO (model)](https://img.shields.io/badge/License-MIT%20(code)%20%2F%20RUO%20(model)-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.129-009688.svg)](https://fastapi.tiangolo.com)

SteppeDNA classifies missense variants in 5 Homologous Recombination DNA repair genes as **pathogenic** or **benign** using an ensemble of XGBoost and Multi-Layer Perceptron models with isotonic calibration.

## Key Metrics

| Metric | Value |
|--------|-------|
| ROC-AUC (overall) | **0.985** (per-gene: 0.605–0.994)* |
| Macro-Averaged AUC | **0.791** (equal-weight per-gene mean) |
| MCC | **0.928** |
| Balanced Accuracy | **96.5%** |
| 10-Fold CV | 0.9858 +/- 0.0021 |
| Training Variants | 19,223 |
| Features | 120 |

*Overall AUC is weighted by test set composition; BRCA2 comprises 52% of test variants. See per-gene breakdown below.

Achieves higher test-set ROC-AUC than REVEL (0.725), BayesDel (0.721), and CADD (0.539) on SteppeDNA's own held-out test set. Competitor tools were not trained on this distribution, giving SteppeDNA a methodological advantage. Evaluated on SteppeDNA's own test set -- see [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for independent benchmark results. On independent benchmarks (ProteinGym DMS + ClinVar Expert Panel), SteppeDNA achieves AUC 0.750-0.801.

## Supported Genes

| Gene | Test AUC | Training Variants |
|------|----------|-------------------|
| BRCA2 | 0.994 | 10,085 |
| RAD51D | 0.824 | 410 |
| RAD51C | 0.785 | 675 |
| BRCA1 | 0.747 | 5,432 |
| PALB2 | 0.605 | 2,621 |

## Features

ML model handles missense variants only. Nonsense, frameshift, and splice variants use rule-based classification. Germline variants only.

- **Ensemble ML**: XGBoost + MLP with per-gene weights and isotonic probability calibration
- **120 Engineered Features**: BLOSUM62, ESM-2 embeddings, EVE, MAVE/DMS, PhyloP, SpliceAI, AlphaFold 3D structure, gnomAD frequencies
- **SHAP Explanations**: Every prediction shows which features drove the classification
- **ACMG Evidence Codes**: Computational ACMG criteria (PP3, PM1, BP4, etc.)
- **Live Database Lookups**: Real-time ClinVar and gnomAD queries
- **VCF Batch Upload**: Analyze entire VCF files at once with CSV export
- **PDF Reports**: Professional variant reports for documentation
- **Protein Domain Visualization**: Lollipop plot showing mutation position on gene architecture
- **Multilingual UI**: English, Kazakh, Russian with full i18n support
- **Dark Mode**: System-aware theme toggle
- **Responsive Design**: Works on desktop, tablet, and mobile

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/elnarabzhakhanov/SteppeDNA.git
cd SteppeDNA
pip install -r requirements.txt
```

### Run the Backend

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Open the Frontend

Open `frontend/index.html` in your browser. The frontend connects to `http://localhost:8000` by default.

### Docker

```bash
docker-compose up --build
```

This starts the FastAPI backend on port 8000 and an nginx frontend on port 3000.

## Known Limitations

SteppeDNA has 20 documented limitations — see [VALIDATION_REPORT.md](VALIDATION_REPORT.md#9-known-limitations) for full details. Key caveats:

- **Population bias:** Training data predominantly European ancestry (ClinVar submission bias)
- **BRCA2 dominance:** Headline AUC 0.985 is sample-weighted; macro-averaged across genes: 0.791
- **Temporal generalization:** Non-BRCA2 temporal AUCs are near-random (0.51–0.61)
- **AlphaMissense removed:** AM was removed in v5.4 due to indirect label leakage (ClinVar circularity)
- **Research use only:** Not validated for clinical diagnostic decisions

## Project Structure

```
SteppeDNA/
+-- backend/
|   +-- main.py                 # FastAPI server + prediction endpoints
|   +-- middleware.py            # Rate limiter, API key, security headers, CSP
|   +-- models.py               # Model loading and weight management
|   +-- features.py             # Gene cache, build_feature_vector, NICE_NAMES
|   +-- feature_engineering.py  # Training-time 120-feature pipeline
|   +-- explanations.py         # Data scarcity, contrastive, bootstrap CI, conformal
|   +-- acmg_rules.py           # ACMG rule engine (13 criteria, gene-specific thresholds)
|   +-- vcf.py                  # VCF parsing, batch prediction endpoint
|   +-- external_api.py         # ClinVar/gnomAD lookups, caching
|   +-- cohort.py               # UMAP, metrics, cohort, history, research priorities
|   +-- database.py             # SQLite persistence
|   +-- constants.py            # CODON_TABLE, COMPLEMENT
+-- frontend/
|   +-- index.html              # Main UI
|   +-- app.js                  # Application logic
|   +-- lang.js                 # i18n (EN/KK/RU)
|   +-- styles.css              # Responsive stylesheet
|   +-- api.js                  # API configuration
+-- data/
|   +-- master_training_dataset.csv  # 19,223 training variants
|   +-- model_metrics.json           # Test set performance
|   +-- sota_comparison.json         # REVEL/CADD/BayesDel comparison
|   +-- benchmark/                   # Gold-standard benchmark data
+-- scripts/
|   +-- vus_reclassification_multigene.py  # VUS reclassification analysis
|   +-- build_master_dataset.py      # Dataset builder with gnomAD
|   +-- sota_comparison.py           # SOTA comparison pipeline
|   +-- evaluate_benchmark.py        # Gold-standard evaluation
|   +-- mave_ablation.py             # MAVE leakage assessment
|   +-- delong_test.py               # DeLong statistical significance test
|   +-- fair_sota_benchmark.py       # Fair SOTA benchmark pipeline
|   +-- kz_population_summary.py     # Kazakhstan population analysis
+-- data_pipelines/
|   +-- fetch_gnomad_proxy_benign.py # gnomAD v4 GraphQL fetcher
|   +-- fetch_dbnsfp_scores.py       # REVEL/CADD/BayesDel via myvariant.info
|   +-- build_gold_standard_benchmark.py
+-- notebooks/
|   +-- esm2_lora_finetuning.ipynb   # ESM-2 LoRA fine-tuning (GPU)
|   +-- esm2_650m_embeddings.ipynb   # ESM-2 650M upgrade (GPU)
|   +-- gnn_alphafold_structure.ipynb # GNN on AlphaFold 3D (GPU)
+-- tests/                           # 200 tests (pytest)
+-- Dockerfile
+-- docker-compose.yml
+-- render.yaml                      # Render.com deployment
+-- requirements.txt
+-- VALIDATION_REPORT.md
+-- LICENSE
```

## Architecture

```
User Input (Gene + AA Change)
        |
        v
Feature Engineering (120 features)
  - BLOSUM62, volume/hydro/charge diffs
  - ESM-2 protein language model (20 PCA components)
  - EVE, MAVE/DMS, PhyloP scores
  - AlphaFold 3D structure features
  - SpliceAI splice predictions
        |
        v
Ensemble Prediction
  - XGBoost (per-gene weight)
  - MLP Neural Network (per-gene weight)
        |
        v
Isotonic Calibration --> Probability [0.5% - 99.5%]
        |
        v
SHAP Explanation + ACMG Evidence + Confidence Interval
```

## Validation

- **SOTA Comparison**: Achieves higher ROC-AUC than REVEL, BayesDel, and CADD on SteppeDNA's held-out test set (competitor tools were not trained on this distribution)
- **Gold-Standard Benchmark**: Evaluated on 2,234 variants from ProteinGym DMS + ClinVar Expert Panel
- **MAVE Leakage Assessment**: Ablation shows minimal dependence on MAVE features (dAUC = -0.0017)
- **Cross-Validation**: 10-fold CV AUC = 0.9858 +/- 0.0021

See [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for full details.

## API Documentation

When the backend is running, interactive API docs are available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Deployment

### Render.com

The included `render.yaml` configures automatic deployment:

1. Push to GitHub
2. Connect your repo on [render.com](https://render.com)
3. Click "New Blueprint" and select your repo
4. Render reads `render.yaml` and deploys both services

### Vercel (Frontend Only)

```bash
cd frontend
npx vercel --prod
```

Set `window.STEPPEDNA_API_BASE` to your Render backend URL.

## Research Use Disclaimer

SteppeDNA is a **research tool** and is **NOT** a clinical diagnostic. All predictions are computational approximations and should not be used as the sole basis for medical decisions. ACMG evidence codes are automated estimates, not expert classifications.

## License

Code: [MIT](LICENSE). Model artifacts and predictions: Research Use Only -- see [LICENSE-MODEL](LICENSE-MODEL).
