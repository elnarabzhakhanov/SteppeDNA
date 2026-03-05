# SteppeDNA

**Pan-Gene Variant Pathogenicity Classifier for HR DNA Repair Genes**

[![CI](https://github.com/elnarabzhakhanov/SteppeDNA/actions/workflows/ci.yml/badge.svg)](https://github.com/elnarabzhakhanov/SteppeDNA/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.129-009688.svg)](https://fastapi.tiangolo.com)

SteppeDNA classifies missense variants in 5 Homologous Recombination DNA repair genes as **pathogenic** or **benign** using an ensemble of XGBoost and Multi-Layer Perceptron models with isotonic calibration.

## Key Metrics

| Metric | Value |
|--------|-------|
| ROC-AUC (overall) | **0.978** (per-gene: 0.641–0.983)* |
| Macro-Averaged AUC | **0.775** (equal-weight per-gene mean) |
| MCC | **0.881** |
| Balanced Accuracy | **94.1%** |
| 10-Fold CV | 0.9797 +/- 0.0031 |
| Training Variants | 19,223 |
| Features | 103 |

*Overall AUC is weighted by test set composition; BRCA2 comprises 52% of test variants. See per-gene breakdown below.

Outperforms REVEL (0.725), BayesDel (0.721), and CADD (0.539) on SteppeDNA's own held-out test set. Competitor tools were not trained on this distribution, giving SteppeDNA a methodological advantage. On independent benchmarks (ProteinGym DMS + ClinVar Expert Panel), SteppeDNA achieves AUC 0.719–0.793.

## Supported Genes

| Gene | Test AUC | Training Variants |
|------|----------|-------------------|
| BRCA2 | 0.983 | 14,200+ |
| RAD51D | 0.804 | 128 gnomAD-augmented |
| RAD51C | 0.743 | 86 gnomAD-augmented |
| BRCA1 | 0.706 | 97 gnomAD-augmented |
| PALB2 | 0.641 | 60 gnomAD-augmented |

## Features

- **Ensemble ML**: XGBoost (60%) + MLP (40%) with isotonic probability calibration
- **103 Engineered Features**: BLOSUM62, ESM-2 embeddings, AlphaMissense, MAVE, PhyloP, SpliceAI, AlphaFold 3D structure
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

## Project Structure

```
SteppeDNA/
+-- backend/
|   +-- main.py                 # FastAPI server + prediction endpoints
|   +-- feature_engineering.py  # 103-feature pipeline
|   +-- models/                 # Trained model artifacts
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
|   +-- train_universal_model.py     # Model training (v4)
|   +-- build_master_dataset.py      # Dataset builder with gnomAD
|   +-- sota_comparison.py           # SOTA comparison pipeline
|   +-- evaluate_benchmark.py        # Gold-standard evaluation
|   +-- mave_ablation.py             # MAVE leakage assessment
+-- data_pipelines/
|   +-- fetch_gnomad_proxy_benign.py # gnomAD v4 GraphQL fetcher
|   +-- fetch_dbnsfp_scores.py       # REVEL/CADD/BayesDel via myvariant.info
|   +-- build_gold_standard_benchmark.py
+-- notebooks/
|   +-- esm2_lora_finetuning.ipynb   # ESM-2 LoRA fine-tuning (GPU)
|   +-- esm2_650m_embeddings.ipynb   # ESM-2 650M upgrade (GPU)
|   +-- gnn_alphafold_structure.ipynb # GNN on AlphaFold 3D (GPU)
+-- tests/                           # Pytest test suite
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
Feature Engineering (103 features)
  - BLOSUM62, volume/hydro/charge diffs
  - ESM-2 protein language model (20 PCA components)
  - AlphaMissense, MAVE, PhyloP scores
  - AlphaFold 3D structure features
  - SpliceAI splice predictions
        |
        v
Ensemble Prediction
  - XGBoost (60% weight)
  - MLP Neural Network (40% weight)
        |
        v
Isotonic Calibration --> Probability [0.5% - 99.5%]
        |
        v
SHAP Explanation + ACMG Evidence + Confidence Interval
```

## Validation

- **SOTA Comparison**: Outperforms REVEL, BayesDel, and CADD on SteppeDNA's held-out test set (competitor tools were not trained on this distribution)
- **Gold-Standard Benchmark**: Evaluated on 2,234 variants from ProteinGym DMS + ClinVar Expert Panel
- **MAVE Leakage Assessment**: Ablation shows minimal dependence on MAVE features (dAUC = -0.0017)
- **Cross-Validation**: 10-fold CV AUC = 0.9797 +/- 0.0031

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

[MIT](LICENSE)
