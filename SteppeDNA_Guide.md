# SteppeDNA

A tool that predicts whether BRCA2 gene mutations are dangerous or harmless, so patients don't have to wait months for expensive lab tests to find out.

## What This Project Does

Genetic testing for breast cancer risk is getting cheaper, but interpreting the results isn't. When someone's BRCA2 test comes back with a mutation that hasn't been seen before, doctors can't tell them whether it's dangerous — these are called Variants of Uncertain Significance (VUS). The only way to classify them right now is through wet-lab functional assays that cost thousands of dollars and take months.

SteppeDNA tries to solve this computationally. You give it a BRCA2 variant (manually or via a VCF file) and it predicts the probability of it being pathogenic, using 99 features I put together from 5 public biological databases.

## How I Built It

This project went through several major rewrites as I learned more:

1. **Started simple** — a basic neural network with just BLOSUM62 substitution scores
2. **Realized 1D features aren't enough** — added 3D structural distances from the actual PDB crystal structure (1MIU), evolutionary conservation from PhyloP, population frequencies from gnomAD, and functional assay data from MAVE
3. **Explored advanced Deep Learning architectures** — built a custom Dual-Head Multi-Task Neural Network in PyTorch/Keras designed to predict both clinical pathogenicity (classification) *and* the exact physical wet-lab score (regression) simultaneously. I hoped this would constrain the latent space to learn the fundamental biology.
4. **Discovered Negative Transfer** — the 100-feature dual-head network achieved a flawless 0.9997 internal ROC-AUC but collapsed to 0.7620 when tested on truly independent MAVE holdout data. The exact continuous lab score prediction objective was too noisy and dragged down the clinical accuracy.
5. **Settled on a Heterogeneous Stacking Ensemble** — returning to simpler roots, I fused XGBoost (for non-linear decision trees), an MLP (for smooth probability curves), and an SVM (for margin maximization). This ensemble blew past the Deep Learning architecture, achieving a highly stable **0.8445 independent ROC-AUC**. Wait times for predictions dropped from seconds to milliseconds.
6. **Prioritized safety over accuracy** — instead of optimizing for balanced accuracy (which treats false positives and false negatives equally), I shifted the decision threshold of the final ensemble to guarantee high sensitivity for pathogenic variants, because missing a pathogenic variant is much worse than flagging a benign one

## How to Run It

### Option A: Docker
```bash
docker-compose up
```
Then open `frontend/index.html` in your browser. The backend will be running at `http://localhost:8000`.

### Option B: Manual Setup
```bash
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
Then open `frontend/index.html` in your browser.

## Quick Test

Use the test VCF file included in the `tests/` folder — drag and drop `test_variants.vcf` into the upload zone on the frontend to see batch analysis in action.


## Key Results

- **95.1% Pathogenic Sensitivity** on the ClinVar holdout set
- **0.989 ROC-AUC** (calibrated)  
- **0.912 ROC-AUC** on completely independent MAVE wet-lab validation data (MAVE-blind model)
- Every prediction comes with SHAP feature attribution and ACMG evidence codes

## Project Structure

```
backend/       — FastAPI server, feature engineering, ACMG rule engine
frontend/      — Single-page web interface
data/          — Trained model weights, pre-computed feature dictionaries
data_pipelines/— Scripts that fetch data from external databases
scripts/       — Training, benchmarking, and validation scripts
tests/         — API tests and sample VCF file
utils/         — Helper scripts for data preparation
```

## Disclaimer

This is a research project. It is not a medical device and should not be used for clinical decision-making.
