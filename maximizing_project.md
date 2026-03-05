# SteppeDNA v5.2 — Maximizing Project Potential

> Master roadmap for fixes, improvements, novelty additions, and competition preparation.
> Every item from the full review session is included. Nothing is skipped.
> Last updated: 2026-03-03

---

## TABLE OF CONTENTS

1. [Current Flaws & Methodological Issues](#1-current-flaws--methodological-issues)
2. [Backend Fixes](#2-backend-fixes)
3. [Frontend Fixes](#3-frontend-fixes)
4. [Documentation & Framing Fixes](#4-documentation--framing-fixes)
5. [Novelty Additions (Algorithmic)](#5-novelty-additions-algorithmic)
6. [Data & Training Improvements](#6-data--training-improvements)
7. [Clinical Validation & Wet Lab](#7-clinical-validation--wet-lab)
8. [External Validation & Letters](#8-external-validation--letters)
9. [Competition Preparation (Infomatrix Asia — 3 weeks)](#9-competition-preparation-infomatrix-asia--3-weeks)
10. [Competition Preparation (Infomatrix Worlds)](#10-competition-preparation-infomatrix-worlds)
11. [Competition Preparation (ISEF 2027)](#11-competition-preparation-isef-2027)
12. [Engineering & Infrastructure](#12-engineering--infrastructure)
13. [Publication Pathway](#13-publication-pathway)
14. [Priority Timeline](#14-priority-timeline)

---

## 1. CURRENT FLAWS & METHODOLOGICAL ISSUES

These are problems that currently exist and could be challenged by judges or reviewers.

### 1.1 BRCA2 Dominance Distorts Headline Metric (CRITICAL)

**Problem:** The 0.978 overall AUC is driven by BRCA2 (52% of test data, AUC 0.983). Equal-weight per-gene average AUC is ~0.775. The headline number is technically correct but misleading for a "pan-gene" classifier.

**Impact:** A judge who asks "what's the per-gene breakdown?" will see the gap immediately. This looks like you're hiding weak performance behind a strong gene.

**Fix:**
- Report BOTH: overall weighted AUC (0.978) AND equal-weight per-gene mean AUC (~0.775)
- Add a prominent footnote: "Overall AUC weighted by test set composition; BRCA2 comprises 52% of test variants"
- In the README hero metrics, add per-gene AUC range: "Per-gene: 0.641-0.983"
- In presentations, lead with the per-gene breakdown chart, not the headline number
- Consider adding a macro-averaged AUC metric to `/predict` response and model card

**Where to change:**
- `README.md` — hero metrics section
- `MODEL_CARD.md` — evaluation metrics
- `frontend/index.html` — hero stats
- Poster/presentation materials

---

### 1.2 Temporal Generalization Failure for Non-BRCA2 (CRITICAL)

**Problem:** Temporal validation (pre-2024 train -> 2024+ test) shows near-random performance for 4/5 genes:
- BRCA1: 0.527 (near random)
- PALB2: 0.513 (near random)
- RAD51C: 0.561
- RAD51D: 0.608
- BRCA2: 0.983 (excellent)

**Impact:** This means the model does NOT generalize to newly classified variants for non-BRCA2 genes. It has essentially memorized ClinVar's existing classification patterns rather than learning biology.

**Fix:**
- Must be prominently disclosed in all competition materials
- Frame honestly: "Temporal stability validated for BRCA2; non-BRCA2 genes require more training data"
- Prepare a rehearsed answer for when judges ask about this
- The meta-learning novelty addition (see 5.2) directly addresses this

**Where to change:**
- `VALIDATION_REPORT.md` — already disclosed but ensure prominence
- Poster/presentation — must address proactively, not defensively
- `MODEL_CARD.md` — limitations section

---

### 1.3 SOTA Comparison Methodology (IMPORTANT)

**Problem:** Claiming SteppeDNA (0.978) > REVEL (0.725) > BayesDel (0.721) > CADD (0.539) is unfair. SteppeDNA was trained on data from the same distribution (ClinVar). REVEL, BayesDel, and CADD were not — they're general-purpose tools evaluated out-of-distribution on YOUR test set.

**Impact:** A sophisticated judge will recognize this as an apples-to-oranges comparison. It undermines credibility if framed as "we beat everything."

**Fix:**
- Add prominent caveat: "Evaluated on SteppeDNA's own held-out test set. Competitor tools were not trained on this distribution."
- Emphasize the independent benchmark results instead: "On independent gold-standard benchmark (ProteinGym DMS + ClinVar Expert Panel), SteppeDNA achieves AUC 0.719-0.793"
- Reframe: instead of "we're better than REVEL," say "on our specialized domain (5 HR genes), our domain-specific model outperforms general-purpose tools as expected; on independent benchmarks we remain competitive"
- The caveat already exists in VALIDATION_REPORT.md but needs to be front-and-center in presentations

**Where to change:**
- `README.md` — SOTA comparison section
- `frontend/index.html` — if SOTA comparison is displayed
- Poster/presentation materials — biggest risk of judge pushback

---

### 1.4 Independent Benchmark Performance Gap (IMPORTANT)

**Problem:** Own test set AUC is 0.978. Independent benchmark AUC is 0.719-0.793. That's a ~20% gap. While some drop is expected (different distributions), the magnitude suggests overfitting to ClinVar patterns.

**Impact:** Judges who see both numbers will question generalization.

**Fix:**
- Frame as: "The gap between internal test (0.978) and external benchmark (0.719-0.793) reflects distributional shift and the challenge of generalizing across data sources — a known issue in clinical genomics ML"
- The DMS correlation (Spearman r=-0.319, p=1.45e-10) shows the model captures biological signal despite the AUC gap
- The Expert Panel BRCA2 AUC of 0.918 shows the model generalizes well for BRCA2
- Prepare a slide specifically addressing this gap

---

### 1.5 Circular Validation Concern (MODERATE)

**Problem:** Training on ClinVar labels and testing on ClinVar labels (even with proper splits) means the model may learn ClinVar's classification biases. ClinVar labels are assigned by human curators using specific criteria — the model may be predicting "what a ClinVar curator would decide" rather than true biological pathogenicity.

**Impact:** Undermines the claim of predicting "pathogenicity" — it may be predicting "ClinVar classification."

**Fix:**
- Acknowledge explicitly: "Our model predicts concordance with ClinVar expert classifications, which is a proxy for pathogenicity"
- The ProteinGym DMS benchmark (experimental functional data) provides independent validation of biological signal
- Frame the temporal validation as a test of this concern (and be honest about the results)

---

### 1.6 "Pan-Gene" Label is Overstated (MODERATE)

**Problem:** Calling it a "Pan-Gene Variant Pathogenicity Classifier for 5 HR genes" when 3/5 genes have AUC < 0.75 is generous framing. It's really a BRCA2 classifier with exploratory coverage of 4 additional genes.

**Fix:**
- Reframe in presentations: "BRCA2-optimized with expanding coverage to 4 additional HR genes"
- Or: "Pan-gene architecture with gene-specific reliability tiers"
- Keep the gene reliability tiers prominent — they exist precisely for this reason
- Don't claim "pan-gene accuracy of 0.978" — claim "architecture supports 5 genes with per-gene performance transparency"

---

### 1.7 gnomAD Proxy-Benign Label Noise (MINOR)

**Problem:** Using 485 common gnomAD variants (AC >= 2) as benign proxies introduces potential label noise. Some common variants may be low-penetrance pathogenic or have population-specific effects.

**Fix:**
- Run a sensitivity analysis: retrain without the 485 gnomAD variants and report AUC delta
- If delta is small (< 0.01), it confirms gnomAD augmentation helps without introducing noise
- If delta is large, investigate which gnomAD variants are borderline
- Add to VALIDATION_REPORT.md

**Implementation:**
```python
# In a new script: scripts/gnomad_sensitivity.py
# 1. Load master_training_dataset.csv
# 2. Remove rows where source == 'gnomad_proxy_benign'
# 3. Retrain XGBoost + MLP on remaining 18,738 variants
# 4. Evaluate on same test set
# 5. Compare per-gene AUC
```

---

### 1.8 Population Bias (MODERATE)

**Problem:** Training data is predominantly European-ancestry. ClinVar has geographic bias toward US/European submissions. No population-stratified allele frequencies in training.

**Impact:** Especially problematic given the Kazakh cultural angle — Kazakh founder mutations in BRCA1/2 are NOT represented in training data.

**Fix:**
- Add explicit disclaimer: "Training data reflects predominantly European-ancestry variant ascertainment"
- Source Kazakh-specific variant data (see Section 7)
- If model fails on Kazakh variants, that's also valuable — confirms population bias and motivates further work
- Use gnomAD v4 ancestry-specific allele frequencies as additional features

**Where to change:**
- `MODEL_CARD.md` — ethical considerations (partially done)
- Poster/presentation materials
- `VALIDATION_REPORT.md` — population equity section (partially done)

---

### 1.9 Feature Concentration (MINOR)

**Problem:** ~15 of 103 features drive most predictions. The other 88 contribute minimally. This raises the question: is the feature engineering over-engineered?

**Fix:**
- Document which 15 features are most important (from SHAP global importances)
- Frame as: "103 features provide comprehensive coverage; top 15 by SHAP importance account for X% of prediction variance"
- The remaining features may help on edge cases even if globally unimportant
- Consider showing a feature ablation: performance with top-15 vs all-103

---

### 1.10 Kazakh Medical Terminology (MINOR)

**Problem:** Kazakh language (KK) medical terminology in `lang.js` has not been verified by a Kazakh-speaking medical professional.

**Fix:**
- Ask a Kazakh medical professional to review the KK translations
- Key terms to verify: pathogenic, benign, variant of uncertain significance, allele frequency, missense
- Could potentially ask mentor's contacts at Nazarbayev University

---

## 2. BACKEND FIXES

### 2.1 API Version String Consistency

**Problem:** Verify that version string "5.2.0" is consistent across all files.

**Check:**
- `backend/main.py` — root endpoint version
- `frontend/index.html` — CSS version tag
- `frontend/sw.js` — cache version
- `MODEL_CARD.md` — version field
- PDF report footer

---

### 2.2 gnomAD Sensitivity Analysis Script

**Action:** Create `scripts/gnomad_sensitivity.py`

**Implementation:**
1. Load `master_training_dataset.csv`
2. Identify gnomAD proxy-benign rows (485 variants)
3. Retrain XGBoost + MLP on remaining 18,738 variants
4. Same 60/20/20 split, same hyperparameters
5. Evaluate on same test set
6. Report per-gene AUC delta
7. Save results to `data/gnomad_sensitivity_results.json`

---

### 2.3 Macro-Averaged AUC in Response

**Action:** Add `macro_avg_auc` field to model metadata at root endpoint.

**Implementation:**
- In `backend/main.py` root endpoint, add:
  ```python
  "macro_avg_auc": round(np.mean([0.983, 0.804, 0.743, 0.706, 0.641]), 3)  # = 0.775
  ```

---

### 2.4 Feature Importance Top-15 Endpoint

**Action:** Add an endpoint or field that returns the top 15 features by global SHAP importance.

**Implementation:**
- Precompute from SHAP values on test set
- Store as `data/top_features.json`
- Return in model metadata

---

## 3. FRONTEND FIXES

### 3.1 Hero Metric Clarification

**Action:** Add per-gene AUC range below the headline 0.978.

**Implementation:**
- In `frontend/index.html` hero section, add subtext: "Per-gene: 0.641-0.983"
- Or add a small asterisk: "*weighted by test set composition; see per-gene chart below"

### 3.2 SOTA Comparison Caveat in UI

**Action:** If SOTA comparison is displayed in the frontend, add the caveat.

**Implementation:**
- Add tooltip or footnote: "Evaluated on SteppeDNA test set. General-purpose tools not trained on same distribution."

---

## 4. DOCUMENTATION & FRAMING FIXES

### 4.1 README.md Updates

- [ ] Add per-gene AUC range in hero metrics
- [ ] Add BRCA2 dominance footnote
- [ ] Strengthen SOTA comparison caveat
- [ ] Add macro-averaged AUC
- [ ] Verify all version strings are 5.2.0

### 4.2 MODEL_CARD.md Updates

- [ ] Add macro-averaged per-gene AUC
- [ ] Strengthen population bias section
- [ ] Add temporal validation summary
- [ ] Add Kazakh variant representation gap

### 4.3 VALIDATION_REPORT.md Updates

- [ ] Ensure temporal validation is prominent (not buried)
- [ ] Add gnomAD sensitivity analysis results (when completed)
- [ ] Add feature concentration analysis
- [ ] Add conformal prediction results (when implemented)

### 4.4 COMPETITION_NARRATIVE.md

- [ ] Reframe from "pan-gene classifier with 0.978 AUC" to "trustworthy AI system with integrated uncertainty quantification"
- [ ] Lead with system novelty, not raw performance
- [ ] Address BRCA2 dominance proactively
- [ ] Include mentor endorsement reference

---

## 5. NOVELTY ADDITIONS (ALGORITHMIC)

### 5.1 Split Conformal Prediction (HIGHEST PRIORITY — 1-2 weeks)

**What:** Distribution-free uncertainty quantification that produces prediction SETS with mathematically guaranteed coverage probability.

**Why novel:** No variant pathogenicity classifier uses conformal prediction. Emerging in medical AI (2023-2025) but hasn't reached genomics. Gives you the claim: "First variant classifier with distribution-free coverage guarantees."

**How to implement:**

```python
# New file: scripts/train_conformal.py

import numpy as np
from sklearn.model_selection import train_test_split

def compute_nonconformity_scores(model, X_cal, y_cal):
    """Compute nonconformity scores on calibration set."""
    probs = model.predict_proba(X_cal)
    # Nonconformity = 1 - probability of true class
    scores = []
    for i, y in enumerate(y_cal):
        scores.append(1 - probs[i, int(y)])
    return np.array(scores)

def conformal_predict(model, x_test, cal_scores, alpha=0.10):
    """
    Produce prediction set with 1-alpha coverage guarantee.
    alpha=0.10 means 90% coverage.
    """
    n = len(cal_scores)
    # Quantile with finite-sample correction
    q = np.quantile(cal_scores, np.ceil((n + 1) * (1 - alpha)) / n)

    probs = model.predict_proba(x_test.reshape(1, -1))[0]
    prediction_set = []
    for c in range(len(probs)):
        if probs[c] >= 1 - q:
            prediction_set.append(c)
    return prediction_set, q

def gene_stratified_conformal(model, X_cal, y_cal, genes_cal, alpha=0.10):
    """
    Per-gene conformal calibration for gene-specific coverage guarantees.
    Each gene gets its own quantile threshold.
    """
    gene_thresholds = {}
    for gene in set(genes_cal):
        mask = genes_cal == gene
        if mask.sum() < 20:
            # Not enough data — fall back to global
            continue
        gene_scores = compute_nonconformity_scores(model, X_cal[mask], y_cal[mask])
        n = len(gene_scores)
        q = np.quantile(gene_scores, np.ceil((n + 1) * (1 - alpha)) / n)
        gene_thresholds[gene] = q
    return gene_thresholds
```

**Integration into backend:**
1. Train conformal calibration on existing calibration set (3,845 variants)
2. Save per-gene quantile thresholds to `data/conformal_thresholds.json`
3. At prediction time, produce prediction set alongside probability
4. Add to `/predict` response: `"conformal_set": ["Pathogenic"]` or `"conformal_set": ["Pathogenic", "VUS"]`
5. Frontend: show prediction set badge ("90% guaranteed")

**Key insight for judges:** For PALB2 (low data, AUC 0.641), conformal sets will be LARGER — often {Pathogenic, VUS, Benign} — which honestly communicates "we don't know." For BRCA2 (high data, AUC 0.983), sets will be tight — typically just {Pathogenic} or {Benign}. This is epistemic honesty built into the math.

**New files:**
- `scripts/train_conformal.py`
- `data/conformal_thresholds.json`
- Updates to `backend/main.py` and `frontend/app.js`

---

### 5.2 Meta-Learning (MAML/Reptile) for Data-Scarce Genes (4-6 weeks)

**What:** Use BRCA2 (data-rich, 10,000+ variants) as a meta-training source, then few-shot adapt to PALB2/RAD51C/RAD51D (data-scarce).

**Why novel:** Meta-learning has NOT been applied to variant pathogenicity classification. The framing — "transfer biological signal from data-rich to data-scarce homologous genes" — is biologically motivated and methodologically novel.

**How to implement:**

```python
# New file: scripts/meta_learning_genes.py

import torch
import torch.nn as nn
import learn2learn as l2l

class VariantMLP(nn.Module):
    def __init__(self, n_features=103, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )

    def forward(self, x):
        return self.net(x)

def train_maml(brca2_data, target_gene_data, n_episodes=1000):
    """
    Meta-train on BRCA2 episodes, then fine-tune on target gene.

    Each episode:
    1. Sample support set (K examples per class) from BRCA2
    2. Sample query set from BRCA2
    3. Inner loop: adapt on support set
    4. Outer loop: update based on query set loss
    """
    model = VariantMLP()
    maml = l2l.algorithms.MAML(model, lr=0.01, first_order=True)
    optimizer = torch.optim.Adam(maml.parameters(), lr=0.001)

    for episode in range(n_episodes):
        # Sample support + query from BRCA2
        support_x, support_y = sample_episode(brca2_data, k_shot=20)
        query_x, query_y = sample_episode(brca2_data, k_shot=20)

        learner = maml.clone()
        # Inner loop adaptation
        for _ in range(5):
            loss = nn.CrossEntropyLoss()(learner(support_x), support_y)
            learner.adapt(loss)

        # Outer loop update
        query_loss = nn.CrossEntropyLoss()(learner(query_x), query_y)
        optimizer.zero_grad()
        query_loss.backward()
        optimizer.step()

    # Fine-tune on target gene
    learner = maml.clone()
    target_x, target_y = target_gene_data
    for _ in range(50):
        loss = nn.CrossEntropyLoss()(learner(target_x), target_y)
        learner.adapt(loss)

    return learner
```

**Expected impact:**
- If PALB2 improves from 0.641 to ~0.70: publishable result
- If it doesn't improve: negative result that's still valuable ("HR gene homology doesn't transfer via MAML because feature distributions differ too much")
- Either outcome strengthens the project

**Dependencies:** `learn2learn`, `torch`

**New files:**
- `scripts/meta_learning_genes.py`
- `data/maml_results.json`
- Updates to `VALIDATION_REPORT.md`

---

### 5.3 Prototype Network for Interpretable Classification (3-4 weeks)

**What:** Upgrade the KD-tree contrastive system into a learned prototype network. The model learns class-representative prototypes during training, and predictions are explained by distance to learned prototypes.

**Why novel:** Combines prototype learning (Li et al. 2018) with variant classification. More principled than post-hoc KD-tree retrieval.

**How to implement:**

```python
# New file: scripts/train_prototype_network.py

import torch
import torch.nn as nn

class PrototypeNetwork(nn.Module):
    def __init__(self, n_features=103, n_prototypes_per_class=5, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        # Learnable prototypes: 2 classes x 5 prototypes x hidden_dim
        self.prototypes = nn.Parameter(
            torch.randn(2, n_prototypes_per_class, hidden)
        )

    def forward(self, x):
        z = self.encoder(x)  # [batch, hidden]
        # Distance to each prototype
        dists = torch.cdist(z.unsqueeze(1), self.prototypes.reshape(-1, self.prototypes.shape[-1]).unsqueeze(0))
        # Nearest prototype per class
        # ...classification based on prototype similarity
        return logits, prototype_similarities
```

**Integration:** At prediction time, report which prototype(s) the variant is most similar to, with human-readable prototype descriptions (e.g., "Pathogenic Prototype 3: buried missense in BRC repeat domain").

---

### 5.4 Gene-Conditional Normalizing Flow for Calibration (2-3 weeks)

**What:** Replace isotonic regression with a normalizing flow that takes raw model score + gene identity + data scarcity metrics as input and outputs a calibrated probability distribution.

**Why novel:** Normalizing flows for calibration is cutting-edge (Charpentier et al. 2022). Gene-conditioning is unique.

**How to implement:**

```python
# New file: scripts/train_normalizing_flow_calibrator.py

import torch
import torch.nn as nn
from torch.distributions import TransformedDistribution, Normal

class ConditionalFlow(nn.Module):
    def __init__(self, context_dim=5):  # gene_onehot(5) + scarcity
        super().__init__()
        # Affine coupling layers conditioned on gene + scarcity
        self.layers = nn.ModuleList([
            AffineLayer(context_dim) for _ in range(4)
        ])

    def forward(self, raw_score, context):
        z = raw_score
        log_det = 0
        for layer in self.layers:
            z, ld = layer(z, context)
            log_det += ld
        return z, log_det  # z is calibrated probability
```

**Advantage:** Produces a full posterior distribution, not just a point estimate. You can report mean, mode, and credible intervals.

---

### 5.5 Attention-Based Feature Interaction Discovery (2-3 weeks)

**What:** Replace the 4 manually defined interaction terms with a small self-attention layer that discovers feature interactions automatically.

**How to implement:**

```python
# In feature_engineering.py or a new module

class FeatureInteractionAttention(nn.Module):
    def __init__(self, n_features=103, n_heads=4, hidden=32):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=n_heads
        )
        self.project = nn.Linear(n_features, hidden)
        self.output = nn.Linear(hidden, n_features)

    def forward(self, x):
        # x: [batch, 103]
        # Treat each feature as a "token"
        tokens = self.project(x).unsqueeze(0)  # [1, batch, hidden]
        attn_out, attn_weights = self.attention(tokens, tokens, tokens)
        return self.output(attn_out.squeeze(0)), attn_weights
```

**Key output:** Attention weight matrices that show which features the model considers together. Visualizable as heatmaps. You could discover non-obvious interactions like "ESM-2 cosine similarity interacts with AlphaMissense score specifically in buried residues."

---

### 5.6 Evidential Deep Learning for Uncertainty Decomposition (3-4 weeks)

**What:** Replace the MLP with an evidential neural network that outputs Dirichlet distribution parameters, decomposing uncertainty into aleatoric (inherent data noise) vs. epistemic (lack of training data).

**Why novel:** For VUS predictions, you could distinguish: "uncertain because variant is genuinely borderline" (aleatoric) vs. "uncertain because we haven't seen similar variants" (epistemic). These require different clinical responses.

**How to implement:**

```python
# New file: scripts/train_evidential_nn.py

import torch
import torch.nn as nn

class EvidentialMLP(nn.Module):
    def __init__(self, n_features=103, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)  # Dirichlet concentration params
        )

    def forward(self, x):
        # Output: alpha parameters for Dirichlet distribution
        alpha = torch.exp(self.net(x)) + 1  # Ensure alpha > 1
        S = alpha.sum(dim=-1, keepdim=True)  # Dirichlet strength
        prob = alpha / S  # Expected probability

        # Uncertainty decomposition
        aleatoric = (prob * (1 - prob)) / (S + 1)
        epistemic = prob * (1 - prob) * S / ((S + 1) * (S ** 2))

        return prob, aleatoric, epistemic
```

---

### 5.7 Summary: Novelty Priority Ranking

| # | Novelty | Time | Impact | Feasibility | Priority |
|---|---------|------|--------|-------------|----------|
| 5.1 | Conformal Prediction | 1-2 wk | HIGH | HIGH | **DO FIRST** |
| 5.2 | Meta-Learning (MAML) | 4-6 wk | HIGH | MEDIUM | ISEF 2027 |
| 5.3 | Prototype Network | 3-4 wk | MEDIUM | MEDIUM | ISEF 2027 |
| 5.4 | Normalizing Flow Cal. | 2-3 wk | MEDIUM | MEDIUM | If time allows |
| 5.5 | Attention Interactions | 2-3 wk | MEDIUM | HIGH | If time allows |
| 5.6 | Evidential Deep Learning | 3-4 wk | HIGH | MEDIUM | ISEF 2027 |

---

## 6. DATA & TRAINING IMPROVEMENTS

### 6.1 More Non-BRCA2 Training Data (CRITICAL for ISEF)

**Sources:**
- International ClinVar submissions (non-US/European labs)
- LOVD (Leiden Open Variation Database) — has RAD51C/D variants
- ENIGMA consortium data (if accessible)
- Kazakh National Center for Biotechnology

**Impact:** This is the fundamental bottleneck. More data for PALB2/RAD51C/RAD51D would fix the core weakness.

### 6.2 Population-Stratified Allele Frequencies

**Action:** Use gnomAD v4's ancestry-specific AFs instead of global MAF.

**Implementation:**
- gnomAD v4 API provides AF for: African, East Asian, European, South Asian, etc.
- Add as features: `af_eas`, `af_sas`, `af_eur`, `af_afr`
- Relevant for Kazakh population (Central Asian ancestry)

### 6.3 EVE Scores for BRCA2

**Problem:** EVE scores are available for BRCA1, PALB2, RAD51C, RAD51D but NOT BRCA2 via dbNSFP.

**Fix:** Check alternative sources (EVE official website, direct download). BRCA2 EVE scores may be available from the original publication dataset.

### 6.4 ESM-2 650M with Gene-Specific Regularization

**Problem:** ESM-2 650M improved BRCA1/BRCA2 but degraded PALB2 (-0.120 AUC). The current solution is to not use 650M at all.

**Better fix:** Use 650M embeddings but add L2 regularization proportional to inverse gene training set size. Small genes get more regularization to prevent overfitting to 650M features.

```python
# Gene-specific regularization weight
reg_weight = {
    'BRCA2': 0.01,   # Large dataset, low regularization
    'BRCA1': 0.05,
    'PALB2': 0.10,   # Small dataset, high regularization
    'RAD51C': 0.15,
    'RAD51D': 0.15,
}
```

### 6.5 Feature Ablation Study (Top-15 vs All-103)

**Action:** Train model on only the top 15 SHAP features and compare.

**Implementation:**
```python
# scripts/feature_ablation_top15.py
top_15 = shap_global_importances[:15]  # Get from SHAP
X_reduced = X_train[top_15]
# Retrain and evaluate
```

**Expected result:** If top-15 achieves 0.97+ AUC, the other 88 features are marginal. If it drops to 0.95, the 88 features contribute meaningfully on edge cases.

---

## 7. CLINICAL VALIDATION & WET LAB

### 7.1 VUS Reclassification Tracking (6-12 months, for ISEF)

**What:** Run predictions on current ClinVar VUS variants. In 6-12 months, check if ClinVar reclassifies any of them. If your predictions match the reclassification, that's prospective validation.

**Implementation:**
1. Pull all current VUS variants for 5 HR genes from ClinVar
2. Run SteppeDNA predictions on each
3. Store predictions with timestamps: `data/vus_predictions_dated.csv`
4. In 6 months: re-pull ClinVar, check for reclassifications
5. Compare your prediction vs. new classification

**Impact:** Even 5-10 correct reclassification predictions would be strong evidence. This is the closest thing to prospective validation without a lab.

### 7.2 Clinical Geneticist Review (Before ISEF)

**What:** Have a board-certified clinical geneticist (MD, not just PhD) review:
- Methodology (is the approach sound?)
- ACMG code implementation (are the computational approximations reasonable?)
- Clinical utility (would this be useful in practice?)

**How:** Ask mentor for contacts. Kazakh medical genetics community. Any MD who does variant classification.

**Output:** A letter or written review. Even "we reviewed the methodology and find it sound for research purposes" adds credibility.

### 7.3 Functional Assay Validation (If Lab Access Available)

**What:** Select 10-20 predicted-pathogenic VUS in an underserved gene (PALB2 or RAD51C). Perform functional assay (e.g., HDR assay, minigene splicing assay) to test predictions.

**Impact:** This transforms the project from computational-only to experimental validation. ISEF Grand Award territory if even a few predictions are confirmed.

**How:** This requires wet lab access and expertise. Mentor may have connections at Nazarbayev University or MIT.

**Timeline:** 3-6 months for assay design + execution + analysis.

### 7.4 Kazakh Population-Specific Variant Characterization

**What:** Identify known Kazakh founder mutations in BRCA1/BRCA2. Test model predictions on them. Analyze if training data bias affects accuracy.

**Sources:**
- Nazarbayev University genetics department
- Kazakhstan National Center for Biotechnology
- Published literature on Central Asian BRCA variants

**Impact:** Whether the model succeeds or fails on Kazakh variants, the result is meaningful:
- Success: "Our model generalizes to underrepresented populations"
- Failure: "We identified a specific population gap that motivates targeted data collection"

---

## 8. EXTERNAL VALIDATION & LETTERS

### 8.1 Mentor Scientific Review Letter (THIS WEEK)

**Who:** Nazarbayev University PhD, currently at MIT.

**What to ask:**
1. "Would you be willing to write a brief letter of scientific review/endorsement?"
2. "Could you review our VALIDATION_REPORT.md and flag any methodological concerns?"
3. "Do you have contacts at the Broad Institute or MIT who work on variant classification?"

**Template message:**
```
Dear [Mentor],

I'm preparing SteppeDNA for Infomatrix Asia (3 weeks) and eventually ISEF 2027.
The project is a variant pathogenicity classifier for 5 HR DNA repair genes.

Could you:
1. Review our validation methodology (attached VALIDATION_REPORT.md)?
2. Write a brief endorsement letter if you find the methodology sound?
3. Connect me with anyone at MIT/Broad who works on variant interpretation?

The project has AUC 0.978 overall (BRCA2-driven), 103 features from 10+
biological sources, and novel uncertainty quantification features.

Thank you for your time.
```

### 8.2 MIT/Broad Institute Connection

**Why:** The Broad Institute runs ClinGen (Clinical Genome Resource) and is the world leader in variant classification. A conversation with anyone there would:
- Provide expert feedback on methodology
- Potentially access to data
- A reference or collaboration mention

### 8.3 Clinical Geneticist Review (Before ISEF)

**Who:** Any MD who classifies variants professionally.

**What:** Review the ACMG code implementation and clinical framing.

### 8.4 Kazakh Medical Terminology Review

**Who:** Kazakh-speaking medical professional or translator.

**What:** Verify KK translations in `frontend/lang.js`.

---

## 9. COMPETITION PREPARATION (INFOMATRIX ASIA — 3 WEEKS)

### 9.1 Presentation Narrative

**DO lead with:**
- "First variant classifier with integrated uncertainty quantification system"
- The comparison table (SteppeDNA vs REVEL/CADD/AlphaMissense feature-by-feature)
- Contrastive explanations demo
- 36,613 lines of code, 117 tests, 3 languages
- 16 documented limitations (shows scientific maturity)
- Kazakh language support + Central Asian health relevance
- Per-gene reliability tiers (transparency)

**DON'T lead with:**
- "0.978 AUC" without context
- "We beat REVEL and CADD" without the caveat
- The number of features (103) as the main selling point

### 9.2 Poster/Slides Content

Must include:
- [ ] System architecture diagram (user input -> features -> ensemble -> calibration -> uncertainty -> output)
- [ ] Per-gene AUC chart (honest, with reliability tiers)
- [ ] Feature source diagram (10+ biological databases)
- [ ] Trustworthy AI comparison table (vs REVEL, CADD, etc.)
- [ ] Contrastive explanation example
- [ ] Conformal prediction example (if implemented)
- [ ] Known limitations summary (3-5 key ones)
- [ ] Live demo QR code

### 9.3 Rehearsed Answers for Tough Questions

**Q: "Your non-BRCA2 genes have AUC below 0.75. Does the model actually work?"**
A: "For BRCA2, which is the most clinically important and most studied, yes — AUC 0.983 with temporal stability. For smaller genes, we honestly disclose the limitations through gene reliability tiers. This is a data scarcity problem, not a model problem — with more training data, performance will improve. Our active learning system identifies which variants to prioritize for experimental validation."

**Q: "Why XGBoost + MLP instead of a transformer?"**
A: "We tested larger models (ESM-2 650M, GNN). They improved BRCA2 but degraded small genes by 12-15 AUC points — a classic overfitting problem with limited training data. Our architectural contribution isn't the base model — it's the trustworthy prediction system wrapped around it: conformal prediction, contrastive explanations, data scarcity scoring, gene-specific calibration."

**Q: "Isn't this just ClinVar with extra steps?"**
A: "We validated on independent data: ProteinGym DMS (experimental functional data, AUC 0.719) and ClinVar Expert Panel (AUC 0.793, BRCA2-only 0.918). The DMS correlation (Spearman r=-0.319, p<1e-10) confirms biological signal beyond ClinVar label memorization."

**Q: "What's actually novel?"**
A: "Three things: (1) the first variant classifier with distribution-free coverage guarantees via conformal prediction [if implemented], (2) contrastive explanations using real training variant retrieval instead of synthetic perturbations, (3) per-prediction data scarcity quantification as epistemic uncertainty. No competing tool offers any of these."

### 9.4 Live Demo Script

1. Open SteppeDNA in browser
2. Select BRCA2, enter a known pathogenic variant (e.g., c.5946delT)
3. Show: probability, SHAP waterfall, contrastive explanation, reliability tier
4. Switch to PALB2, enter a variant — show the different reliability warning (LOW tier)
5. Upload a small VCF file — show batch processing + CSV export
6. Switch language to Kazakh — show localization
7. Show model card page — show 16 limitations

### 9.5 Three-Week Timeline

| Day | Action |
|-----|--------|
| Day 1 | Message mentor. Initialize git. Push to GitHub. |
| Days 1-4 | Implement conformal prediction (scripts + backend + frontend) |
| Days 4-5 | Run gnomAD sensitivity analysis |
| Days 5-7 | Fix documentation: headline AUC, SOTA caveat, per-gene framing |
| Days 7-10 | Build poster/slides with "trustworthy AI" narrative |
| Days 10-12 | Practice presentation + tough question rehearsal |
| Days 12-14 | Integrate mentor feedback if received |
| Days 14-18 | Final polish, live demo rehearsal |
| Days 18-21 | Travel, setup, last-minute fixes |

---

## 10. COMPETITION PREPARATION (INFOMATRIX WORLDS)

Everything from Section 9, plus:

- [ ] Tighten narrative for international audience (less regional, more global genomics)
- [ ] Add meta-learning results if time allows
- [ ] Stronger independent validation evidence
- [ ] Consider adding population-stratified AF features
- [ ] More polished poster with professional design

---

## 11. COMPETITION PREPARATION (ISEF 2027)

Everything from Sections 9-10, plus:

### Must-haves:
- [ ] Clinical geneticist review letter
- [ ] VUS reclassification tracking results (6+ months of data)
- [ ] Meta-learning (MAML) implementation and results
- [ ] Conformal prediction (if not done for Infomatrix)
- [ ] Kazakh population variant analysis
- [ ] At least one novel algorithmic contribution with measurable impact
- [ ] Publication (submitted, even if not accepted)

### Nice-to-haves:
- [ ] Functional assay validation (wet lab)
- [ ] Broad Institute/MIT collaboration reference
- [ ] Evidential deep learning implementation
- [ ] Prototype network for interpretable classification
- [ ] Population-stratified allele frequency features

---

## 12. ENGINEERING & INFRASTRUCTURE

### 12.1 Git Version Control (TODAY)

**Action:** Initialize git, create proper .gitignore, push to GitHub.

```bash
cd C:\Users\User\OneDrive\Desktop\SteppeDNA
git init
git add .
git commit -m "Initial commit: SteppeDNA v5.2 — pan-gene variant pathogenicity classifier"
# Create repo on GitHub, then:
git remote add origin https://github.com/elnarabzhakhanov/SteppeDNA.git
git push -u origin main
```

**Important:** The .gitignore should exclude large binary files. Consider git-lfs for model files.

### 12.2 Large File Handling

**Problem:** `data/` directory is ~702 MB. GitHub has a 100 MB file limit.

**Fix options:**
1. Git LFS for model binaries (.pkl, .h5, .json models)
2. Or exclude model files from git and document how to download/generate them
3. Update .gitignore to exclude large files:
   ```
   data/*.pkl
   data/*.h5
   data/variant_summary.txt.gz
   data/esm2_embeddings.pkl
   data/esm2_650m_embeddings.pkl
   ```

### 12.3 Future Engineering Improvements

- [ ] Population-stratified AF display in frontend
- [ ] Compound heterozygosity visualization
- [ ] ML models for splice/frameshift/nonsense (not just rules)
- [ ] Multi-gene VCF with gene-gene interaction awareness
- [ ] User authentication for saved analysis history
- [ ] API documentation with Swagger examples

---

## 13. PUBLICATION PATHWAY

### 13.1 Short Paper: Negative Result (ESM-2 Scaling)

**Title:** "Scaling Protein Language Models Does Not Improve Variant Classification for Data-Scarce Genes"

**Venue:** MLCB Workshop (ML in Computational Biology) or ML4H (ML for Health)

**Content:** ESM-2 8M vs 650M vs LoRA, per-gene analysis, the PALB2 regression finding.

### 13.2 System Paper: Trustworthy Variant Prediction

**Title:** "SteppeDNA: Trustworthy Variant Pathogenicity Prediction with Integrated Uncertainty Quantification"

**Venue:** Bioinformatics, PLOS Computational Biology, or Briefings in Bioinformatics

**Content:** Full system description, conformal prediction, contrastive explanations, data scarcity scoring, per-gene calibration, gold-standard benchmark.

### 13.3 Application Note

**Title:** "SteppeDNA: A Web Tool for Variant Pathogenicity Prediction in Homologous Recombination Genes"

**Venue:** Bioinformatics (Application Notes section — 2-page format)

**Content:** Tool description, usage, deployment, performance summary.

---

## 14. PRIORITY TIMELINE (Master Schedule)

### Immediate (This Week)
1. Message mentor
2. Initialize git + push to GitHub
3. Start conformal prediction implementation

### Before Infomatrix Asia (3 weeks)
4. Complete conformal prediction
5. gnomAD sensitivity analysis
6. Fix headline AUC framing + SOTA caveat
7. Build poster/presentation
8. Practice + rehearse tough questions

### Before Infomatrix Worlds
9. Meta-learning (MAML) implementation
10. Integrate mentor feedback
11. Polish international narrative

### Before ISEF 2027 (12+ months)
12. Clinical geneticist review
13. VUS reclassification tracking
14. Kazakh population variant analysis
15. Publication submission
16. Functional assay validation (if lab access)
17. Additional novelty (evidential DL, prototype network)

---

## APPENDIX: PROJECT STATISTICS

| Metric | Value |
|--------|-------|
| Total source files | 204 |
| Total lines of code | 36,613 |
| Python files/lines | 93 / 23,049 |
| JavaScript files/lines | 4 / 2,768 |
| CSS files/lines | 1 / 2,410 |
| HTML files/lines | 3 / 1,048 |
| Tests | 117 (112 core + 5 flaky) |
| Training variants | 19,223 |
| Features | 103 |
| Model artifacts | 59 files |
| Libraries used | 27 |
| Languages supported | 3 (EN/KK/RU) |
| Documentation | 37KB validation report + model card |
| Known limitations disclosed | 16 |
