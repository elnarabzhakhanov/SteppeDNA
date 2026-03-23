# SteppeDNA Research Poster — Complete Prompt

Generate 3 research poster walls as **self-contained HTML files** (one per wall) for the Infomatrix Asia 2026 science competition. Each poster should be a single HTML file with all CSS inline. The output will be printed as a large-format poster, but design at a small convenient canvas size using the correct **aspect ratio** — the print shop scales it up and since it's vector (HTML/CSS rendered to PDF), quality is lossless.

## Canvas Sizes (use these exact pixel dimensions)

- **Back wall**: 848 × 1000 px (aspect ratio 0.848:1, prints at 140 × 165 cm)
- **Left wall**: 788 × 1000 px (aspect ratio 0.788:1, prints at 130 × 165 cm)
- **Right wall**: 788 × 1000 px (aspect ratio 0.788:1, prints at 130 × 165 cm)

Use `width` and `height` on the body/root container in px. NOT cm. The poster will be opened in a browser and printed/saved as PDF from Chrome (Ctrl+P → Save as PDF).

## Design System (MUST match the SteppeDNA web app dark theme)

```css
/* Exact tokens from the SteppeDNA frontend dark theme */
--primary: #8382ff;
--primary-dark: #6260FF;
--primary-glow: rgba(131, 130, 255, 0.15);
--secondary: #2a2855;
--accent-light: #3d3a8a;
--text-body: #a2a2ca;
--bg: #12121a;
--text-dark: #e8e8f2;     /* main text color */
--card-bg: #1e1e2d;
--card-alt: #262638;
--feat-bg: #2a2a3e;
--track-bg: #31314a;
--input-bg: #2a2a3e;

--success: #10B981;
--danger: #ef6565;
--warning: #fbbf24;
--border: #31314a;

--text: #e8e8f2;
--text-mid: #8888aa;

--radius: 20px;
--shadow: 0 12px 40px -12px rgba(0, 0, 0, 0.4);

/* Font: Inter for body, Playfair Display for title */
/* Hero gradient (use for banner): */
background: linear-gradient(135deg, #1a1a2e 0%, #2a2855 40%, #3d3a8a 70%, #6260FF 100%);
```

The web app uses rounded cards with subtle borders, the dark purple-blue palette above, and clean typography. The poster MUST look like it belongs to the same product family as the web app. Think: polished dark SaaS dashboard aesthetic, not academic LaTeX poster.

## Design Principles

- **~30-40% white space** (dark space in our case). Do NOT cram every centimeter.
- **Visual hierarchy**: Title largest, section headers next, body text readable from 1.5m away at print size.
- **Charts over text**: every result that can be a chart should be a chart.
- **Consistent card containers** with rounded corners, subtle borders, and the card-bg color.
- **No emojis**. Clean, professional.
- Left-to-right or top-to-bottom logical reading flow.
- Max 3–4 accent colors beyond the base palette.

## Project Overview

**SteppeDNA** is a multi-gene variant pathogenicity classifier for 5 homologous recombination (HR) DNA repair genes: BRCA1, BRCA2, PALB2, RAD51C, RAD51D.

- **Authors**: Elnar Abzhakhanov, 16 | Inkar Saule Abzhakhanova, 16
- **School**: K. Satpayev Binom-School, Astana, Kazakhstan
- **Mentor**: Darkhan Shadykul
- **Category**: Applied Science • Infomatrix Asia 2026
- **URL**: steppedna.com

### Technical Details
- **Ensemble**: XGBoost + MLP (TensorFlow) with per-gene isotonic calibration
- **Per-gene weights**: BRCA1 (MLP-only), BRCA2 (60/40), PALB2 (MLP-only), RAD51C (80/20 XGB), RAD51D (55/45)
- **Features**: 120 engineered features (gene-identifying + AlphaMissense features REMOVED due to discovered label leakage)
- **Feature sources**: BLOSUM62 + physicochemical, ESM-2 protein language model embeddings (8M params, 20 PCA components per gene), EVE evolutionary coupling scores, AlphaFold structural features, gnomAD population frequencies, BRCA1 Findlay SGE functional scores
- **Dataset**: 19,223 variants (18,738 ClinVar + 485 gnomAD proxy-benign) across 5 HR genes
- **Split**: 60/20/20 train/cal/test with gene × label stratification (NOT 10-fold CV)
- **Frontend**: HTML/JS (XSS-hardened), trilingual (EN/KK/RU)
- **Backend**: FastAPI (Python)
- **Deployed**: Frontend on Vercel, Backend on Render (steppedna.com)
- **Tests**: 198 automated tests (property-based, ACMG combinatorial, clinical correctness)
- **Tech stack**: Python, FastAPI, XGBoost, TensorFlow, ESM-2, BioPython, SHAP

---

## WALL 1: BACK WALL (848 × 1000 px) — "The Hook"

This is what judges see first. It must answer: **what is this, how well does it work, and why should I care?**

### Sections (top to bottom):

**1. Title Banner** (full width, gradient background)
- "SteppeDNA" in large Playfair Display font
- Subtitle: "Multi-Gene Variant Pathogenicity Classifier for HR DNA Repair Genes"
- Right side: authors, school, mentor
- Pill badge: "APPLIED SCIENCE • INFOMATRIX ASIA 2026"

**2. Problem Statement** (1-2 lines, centered, subtle background)
- "Over 50% of missense variants in HR repair genes remain **Variants of Uncertain Significance**, leaving patients without actionable guidance. Central Asian populations face an additional **11% classification disparity** due to database underrepresentation. SteppeDNA addresses both with a calibrated multi-gene ML ensemble."

**3. Hero Metrics Row** (5 boxes, horizontal)
| Value | Label | Sub-label |
|-------|-------|-----------|
| 0.985 | OVERALL AUC | sample-weighted |
| 0.791 | MACRO AUC | equal per gene |
| 0.928 | MCC | balanced metric |
| 96.5% | BALANCED ACC | sensitivity + specificity |
| 4/5 (green) | GENES BEAT SOTA | vs REVEL, BayesDel, CADD |

**4. Main Results Section** (two-column: chart left 2/3, table right 1/3)

Left: **Grouped Bar Chart** — "Per-Gene AUC: SteppeDNA vs. State of the Art"
- 4 tools × 5 genes = 20 bars
- Colors: SteppeDNA=#8382ff, REVEL=#10B981, BayesDel=#fbbf24, CADD=#ef6565
- Y-axis from 0.4 to 1.0
- Data:

| Gene | SteppeDNA | REVEL | BayesDel | CADD |
|------|-----------|-------|----------|------|
| BRCA2 | **0.994** | 0.891 | 0.949 | 0.908 |
| RAD51D | **0.824** | 0.461 | 0.448 | 0.457 |
| RAD51C | **0.785** | 0.651 | 0.634 | 0.703 |
| BRCA1 | **0.747** | 0.595 | 0.646 | 0.527 |
| PALB2 | 0.605 | **0.732** | 0.432 | 0.564 |

- Below chart, highlight box: "A single multi-gene ensemble with gene-adaptive calibration achieves competitive or superior performance to established tools on 4 of 5 HR repair genes, while providing calibrated probabilities, SHAP explanations, and ACMG evidence codes."

Right: **AUC Comparison Table** (same data as chart but in table form)
- Subtitle: "HELD-OUT TEST SET (N=3,845)"
- Green highlight on best value per row
- Below table: two benchmark boxes in teal/cyan:
  - 0.801 — ClinVar Expert Panel (n=80)
  - 0.750 — ProteinGym DMS (n=443)
- Caveat (small yellow box): "All tools evaluated on our held-out test set via dbNSFP. SteppeDNA has home-court advantage — SOTA trained on different distributions. Independent benchmarks (above) provide unbiased comparison."

**5. System Architecture** (horizontal pipeline with arrows)
Three boxes connected by arrows (→):
1. **Front End & User Input** (primary border): User Input (VCF, manual via HTML/JS) • Dark Mode & Trilingual • SHAP Viz & PDF Export
2. **FastAPI Backend** (primary border, wider): Feature Engine — 120 features (ESM-2, AlphaFold) • ML Ensemble (XGB+MLP, per-gene weights) • External APIs (ClinVar, gnomAD) • ACMG Engine (Rule-based codes PVS1, PM1, BS1)
3. **Clinical Report** (green/success border): P(pathogenic)+95% CI • SHAP explanations • ACMG evidence codes • Conformal prediction set • Data scarcity warnings

**6. Footer** (compact bar)
- Left: tech badges — ML/DL: XGBoost • TensorFlow • ESM-2 | Backend: Python • FastAPI • BioPython | Data: 19,223 variants • 5 genes • 120 features | Features: SHAP • ACMG • VCF • PDF • Trilingual
- Right: QR code + "steppedna.com"

---

## WALL 2: LEFT WALL (788 × 1000 px) — "The Proof"

Judge has stopped at the booth. This wall answers: **how did you build it and does the science hold up?**

### Sections:

**1. Banner** — "SteppeDNA" + "Methodology & Validation" on the right

**2. Two-column top row:**

Left card: **Research Questions**
- RQ1: Can one ensemble classify missense variants across 5 HR genes? → Universal XGBoost+MLP with gene-adaptive isotonic calibration.
- RQ2: Do protein language models improve pathogenicity prediction? → ESM-2 embeddings → 20 PCA components per gene.
- RQ3: Can we provide calibrated, explainable confidence estimates? → Isotonic calibration + conformal prediction + SHAP explanations.

Right card: **Dataset & Training**
- Summary metrics: 19,223 Total Variants | 5 HR Genes | 120 Features
- Horizontal bar chart showing gene distribution:
  - BRCA2: 10,085 (100%)
  - BRCA1: 5,432 (53%)
  - PALB2: 2,621 (26%)
  - RAD51C: 675 (7%)
  - RAD51D: 410 (4%)
- Caveat: "ClinVar + ENIGMA + gnomAD proxy-benign. 60/20/20 train/cal/test, gene×label stratified."

**3. Two-column middle row:**

Left card: **Feature Engineering**
- 4 category boxes in a 2×2 grid: BLOSUM62 + Physicochemical | ESM-2 Embeddings | EVE + MAVE | AlphaFold + Conservation
- Highlight box: "Hybrid approach: Our final scores use XGBoost/MLP for predictive probabilities, but leverage ACMG rule-based systems purely objectively to ensure outputs are explainable to clinicians."

Right card: **Visual Evidence**
- Two images side by side:
  1. Per-Gene ROC Curves (caption below)
  2. SHAP Global Beeswarm (caption below)
- NOTE: For the prompt, use placeholder rectangles with dashed borders labeled "Per-Gene ROC Curves" and "SHAP Global Beeswarm" — the user will replace these with actual images later.

**4. Full-width bottom card: Validation & Scientific Rigor**
Two-column layout inside:

Left column: **Model Evaluation** (bullet list)
- Model Validation: 60/20/20 stratified split, AUC 0.985 on held-out test (N=3,845)
- External Benchmark: ProteinGym DMS AUC 0.750 (n=443), ClinVar EP AUC 0.801 (n=80) — zero training overlap
- Calibration: Isotonic per-gene calibration on dedicated cal set, bootstrap CI, conformal prediction.
- Software Testing: 198 automated tests: property-based, ACMG combinatorial, clinical correctness.

Right column: **AlphaMissense Leakage Discovery**
- Description: "AM features removed in v5.4. We discovered data leakage caused ~0.02 artificial AUC bump."
- Table:

| Gene | Baseline (with AM) | Ablated (without AM) |
|------|-------------------|---------------------|
| BRCA1 | 0.765 | **0.784** |
| PALB2 | 0.570 | **0.586** |
| RAD51C | 0.799 | **0.817** |
| RAD51D | **0.821** | 0.815 |
| BRCA2 | 0.997 | 0.997 |

- Footnote: "MAVE ablation caused ΔAUC -0.0006 overall. Model is robust without AM leakage."

**5. Footer** — QR code + steppedna.com

---

## WALL 3: RIGHT WALL (788 × 1000 px) — "The So What"

Judge is engaged. This wall answers: **why does this matter and what's next?**

### Sections:

**1. Banner** — "SteppeDNA" + "Clinical Impact & Population Equity" on the right

**2. The VUS Crisis** (full-width card)
Two-column inside:
- Left: text paragraphs
  - ">50% of missense variants remain categorized as **Variants of Uncertain Significance (VUS)**. Pathology labs cannot definitively tell if they are dangerous or harmless."
  - "These 5 HR genes protect against cancer by repairing DNA. A pathogenic diagnosis enables **preventive surgery, PARP inhibitors, and enhanced screening**."
  - "A VUS diagnosis leaves patients stranded with **no actionable options**."
- Right: callout box (primary border, centered text)
  - Title: "SteppeDNA translates VUS into actionable data."
  - Body: "Resolves VUS findings with transparent, calibrated probability scores tied to hard ACMG evidence."

**3. Two-column middle section (takes most space):**

Left card: **Kazakhstan Population Equity**
- Bullet stats:
  - **7 founder mutations** mapped to the Kazakhstan phenotype pool.
  - **11% disparity** in clinical utility: 97.4% vs 87.7% PM2 flagging (EAS vs NFE).
  - Result: **1,260 excess PM2 flags** for non-European patients.
  - **0% Kazakh-specific data** currently mapped in global databases.
- Founder mutations table:

| Gene | Variant | Frequency |
|------|---------|-----------|
| BRCA1 | c.5266dup | 3.5% |
| BRCA2 | c.9409dup | 1.2% |
| BRCA2 | c.9253del | 0.8% |
| BRCA1 | c.5278-2del | 0.8% |
| BRCA1 | c.181T>G | 0.4% |

- Source: "Oncotarget 2023, Djansugurova et al. 2023"

Right card: **Clinical Decision Support**
- Feature list:
  - **Explainable AI (XAI):** SHAP values map exact feature importance.
  - **Objective Mapping:** Implementation of 11 ACMG rules.
  - **Reliability Checking:** Conformal prediction + bootstrap CI error bands.
  - **Edge-Case Warnings:** Gene reliability and model disagreement alerts.
  - **Workflow Ready:** Live API hooks (ClinVar, gnomAD), VCF batch processing, instant PDF report generation.
  - **Accessible:** Full EN / KK / RU localization.
- Caveat: "Research Use Only (RUO) — not approved for primary diagnosis."

**4. Full-width: Conclusions & Future Work** (3-column)

Column 1: **Novel Contributions**
- Multi-gene ensemble outperforms REVEL/BayesDel/CADD on 4/5 genes.
- AlphaMissense label leakage discovered and removed (+0.02 AUC on 3 genes).
- Per-gene isotonic calibration eliminates systematic bias.
- 7 Kazakh founder mutations integrated for population equity.

Column 2: **Limitations**
- European training bias limits trans-ethnic utility.
- BRCA2 dominates dataset (52% of variants).
- PALB2 predictive performance trails REVEL.
- Temporal generalization limited for rare genes; BRCA2 maintains 0.983 AUC.

Column 3: **Future Roadmap**
- Scale up to ESM-2 (650M parameters).
- Transfer learning path: BRCA2 → PALB2.
- Clinical expert review queue implementation.
- Population-aware thresholds per ethnic group.
- Prospective clinical validation: Sept 2026.

**5. Footer** — QR code + steppedna.com

---

## Critical Rules

1. **All numbers are verified** — do NOT change any metric, AUC value, or statistic. Use them exactly as provided.
2. **Do NOT add content** that isn't listed above. No filler text, no placeholder lorem ipsum.
3. Each poster is a **single self-contained HTML file** — all CSS inline in a `<style>` tag, no external dependencies except Google Fonts (Inter + Playfair Display).
4. The chart on the back wall should be rendered as an **inline SVG** — not an image, not a canvas.
5. Use **flexbox** for layout. The body should be `display: flex; flex-direction: column;` with the content filling the exact canvas dimensions.
6. Make sure all text is **large enough to read from 1.5 meters** when the poster is printed at full size (140cm or 130cm wide). Err on the side of larger.
7. The design should look **polished and professional** — like a dark-mode SaaS product dashboard, not a hastily-made academic poster. Rounded corners, subtle gradients, consistent spacing.
8. Generate **one wall at a time**, starting with the back wall. Do NOT generate all three at once.
