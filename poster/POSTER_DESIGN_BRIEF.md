# SteppeDNA — 3-Wall Poster Design Brief

## Competition Context

**Event:** Infomatrix Asia 2026 (March 25-27, Spectrum International School, Astana, Kazakhstan)
**Category:** Applied Science (#17 on finalist list)
**Team:** "Double Helix" — Elnar Abzhakhanov (16) & Inkar Saule Abzhakhanova (16)
**School:** K. Satpayev Binom-School, Astana
**Mentor:** Darkhan Shadykul
**Website:** steppedna.com (live, deployed)

## What SteppeDNA Is

A machine learning system that predicts whether DNA mutations in 5 cancer-related genes (BRCA1, BRCA2, PALB2, RAD51C, RAD51D) are dangerous (pathogenic) or harmless (benign). It's a web application — you input a mutation and get a probability, confidence interval, explanation of why (SHAP), and clinical evidence codes (ACMG). Trained on 19,223 variants with 120 engineered features from 7 biological data sources. Uses an XGBoost + MLP neural network ensemble with per-gene weights and isotonic calibration. Beats REVEL, BayesDel, and CADD (the three most widely used existing tools) on 4 out of 5 genes.

## Booth Setup

- **Booth dimensions:** 2.0m high × 1.5m wide × 1.4m deep
- **Three internal walls** — 1 back wall + 2 side walls
- **Table provided** by organizers (for laptop demo)
- Stand is scored: **up to 10% of final score**
- No double-sided tape (regular tape only)
- No decorations outside the stand

## Poster Dimensions

| Wall | Width | Height |
|------|-------|--------|
| **Back wall** | 140cm | 165cm |
| **Left side wall** | 130cm | 165cm |
| **Right side wall** | 130cm | 165cm |

## Design System (MUST follow — this is proven and looks good)

The project has an existing poster (`posterv2.html`, 165×125cm) that looks cohesive and professional. All three new posters MUST use the same design system:

### Colors (CSS variables)
```
--bg: #12121a (dark background)
--card-bg: #1e1e2d (card background)
--feat-bg: #2a2a3e (feature/metric box background)
--track-bg: #31314a
--border: #31314a
--primary: #8382ff (purple accent — main brand color)
--primary-dark: #6260FF
--text-main: #e8e8f2 (headings, strong text)
--text-body: #c8c8e0 (body text)
--text-muted: #8888aa (labels, captions)
--success: #10B981 (green — best scores)
--danger: #ef6565 (red — losses)
--warning: #fbbf24 (yellow — caveats)
--bench: #38bdf8 (blue — independent benchmarks)
--hero-gradient: linear-gradient(135deg, #1a1a2e 0%, #2a2855 40%, #3d3a8a 70%, #6260FF 100%)
```

### Typography
- **Fonts:** Inter (body) + Playfair Display (title only)
- **Base font size:** 45px (html root) — all sizes in rem
- **Title:** Playfair Display, ~3rem, 800 weight, white
- **Card titles:** Inter, ~1rem, 700 weight, --text-main
- **Body text:** Inter, ~0.7rem, 400 weight, --text-body
- **Labels/captions:** Inter, ~0.5rem, 600 weight, uppercase, --text-muted
- **Metric values:** Inter, ~1rem, 800 weight, --primary

### Card Style
- Background: --card-bg
- Border: 1px solid --border
- Border-radius: 32px
- Padding: ~1.2rem 1.5rem
- Box-shadow: 0 12px 40px -12px rgba(0,0,0,0.4)
- Card header: icon circle (gradient purple) + title, with 2px purple bottom border

### Layout Principles
- CSS Grid with consistent gap (~1.5rem)
- Cards span 1 or 2 columns
- Everything should fill its space — NO large empty voids
- Consistent padding and spacing between all elements
- Metric boxes: --feat-bg background, 12px border-radius, centered text
- Highlight boxes: subtle purple gradient border for key findings
- Warning/caveat boxes: subtle yellow border

### Critical Design Rules
- **Minimal text** — readable from 3 meters away
- **Large fonts** — no tiny unreadable labels
- **Cohesive proportions** — all cards similar sizes, no jarring contrasts
- **No empty voids** — every area should have content filling it naturally
- **Visual hierarchy** — title > metrics > charts/tables > body text > captions
- The three walls should look like they BELONG TOGETHER

---

## BACK WALL (140cm × 165cm) — "The Hero Wall"

First thing judges see when they walk up. Must hook them in 5 seconds. Big numbers, clear results.

### Row 1 — Title Banner (full width)
- Left side: "SteppeDNA" in Playfair Display, large + subtitle "Multi-Gene Variant Pathogenicity Classifier for HR DNA Repair Genes"
- Right side: "Elnar Abzhakhanov, 16" / "Inkar Saule Abzhakhanova, 16" / "K. Satpayev Binom-School, Astana" / "Mentor: Darkhan Shadykul" / badge "Applied Science • Infomatrix Asia 2026"

### Row 2 — Five Metric Boxes (full width, evenly spaced horizontal row)
| Value | Label | Sublabel |
|-------|-------|----------|
| 0.985 | Overall AUC | sample-weighted |
| 0.791 | Macro AUC | equal per gene |
| 0.928 | MCC | balanced metric |
| 96.5% | Balanced Acc | sensitivity + specificity |
| 4/5 | Genes Beat SOTA | vs REVEL, BayesDel, CADD |

### Row 3 — Two cards side by side (this is the main content area)

**Left card (spans 2/3 width): SOTA Bar Chart**
- Title: "Per-Gene AUC: SteppeDNA vs. State of the Art"
- Grouped bar chart with 5 gene groups, 4 bars each:
  - SteppeDNA (purple #8382ff): BRCA2=0.994, RAD51D=0.824, RAD51C=0.785, BRCA1=0.747, PALB2=0.605
  - REVEL (green #10B981): 0.891, 0.461, 0.651, 0.595, 0.732
  - BayesDel (yellow #fbbf24): 0.949, 0.448, 0.634, 0.646, 0.432
  - CADD (red #ef6565): 0.908, 0.457, 0.703, 0.527, 0.564
- Y-axis from 0.4 to 1.0
- Gene names below each group: BRCA2, RAD51D, RAD51C, BRCA1, PALB2
- Legend at bottom: colored squares + labels
- SteppeDNA value labels above each purple bar (.994, .824, .785, .747, .605)
- At bottom of card, a highlight box: "**Key Finding:** A single multi-gene ensemble with gene-adaptive calibration achieves competitive or superior performance to established tools on *4 of 5 HR repair genes*, while providing calibrated probabilities, SHAP explanations, and ACMG evidence codes."

**Right card (1/3 width): AUC Comparison Table + Independent Validation**
- Title: "AUC Comparison"
- Subtitle: "HELD-OUT TEST SET (N=3,845)"
- Table with columns: Gene | Ours | REVEL | BayesDel | CADD
  - BRCA2: **0.994** | 0.891 | 0.949 | 0.908 (our value in green = best)
  - RAD51D: **0.824** | 0.461 | 0.448 | 0.457
  - RAD51C: **0.785** | 0.651 | 0.634 | 0.703
  - BRCA1: **0.747** | 0.595 | 0.646 | 0.527
  - PALB2: 0.605 (red = we lose) | **0.732** (green) | 0.432 | 0.564
- Below table, two blue benchmark boxes:
  - 0.801 — ClinVar Expert Panel
  - 0.750 — ProteinGym DMS
- Below benchmarks, yellow caveat box: "Independent validation — zero training overlap. SOTA evaluated on our test set (home-court advantage)."

### Row 4 — Architecture Card (full width)
- Title: "System Architecture"
- Horizontal flow diagram (SVG):
  - **User** box (variant query or VCF upload) →arrow→
  - **Frontend** box (HTML/JavaScript, Dark Mode • Trilingual, SHAP Viz • PDF Export) →arrow→
  - **FastAPI Backend** large box containing 4 sub-boxes:
    - Feature Engine (120 features, ESM-2 • AlphaFold)
    - ML Ensemble (XGB + MLP, Per-gene weights)
    - External APIs (ClinVar • gnomAD, Cached lookups)
    - ACMG Engine (Rule-based codes, PVS1 • PM1 • BS1...)
  - →arrow→ **Clinical Report** box (green border) (P(pathogenic) + 95% CI, SHAP explanations, ACMG evidence codes, Conformal prediction set, Data scarcity warnings)

### Row 5 — Footer Banner (full width)
- Four tech badges in a row:
  - ML/DL: XGBoost • TensorFlow • ESM-2
  - Backend: Python • FastAPI • BioPython
  - Data: 19,223 variants • 5 genes • 120 features
  - Live Features: SHAP • ACMG • VCF • PDF • Trilingual
- QR code (links to steppedna.com) + "steppedna.com" text

---

## LEFT WALL (130cm × 165cm) — "The Science Wall"

Methodology deep-dive for judges who lean in and ask technical questions.

### Row 1 — Title Banner (full width)
- Same style/colors as back wall but can be slightly more compact
- "SteppeDNA" + subtitle "Methodology & Validation"
- Same author info on right

### Row 2 — Two cards side by side

**Left card: Research Questions**
- Title: "Research Questions"
- Four research questions, each with bold question + 1-line answer:
  - **RQ1:** Can a single ML model classify missense variants across multiple HR genes?
    - "We train one universal model on all 5 genes with gene-adaptive calibration."
  - **RQ2:** Do protein language model embeddings improve classification?
    - "ESM-2 (8M parameters) generates per-gene embeddings reduced to 20 PCA components."
  - **RQ3:** Can we provide calibrated confidence with transparent limitations?
    - "Isotonic calibration, bootstrap CI, and conformal prediction sets."
  - **RQ4:** How does SteppeDNA compare to existing tools?
    - "We benchmark against REVEL, BayesDel, and CADD on held-out test data."

**Right card: Dataset & Training**
- Title: "Dataset & Training"
- Three metric boxes at top: 19,223 Variants | 5 HR Genes | 120 Features
- Horizontal bar chart showing gene distribution:
  - BRCA2: 10,085 (52.5% width, purple bar)
  - BRCA1: 5,432 (28.3% width, yellow bar)
  - PALB2: 2,621 (13.6% width, red bar)
  - RAD51C: 675 (3.5% width, pink bar, label outside bar)
  - RAD51D: 410 (2.1% width, green bar, label outside bar)
- Caption: "Sources: 18,738 ClinVar + 485 gnomAD proxy-benign. Split: 60/20/20 train/cal/test with gene × label stratification."

### Row 3 — Two cards side by side

**Left card: Feature Engineering**
- Title: "Feature Engineering"
- 2×2 grid of feature source boxes:
  - BLOSUM62 + Physicochemical: "Substitution scores, volume, hydrophobicity, charge"
  - ESM-2 Embeddings: "8M-param protein language model, 20 PCA components per gene"
  - EVE + MAVE Scores: "Evolutionary variant effect + functional assay data"
  - AlphaFold 3D + Conservation: "pLDDT, RSA, domain proximity, PhyloP"
- Highlight box: "Rule-based for nonsense/frameshift/splice. ML handles missense only. ACMG codes for every prediction."

**Right card: Visual Evidence**
- Title: "Visual Evidence"
- Two images side by side (embedded as base64 PNGs from visual_proofs/ directory):
  - Left image: `1_PerGene_ROC_Curves.png` — caption "Per-Gene ROC Curves"
  - Right image: `4_SHAP_Global_Beeswarm.png` — caption "SHAP Global Feature Importance"
- Caption below: "Left: AUC curves per gene show strong BRCA2 performance and data-scarcity effects. Right: SHAP beeswarm — BLOSUM62, PhyloP, and ESM-2 PCA among top contributors."

### Row 4 — Full-width card: Validation & Rigor
- Title: "Validation & Scientific Rigor"
- Two-column layout inside the card:

**Column 1: Internal & External Validation**
- Internal Validation:
  - 10-Fold Cross-Validation: AUC = 0.9797 ± 0.0031
  - Gene × label stratified split
  - 198 automated tests (ACMG combinatorial, clinical correctness, property-based Hypothesis)
- External Benchmark (n=2,234):
  - ProteinGym DMS (BRCA1) + ClinVar Expert Panel (ENIGMA)
  - Independent AUC: 0.750–0.801 (zero training overlap)
- Calibration & Uncertainty:
  - Isotonic calibration per gene for accurate probabilities
  - Bootstrap CI (1,000 iterations)
  - Conformal prediction sets for coverage guarantees
  - Data scarcity warnings for under-represented genes

**Column 2: Feature Leakage Assessment**
- AlphaMissense removed in v5.4: ΔAUC = +0.02 for BRCA1/PALB2/RAD51C — label leakage confirmed, features deleted
  - AlphaMissense (DeepMind, 2023) was partially trained on ClinVar labels — our training target
  - Removing it IMPROVED accuracy — smoking gun for leakage
  - 3 features deleted: am_score, am_pathogenic, am_x_phylop
- Per-gene ablation results:
  - BRCA1: 0.765 → 0.784 (+0.020)
  - PALB2: 0.570 → 0.586 (+0.015)
  - RAD51C: 0.799 → 0.817 (+0.018)
  - RAD51D: 0.821 → 0.815 (-0.007)
  - BRCA2: 0.997 → 0.997 (negligible)
- MAVE removed: ΔAUC = −0.0006 — minimal (3.5% coverage)

### Row 5 — Footer (full width)
- Compact: "steppedna.com" + Applied Science badge

---

## RIGHT WALL (130cm × 165cm) — "The Impact Wall"

Why this matters — the Kazakhstan equity story + clinical utility. Judges score "Practical Benefits to Society."

### Row 1 — Title Banner (full width)
- "SteppeDNA" + subtitle "Clinical Impact & Population Equity"
- Same author info

### Row 2 — Full-width card: The VUS Crisis
- Title: "The Problem: Variants of Uncertain Significance"
- Key message: When patients get genetic tests for cancer risk, >50% of missense variants come back as VUS — the lab can't tell if it's dangerous or harmless
- These 5 genes (BRCA1, BRCA2, PALB2, RAD51C, RAD51D) protect against cancer by repairing DNA. Broken = dramatically increased cancer risk
- If pathogenic → preventive surgery, PARP inhibitor drugs, enhanced screening
- If VUS → no actionable options, patient and doctor stuck
- SteppeDNA helps resolve VUS by providing probability scores with evidence

### Row 3 — Two cards side by side

**Left card: Kazakhstan Population Equity**
- Title: "Population Equity: Kazakhstan"
- Key stats (large, prominent):
  - 7 founder mutations integrated and searchable
  - 97.4% vs 87.7% PM2 flagging rate (EAS vs NFE) — 11% disparity
  - 1,260 excess PM2 flags for East Asian cohort vs European
  - 0% Kazakh-specific frequency data in global databases (gnomAD, ExAC)
- Mini table of top founder mutations:
  - BRCA1 c.5266dup (5382insC) — 3.5% frequency — most common Kazakh founder
  - BRCA2 c.9409dup — 1.2% frequency
  - BRCA2 c.9253del — 0.8% frequency
  - BRCA1 c.5278-2del — 0.8% frequency (splice)
  - BRCA1 c.181T>G (p.Cys61Gly) — 0.4% frequency (RING domain missense)
- Source: "Oncotarget 2023 (224 Kazakh women), Djansugurova et al. 2023"
- Narrative: "East Asian/Central Asian patients receive more VUS results than European patients for the same genes — purely because population frequency data is missing."

**Right card: Clinical Features**
- Title: "Clinical Decision Support"
- Feature list with brief descriptions:
  - SHAP Explanations — shows which features drove each prediction (why, not just what)
  - ACMG Evidence Codes — 11 of 28+ clinical classification codes implemented
  - Conformal Prediction — 90% coverage guarantee on prediction sets
  - Bootstrap 95% CI — confidence intervals from 1,000 model iterations
  - Gene Reliability Warnings — data scarcity tiers (High/Moderate/Low per gene)
  - Model Disagreement Alerts — when XGBoost and MLP disagree significantly
  - Live ClinVar + gnomAD Lookups — real-time database cross-reference
  - VCF Batch Upload — analyze multiple variants at once
  - PDF Clinical Reports — exportable prediction summaries
  - Trilingual UI — English, Kazakh, Russian
  - Research Use Only — prominent RUO disclaimer, not for clinical diagnosis

### Row 4 — Full-width card: Conclusions, Limitations, Future Work
- Title: "Conclusions & Future Work"
- Three-column grid:

**Column 1: Conclusions**
- ESM-2 protein language model features rank among top-20 global SHAP contributors (RQ2 confirmed)
- Per-gene isotonic calibration compensates for class imbalance across genes
- SHAP explanations + ACMG codes make predictions interpretable for clinicians
- AlphaMissense removed due to label leakage (ΔAUC +0.02 for non-BRCA2 genes)
- 7 Kazakh founder mutations correctly identified (2/2 testable = pathogenic)

**Column 2: Known Limitations**
- Population bias: training data predominantly European ancestry (ClinVar submission bias)
- Data imbalance: BRCA2 = 52% of data; RAD51C/RAD51D have <700 variants each
- PALB2 performance: REVEL beats SteppeDNA (0.732 vs 0.605)
- Temporal generalization: non-BRCA2 temporal AUCs 0.51–0.61
- Research Use Only: not validated for clinical diagnostic decisions

**Column 3: Future Directions**
- Upgrade ESM-2 to 650M parameters for richer protein embeddings
- Transfer learning: BRCA2 (0.994 AUC, 10K variants) → PALB2 (0.605 AUC, 2.6K variants)
- Clinical expert review and geneticist endorsement letter
- Population-aware ACMG thresholds using Kazakh WGS data
- Prospective validation: re-check ClinVar VUS predictions in September 2026
- Expand gnomAD benign mining for BRCA1/PALB2 class balance

### Row 5 — Footer (full width)
- Same compact footer: "steppedna.com" + badge

---

## Implementation Notes

### Technical
- All three posters are generated as standalone HTML files
- Rendered to PDF via Playwright (headless Chromium) using `posterv2_to_pdf.py`
- Visual proof images embedded as base64 data URIs (from `visual_proofs/` directory):
  - `1_PerGene_ROC_Curves.png`
  - `4_SHAP_Global_Beeswarm.png`
- QR code SVG for steppedna.com is already available (inline SVG path data)
- Architecture diagram is an inline SVG (no external images)
- SOTA bar chart is an inline SVG (no external images)

### File structure
- `poster/generate_wall_back.py` → `poster/wall_back.html` → `poster/wall_back.pdf`
- `poster/generate_wall_left.py` → `poster/wall_left.html` → `poster/wall_left.pdf`
- `poster/generate_wall_right.py` → `poster/wall_right.html` → `poster/wall_right.pdf`
- PDF conversion: `python poster/posterv2_to_pdf.py poster/wall_back.html --width 140cm --height 165cm`

### What the existing posterv2.html looks like (for reference)
- Dark background (#12121a)
- Cards with subtle dark borders on slightly lighter dark background
- Purple accent color throughout (headers, icons, metric values)
- Title in serif font (Playfair Display), everything else in Inter
- Rounded corners (32px radius) on all cards
- Subtle box shadows
- Purple gradient title banner
- Green for "best" scores, red for losses, yellow for warnings, blue for independent benchmarks
- Each card has a header with a circular purple gradient icon + title text, with a purple bottom border line
- Compact, dense, no wasted space
- Font sizes are all proportional and readable

### Additional deliverables (not in this brief)
- Flyer/brochure for jury takeaway (A4 or tri-fold)
- Top-of-stand abstract card (names, ages, school, city, region)
- Presentation slides (10-min formal interview)
