# Prompt: Generate 3 Competition Poster HTMLs for SteppeDNA

You are a world-class scientific poster designer with expertise in CSS Grid, SVG data visualization, and large-format print design. You will generate three standalone HTML files — one for each wall of a competition booth — for a high school science project called SteppeDNA. These posters will be printed at large format and displayed at Infomatrix Asia 2026 in Astana, Kazakhstan.

The stand design is scored (up to 10% of total score), so aesthetic quality, visual cohesion, readability, and professional layout are critical.

---

## YOUR TASK

Generate THREE standalone HTML files (no external dependencies except Google Fonts). Each file must:
1. Be a single self-contained HTML file with inline CSS and inline SVGs
2. Render correctly at the specified physical dimensions when opened in a browser and printed/PDF'd
3. Use the exact design system specified below — all three walls must look like they belong together
4. Fill the entire poster area with content — NO large empty voids or dead space
5. Be readable from 3 meters away — large fonts, minimal dense text, visual hierarchy
6. Use dark theme throughout

---

## CRITICAL DESIGN REQUIREMENTS

### What went wrong in previous attempts (DO NOT repeat these mistakes):
- **Empty voids:** Grid rows with `1fr` expanded but card content didn't fill the space, leaving massive dark empty areas. FIX: Either use fixed/proportional heights that match content, or ensure content (SVGs, tables, flex children) actually stretches to fill available space using `flex:1`, `height:100%`, etc.
- **Disproportionate elements:** Some elements were tiny (footer badges, table text, captions) while others were oversized (title banner, metric boxes). FIX: Keep all elements within a narrow, cohesive size range. Look at the reference poster's proportions.
- **Content not filling cards:** SVGs rendered at fixed small size inside large cards. FIX: SVGs should use `width:100%` and expand naturally within their flex containers. Don't set `height:auto` on SVGs inside flex containers — let them grow.
- **Chart and table side-by-side with bad proportions:** The SOTA table was microscopic next to a huge chart. FIX: Give the table card enough width (at least 1/3) and use readable font sizes (0.55rem+ for table cells).

### What works well (the reference poster posterv2.html):
- 4-column CSS grid with cards spanning 1-2 columns
- Consistent card sizes — no card is dramatically larger or smaller than others
- Cards are dense but not cramped — content fills each card naturally
- All font sizes are proportional and readable at poster scale
- Purple accent color ties everything together
- Metric boxes are compact, not oversized
- Title banner is slim, not a huge empty gradient block
- Footer badges are readable

---

## DESIGN SYSTEM (use exactly)

### Page Setup
```css
@page { size: [WIDTH] [HEIGHT]; margin: 0; }
html { font-size: 45px; } /* All rem values scale from this */
body { font-family: 'Inter', -apple-system, sans-serif; background: #12121a; color: #c8c8e0; line-height: 1.75; }
```
Google Fonts import: `Inter:wght@300;400;500;600;700;800;900` and `Playfair+Display:wght@700;800`

### CSS Variables
```css
:root {
  --bg: #12121a;
  --card-bg: #1e1e2d;
  --feat-bg: #2a2a3e;
  --track-bg: #31314a;
  --border: #31314a;
  --primary: #8382ff;
  --primary-dark: #6260FF;
  --text-main: #e8e8f2;
  --text-body: #c8c8e0;
  --text-muted: #8888aa;
  --success: #10B981;
  --danger: #ef6565;
  --warning: #fbbf24;
  --bench: #38bdf8;
  --radius: 32px;
  --shadow: 0 12px 40px -12px rgba(0,0,0,0.4);
  --hero-gradient: linear-gradient(135deg, #1a1a2e 0%, #2a2855 40%, #3d3a8a 70%, #6260FF 100%);
}
```

### Card Style
```css
.card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.2rem 1.5rem;
  box-shadow: var(--shadow);
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  overflow: hidden;
}
.card-header {
  display: flex; align-items: center; gap: 0.5rem;
  margin-bottom: 0.25rem; padding-bottom: 0.3rem;
  border-bottom: 2px solid var(--primary);
}
```
Each card header has a circular icon (purple gradient circle with white SVG icon inside) + title text.

### Typography Scale (in rem, base 45px)
| Element | Size | Weight | Color |
|---------|------|--------|-------|
| Poster title ("SteppeDNA") | 3rem | 800 | white, Playfair Display |
| Subtitle | 0.85rem | 400 | rgba(255,255,255,0.85) |
| Card title | 1.05rem | 700 | --text-main |
| Body text | 0.78rem | 400 | --text-body |
| Metric value | 1rem | 800 | --primary |
| Metric label | 0.52rem | 600 | --text-muted, uppercase |
| Table cell | 0.65rem | 400 | --text-body |
| Table header | 0.6rem | 700 | --text-main, uppercase |
| Caption/footnote | 0.48rem | 400 | --text-muted |
| Footer badge value | 0.65rem | 700 | --text-main |
| Footer badge label | 0.5rem | 600 | --text-muted, uppercase |

### Metric Box
```css
.metric-box {
  flex: 1; min-width: 5rem;
  background: var(--feat-bg);
  border-radius: 12px;
  padding: 0.5rem 0.6rem;
  text-align: center;
}
```

### Highlight Box (for key findings)
```css
.highlight-box {
  background: linear-gradient(135deg, rgba(131,130,255,0.12) 0%, rgba(98,96,255,0.06) 100%);
  border: 1px solid rgba(131,130,255,0.3);
  border-radius: 12px;
  padding: 0.5rem 0.8rem;
}
```

### Warning/Caveat Box
```css
.caveat {
  background: rgba(251,191,36,0.08);
  border: 1px solid rgba(251,191,36,0.25);
  border-radius: 12px;
  padding: 0.25rem 0.5rem;
}
```

### Independent Benchmark Box
```css
.bench-box {
  background: rgba(56,189,248,0.08);
  border: 1px solid rgba(56,189,248,0.25);
  border-radius: 12px;
  padding: 0.35rem 0.5rem;
  text-align: center;
}
/* Value: font-size 1rem, font-weight 800, color #38bdf8 */
/* Label: font-size 0.48rem, uppercase, color rgba(56,189,248,0.7) */
```

### Table Style
```css
.data-table {
  width: 100%; border-collapse: separate; border-spacing: 0;
  font-size: 0.65rem;
}
.data-table th {
  background: var(--feat-bg); color: var(--text-main);
  font-weight: 700; padding: 0.4rem 0.6rem;
  text-transform: uppercase; letter-spacing: 0.04em;
}
.data-table td { padding: 0.4rem 0.6rem; border-bottom: 1px solid var(--border); }
.best { color: #10B981; font-weight: 700; } /* winning scores */
```

### Color Coding
- **Green (#10B981):** Best/winning scores, success states
- **Red (#ef6565):** Losing scores, danger states
- **Yellow (#fbbf24):** Warnings, caveats
- **Blue (#38bdf8):** Independent benchmarks
- **Purple (#8382ff):** Brand accent, our metric values, card headers

---

## POSTER 1: BACK WALL (140cm wide × 165cm tall) — "The Hero Wall"

**File:** `wall_back.html`
**Purpose:** First thing judges see. Must hook them in 5 seconds. Big numbers, clear results, system overview.
**Grid:** 3 columns (or 4 if it fits better), 5 rows. Use CSS Grid.

### Content (top to bottom):

**Row 1 — Title Banner (spans full width)**
- Hero gradient background
- Left: "SteppeDNA" in Playfair Display (large) + subtitle "Multi-Gene Variant Pathogenicity Classifier for HR DNA Repair Genes"
- Right: "Elnar Abzhakhanov, 16" / "Inkar Saule Abzhakhanova, 16" / "K. Satpayev Binom-School, Astana" / "Mentor: Darkhan Shadykul" / pill badge "APPLIED SCIENCE • INFOMATRIX ASIA 2026"

**Row 2 — Five Metric Boxes (spans full width, horizontal row)**
| Value | Label | Sublabel |
|-------|-------|----------|
| 0.985 | OVERALL AUC | sample-weighted |
| 0.791 | MACRO AUC | equal per gene |
| 0.928 | MCC | balanced metric |
| 96.5% | BALANCED ACC | sensitivity + specificity |
| 4/5 | GENES BEAT SOTA | vs REVEL, BayesDel, CADD |

**Row 3 — Main content, two cards**

*Left card (2/3 width): "Per-Gene AUC: SteppeDNA vs. State of the Art"*
- Inline SVG grouped bar chart. 5 gene groups × 4 bars each.
- Data:
  - BRCA2: SteppeDNA=0.994, REVEL=0.891, BayesDel=0.949, CADD=0.908
  - RAD51D: 0.824, 0.461, 0.448, 0.457
  - RAD51C: 0.785, 0.651, 0.634, 0.703
  - BRCA1: 0.747, 0.595, 0.646, 0.527
  - PALB2: 0.605, 0.732, 0.432, 0.564
- Colors: SteppeDNA=#8382ff, REVEL=#10B981, BayesDel=#fbbf24, CADD=#ef6565
- Y-axis 0.4 to 1.0, gene names below, legend at bottom, SteppeDNA value labels above purple bars
- Below chart: highlight box with key finding text: "A single multi-gene ensemble with gene-adaptive calibration achieves competitive or superior performance to established tools on 4 of 5 HR repair genes, while providing calibrated probabilities, SHAP explanations, and ACMG evidence codes."

*Right card (1/3 width): "AUC Comparison"*
- Subtitle: "HELD-OUT TEST SET (N=3,845)"
- Comparison table: Gene | Ours | REVEL | BayesDel | CADD (5 rows, green for best, red for loss on PALB2)
- Two blue benchmark boxes: 0.801 ClinVar Expert Panel | 0.750 ProteinGym DMS
- Yellow caveat: "Independent validation — zero training overlap. SOTA evaluated on our test set (home-court advantage)."

**Row 4 — Architecture Card (full width)**
- "System Architecture" header
- Inline SVG horizontal flow: User → Frontend (HTML/JS, Dark Mode, Trilingual, SHAP Viz, PDF Export) → FastAPI Backend (containing: Feature Engine 120 features ESM-2•AlphaFold, ML Ensemble XGB+MLP Per-gene weights, External APIs ClinVar•gnomAD, ACMG Engine Rule-based codes PVS1•PM1•BS1) → Clinical Report (green border: P(pathogenic)+95% CI, SHAP explanations, ACMG evidence codes, Conformal prediction set, Data scarcity warnings)

**Row 5 — Footer Banner (full width)**
- Tech badges: ML/DL (XGBoost•TensorFlow•ESM-2) | Backend (Python•FastAPI•BioPython) | Data (19,223 variants•5 genes•120 features) | Features (SHAP•ACMG•VCF•PDF•Trilingual)
- QR code SVG (for steppedna.com) + "steppedna.com" text

QR code SVG path data (paste this exactly):
```
<svg width="100%" height="100%" viewBox="0 0 27 27" xmlns="http://www.w3.org/2000/svg"><path d="M1,1H2V2H1zM2,1H3V2H2zM3,1H4V2H3zM4,1H5V2H4zM5,1H6V2H5zM6,1H7V2H6zM7,1H8V2H7zM11,1H12V2H11zM13,1H14V2H13zM15,1H16V2H15zM17,1H18V2H17zM19,1H20V2H19zM20,1H21V2H20zM21,1H22V2H21zM22,1H23V2H22zM23,1H24V2H23zM24,1H25V2H24zM25,1H26V2H25zM1,2H2V3H1zM7,2H8V3H7zM9,2H10V3H9zM11,2H12V3H11zM14,2H15V3H14zM15,2H16V3H15zM17,2H18V3H17zM19,2H20V3H19zM25,2H26V3H25zM1,3H2V4H1zM3,3H4V4H3zM4,3H5V4H4zM5,3H6V4H5zM7,3H8V4H7zM9,3H10V4H9zM10,3H11V4H10zM12,3H13V4H12zM13,3H14V4H13zM16,3H17V4H16zM19,3H20V4H19zM21,3H22V4H21zM22,3H23V4H22zM23,3H24V4H23zM25,3H26V4H25zM1,4H2V5H1zM3,4H4V5H3zM4,4H5V5H4zM5,4H6V5H5zM7,4H8V5H7zM9,4H10V5H9zM10,4H11V5H10zM11,4H12V5H11zM13,4H14V5H13zM14,4H15V5H14zM17,4H18V5H17zM19,4H20V5H19zM21,4H22V5H21zM22,4H23V5H22zM23,4H24V5H23zM25,4H26V5H25zM1,5H2V6H1zM3,5H4V6H3zM4,5H5V6H4zM5,5H6V6H5zM7,5H8V6H7zM11,5H12V6H11zM12,5H13V6H12zM13,5H14V6H13zM17,5H18V6H17zM19,5H20V6H19zM21,5H22V6H21zM22,5H23V6H22zM23,5H24V6H23zM25,5H26V6H25zM1,6H2V7H1zM7,6H8V7H7zM10,6H11V7H10zM11,6H12V7H11zM12,6H13V7H12zM13,6H14V7H13zM14,6H15V7H14zM19,6H20V7H19zM25,6H26V7H25zM1,7H2V8H1zM2,7H3V8H2zM3,7H4V8H3zM4,7H5V8H4zM5,7H6V8H5zM6,7H7V8H6zM7,7H8V8H7zM9,7H10V8H9zM11,7H12V8H11zM13,7H14V8H13zM15,7H16V8H15zM17,7H18V8H17zM19,7H20V8H19zM20,7H21V8H20zM21,7H22V8H21zM22,7H23V8H22zM23,7H24V8H23zM24,7H25V8H24zM25,7H26V8H25zM9,8H10V9H9zM11,8H12V9H11zM12,8H13V9H12zM13,8H14V9H13zM15,8H16V9H15zM16,8H17V9H16zM17,8H18V9H17zM1,9H2V10H1zM7,9H8V10H7zM9,9H10V10H9zM10,9H11V10H10zM11,9H12V10H11zM13,9H14V10H13zM14,9H15V10H14zM15,9H16V10H15zM18,9H19V10H18zM19,9H20V10H19zM22,9H23V10H22zM23,9H24V10H23zM24,9H25V10H24zM5,10H6V11H5zM8,10H9V11H8zM13,10H14V11H13zM14,10H15V11H14zM15,10H16V11H15zM16,10H17V11H16zM18,10H19V11H18zM20,10H21V11H20zM21,10H22V11H21zM22,10H23V11H22zM23,10H24V11H23zM24,10H25V11H24zM1,11H2V12H1zM3,11H4V12H3zM4,11H5V12H4zM5,11H6V12H5zM7,11H8V12H7zM9,11H10V12H9zM10,11H12V12H10zM12,11H13V12H12zM14,11H15V12H14zM15,11H16V12H15zM16,11H17V12H16zM18,11H19V12H18zM19,11H20V12H19zM20,11H21V12H20zM21,11H22V12H21zM22,11H23V12H22zM24,11H25V12H24zM25,11H26V12H25z" fill="#000"/></svg>
```
(Note: this is truncated — use the full QR path from the codebase or generate a QR for https://steppedna.com)

---

## POSTER 2: LEFT WALL (130cm wide × 165cm tall) — "The Science Wall"

**File:** `wall_left.html`
**Purpose:** Methodology deep-dive. Research questions, dataset, features, validation, AlphaMissense leakage discovery.
**Grid:** 2 columns, 5 rows.

### Content:

**Row 1 — Title Banner (full width):** "SteppeDNA" + "Methodology & Validation" + author info

**Row 2 — Two cards:**
- Left: Research Questions (RQ1-RQ4, each bold question + 1-line answer)
  - RQ1: Can a single ML model classify missense variants across multiple HR genes? → "One universal model on all 5 genes with gene-adaptive calibration."
  - RQ2: Do protein language model embeddings improve classification? → "ESM-2 (8M params) generates per-gene embeddings reduced to 20 PCA components."
  - RQ3: Can we provide calibrated confidence with transparent limitations? → "Isotonic calibration, bootstrap CI, and conformal prediction sets."
  - RQ4: How does SteppeDNA compare to existing tools? → "Benchmark against REVEL, BayesDel, CADD on held-out test data."
- Right: Dataset & Training — metric boxes (19,223 / 5 / 120), horizontal bar chart (BRCA2=10,085 / BRCA1=5,432 / PALB2=2,621 / RAD51C=675 / RAD51D=410), caption about sources and split

**Row 3 — Two cards:**
- Left: Feature Engineering — 2×2 grid boxes (BLOSUM62+Physicochemical / ESM-2 Embeddings / EVE+MAVE / AlphaFold+Conservation) + highlight about rule-based vs ML
- Right: Visual Evidence — two placeholder image boxes labeled "Per-Gene ROC Curves" and "SHAP Global Beeswarm" (use colored placeholder rectangles with labels since we can't embed actual PNGs in this prompt — the developer will replace with base64 images)

**Row 4 — Full-width card: Validation & Scientific Rigor** (two-column internal layout)
- Column 1: Internal validation (10-fold CV 0.9797±0.0031, 198 tests, stratified split), External benchmark (ProteinGym DMS + ClinVar EP, AUC 0.750-0.801), Calibration (isotonic per-gene, bootstrap CI, conformal prediction)
- Column 2: AlphaMissense Leakage Discovery — AM removed in v5.4, ΔAUC +0.02 for BRCA1/PALB2/RAD51C, per-gene ablation table (BRCA1: 0.765→0.784, PALB2: 0.570→0.586, RAD51C: 0.799→0.817, RAD51D: 0.821→0.815, BRCA2: 0.997→0.997), MAVE ablation (ΔAUC -0.0006)

**Row 5 — Footer:** compact "steppedna.com"

---

## POSTER 3: RIGHT WALL (130cm wide × 165cm tall) — "The Impact Wall"

**File:** `wall_right.html`
**Purpose:** Why this matters. VUS crisis, Kazakhstan equity story, clinical features, conclusions. Judges score "Practical Benefits to Society."
**Grid:** 2 columns, 5 rows.

### Content:

**Row 1 — Title Banner:** "SteppeDNA" + "Clinical Impact & Population Equity" + author info

**Row 2 — Full-width card: The VUS Crisis**
- >50% of missense variants are VUS — lab can't tell if dangerous or harmless
- These 5 genes protect against cancer by repairing DNA
- Pathogenic → preventive surgery, PARP inhibitors, enhanced screening
- VUS → no actionable options, patient stuck
- SteppeDNA resolves VUS with probability scores + evidence

**Row 3 — Two cards:**
- Left: Kazakhstan Population Equity
  - 7 founder mutations integrated
  - 97.4% vs 87.7% PM2 flagging (EAS vs NFE) — 11% disparity
  - 1,260 excess PM2 flags for non-European patients
  - 0% Kazakh-specific data in global databases
  - Mini founder mutation table (BRCA1 c.5266dup 3.5%, BRCA2 c.9409dup 1.2%, BRCA2 c.9253del 0.8%, BRCA1 c.5278-2del 0.8%, BRCA1 c.181T>G 0.4%)
  - Source: Oncotarget 2023, Djansugurova et al. 2023
- Right: Clinical Decision Support — feature list (SHAP, ACMG 11 codes, conformal prediction, bootstrap CI, gene reliability warnings, model disagreement alerts, live ClinVar+gnomAD, VCF batch, PDF reports, trilingual EN/KK/RU, RUO disclaimer)

**Row 4 — Full-width card: Conclusions & Future Work** (3-column grid)
- Conclusions: ESM-2 top-20 SHAP, per-gene calibration works, AM leakage caught, 7 KZ founders identified
- Limitations: European training bias, BRCA2=52%, PALB2 trails REVEL, temporal AUCs 0.51-0.61, RUO
- Future: ESM-2 650M, transfer learning BRCA2→PALB2, clinical expert review, population-aware thresholds, prospective validation Sept 2026

**Row 5 — Footer:** compact "steppedna.com"

---

## FINAL CHECKLIST

Before outputting, verify each poster against these criteria:
- [ ] No large empty voids — every grid cell has content that fills it
- [ ] Font sizes are consistent and proportional across all three posters
- [ ] All three posters use identical CSS variables, card styles, and typography
- [ ] Title banners are slim (not giant empty gradient blocks)
- [ ] Metric boxes are compact (not oversized with excessive padding)
- [ ] Tables are readable (0.55rem+ cell text)
- [ ] Footer badges are readable (0.5rem+ text)
- [ ] SVG charts fill their containers naturally
- [ ] Cards have consistent padding and border-radius
- [ ] Green for wins, red for losses, yellow for caveats, blue for benchmarks, purple for brand
- [ ] All data values are exactly as specified (no rounding, no changes)
- [ ] Grid gap is consistent (~1.5rem) across all posters
- [ ] The three walls would look cohesive displayed side by side in a booth

Output each poster as a complete HTML file wrapped in a code block, clearly labeled.
