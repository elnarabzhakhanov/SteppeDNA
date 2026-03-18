# SteppeDNA: Roadmap to ICOR/ISEF-Level Project

## The Honest Diagnosis: Where You Stand Right Now

Your project is **technically impressive** but has a critical identity problem that would hold it back at Infomatrix Asia or any top-tier competition. Here's the brutal truth:

### What's Strong
- Production-quality software (Docker, CI/CD, PWA, i18n, 198 tests)
- Real ML pipeline with XGBoost + MLP ensemble, SHAP explanations, ACMG codes
- Honest validation (temporal splits, gold-standard benchmarks, SOTA comparison)
- Multilingual support (EN/KK/RU) — unique differentiator

### What's Holding You Back

1. **No novel scientific contribution.** You built a well-engineered *tool*, but you didn't *discover* anything. ICOR won at ISEF because Rishab Jain didn't just build software — he proved his RNN outperformed existing codon optimization methods on *actual E. coli protein expression* and got published in BMC Bioinformatics. Krish Lal's Microby discovered two new plastic-eating bacteria. Judges want: "What did the world learn from your work that it didn't know before?"

2. **BRCA2 carries everything.** Your AUC of 0.978 is inflated — BRCA2 is 52% of your test set with 0.983 AUC. Your macro-averaged AUC is 0.775, and temporal validation shows BRCA1/PALB2 near-random (0.527, 0.513). A sharp judge will see this immediately.

3. **No population-level novelty.** Despite having Kazakh language support, there's zero Kazakh-specific genomic analysis. This is a missed opportunity — the literature on Kazakh population genetics is *thin*, meaning there's actual room for novel findings.

4. **Competing against established tools.** REVEL, AlphaMissense, ClinVar already exist for variant pathogenicity. Your advantage over them on your own test set (0.978 vs 0.725) is partly due to training distribution overlap.

---

## The Pivot: What ICOR-Level Actually Means

ICOR-level projects share these traits:

| Trait | ICOR (Rishab Jain) | Microby (Krish Lal) | ISEF CBIO 2024 Winner |
|-------|--------------------|--------------------|----------------------|
| **Novel method** | RNN for codon optimization (first of its kind) | ML to predict plastic-degrading microbes | Novel drug target identification scheme |
| **Real discovery** | Outperformed existing methods on protein expression | Found 2 new plastic-eating bacteria | Identified CtBP2 gene mechanism |
| **Published** | BMC Bioinformatics | — | — |
| **Testable claim** | "ICOR-optimized codons produce X% more protein" | "Microby predicts plastic degradation with 92% accuracy" | "Phenothiazines inhibit CtBP2 in AD" |
| **Beyond software** | Validated with wet-lab data | Validated predictions experimentally | Validated with gene expression data |

**The pattern:** Build something novel → use it to discover something → validate it rigorously → tell a compelling story.

---

## Three Strategic Options (Pick One)

### Option A: "SteppeDNA-KZ" — Kazakh Population Genomics (RECOMMENDED)

**Thesis:** *"Population-specific variant pathogenicity prediction reveals classification disparities and potential founder mutations in the Kazakh population"*

**Why this wins:**
- Fills a **genuine gap** in the literature — there is almost no computational genomics work specific to Kazakh/Central Asian populations
- Directly addresses a **hot topic** in genomics: VUS disparities in non-European populations (published in Pediatrics 2025, Genome Medicine 2024)
- You already have the infrastructure (multilingual, ACMG codes, ClinVar integration)
- Scientifically **novel** — nobody has built a population-aware variant classifier for Central Asians
- **Locally relevant** at Infomatrix Asia (Kazakhstan-hosted competition)
- Aligns with real published research: Kazakh founder mutations BRCA1 c.5266dup, BRCA2 c.9409dup identified in 2023

**What you'd actually build and discover:**

1. **Population-Adjusted Allele Frequency Module**
   - Pull Kazakh-specific allele frequencies from the 2022 Whole-Genome Sequencing study of Kazakh individuals (Frontiers in Genetics)
   - Recalibrate BA1/BS1 ACMG thresholds using Central Asian population data instead of gnomAD global frequencies
   - **Novel finding:** Show how many variants currently classified as "rare" (and therefore suspicious) in gnomAD are actually common in Kazakhs — and therefore likely benign

2. **Founder Mutation Database**
   - Compile the 38 pathogenic variants from the 2023 Oncotarget study on 224 Kazakh women
   - Integrate the 12 recurrent variants (BRCA1 c.5266dup, c.5278-2del, c.2T>C; BRCA2 c.9409dup, c.9253del)
   - Build a "Kazakh Founder Mutation Panel" feature in SteppeDNA
   - **Novel finding:** Cross-reference these with your model's predictions — do any get misclassified? This reveals model bias

3. **VUS Disparity Analysis**
   - Take all VUS in ClinVar for your 5 genes
   - Stratify by population origin where available
   - Show that Central Asian populations have disproportionately more VUS
   - **Novel finding:** Quantify the "classification gap" for Kazakh patients vs. European patients

4. **Population-Aware Prediction**
   - Add population as a feature to your model (or build a population-specific calibration layer)
   - Show that population-aware predictions reduce VUS rates for Kazakh patients
   - **Novel finding:** "Population-calibrated SteppeDNA reclassifies X% of Kazakh VUS"

**Presentation narrative:** *"70% of genomic databases represent European populations. A Kazakh patient receiving genetic testing today faces 2-3x more inconclusive results. SteppeDNA-KZ is the first tool to address this disparity for Central Asian populations, and in doing so, we discovered [X founder mutations / Y misclassified variants / Z reclassifiable VUS]."*

---

### Option B: "SteppeDNA-ACMG-KZ" — Localized ACMG Framework

**Thesis:** *"Adapting ACMG/AMP variant classification guidelines for Central Asian populations: a computational framework and empirical validation"*

**Why this could work:**
- UK has ACGS (localized ACMG). No Central Asian equivalent exists.
- You already implement ~10 ACMG criteria — extend to all 28
- Population-specific thresholds are a recognized need

**What you'd build:**
1. Full 28-criteria ACMG engine with population-adjustable thresholds
2. Central Asian allele frequency calibration
3. Comparison: standard ACMG vs KZ-ACMG on Kazakh patient variants
4. Show how many variants change classification with localized thresholds

**Risk:** More incremental, less "wow factor." Better as a component of Option A.

---

### Option C: "Cross-Gene Transfer Learning" — Pure ML Novelty

**Thesis:** *"Transfer learning from data-rich to data-scarce genes improves variant pathogenicity prediction for understudied cancer genes"*

**Why this could work:**
- Your BRCA2 model is strong (0.983 AUC) but PALB2 is weak (0.641)
- BRCA2 and PALB2 are functionally related (PALB2 is literally "Partner And Localizer of BRCA2")
- Could you transfer BRCA2 knowledge to improve PALB2?

**What you'd build:**
1. Domain-adaptive transfer learning: pretrain on BRCA2, fine-tune on PALB2
2. Show improvement over training PALB2 from scratch
3. Validate on held-out temporal set

**Risk:** You already have `cross_gene_transfer.py` — if the results aren't good, this won't work. Also less unique since transfer learning is well-established.

---

## Recommended Path: Option A with elements of B

**Rebrand: SteppeDNA → "SteppeDNA: Population-Aware Variant Classification for Central Asian Genomics"**

---

## Implementation Roadmap

### Phase 1: Data Foundation (Week 1-2)

- [ ] **Download and process Kazakh WGS data**
  - Source: Frontiers in Genetics 2022 study — supplementary data
  - Extract Central Asian allele frequencies
  - Build `data/kazakh_population_frequencies.csv`

- [ ] **Compile Kazakh founder mutation database**
  - Source: Oncotarget 2023 — 38 pathogenic variants in 13 genes
  - Source: CAJGH editorial on BRCA1/BRCA2 in Kazakhstan
  - Cross-reference with ClinVar entries
  - Build `data/kazakh_founder_mutations.json`

- [ ] **Collect VUS data stratified by ancestry**
  - Query ClinVar for all VUS in BRCA1/BRCA2/PALB2/RAD51C/RAD51D
  - Tag with submitter country/population where available
  - Build `data/vus_population_stratification.csv`

### Phase 2: Population-Aware Engine (Week 2-3)

- [ ] **Recalibrate ACMG population criteria**
  - Modify `backend/acmg_rules.py` to accept population parameter
  - Implement KZ-specific thresholds for BA1, BS1, PM2 (allele frequency codes)
  - Add PS4 (case-control) with Kazakh cohort data
  - Add population-specific hotspot regions

- [ ] **Build Founder Mutation Module**
  - New endpoint: `/predict/founder-check`
  - Flag known Kazakh founder mutations with population-specific OR/RR
  - Visual indicator in frontend for founder mutations

- [ ] **Population feature in ML model**
  - Add population-origin feature to feature engineering
  - Or: build population-specific calibration layer (isotonic per population)
  - Retrain and validate

### Phase 3: The Discovery (Week 3-4) — THIS IS WHAT WINS

- [ ] **VUS Disparity Quantification**
  - Run all Kazakh-origin VUS through standard SteppeDNA vs. SteppeDNA-KZ
  - Measure: how many VUS get reclassified with population-aware thresholds?
  - Generate figure: "VUS Reclassification Rate by Population"
  - **This is your paper figure 1**

- [ ] **Misclassification Analysis**
  - Run Kazakh founder mutations through major tools (REVEL, CADD, AlphaMissense)
  - Show which tools misclassify population-specific variants
  - Show SteppeDNA-KZ correctly classifies them
  - **This is your paper figure 2**

- [ ] **Allele Frequency Recalibration Impact**
  - Show how many variants shift from "rare" to "common" when using KZ frequencies vs. gnomAD global
  - Quantify the clinical impact (potential misdiagnosis avoidance)
  - **This is your paper figure 3**

### Phase 4: Validation & Presentation Polish (Week 4-5)

- [ ] **Independent validation**
  - Use the 224-patient Kazakh cohort as an independent test set
  - Report sensitivity/specificity/AUC on this cohort specifically
  - Compare standard vs. population-aware classification

- [ ] **Complete ACMG criteria**
  - Expand from ~10 to 20+ criteria
  - Document which criteria have population-specific adjustments
  - Add a "classification comparison" view (standard vs. KZ-adapted)

- [ ] **Presentation materials**
  - Before/after comparison dashboard
  - Real case study: "Patient X has variant Y, classified VUS by standard tools, reclassified as Likely Benign by SteppeDNA-KZ"
  - Impact statement with numbers

### Phase 5: Publication-Ready Documentation (Week 5-6)

- [ ] **Write up as a short paper / preprint**
  - Title: "SteppeDNA-KZ: Population-Aware Variant Classification Reveals Disparities in Central Asian Genomics"
  - Submit to bioRxiv or medRxiv as preprint
  - **Having a preprint URL on your poster is ISEF-tier credibility**

- [ ] **Update MODEL_CARD.md and VALIDATION_REPORT.md**
  - Add population-specific validation section
  - Document Kazakh founder mutation database
  - Add VUS disparity analysis results

---

## What the Final Project Looks Like at Presentation

### Title Slide
**"SteppeDNA-KZ: Bridging the Genomic Divide — Population-Aware Variant Classification for Central Asian Hereditary Cancer"**

### The Story (5-minute pitch)

> "70% of genomic data comes from European populations. When a Kazakh patient gets genetic testing for hereditary breast cancer, they receive 2-3x more inconclusive 'variant of uncertain significance' results than a European patient. This means their doctors can't tell them if they carry a dangerous mutation.
>
> We built SteppeDNA-KZ — the first variant pathogenicity classifier calibrated for Central Asian populations. By integrating Kazakh-specific allele frequencies, founder mutation data from recent NGS studies, and population-adjusted ACMG classification criteria, we:
>
> 1. **Discovered** that [X]% of Kazakh VUS can be reclassified using population-aware thresholds
> 2. **Identified** [Y] variants commonly misclassified by existing tools due to European-centric training data
> 3. **Validated** our approach on an independent cohort of 224 Kazakh breast cancer patients
> 4. **Built** an open-source, multilingual platform accessible to Central Asian clinicians and researchers
>
> Our tool doesn't just predict pathogenicity — it quantifies and corrects the classification bias that affects 4 billion people in underrepresented populations."

### Demo Flow
1. Enter a Kazakh founder mutation → show it's correctly flagged with population context
2. Enter a VUS → show standard classification vs. KZ-adjusted classification
3. Show the disparity dashboard: VUS rates by population
4. Show 3D protein structure with mutation highlighted
5. Generate PDF report in Kazakh language

### Poster/Slides Key Figures
1. **Bar chart:** VUS rates by population (European vs. Central Asian vs. Global)
2. **Sankey diagram:** VUS reclassification flow (VUS → Likely Benign / Likely Pathogenic)
3. **ROC curves:** Standard SteppeDNA vs. SteppeDNA-KZ on Kazakh cohort
4. **Map:** Genomic data representation by country (Kazakhstan highlighted as underrepresented)
5. **Table:** Founder mutations with SteppeDNA-KZ vs. REVEL vs. CADD classification

---

## Why This Beats Your Current Version at Every Level

| Judging Criterion | Current SteppeDNA | SteppeDNA-KZ |
|---|---|---|
| **Novelty** | Another variant classifier | First population-aware classifier for Central Asia |
| **Discovery** | None — tool only | VUS disparity quantification, reclassification rates |
| **Real-world impact** | Generic research tool | Addresses documented health equity gap |
| **Scientific rigor** | Good validation | Independent cohort validation + population analysis |
| **Local relevance** | Has Kazakh UI | Has Kazakh UI + Kazakh genomic data + Kazakh founder mutations |
| **Publishable?** | No (incremental) | Yes (novel population-specific analysis) |
| **Judge question: "What did you discover?"** | "Our model has 0.978 AUC" | "X% of Kazakh VUS are reclassifiable; Y founder mutations are misclassified by standard tools" |

---

## Key Literature to Cite

1. **Kazakh breast cancer genetics:** Determination of genetic predisposition to early breast cancer in women of Kazakh ethnicity (Oncotarget 2023)
2. **Kazakh WGS:** Whole-Genome Sequencing and Genomic Variant Analysis of Kazakh Individuals (Frontiers in Genetics 2022)
3. **VUS disparities:** Germline Genetic Variant Classification Requires More Equitable Reference Database Representation (Pediatrics 2025)
4. **MAVE for equity:** Using multiplexed functional data to reduce variant classification inequities in underrepresented populations (Genome Medicine 2024)
5. **ACMG guidelines:** Standards and Guidelines for the Interpretation of Sequence Variants (ACMG/AMP 2015)
6. **Classification disparities:** Defining and Reducing Variant Classification Disparities (PMC 2024)

---

## What NOT to Do

- **Don't** keep framing this as "we built a better classifier" — that's an engineering project, not science
- **Don't** lead with your AUC numbers — judges know these are inflated by BRCA2 dominance
- **Don't** ignore your weak genes — instead, explain WHY they're weak (data scarcity in non-European populations) and make that your research question
- **Don't** treat the Kazakh angle as decoration — make it the scientific core
- **Don't** try to compete with AlphaMissense/REVEL on their terms — compete on *your* terms (population specificity)

---

## Summary

Your current project is a well-built car without a destination. The Kazakh population genomics pivot gives it a destination that is:
- **Scientifically novel** (no one has done this)
- **Socially impactful** (health equity for underrepresented populations)
- **Locally relevant** (Infomatrix Asia is in Kazakhstan)
- **Publishable** (preprint-worthy)
- **Defensible** (backed by real literature and data)

This is how you go from "impressive software project" to "ICOR-level science project."
