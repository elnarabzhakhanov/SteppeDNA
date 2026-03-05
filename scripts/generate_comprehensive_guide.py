"""
SteppeDNA — Comprehensive Project Guide (PDF Generator)
========================================================
Generates a richly-formatted PDF document that explains every aspect
of the SteppeDNA project: architecture, codebase metrics, ML pipeline,
features, external data, APIs, frontend, tests, and deployment.

Usage:
    python scripts/generate_comprehensive_guide.py
"""

import os, sys, textwrap, datetime

# ── PDF library ──────────────────────────────────────────────────────────────
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm, cm
    from reportlab.lib.colors import HexColor, white, black
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, KeepTogether, HRFlowable,
    )
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
except ImportError:
    print("reportlab not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm, cm
    from reportlab.lib.colors import HexColor, white, black
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, KeepTogether, HRFlowable,
    )

# ── Colours ──────────────────────────────────────────────────────────────────
BRAND       = HexColor("#6260FF")
BRAND_DARK  = HexColor("#4e4cdf")
BRAND_LIGHT = HexColor("#d9d9fc")
DARK_BG     = HexColor("#1a1a2e")
TEXT_DARK    = HexColor("#222244")
TEXT_BODY    = HexColor("#555577")
LIGHT_BG    = HexColor("#f5f5ff")
WHITE       = white
DANGER      = HexColor("#EF4444")
SUCCESS     = HexColor("#22C55E")

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "SteppeDNA_Comprehensive_Guide.pdf")

# ── Styles ───────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def _ps(name, parent="Normal", **kw):
    return ParagraphStyle(name, parent=styles[parent], **kw)

sTitle = _ps("CoverTitle", fontSize=32, leading=40, textColor=BRAND,
             fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=8)
sSubtitle = _ps("CoverSub", fontSize=14, leading=20, textColor=TEXT_BODY,
                fontName="Helvetica", alignment=TA_CENTER, spaceAfter=4)
sH1 = _ps("H1", fontSize=22, leading=28, textColor=BRAND,
          fontName="Helvetica-Bold", spaceBefore=18, spaceAfter=10,
          borderWidth=0, borderPadding=0)
sH2 = _ps("H2", fontSize=16, leading=22, textColor=BRAND_DARK,
          fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6)
sH3 = _ps("H3", fontSize=13, leading=18, textColor=TEXT_DARK,
          fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4)
sBody = _ps("Body2", fontSize=10.5, leading=15, textColor=TEXT_DARK,
            fontName="Helvetica", alignment=TA_JUSTIFY, spaceAfter=6)
sBullet = _ps("Bullet", fontSize=10.5, leading=15, textColor=TEXT_DARK,
              fontName="Helvetica", leftIndent=18, bulletIndent=6,
              spaceAfter=3)
sCode = _ps("Code2", fontSize=9, leading=13, textColor=HexColor("#333355"),
            fontName="Courier", backColor=LIGHT_BG, leftIndent=12,
            rightIndent=12, spaceBefore=4, spaceAfter=6,
            borderWidth=0.5, borderColor=BRAND_LIGHT, borderPadding=6)
sCaption = _ps("Caption", fontSize=9, leading=12, textColor=TEXT_BODY,
               fontName="Helvetica-Oblique", alignment=TA_CENTER, spaceAfter=8)
sSmall = _ps("Small", fontSize=9, leading=12, textColor=TEXT_BODY,
             fontName="Helvetica", spaceAfter=4)

# ── Helpers ──────────────────────────────────────────────────────────────────
def h1(t):  return Paragraph(t, sH1)
def h2(t):  return Paragraph(t, sH2)
def h3(t):  return Paragraph(t, sH3)
def p(t):   return Paragraph(t, sBody)
def b(t):   return Paragraph(f"• {t}", sBullet)
def sp(n=6):return Spacer(1, n)
def hr():   return HRFlowable(width="100%", thickness=1, color=BRAND_LIGHT,
                              spaceBefore=6, spaceAfter=6)

def tbl(data, col_widths=None, header=True):
    """Create a branded table."""
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    cmds = [
        ("BACKGROUND",  (0,0), (-1,0), BRAND),
        ("TEXTCOLOR",   (0,0), (-1,0), WHITE),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,0), 10),
        ("FONTNAME",    (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,1), (-1,-1), 9.5),
        ("ALIGN",       (0,0), (-1,-1), "LEFT"),
        ("VALIGN",      (0,0), (-1,-1), "TOP"),
        ("GRID",        (0,0), (-1,-1), 0.4, BRAND_LIGHT),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, LIGHT_BG]),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING",(0,0), (-1,-1), 6),
    ]
    t.setStyle(TableStyle(cmds))
    return t

# ── Page callbacks ───────────────────────────────────────────────────────────
def _header_footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    # Footer
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(TEXT_BODY)
    canvas.drawString(2*cm, 1.2*cm, "SteppeDNA — Comprehensive Project Guide")
    canvas.drawRightString(w - 2*cm, 1.2*cm, f"Page {doc.page}")
    # Top accent line
    canvas.setStrokeColor(BRAND)
    canvas.setLineWidth(2)
    canvas.line(2*cm, h - 1.5*cm, w - 2*cm, h - 1.5*cm)
    canvas.restoreState()

# ══════════════════════════════════════════════════════════════════════════════
#  CONTENT
# ══════════════════════════════════════════════════════════════════════════════
def build_content():
    S = []   # story list

    # ── COVER PAGE ───────────────────────────────────────────────────────────
    S.append(Spacer(1, 80))
    S.append(Paragraph("SteppeDNA", sTitle))
    S.append(Paragraph("Pan-Gene Variant Pathogenicity Classifier", sSubtitle))
    S.append(sp(12))
    S.append(Paragraph("Comprehensive Project Guide", _ps("x", fontSize=18,
             leading=24, textColor=TEXT_DARK, fontName="Helvetica-Bold",
             alignment=TA_CENTER)))
    S.append(sp(20))
    S.append(hr())
    S.append(Paragraph(
        f"Generated {datetime.datetime.now().strftime('%B %d, %Y')}",
        _ps("dt", fontSize=11, alignment=TA_CENTER, textColor=TEXT_BODY)))
    S.append(sp(8))
    S.append(Paragraph(
        "128 Features · 5 HR-Pathway Genes · Stacked Ensemble ML · SHAP Explanations · ACMG/AMP Evidence",
        _ps("tag", fontSize=10, alignment=TA_CENTER, textColor=BRAND)))
    S.append(sp(30))
    S.append(Paragraph(
        "This document provides an in-depth, section-by-section breakdown of every component, "
        "feature, data source, method, and file within the SteppeDNA project. It is intended as "
        "a preparation guide for understanding the full scope and technical depth of the work.",
        _ps("intro", fontSize=10.5, leading=16, alignment=TA_CENTER,
            textColor=TEXT_BODY, leftIndent=40, rightIndent=40)))
    S.append(PageBreak())

    # ── TABLE OF CONTENTS (manual) ───────────────────────────────────────────
    S.append(h1("Table of Contents"))
    toc_items = [
        "1. Project Overview & Purpose",
        "2. Codebase Metrics at a Glance",
        "3. Languages, Frameworks & Dependencies",
        "4. Complete File Inventory",
        "5. System Architecture",
        "6. Machine Learning Pipeline",
        "7. All 128 Features Explained",
        "8. External Datasets & APIs",
        "9. Backend API Endpoints",
        "10. Frontend & User Interface",
        "11. Data Pipelines",
        "12. Testing & Validation",
        "13. Security Measures",
        "14. Deployment (Docker)",
        "15. Complete Feature List (User-Facing)",
        "16. Glossary of Key Terms",
    ]
    for item in toc_items:
        S.append(Paragraph(item, _ps("toc", fontSize=11, leading=18,
                 textColor=TEXT_DARK, leftIndent=20, spaceAfter=2)))
    S.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. PROJECT OVERVIEW
    # ═══════════════════════════════════════════════════════════════════════════
    S.append(h1("1. Project Overview &amp; Purpose"))
    S.append(p(
        "SteppeDNA is a web-based bioinformatics tool that predicts whether a missense DNA "
        "mutation in one of five Homologous Recombination (HR) DNA-repair genes is likely to "
        "be <b>pathogenic</b> (disease-causing) or <b>benign</b> (harmless). "
        "The five supported genes — <b>BRCA1, BRCA2, PALB2, RAD51C, RAD51D</b> — are all "
        "part of the HR pathway, which is critical for repairing double-strand DNA breaks. "
        "Mutations in these genes are strongly associated with hereditary breast and ovarian "
        "cancer syndromes."
    ))
    S.append(p(
        "The system uses a <b>stacked ensemble</b> of machine learning models (XGBoost + "
        "Deep Neural Network) trained on 750+ clinically-annotated missense variants from "
        "ClinVar, enriched with 128 engineered features from 8 external biological databases "
        "and computational tools. Each prediction is accompanied by:"
    ))
    S.append(b("A calibrated pathogenicity probability (0.0 – 1.0)"))
    S.append(b("A confidence interval derived from Beta-distribution approximation"))
    S.append(b("SHAP (SHapley Additive exPlanations) feature attributions showing <i>why</i> the model decided"))
    S.append(b("ACMG/AMP clinical evidence codes (PM1, PP3, BP4, BS1, PVS1)"))
    S.append(b("Live cross-referencing against ClinVar and gnomAD databases"))
    S.append(sp())
    S.append(p(
        "<b>Key differentiator:</b> SteppeDNA is not just a black-box classifier. It includes "
        "a Tier 1 rule-based interceptor for obvious truncating (nonsense/frameshift) mutations, "
        "and a Tier 2 ML engine for the harder missense cases. Every result is explainable."
    ))
    S.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. CODEBASE METRICS
    # ═══════════════════════════════════════════════════════════════════════════
    S.append(h1("2. Codebase Metrics at a Glance"))
    S.append(p("The following table shows the breakdown of source code by language:"))
    S.append(tbl([
        ["Language / Format", "Files", "Lines of Code", "Size (KB)"],
        ["Python (.py)",       "69",  "10,955",          "567.5"],
        ["JavaScript (.js)",   "3",   "975",             "50.8"],
        ["CSS (.css)",         "1",   "1,436",           "28.4"],
        ["HTML (.html)",       "3",   "552",             "41.8"],
        ["JSON (.json)",       "10",  "188",             "1,585.1"],
        ["Markdown (.md)",     "5",   "187",             "23.6"],
        ["VCF (.vcf)",         "2",   "1,086",           "12.6"],
        ["YAML (.yml)",        "1",   "10",              "0.2"],
        ["Text (.txt)",        "7",   "51",              "22.9"],
        ["TOTAL",              "101", "15,440",          "~2,332"],
    ], col_widths=[140, 50, 90, 70]))
    S.append(sp())
    S.append(p(
        "<b>Primary language:</b> Python (75.6% of all code). "
        "The project also includes serialized model artifacts (.pkl, .h5, .json) totalling "
        "~60 MB in the <font face='Courier'>data/</font> directory."
    ))
    S.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. LANGUAGES, FRAMEWORKS & DEPENDENCIES
    # ═══════════════════════════════════════════════════════════════════════════
    S.append(h1("3. Languages, Frameworks &amp; Dependencies"))

    S.append(h2("3.1 Backend (Python)"))
    S.append(tbl([
        ["Package", "Version", "Purpose"],
        ["FastAPI", "0.129.0", "High-performance async web framework for REST API"],
        ["Uvicorn", "0.41.0", "ASGI server to run FastAPI"],
        ["XGBoost", "2.1.4", "Gradient-boosted decision tree model (Tier 2 ML)"],
        ["TensorFlow", "≥2.15.0", "Deep Neural Network (ensemble member + ESM-2 fallback)"],
        ["scikit-learn", "1.7.2", "StandardScaler, IsotonicRegression calibration, metrics"],
        ["SHAP", "0.49.1", "Model explainability via Shapley values"],
        ["pandas", "2.3.2", "Tabular data manipulation in training scripts"],
        ["NumPy", "1.26.4", "Numerical computation and feature vectors"],
        ["SciPy", "≥1.12.0", "Beta distribution for confidence intervals"],
        ["BioPython", "1.85", "CDS translation, codon table, sequence manipulation"],
        ["httpx", "≥0.27.0", "Async HTTP client for ClinVar/gnomAD live lookups"],
        ["imbalanced-learn", "0.14.1", "SMOTE oversampling for class imbalance"],
        ["python-multipart", "—", "File upload parsing for VCF endpoint"],
    ], col_widths=[100, 60, 300]))
    S.append(sp())

    S.append(h2("3.2 Frontend"))
    S.append(b("<b>HTML5</b> — semantic markup with accessibility attributes (ARIA roles)"))
    S.append(b("<b>Vanilla CSS</b> — 1,436 lines with CSS custom properties, dark/light theme, gradient animations"))
    S.append(b("<b>Vanilla JavaScript</b> — no frameworks, 875 lines of hand-written logic"))
    S.append(b("<b>Inter font</b> — loaded from Google Fonts for premium typography"))
    S.append(sp())

    S.append(h2("3.3 Infrastructure"))
    S.append(b("<b>Docker</b> — Dockerfile + docker-compose.yml for containerized deployment"))
    S.append(b("<b>Python 3.10-slim</b> — base Docker image"))
    S.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. COMPLETE FILE INVENTORY
    # ═══════════════════════════════════════════════════════════════════════════
    S.append(h1("4. Complete File Inventory"))

    S.append(h2("4.1 Backend (backend/)"))
    S.append(tbl([
        ["File", "Lines", "Purpose"],
        ["main.py", "1,311", "FastAPI app: 8 endpoints, ML inference, VCF parsing, ClinVar/gnomAD proxy, middleware"],
        ["feature_engineering.py", "472", "Shared bio-lookup tables (BLOSUM62, hydrophobicity, volumes), engineer_features() for training"],
        ["acmg_rules.py", "55", "ACMG/AMP clinical rule engine: PM1, PP3, BP4, BS1 evidence evaluation"],
        ["constants.py", "27", "Codon translation table (64 codons), DNA complement mapping"],
        ["gene_configs/*.json (×5)", "~165", "Per-gene config: chromosome, strand, CDS length, AA length, functional domains"],
    ], col_widths=[130, 40, 290]))
    S.append(sp())

    S.append(h2("4.2 Frontend (frontend/)"))
    S.append(tbl([
        ["File", "Lines", "Purpose"],
        ["index.html", "315", "Main single-page application with semantic HTML5 structure"],
        ["index-split.html", "237", "Alternative layout variant"],
        ["app.js", "735", "Core application logic: form submission, VCF upload, result rendering"],
        ["api.js", "16", "Single source of truth for all backend URL constants"],
        ["lang.js", "224", "Trilingual i18n system: English, Kazakh, Russian"],
        ["styles.css", "1,436", "Complete design system: CSS variables, themes, animations"],
    ], col_widths=[110, 40, 310]))
    S.append(sp())

    S.append(h2("4.3 Training &amp; Analysis Scripts (scripts/)"))
    S.append(tbl([
        ["File", "Lines", "Purpose"],
        ["cross_validate.py", "336", "10-fold stratified CV with bootstrap CIs, ROC/PR curves"],
        ["train_ensemble_baseline.py", "393", "SMOTE + focal-loss DNN ensemble (5 seeds) + isotonic calibration"],
        ["train_ensemble_blind.py", "~170", "Blind-test ensemble variant"],
        ["train_xgboost.py", "~110", "Standalone XGBoost trainer with Optuna-tuned hyperparams"],
        ["tune_xgboost.py / tune_xgboost_blind.py", "~260", "Optuna Bayesian hyperparameter search"],
        ["train_multitask_blind.py", "~240", "Multi-task neural network variant"],
        ["train_universal_model.py", "~155", "Universal (pan-gene) model trainer"],
        ["vus_reclassification.py", "513", "ClinVar VUS fetch + SteppeDNA reclassification + visualization"],
        ["ablation_study_xgb.py", "~450", "Leave-one-feature-group-out ablation analysis"],
        ["sota_comparison.py", "~480", "Benchmarks against REVEL, CADD, AlphaMissense, PolyPhen-2"],
        ["external_validation*.py (×3)", "~700", "MAVE holdout validation for XGBoost, ensemble, multitask"],
        ["run_benchmark.py", "~370", "End-to-end benchmark runner"],
        ["generate_*_pdf.py (×8)", "~1,400", "Various PDF report generators"],
        ["check_*.py (×5)", "~350", "Data integrity checks (leakage, mutations, VCF, known variants)"],
    ], col_widths=[155, 40, 265]))
    S.append(sp())

    S.append(h2("4.4 Data Pipelines (data_pipelines/)"))
    S.append(tbl([
        ["File", "Lines", "Purpose"],
        ["fetch_alphafold.py", "321", "3D structural features from PDB 1MIU + ESMFold (RSA, B-factor, secondary structure, distances)"],
        ["fetch_alphamissense.py", "~180", "DeepMind AlphaMissense pathogenicity scores (Cheng et al. 2023)"],
        ["fetch_gnomad.py", "~350", "gnomAD v4 allele frequencies (global + 4 sub-populations)"],
        ["fetch_phylop.py", "~240", "PhyloP 100-way vertebrate conservation scores from UCSC"],
        ["fetch_dbnsfp.py", "~520", "dbNSFP aggregated functional predictions"],
        ["fetch_mave.py", "~140", "MAVE HDR functional assay scores (Hu C et al. 2024)"],
        ["fetch_spliceai.py", "~170", "SpliceAI delta scores for splice-site disruption"],
        ["generate_esm2_embeddings.py", "246", "ESM-2 protein language model embeddings (Meta AI)"],
        ["prepare_brca1.py / prepare_all_genes.py", "~440", "Gene-specific dataset preparation and unification"],
    ], col_widths=[170, 40, 250]))
    S.append(sp())

    S.append(h2("4.5 Tests (tests/)"))
    S.append(tbl([
        ["File", "Lines", "Purpose"],
        ["conftest.py", "~45", "Pytest fixtures and shared test configuration"],
        ["test_acmg_rules.py", "~180", "ACMG evidence code evaluation tests"],
        ["test_feature_engineering.py", "~180", "Feature vector construction correctness tests"],
        ["test_negative_cases.py", "~165", "Edge cases: invalid inputs, boundary conditions"],
        ["test_shap_stability.py", "~185", "SHAP value determinism and stability tests"],
        ["test_variants.vcf", "—", "Sample VCF file for integration tests"],
    ], col_widths=[160, 40, 260]))
    S.append(sp())

    S.append(h2("4.6 Other Files"))
    S.append(tbl([
        ["File", "Purpose"],
        ["Dockerfile", "Container image definition (Python 3.10-slim base)"],
        ["docker-compose.yml", "Single-service orchestration config"],
        ["requirements.txt", "13 Python dependencies pinned"],
        ["comprehensive_test_suite.vcf", "224-line VCF with pathogenic + benign + edge-case variants"],
        ["brca2_missense_dataset_2.csv", "759-row ClinVar training dataset"],
        ["gen_vcf.py", "Script to generate test VCF files"],
        ["SteppeDNA_Guide.md", "Quick-start user guide"],
        ["visual_proofs/ (13 files)", "Publication-quality figures: ROC curves, ablation, VUS, MAVE validation"],
    ], col_widths=[180, 280]))
    S.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. SYSTEM ARCHITECTURE
    # ═══════════════════════════════════════════════════════════════════════════
    S.append(h1("5. System Architecture"))
    S.append(p("SteppeDNA follows a classic <b>client-server</b> architecture:"))
    S.append(sp())
    S.append(Paragraph(
        "<font face='Courier' size='9'>"
        "┌──────────────────────────────────────────────────┐\n"
        "│       FRONTEND  (Static HTML/CSS/JS)             │\n"
        "│ index.html + app.js + styles.css + lang.js       │\n"
        "│ ─ Manual form input (cDNA, AA_ref, AA_alt)       │\n"
        "│ ─ VCF file drag-and-drop upload                  │\n"
        "│ ─ Dark/light theme toggle                        │\n"
        "│ ─ Trilingual (EN / KK / RU)                      │\n"
        "└─────────┬────────────────────────────────────────┘\n"
        "          │  HTTP POST / GET (JSON)\n"
        "          ▼\n"
        "┌──────────────────────────────────────────────────┐\n"
        "│     BACKEND  (FastAPI + Uvicorn, port 8000)      │\n"
        "│ Middleware: CORS → API Key → Rate Limit → Req ID │\n"
        "│ TIER 1: Rule Engine (Nonsense/Frameshift)        │\n"
        "│    └─ PVS1 auto-pathogenic                       │\n"
        "│ TIER 2: ML Engine                                │\n"
        "│    ├─ build_feature_vector() → 128 features      │\n"
        "│    ├─ StandardScaler normalization                │\n"
        "│    ├─ XGBoost prediction (60% weight)            │\n"
        "│    ├─ DNN prediction (40% weight)                │\n"
        "│    ├─ Blended → Isotonic calibration             │\n"
        "│    ├─ Beta-distribution confidence interval       │\n"
        "│    ├─ SHAP top-8 feature attributions            │\n"
        "│    └─ ACMG/AMP evidence evaluation               │\n"
        "│ DATA LAYER: Pickle/JSON/H5 model artifacts       │\n"
        "│    Per-gene lazy loading with LRU caching         │\n"
        "└──────────────────────────────────────────────────┘\n"
        "</font>", sBody))
    S.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # 6. ML PIPELINE
    # ═══════════════════════════════════════════════════════════════════════════
    S.append(h1("6. Machine Learning Pipeline"))

    S.append(h2("6.1 Training Data"))
    S.append(p(
        "The master training dataset (<font face='Courier'>brca2_missense_dataset_2.csv</font>) "
        "contains <b>759 missense variants</b> from ClinVar with binary labels: "
        "Pathogenic (1) vs Benign (0). The class distribution is imbalanced (~11% pathogenic), "
        "which is addressed through SMOTE oversampling and focal loss."
    ))

    S.append(h2("6.2 Feature Engineering (128 Features)"))
    S.append(p(
        "Each variant is converted into a 128-dimensional feature vector. Features are grouped into "
        "8 categories (see Section 7 for the full list). The same <font face='Courier'>build_feature_vector()</font> "
        "function is used in both training and inference for consistency."
    ))

    S.append(h2("6.3 Model Architecture — Stacked Ensemble"))
    S.append(p("The Tier 2 ML engine uses a <b>stacked ensemble</b> of two models:"))
    S.append(b("<b>XGBoost</b> (gradient-boosted trees) — 60% blend weight. Hyperparameters tuned via Optuna Bayesian search."))
    S.append(b("<b>Deep Neural Network</b> (TensorFlow/Keras) — 40% blend weight. Multi-layer perceptron with dropout, trained with focal loss and class weighting. Ensemble of 5 models with different random seeds."))
    S.append(p(
        "The blended raw probability is passed through <b>Isotonic Regression</b> calibration "
        "(trained on the test split) to produce well-calibrated probabilities."
    ))

    S.append(h2("6.4 Evaluation & Performance"))
    S.append(tbl([
        ["Metric", "Value", "Method"],
        ["ROC-AUC (cross-val)", "~0.73", "10-fold stratified CV with 1000-bootstrap CIs"],
        ["Training set size", "759 variants", "ClinVar BRCA2 missense, deduplicated"],
        ["Feature count", "128", "8 biological data sources"],
        ["Validation", "External MAVE holdout", "Hu C et al. 2024 functional assay data"],
    ], col_widths=[140, 90, 230]))
    S.append(sp())

    S.append(h2("6.5 Overfitting Prevention"))
    S.append(b("Stratified K-fold cross-validation (10 folds) — no data leakage between folds"))
    S.append(b("Separate scaler fitted on each fold's training data only"))
    S.append(b("Early stopping with patience (DNN training)"))
    S.append(b("Data leakage checks (<font face='Courier'>check_leakage.py, check_leakage_brca2.py</font>)"))
    S.append(b("VUS deduplication against training set before reclassification"))
    S.append(b("External validation on MAVE holdout set (never seen during training)"))

    S.append(h2("6.6 Two-Tier Prediction System"))
    S.append(p("<b>Tier 1 — Rule Interceptor:</b> Truncating mutations (nonsense, frameshift, indels) "
               "are automatically classified as Pathogenic with p=0.9999 and PVS1 ACMG evidence. "
               "These don't need ML — the biology is clear."))
    S.append(p("<b>Tier 2 — ML Engine:</b> Missense variants go through the full 128-feature "
               "pipeline with stacked ensemble prediction, confidence estimation, and SHAP explanation."))
    S.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # 7. ALL 128 FEATURES
    # ═══════════════════════════════════════════════════════════════════════════
    S.append(h1("7. All 128 Features Explained"))
    S.append(p("Every variant is encoded into exactly 128 features from 8 groups:"))
    S.append(sp())

    feature_groups = [
        ("Group 1: Amino Acid Properties (11 features)", [
            ("blosum62_score", "BLOSUM62 substitution matrix score — measures evolutionary similarity of the amino acid change"),
            ("volume_diff", "Absolute difference in molecular volume between reference and alternate AA"),
            ("hydro_diff / hydro_delta / ref_hydro / alt_hydro", "Hydrophobicity metrics (Kyte-Doolittle scale)"),
            ("charge_change", "Binary: did the electric charge class change? (positive/negative/nonpolar)"),
            ("nonpolar_to_charged", "Binary: specifically nonpolar → charged transition (drastic)"),
            ("is_nonsense / is_transition / is_transversion", "Mutation type flags"),
        ]),
        ("Group 2: Positional & Domain (9 features)", [
            ("cDNA_pos / AA_pos", "Raw nucleotide and amino acid positions"),
            ("relative_cdna_pos / relative_aa_pos", "Position normalized to gene length (0.0 – 1.0)"),
            ("in_critical_repeat_region", "In BRC repeats, WD40, BRCT, SCD domains"),
            ("in_DNA_binding", "In DNA binding domain"),
            ("in_OB_folds", "In oligonucleotide-binding folds, RING, Walker A/B"),
            ("in_NLS", "In nuclear localization signal"),
            ("in_primary_interaction", "In PALB2/BRCA1 interaction domain"),
        ]),
        ("Group 3: Conservation — PhyloP (4 features)", [
            ("phylop_score", "PhyloP 100-way vertebrate conservation score (higher = more conserved)"),
            ("high_conservation", "Binary: PhyloP > 4.0"),
            ("ultra_conservation", "Binary: PhyloP > 7.0"),
            ("conserv_x_blosum", "Cross-feature: conservation × BLOSUM62 score"),
        ]),
        ("Group 4: Functional Assay — MAVE (4 features)", [
            ("mave_score", "HDR functional assay score from Hu C et al. 2024"),
            ("has_mave", "Binary: has experimental MAVE data"),
            ("mave_abnormal", "Binary: abnormal function (score 0.01 – 1.49)"),
            ("mave_x_blosum", "Cross-feature: MAVE × BLOSUM62"),
        ]),
        ("Group 5: AlphaMissense (3 features)", [
            ("am_score", "DeepMind AlphaMissense pathogenicity prediction (0–1)"),
            ("am_pathogenic", "Binary: AM score > 0.564 (pathogenic threshold)"),
            ("am_x_phylop", "Cross-feature: AlphaMissense × PhyloP"),
        ]),
        ("Group 6: 3D Structure — AlphaFold/PDB (10 features)", [
            ("rsa / is_buried", "Relative solvent accessibility; binary buried flag (<0.25)"),
            ("bfactor", "AlphaFold structural confidence (pLDDT)"),
            ("dist_dna / dist_palb2", "3D distance to DNA binding interface / PALB2 site (Angstroms)"),
            ("is_dna_contact", "Binary: within 5Å of DNA"),
            ("ss_helix / ss_sheet", "Secondary structure: alpha-helix or beta-sheet"),
            ("buried_x_blosum / dna_contact_x_blosum", "Cross-features with BLOSUM62"),
        ]),
        ("Group 7: Population Frequency — gnomAD (9 features)", [
            ("gnomad_af / gnomad_af_log", "Global allele frequency and its log-transform"),
            ("gnomad_popmax", "Maximum frequency across all sub-populations"),
            ("gnomad_afr / gnomad_amr / gnomad_eas / gnomad_nfe", "African, American, East Asian, Non-Finnish European AF"),
            ("is_rare", "Binary: AF < 0.001"),
            ("af_x_blosum", "Cross-feature: frequency × BLOSUM62"),
        ]),
        ("Group 8: ESM-2 + SpliceAI + Encodings (78 features)", [
            ("esm2_cosine_sim", "Cosine similarity of WT vs mutant ESM-2 embeddings"),
            ("esm2_l2_shift", "L2 norm of embedding difference vector"),
            ("esm2_pca_0 … esm2_pca_19", "20 PCA components of ESM-2 difference vectors"),
            ("spliceai_score / splice_pathogenic", "SpliceAI delta score + binary pathogenic flag"),
            ("Mutation_A>C … Mutation_T>G", "12 one-hot encoded nucleotide mutations"),
            ("AA_ref_Ala … AA_ref_Val", "21 one-hot encoded reference amino acids"),
            ("AA_alt_Ala … AA_alt_Val", "21 one-hot encoded alternate amino acids"),
        ]),
    ]

    for group_title, features in feature_groups:
        S.append(h3(group_title))
        for fname, desc in features:
            S.append(Paragraph(
                f"<b><font face='Courier' size='9'>{fname}</font></b> — {desc}",
                sBullet))
        S.append(sp(4))
    S.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # 8. EXTERNAL DATA
    # ═══════════════════════════════════════════════════════════════════════════
    S.append(h1("8. External Datasets &amp; APIs"))
    S.append(tbl([
        ["Source", "Type", "What It Provides", "Pipeline Script"],
        ["ClinVar (NCBI)", "Database", "Clinically-annotated variants with pathogenic/benign labels (training labels)", "Direct CSV download"],
        ["gnomAD v4", "Database + API", "Population allele frequencies across 6 sub-populations", "fetch_gnomad.py"],
        ["PhyloP (UCSC)", "Database", "100-way vertebrate conservation scores per nucleotide position", "fetch_phylop.py"],
        ["AlphaMissense", "ML Model (DeepMind)", "Pre-computed pathogenicity scores for all possible missense mutations", "fetch_alphamissense.py"],
        ["MAVE / MaveDB", "Experimental", "HDR functional assay scores measuring actual protein function", "fetch_mave.py"],
        ["AlphaFold / PDB 1MIU", "Structural DB", "3D protein structure for RSA, B-factor, distances, secondary structure", "fetch_alphafold.py"],
        ["ESM-2 (Meta AI)", "Protein LLM", "Contextual protein sequence embeddings (6-layer, 8M param model)", "generate_esm2_embeddings.py"],
        ["SpliceAI", "ML Model", "Deep-learning splice-site disruption predictions", "fetch_spliceai.py"],
        ["dbNSFP", "Aggregated DB", "Pre-computed functional predictions from 30+ tools", "fetch_dbnsfp.py"],
    ], col_widths=[80, 70, 165, 145]))
    S.append(sp())
    S.append(p(
        "The ClinVar and gnomAD APIs are also used at <b>runtime</b> (live lookup endpoints) "
        "to cross-reference predictions against the latest clinical annotations."
    ))
    S.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # 9. API ENDPOINTS
    # ═══════════════════════════════════════════════════════════════════════════
    S.append(h1("9. Backend API Endpoints"))
    S.append(tbl([
        ["Method", "Path", "Purpose"],
        ["POST", "/predict", "Single-variant pathogenicity prediction (main endpoint)"],
        ["POST", "/predict/vcf", "Batch VCF file upload — parses and predicts all missense variants"],
        ["GET", "/", "API status and model info"],
        ["GET", "/health", "Deep health check — validates all models, scalers, and data are loaded"],
        ["GET", "/lookup/clinvar/{variant}", "Live ClinVar lookup via NCBI E-utilities API"],
        ["GET", "/lookup/gnomad/{variant}", "Live gnomAD v4 lookup via GraphQL API"],
    ], col_widths=[45, 160, 255]))
    S.append(sp())
    S.append(h2("9.1 Middleware Stack"))
    S.append(b("<b>CORS</b> — Configurable allowed origins for cross-origin requests"))
    S.append(b("<b>API Key</b> — Optional header-based authentication (required in production)"))
    S.append(b("<b>Rate Limiter</b> — In-memory per-IP rate limiting (60 req/min default, separate limit for external lookups)"))
    S.append(b("<b>Request ID</b> — UUID attached to every request for tracing and debugging"))
    S.append(sp())
    S.append(h2("9.2 Caching"))
    S.append(p("ClinVar and gnomAD lookups use an in-memory <b>LRU cache</b> with 1-hour TTL and 1,000-entry capacity, "
               "avoiding redundant external API calls."))
    S.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # 10. FRONTEND
    # ═══════════════════════════════════════════════════════════════════════════
    S.append(h1("10. Frontend &amp; User Interface"))
    S.append(h2("10.1 Design System"))
    S.append(b("Purple-themed colour palette with CSS custom properties (30+ variables)"))
    S.append(b("Light and dark mode with smooth 0.4s cubic-bezier transitions on all elements"))
    S.append(b("Glassmorphism cards with gradient backgrounds and hover shadows"))
    S.append(b("Inter font family (Google Fonts) for premium typography"))
    S.append(b("Responsive grid layout (max-width 920px, 2-column grid)"))
    S.append(b("Floating hero section with animated radial gradients"))
    S.append(sp())
    S.append(h2("10.2 User Input Methods"))
    S.append(b("<b>Manual entry form:</b> Gene selector → cDNA position → Reference AA → Alternate AA → Mutation type → Nucleotide change"))
    S.append(b("<b>VCF file upload:</b> Drag-and-drop or click-to-browse with loading spinner"))
    S.append(sp())
    S.append(h2("10.3 Results Display"))
    S.append(b("Pathogenic/Benign badge with colour-coded probability bar"))
    S.append(b("Confidence interval display (Beta-distribution CI)"))
    S.append(b("Data source cards: PhyloP, MAVE, AlphaMissense, Structure details"))
    S.append(b("SHAP feature attribution bar chart (top 8 features, red/green)"))
    S.append(b("ACMG evidence codes with human-readable rationale"))
    S.append(b("Live ClinVar and gnomAD fetch buttons"))
    S.append(b("VCF batch results table with sortable columns"))
    S.append(sp())
    S.append(h2("10.4 Internationalisation (i18n)"))
    S.append(p("Full trilingual support with crossfade animation on language switch:"))
    S.append(b("<b>English</b> — Default language"))
    S.append(b("<b>Kazakh (Қазақша)</b> — Full UI translation"))
    S.append(b("<b>Russian (Русский)</b> — Full UI translation"))
    S.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # 11. DATA PIPELINES
    # ═══════════════════════════════════════════════════════════════════════════
    S.append(h1("11. Data Pipelines"))
    S.append(p(
        "The <font face='Courier'>data_pipelines/</font> directory contains parameterized scripts "
        "that fetch and process external biological data. Each supports a <font face='Courier'>--gene</font> "
        "argument and reads gene-specific configurations from <font face='Courier'>backend/gene_configs/</font>."
    ))
    S.append(b("<b>fetch_alphafold.py</b> — Downloads PDB structures, computes per-residue RSA, B-factor, secondary structure, distances to DNA/PALB2 interfaces"))
    S.append(b("<b>fetch_alphamissense.py</b> — Downloads and indexes AlphaMissense scores by variant key"))
    S.append(b("<b>fetch_gnomad.py</b> — Queries gnomAD v4 for allele frequencies across 6 sub-populations"))
    S.append(b("<b>fetch_phylop.py</b> — Extracts PhyloP 100-way scores from UCSC bigWig files"))
    S.append(b("<b>fetch_mave.py</b> — Downloads MAVE HDR functional assay data"))
    S.append(b("<b>fetch_spliceai.py</b> — Retrieves SpliceAI delta scores"))
    S.append(b("<b>fetch_dbnsfp.py</b> — Extracts predictions from dbNSFP aggregated database"))
    S.append(b("<b>generate_esm2_embeddings.py</b> — Runs ESM-2 protein language model for per-variant embeddings + PCA"))
    S.append(b("<b>prepare_brca1.py / prepare_all_genes.py</b> — Unifies datasets across genes into training-ready format"))
    S.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # 12. TESTING
    # ═══════════════════════════════════════════════════════════════════════════
    S.append(h1("12. Testing &amp; Validation"))
    S.append(h2("12.1 Unit Tests (pytest)"))
    S.append(b("<b>test_acmg_rules.py</b> — Verifies PM1, PP3, BP4, BS1 evidence codes fire correctly"))
    S.append(b("<b>test_feature_engineering.py</b> — Validates feature vector construction, BLOSUM62 lookups, domain mapping"))
    S.append(b("<b>test_negative_cases.py</b> — Edge cases: invalid amino acids, out-of-range positions, malformed input"))
    S.append(b("<b>test_shap_stability.py</b> — Ensures SHAP values are deterministic across repeated runs"))
    S.append(sp())
    S.append(h2("12.2 ML Validation"))
    S.append(b("10-fold stratified cross-validation with 1,000 bootstrap confidence intervals"))
    S.append(b("External validation on MAVE holdout set (functional assay data never used in training)"))
    S.append(b("Ablation study: leave-one-feature-group-out to measure each group's contribution"))
    S.append(b("SOTA comparison benchmarks against REVEL, CADD, AlphaMissense, PolyPhen-2"))
    S.append(b("Data leakage detection scripts to ensure no test-set contamination"))
    S.append(sp())
    S.append(h2("12.3 VCF Integration Tests"))
    S.append(p("A 224-line comprehensive VCF test suite covers pathogenic, benign, synonymous, "
               "multi-allelic, indel, and edge-case variants to validate end-to-end VCF parsing."))
    S.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # 13. SECURITY
    # ═══════════════════════════════════════════════════════════════════════════
    S.append(h1("13. Security Measures"))
    S.append(b("XSS prevention: HTML escaping in frontend (escapeHtml function)"))
    S.append(b("Input validation: Pydantic validators on all API inputs (gene name, cDNA pos, AA codes, mutation format)"))
    S.append(b("Path traversal protection: os.path.basename normalization on pickle file loads"))
    S.append(b("Rate limiting: Per-IP in-memory rate limiter (general + external API specific)"))
    S.append(b("CORS: Configurable allowed origins (not wildcard in production)"))
    S.append(b("API key: Header-based authentication enforced in production mode"))
    S.append(b("Query injection prevention: Regex validation on ClinVar/gnomAD lookup parameters"))
    S.append(b("File size limits: VCF uploads capped at 50 MB"))
    S.append(b("Atomic file writes: Thread-locked writes to needs_wetlab_assay.csv"))
    S.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # 14. DEPLOYMENT
    # ═══════════════════════════════════════════════════════════════════════════
    S.append(h1("14. Deployment (Docker)"))
    S.append(p("The project includes Docker support for reproducible deployment:"))
    S.append(Paragraph("<font face='Courier' size='9'>docker-compose up --build</font>", sCode))
    S.append(b("Base image: python:3.10-slim"))
    S.append(b("Exposes port 8000"))
    S.append(b("Auto-restart policy: unless-stopped"))
    S.append(b("All dependencies installed via requirements.txt"))
    S.append(b("Server: uvicorn backend.main:app --host 0.0.0.0 --port 8000"))
    S.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # 15. USER-FACING FEATURES
    # ═══════════════════════════════════════════════════════════════════════════
    S.append(h1("15. Complete Feature List (User-Facing)"))
    S.append(p("Total count: <b>25+ distinct user-facing features</b>"))
    S.append(sp())
    features_list = [
        ("Single Variant Prediction", "Enter a mutation manually and get instant pathogenicity classification"),
        ("VCF Batch Analysis", "Upload a VCF file to predict all missense variants at once"),
        ("5-Gene Support", "BRCA1, BRCA2, PALB2, RAD51C, RAD51D — all HR-pathway genes"),
        ("Calibrated Probabilities", "Output is a well-calibrated probability (0–1), not just a binary label"),
        ("Confidence Intervals", "Beta-distribution CIs show prediction certainty"),
        ("SHAP Explanations", "Visual bar chart of top 8 features driving each prediction"),
        ("ACMG/AMP Evidence", "Automated clinical evidence codes: PVS1, PM1, PP3, BP4, BS1"),
        ("Tier 1 Rule Engine", "Instant classification of truncating mutations without ML"),
        ("Live ClinVar Lookup", "Real-time cross-reference against NCBI ClinVar database"),
        ("Live gnomAD Lookup", "Real-time allele frequency from gnomAD v4 GraphQL API"),
        ("Data Source Indicators", "Shows PhyloP, MAVE, AlphaMissense, structure info per variant"),
        ("Dark / Light Mode", "Toggle with smooth 0.4s transition on all UI elements"),
        ("Trilingual Interface", "English, Kazakh, Russian with animated language switching"),
        ("Wet-Lab Triage", "Low-confidence or pathogenic predictions logged for follow-up"),
        ("VUS Reclassification", "Bulk ClinVar VUS analysis with confidence-tiered reclassification"),
        ("Keyboard Accessibility", "Custom dropdowns with ArrowKey/Enter/Escape navigation"),
        ("Auto-Capitalisation", "Input fields auto-correct amino acid codes (ala → Ala)"),
        ("Gene-Aware Validation", "cDNA position range adapts per selected gene"),
        ("Genomic Position Mapping", "cDNA → genomic coordinate mapping for gnomAD lookups"),
        ("Strand-Aware VCF Parsing", "Handles both + and − strand genes correctly"),
        ("Multi-Allelic VCF Support", "Comma-separated ALT alleles parsed individually"),
        ("Health Check Endpoint", "Deep system health validation of all ML components"),
        ("Request ID Tracing", "UUID per request for debugging and logging"),
        ("LRU Response Cache", "1-hour TTL cache for external API responses"),
        ("Docker Deployment", "One-command containerized deployment"),
    ]
    for name, desc in features_list:
        S.append(Paragraph(f"<b>{name}</b> — {desc}", sBullet))
    S.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # 16. GLOSSARY
    # ═══════════════════════════════════════════════════════════════════════════
    S.append(h1("16. Glossary of Key Terms"))
    glossary = [
        ("Missense variant", "A single nucleotide change that results in a different amino acid in the protein"),
        ("Pathogenic", "A variant that is disease-causing (disrupts protein function)"),
        ("Benign", "A variant that is harmless (does not affect protein function)"),
        ("VUS", "Variant of Uncertain Significance — not enough evidence to classify"),
        ("HR pathway", "Homologous Recombination — the DNA repair mechanism these 5 genes are part of"),
        ("SHAP", "SHapley Additive exPlanations — a method to explain individual ML predictions"),
        ("ACMG/AMP", "American College of Medical Genetics — clinical variant classification guidelines"),
        ("BLOSUM62", "BLOcks SUbstitution Matrix — measures evolutionary likelihood of amino acid substitutions"),
        ("ESM-2", "Evolutionary Scale Modeling — Meta AI's protein language model"),
        ("PhyloP", "Phylogenetic P-value — measures evolutionary conservation across 100 vertebrate species"),
        ("gnomAD", "Genome Aggregation Database — population allele frequencies from 800,000+ individuals"),
        ("AlphaMissense", "DeepMind's pre-computed pathogenicity predictions for all possible missense mutations"),
        ("MAVE", "Multiplexed Assays of Variant Effect — experimental functional data"),
        ("XGBoost", "eXtreme Gradient Boosting — a tree-based ML algorithm"),
        ("Isotonic Regression", "Non-parametric calibration method to align predicted probabilities with true outcomes"),
        ("CDS", "Coding DNA Sequence — the protein-coding portion of a gene"),
        ("cDNA", "Complementary DNA — a DNA copy of mRNA, used for position numbering"),
        ("VCF", "Variant Call Format — standard file format for storing genetic variants"),
    ]
    S.append(tbl(
        [["Term", "Definition"]] + [[t, d] for t, d in glossary],
        col_widths=[120, 340]
    ))
    S.append(sp(20))
    S.append(hr())
    S.append(Paragraph(
        "End of Document — SteppeDNA Comprehensive Project Guide",
        _ps("end", fontSize=10, alignment=TA_CENTER, textColor=TEXT_BODY)))

    return S


# ══════════════════════════════════════════════════════════════════════════════
#  BUILD PDF
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  SteppeDNA — Comprehensive Project Guide Generator")
    print("=" * 60)

    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=A4,
        topMargin=2*cm,
        bottomMargin=2*cm,
        leftMargin=2*cm,
        rightMargin=2*cm,
        title="SteppeDNA — Comprehensive Project Guide",
        author="SteppeDNA Team",
    )

    story = build_content()

    print(f"\n  Building PDF with {len(story)} elements...")
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)

    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"\n  Output: {OUTPUT_PATH}")
    print(f"  Size:   {size_mb:.2f} MB")
    print(f"\n  Done!")


if __name__ == "__main__":
    main()

