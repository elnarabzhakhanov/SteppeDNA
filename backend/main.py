"""
SteppeDNA - FastAPI Backend (main.py)
=====================================

This is the entry point for ALL API requests to SteppeDNA, a DNA variant
pathogenicity predictor for 5 homologous-recombination (HR) cancer genes:
BRCA1, BRCA2, PALB2, RAD51C, and RAD51D.

REQUEST FLOW OVERVIEW
---------------------
1. A user submits a variant (e.g., BRCA2 c.8023A>G p.Ile2675Val) via the
   frontend or API.

2. The /predict endpoint receives the variant and decides which tier to use:

   TIER 1 - Rule Interceptor (no ML needed):
     If the variant is truncating (nonsense, frameshift, start-loss), it is
     automatically classified as Pathogenic via the PVS1 rule. These variants
     destroy the protein, so no machine learning is needed.

   TIER 2 - ML Engine (full pipeline):
     Missense variants (single amino acid changes) go through the full
     prediction pipeline:
       a) build_feature_vector() - Create 120 engineered features from the
          variant (conservation scores, protein properties, structural data,
          ESM-2 embeddings, EVE scores, gnomAD frequencies, etc.)
       b) Scale the features using a pre-fitted StandardScaler
       c) Run the ensemble: XGBoost and MLP (neural network) each produce a
          pathogenicity probability, then they are blended using gene-specific
          weights (e.g., BRCA2 uses 60% XGBoost + 40% MLP)
       d) Calibrate the blended score using isotonic regression so the output
          probability is well-calibrated (i.e., 0.8 means ~80% chance of being
          truly pathogenic)
       e) Compute confidence intervals via bootstrap resampling or Beta
          distribution approximation
       f) Generate SHAP explanations showing which features pushed the
          prediction toward pathogenic vs. benign
       g) Evaluate ACMG clinical evidence rules (the standard framework
          geneticists use to classify variants)
       h) Return a rich JSON response with prediction, probability, SHAP
          values, ACMG evidence, confidence intervals, data source scores,
          and warnings

3. Other endpoints: /health (liveness), /health/ready (model readiness),
   /model_metrics (test-set performance), /structure (AlphaFold PDB files),
   and static file serving for the frontend.

STARTUP (lifespan):
  On server start, all ML models (XGBoost, MLP, scalers, calibrators) are
  loaded into memory. Gene configs are validated, model checksums verified,
  and optional heavy indexes (training index, contrastive pairs, bootstrap
  models) are built unless LOW_MEMORY mode is active.
"""

# ─── Standard Library & Third-Party Imports ──────────────────────────────────
from contextlib import asynccontextmanager  # For the lifespan context manager (startup/shutdown)
from fastapi import FastAPI, Request        # FastAPI framework and per-request object
from fastapi.middleware.cors import CORSMiddleware  # Cross-Origin Resource Sharing (lets the frontend talk to the backend)
from fastapi.responses import JSONResponse, FileResponse  # Response types for JSON data and file downloads
from fastapi.staticfiles import StaticFiles  # Serves the frontend HTML/JS/CSS as static files
from pydantic import BaseModel, field_validator  # Input validation - ensures API inputs are safe and correctly typed
import numpy as np
import sys
import os
import re

# Ensure the root project directory is in the Python path so that
# "from backend.xxx import yyy" works when running the file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import time
import hashlib
import asyncio
import threading
import xgboost as xgb              # XGBoost gradient-boosted tree model (one half of our ensemble)
from scipy.stats import beta as beta_dist  # Beta distribution for confidence interval fallback

# ─── Logging Setup ────────────────────────────────────────────────────────────
# Supports two log formats:
#   - "text" (default): human-readable for local development
#   - "json": structured JSON for production log aggregation tools (e.g., Datadog)
_LOG_FORMAT = os.getenv("LOG_FORMAT", "text")
if _LOG_FORMAT == "json":
    class _JsonFormatter(logging.Formatter):
        def format(self, record):
            return json.dumps({
                "ts": self.formatTime(record, self.datefmt),
                "level": record.levelname,
                "msg": record.getMessage(),
                "module": record.module,
            })
    _handler = logging.StreamHandler()
    _handler.setFormatter(_JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[_handler])
else:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("steppedna")

# Server-level metrics: track total predictions and cumulative latency
# for monitoring and the /health endpoint
_server_start_time = time.monotonic()
_metrics = {"predictions": 0, "vcf_uploads": 0, "total_predict_ms": 0.0}

# ─── Imports from refactored modules ─────────────────────────────────────────
# The backend is split into focused modules. Here we import what main.py needs:

# models.py: ML model loading, constants, gene configuration
from backend.models import (
    DATA_DIR, SUPPORTED_GENES, MAX_CDNA_LENGTHS, GENE_MAX_AA,
    GENE_RELIABILITY, XGB_WEIGHT, NN_WEIGHT,
    N_EFFECTIVE, VALID_AA_CODES,
    _GENE_CALIBRATORS,
    _load_gene_calibrators, _GENE_ENSEMBLE_WEIGHTS,
    _load_gene_ensemble_weights, _load_bootstrap_models, _load_active_learning_priorities,
    _load_conformal_thresholds, _get_universal_models,
)

# explanations.py: SHAP-adjacent explainability features
#   - data_support: how many similar training examples exist (data scarcity)
#   - contrastive: "what would need to change to flip the prediction?"
#   - bootstrap_ci: confidence intervals from resampled XGBoost models
#   - conformal_set: prediction sets with coverage guarantees
from backend.explanations import (
    _build_training_index, compute_data_support, _build_contrastive_index,
    find_contrastive_explanation, compute_bootstrap_ci, compute_conformal_set,
)

# features.py: builds the 120-dimensional feature vector from a raw variant
from backend.features import (
    get_gene_data, build_feature_vector,
    NICE_NAMES, _safe_critical_domain,
    get_frequency_independence_stats,
)

# external_api.py: ClinVar/gnomAD lookups + prediction cache
from backend.external_api import (
    _pred_cache_get, _pred_cache_set,
)

# vcf.py: VCF batch upload endpoint + risk tier helper
from backend.vcf import (
    _compute_risk_tier,
    GENE_TRANSCRIPTS,
)

# feature_engineering.py: amino acid property lookups used in the response
from backend.feature_engineering import (
    AA_HYDROPHOBICITY, AA_VOLUME, get_blosum62,
    get_charge,
)

# acmg_rules.py: ACMG/AMP variant classification framework
# (the standard set of rules clinical geneticists use to classify variants)
from backend.acmg_rules import evaluate_acmg_rules, combine_acmg_evidence, UNDERREPRESENTED_POPULATIONS
from backend import __version__ as STEPPEDNA_VERSION
# Database storage disabled -- privacy: no variant data is stored server-side
# from backend.database import init_db, record_analysis


# ─── Gene Config Validation ──────────────────────────────────────────────────
# Each gene has a JSON config file (e.g., backend/gene_configs/brca2.json) that
# describes its CDS length, protein length, functional domains, chromosome, and
# strand. We validate these at startup so we fail fast if a config is missing or
# malformed, rather than crashing mid-prediction.
_GENE_CONFIG_REQUIRED_KEYS = {"cds_length", "aa_length", "domains", "chromosome", "strand"}
_unavailable_genes: set = set()  # Genes whose configs failed validation


def _validate_gene_configs():
    """Validate all gene config JSONs have required keys on startup."""
    configs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gene_configs")
    for gene in SUPPORTED_GENES:
        config_path = os.path.join(configs_dir, f"{gene.lower()}.json")
        if not os.path.exists(config_path):
            logger.error(f"[CONFIG] Gene config missing for {gene}: {config_path}")
            _unavailable_genes.add(gene)
            continue
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            missing = _GENE_CONFIG_REQUIRED_KEYS - set(config.keys())
            if missing:
                logger.error(f"[CONFIG] Gene config for {gene} missing required keys: {missing}")
                _unavailable_genes.add(gene)
            else:
                logger.info(f"[CONFIG] {gene} config validated OK")
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"[CONFIG] Failed to load gene config for {gene}: {e}")
            _unavailable_genes.add(gene)

# ─── Model Checksum Verification ─────────────────────────────────────────────


def _verify_model_checksums():
    """Verify SHA256 checksums of model files against data/checksums.json.

    WHY: Detects corrupted or tampered model files. In production, a checksum
    mismatch raises a RuntimeError to prevent serving predictions from a
    compromised model. In development, it just logs a warning.
    """
    checksums_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "checksums.json"
    )
    if not os.path.exists(checksums_path):
        if os.getenv("ENVIRONMENT", "").lower() == "production":
            raise RuntimeError("[CHECKSUM] checksums.json REQUIRED in production — run scripts/generate_checksums.py")
        logger.warning("[CHECKSUM] data/checksums.json not found -- skipping verification. Run scripts/generate_checksums.py to create it.")
        return
    try:
        with open(checksums_path, "r") as f:
            expected = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"[CHECKSUM] Failed to load checksums.json: {e}")
        return
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    for filename, expected_hash in expected.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            logger.warning(f"[CHECKSUM] Model file missing: {filename}")
            continue
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        actual_hash = sha256.hexdigest()
        if actual_hash != expected_hash:
            if os.getenv("ENVIRONMENT", "").lower() == "production":
                raise RuntimeError(f"[CHECKSUM] BLOCKED: {filename} checksum mismatch — possible tampering")
            logger.warning(f"[CHECKSUM] Mismatch for {filename}: expected {expected_hash[:12]}... got {actual_hash[:12]}...")
        else:
            logger.info(f"[CHECKSUM] {filename} OK")


# ─── Kazakh Founder Mutation Data ───────────────────────────────────────────
# Founder mutations are pathogenic variants that are unusually common in a
# specific population due to a shared ancestor. SteppeDNA highlights these
# for the Kazakh/Central Asian population, which is the project's focus.
_FOUNDER_MUTATIONS = {}  # gene -> {hgvs_c -> mutation_info_dict}


def _load_founder_mutations():
    """Load Kazakh/Central Asian founder mutation data from JSON.

    Builds a lookup table so the /predict endpoint can flag when a submitted
    variant matches a known founder mutation and provide population-specific
    frequency and clinical context.
    """
    fpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kazakh_founder_mutations.json")
    if os.path.exists(fpath):
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            # Build lookup: gene -> {hgvs_c -> mutation_info}
            mutations_list = data.get("mutations", []) if isinstance(data, dict) else data
            for entry in mutations_list:
                gene = entry.get("gene", "").upper()
                hgvs = entry.get("hgvs_c", entry.get("hgvs_cdna", ""))
                if gene and hgvs:
                    if gene not in _FOUNDER_MUTATIONS:
                        _FOUNDER_MUTATIONS[gene] = {}
                    _FOUNDER_MUTATIONS[gene][hgvs] = {
                        "name": entry.get("common_name", hgvs),
                        "population": entry.get("population", "Kazakh"),
                        "frequency": entry.get("frequency_kz", entry.get("estimated_frequency", "unknown")),
                        "source": entry.get("notes", entry.get("source", "")),
                        "clinvar_id": entry.get("clinvar_id", ""),
                        "mutation_type": entry.get("type", entry.get("mutation_type", "")),
                        "pathogenicity": entry.get("pathogenicity", ""),
                        "hgvs_p": entry.get("hgvs_p", ""),
                    }
            logger.info("Loaded founder mutation reference data")
        except Exception as e:
            logger.warning(f"Failed to load founder mutations: {e}")


# ─── Lifespan & App Creation ─────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app):
    """Application lifespan: pre-load models on startup, clean up on shutdown.

    This runs BEFORE the server starts accepting requests. It loads all ML
    model artifacts into memory so that the first /predict call is fast.
    The 'yield' in the middle is where the server runs; after yield, we
    handle graceful shutdown.
    """
    # Step 1: Load core ML models (XGBoost, MLP, scaler, calibrator, feature names)
    logger.info("Initializing universal ML models...")
    _get_universal_models()

    # Step 2: Validate gene configs and model file integrity
    _validate_gene_configs()
    _verify_model_checksums()

    # Step 3: Load per-gene calibrators and ensemble weights
    # Each gene has its own isotonic calibrator and XGB/MLP blend ratio
    _load_gene_calibrators()
    _load_gene_ensemble_weights()

    # Step 4: Load conformal prediction thresholds and founder mutation database
    _load_conformal_thresholds()
    _load_founder_mutations()

    # Step 5: Heavy startup tasks - skip on low-memory deployments (Render free tier = 512MB)
    # These build in-memory indexes for explainability features that are nice-to-have
    # but not required for core predictions
    _low_mem = os.getenv("LOW_MEMORY", "").lower() in ("1", "true")
    if _low_mem:
        logger.info("LOW_MEMORY=1: skipping training index, contrastive index, bootstrap models")
    else:
        _build_training_index()        # Index of training data for data scarcity estimation
        _build_contrastive_index()     # Index for "what would need to change?" explanations
        _load_bootstrap_models()       # Multiple XGBoost models for confidence intervals
        _load_active_learning_priorities()  # Which variants would be most valuable to label next
    logger.info("Server ready.")

    # Background task: periodically clean up stale entries from the rate limiter
    # to prevent unbounded memory growth from tracking old IP addresses
    async def _periodic_rate_cleanup():
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            now = time.monotonic()
            with _rate_lock:
                stale = [k for k, v in _rate_counts.items()
                         if not v or (now - max(v)) > RATE_WINDOW * 2]
                for k in stale:
                    del _rate_counts[k]
                if stale:
                    logger.info(f"[RATE-CLEANUP] Purged {len(stale)} stale IPs")
    cleanup_task = asyncio.create_task(_periodic_rate_cleanup())

    yield  # <--- Server is running and accepting requests here

    # Shutdown: cancel the background cleanup task
    cleanup_task.cancel()
    logger.info("Server shutting down.")

# ─── FastAPI App Creation ─────────────────────────────────────────────────────
# In production, the interactive /docs and /redoc endpoints are disabled
# for security (they expose the full API schema to anyone).
_is_production = os.getenv("ENVIRONMENT", "").lower() == "production"

app = FastAPI(
    title="SteppeDNA API",
    description="Multi-gene HR variant pathogenicity classifier for Homologous Recombination DNA repair genes (BRCA1, BRCA2, PALB2, RAD51C, RAD51D). "
                "Ensemble of XGBoost + MLP with isotonic calibration, trained on 19,000+ ClinVar & gnomAD variants across 120 engineered features.",
    version=STEPPEDNA_VERSION,
    docs_url="/docs" if not _is_production else None,
    redoc_url="/redoc" if not _is_production else None,
    contact={"name": "SteppeDNA Team"},
    license_info={"name": "Research Use Only"},
    lifespan=lifespan,  # Hooks into the lifespan function above for startup/shutdown
)

# ─── CORS (Cross-Origin Resource Sharing) ─────────────────────────────────────
# CORS controls which websites can make requests to this API.
# Without this, a browser would block the frontend (e.g., steppedna.vercel.app)
# from calling the backend (steppedna-api.onrender.com) because they are on
# different domains. We whitelist allowed origins here.
_ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost,http://127.0.0.1,http://localhost:8000,http://127.0.0.1:8000,http://localhost:8080,http://127.0.0.1:8080,http://localhost:5500,http://127.0.0.1:5500,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _ALLOWED_ORIGINS],
    allow_credentials=False,         # No cookies needed
    allow_methods=["GET", "POST"],   # Only allow GET and POST (no PUT/DELETE)
    allow_headers=["Content-Type", "X-API-Key"],
)

# ─── Middleware ────────────────────────────────────────────────────────────────
# Registers rate limiting, API key auth, security headers (CSP, X-Frame-Options,
# etc.) to protect the API from abuse and common web vulnerabilities.
from backend.middleware import register_middleware, _rate_lock, _rate_counts, RATE_WINDOW
register_middleware(app)

_metrics_lock = threading.Lock()  # Thread-safe lock for updating prediction metrics

# ─── Include Routers from refactored modules ────────────────────────────────
# FastAPI "routers" let us split endpoints across multiple files while keeping
# them all accessible under the same app. Each router adds its own endpoints.
from backend.external_api import router as external_api_router  # /clinvar, /gnomad lookups
from backend.vcf import router as vcf_router                    # /vcf batch upload
from backend.cohort import router as cohort_router              # /cohort, /umap, /history

app.include_router(external_api_router)
app.include_router(vcf_router)
app.include_router(cohort_router)


# ============================================================
# API - Prediction Endpoint
# ============================================================
# This is the core of SteppeDNA. The /predict endpoint receives a single
# variant and returns a rich JSON response with the pathogenicity prediction.

class MutationInput(BaseModel):
    """Input schema for the /predict endpoint.

    Pydantic validates and sanitizes all fields before they reach the
    endpoint function. This prevents invalid data from causing crashes
    deeper in the pipeline.

    Fields:
        cDNA_pos:    Position in the coding DNA sequence (e.g., 8023 for c.8023A>G)
        AA_ref:      Reference (original) amino acid in 3-letter code (e.g., "Ile")
        AA_alt:      Alternate (mutated) amino acid in 3-letter code (e.g., "Val")
                     Special values: "Ter" (stop codon), "Fs" (frameshift)
        Mutation:    Nucleotide change (e.g., "A>G") or variant type ("FS", "DEL")
        AA_pos:      Amino acid position (auto-computed from cDNA_pos if not provided)
        gene_name:   Which of the 5 HR genes this variant is in
        population:  Optional gnomAD population code for population-specific ACMG rules
    """
    cDNA_pos: int
    AA_ref: str
    AA_alt: str
    Mutation: str = "Unknown"
    AA_pos: int = 0

    gene_name: str = "BRCA2"
    population: str | None = None  # gnomAD population code (afr, amr, asj, eas, fin, nfe, sas)

    @field_validator('gene_name')
    @classmethod
    def validate_gene_name(cls, v):
        """Ensure the gene is one of the 5 supported HR genes."""
        if v.upper() not in SUPPORTED_GENES:
            raise ValueError(f'Unsupported gene: {v}. Supported: {", ".join(sorted(SUPPORTED_GENES))}')
        return v.upper()

    @field_validator('cDNA_pos')
    @classmethod
    def validate_cdna_pos(cls, v):
        """cDNA positions start at 1 (there is no position 0 in biology)."""
        if v < 1:
            raise ValueError('cDNA_pos must be positive')
        return v

    @field_validator('AA_ref', 'AA_alt')
    @classmethod
    def validate_aa(cls, v):
        """Validate and normalize amino acid codes to title case (e.g., 'ALA' -> 'Ala')."""
        if len(v) >= 2:
            v = v[0].upper() + v[1:].lower()
        if v not in VALID_AA_CODES:
            raise ValueError(f'Invalid amino acid code: {v}. Must be a 3-letter code (e.g. Ala, Val, Gly)')
        return v

    @field_validator('Mutation')
    @classmethod
    def validate_mutation(cls, v):
        """Accept nucleotide substitutions (A>G), variant types (FS/DEL/INS/DUP), or 'Unknown'."""
        if v == "Unknown":
            return v
        v_upper = v.upper()
        if v_upper in {"FS", "DEL", "INS", "DUP"}:
            return v_upper
        if not re.match(r'^[ACGT]>[ACGT]$', v_upper):
            raise ValueError('Mutation must be "Unknown", a variant type (fs/del/ins/dup), or format X>Y (e.g. A>G)')
        return v_upper


@app.post("/predict", tags=["Prediction"], summary="Predict variant pathogenicity",
          description="Accepts a single missense variant and returns pathogenicity probability, SHAP explanations, ACMG evidence, and data source scores.")
async def predict(mutation_data: MutationInput, request: Request):  # noqa: C901
    """THE main endpoint. Predicts whether a DNA variant is pathogenic (disease-causing)
    or benign (harmless).

    Two-tier architecture:
      - Tier 1 (Rule Interceptor): Truncating variants (nonsense, frameshift, start-loss)
        are automatically pathogenic -- no ML needed. These destroy the protein entirely.
      - Tier 2 (ML Engine): Missense variants (single amino acid change) go through the
        full ensemble pipeline: feature engineering -> XGBoost + MLP -> calibration ->
        SHAP explanations -> ACMG evidence -> confidence intervals.
    """
    t_start = time.perf_counter()  # Start the latency timer

    # Sanity check: reject cDNA positions beyond the gene's coding sequence length
    max_len = MAX_CDNA_LENGTHS.get(mutation_data.gene_name.upper(), 15000)
    if mutation_data.cDNA_pos > max_len:
        return JSONResponse(status_code=400, content={"error": f"cDNA position cannot exceed {max_len} for {mutation_data.gene_name}"})

    # ─── Prediction cache ──────────────────────────────────────────────────
    # If we have already predicted this exact variant, return the cached result
    # immediately. This avoids redundant computation for repeated lookups.
    pred_cache_key = f"pred_{mutation_data.gene_name}_{mutation_data.cDNA_pos}_{mutation_data.AA_ref}_{mutation_data.AA_alt}_{mutation_data.Mutation}"
    cached_pred = _pred_cache_get(pred_cache_key)
    if cached_pred is not None:
        return cached_pred

    # Convert cDNA position to amino acid position if not explicitly provided.
    # Formula: each codon (3 nucleotides) encodes one amino acid.
    # Example: cDNA pos 100 -> (100-1)//3 + 1 = AA pos 34
    aa_pos = mutation_data.AA_pos if mutation_data.AA_pos > 0 else (mutation_data.cDNA_pos - 1) // 3 + 1

    # Validate AA position against protein length (e.g., BRCA2 protein is 3418 AA)
    max_aa = GENE_MAX_AA.get(mutation_data.gene_name.upper(), 5000)
    if aa_pos > max_aa:
        return JSONResponse(status_code=400, content={
            "error": f"AA position {aa_pos} exceeds {mutation_data.gene_name} protein length ({max_aa} AA)"
        })

    # Load all gene-specific data: models, scalers, calibrators, structural features,
    # conservation scores, gnomAD frequencies, EVE scores, etc.
    gene_data = get_gene_data(mutation_data.gene_name)

    # ─── TIER 1: RULE INTERCEPTOR ───────────────────────────────────────────────
    # Truncating variants destroy the protein entirely. They are automatically
    # classified as Pathogenic via the PVS1 rule (a very strong ACMG evidence
    # criterion). No machine learning is needed -- this is a well-established
    # biological fact: if the protein is destroyed, it cannot function.
    #
    # Types of truncating variants:
    #   - Nonsense: AA_alt is "Ter" (a premature stop codon, cutting the protein short)
    #   - Frameshift: a deletion or insertion that shifts the reading frame
    #   - Start-loss: mutation at the first amino acid (Met1) that prevents translation
    is_truncating = (
        mutation_data.AA_alt in ["Ter", "*", "Fs"] or
        any(x in mutation_data.Mutation.lower() for x in ["fs", "del", "ins", "dup"])
    )
    is_start_loss = (
        aa_pos == 1
        and mutation_data.AA_ref in ("Met", "M")
        and mutation_data.AA_alt not in ("Met", "M")
    )

    # ─── Founder mutation lookup (shared by Tier 1 and Tier 2) ──────────────
    # Check if this variant is a known Kazakh/Central Asian founder mutation.
    # Must happen before ACMG evaluation so PS4 evidence can be included.
    _founder_info = None
    hgvs_c = f"c.{mutation_data.cDNA_pos}{mutation_data.Mutation}"
    gene_founders = _FOUNDER_MUTATIONS.get(mutation_data.gene_name.upper(), {})
    if hgvs_c in gene_founders:
        _founder_info = {**gene_founders[hgvs_c], "is_founder": True}
    else:
        for fhgvs, finfo in gene_founders.items():
            if str(mutation_data.cDNA_pos) in fhgvs:
                _founder_info = {**finfo, "is_founder": True, "match_type": "position_approximate"}
                break

    if is_truncating or is_start_loss:
        # Tier 1 result: hardcoded 99% pathogenic probability (no ML uncertainty)
        label = "Pathogenic"
        probability = 0.99
        # NOTE: The risk tier string "high (Truncating)" distinguishes Tier 1 from
        # Tier 2's "high"/"low"/"uncertain". The frontend checks .includes("high").
        risk = "high (Truncating)"
        if is_start_loss:
            acmg_eval = {"PVS1": "Start-loss variant (p.Met1?) \u2014 loss of initiator methionine abolishes translation initiation."}
        else:
            acmg_eval = {"PVS1": "Pathogenic truncating null variant (Nonsense/Frameshift) in a recognized tumor suppressor."}

        # PS4: If this truncating variant is a known Kazakh founder mutation
        if _founder_info and _founder_info.get("is_founder"):
            freq = _founder_info.get("frequency_kz", _founder_info.get("frequency", 0))
            pop = _founder_info.get("population", "Kazakh")
            source = _founder_info.get("source", "published literature")
            acmg_eval['PS4'] = (
                f"Known {pop} founder mutation (frequency {freq*100:.1f}% in affected individuals). "
                f"Prevalence in cases significantly exceeds controls. Source: {source}"
            )

        # Return immediately -- no need for the ML pipeline
        return {
            "prediction": label,
            "probability": round(probability, 4),
            "risk_tier": risk,
            "threshold": round(float(gene_data.get("threshold", 0.5)), 3),
            "aa_pos": aa_pos,
            "genomic_pos": gene_data.get("cdna_to_genomic", {}).get(mutation_data.cDNA_pos),
            "acmg_evidence": acmg_eval,
            "acmg_classification": "Pathogenic",
            "classification_method": "rule_based",
            "confidence": {
                "probability": round(probability, 4),
                "label": "High Confidence",
                "ci_lower": None,
                "ci_upper": None,
                "std": None,
                "method": "rule_based",
                "ci_note": "Rule-based classification (PVS1 null variant) — no statistical CI available",
            },
            "features_used": {
                "is_nonsense": "Ter" in mutation_data.AA_alt or "*" in mutation_data.AA_alt,
                "is_start_loss": is_start_loss,
                "aa_position": aa_pos,
            },
            "data_sources": {},
            "shap_explanation": [{"feature": "Tier 1 Protocol (PVS1 Null Variant)", "value": 9.99, "direction": "pathogenic"}],
            "founder_mutation": _founder_info,
        }
    # ─── TIER 2: MACHINE LEARNING ENGINE ──────────────────────────────────────
    # If we reach here, the variant is a missense (single amino acid change).
    # We need the full ML pipeline to predict pathogenicity.

    # Warnings accumulate throughout the pipeline and are returned in the response
    # so the user knows about any caveats (e.g., missing ESM-2 embeddings)
    _warnings = []

    # Safety check: ensure the scaler was loaded during startup
    if gene_data.get("scaler") is None:
        return JSONResponse(status_code=500, content={"error": f"ML Model artifacts not available for gene: {mutation_data.gene_name}"})

    # STEP 1: BUILD FEATURE VECTOR
    # Transform the raw variant information (gene, position, amino acids) into
    # a numeric vector of 120 engineered features. These include:
    #   - Conservation scores (PhyloP, BLOSUM62)
    #   - Protein properties (hydrophobicity change, volume change, charge change)
    #   - Structural features (is buried? in critical domain? secondary structure)
    #   - ESM-2 protein language model embeddings (20 PCA components)
    #   - EVE evolutionary coupling scores
    #   - gnomAD allele frequencies
    #   - Gene-specific domain proximity features
    raw_vector = build_feature_vector(mutation_data.cDNA_pos, mutation_data.AA_ref, mutation_data.AA_alt, mutation_data.Mutation, aa_pos, mutation_data.gene_name)
    if raw_vector.shape[1] == 0:
        return JSONResponse(status_code=500, content={"error": f"Feature names not available for gene: {mutation_data.gene_name}"})

    # Feature coverage: count how many of the 120 features are non-zero.
    # This tells the user how much information the model has to work with.
    # A variant with low coverage (many zeros) may have a less reliable prediction.
    _total_features = raw_vector.shape[1]
    _nonzero_features = int(np.count_nonzero(raw_vector[0]))
    _feature_coverage = {
        "nonzero": _nonzero_features,
        "total": _total_features,
        "percentage": round((_nonzero_features / _total_features) * 100, 1) if _total_features > 0 else 0.0,
    }

    # STEP 2: SCALE FEATURES
    # StandardScaler normalizes each feature to mean=0, std=1 so that features
    # with large values (e.g., cDNA position ~10000) do not dominate features
    # with small values (e.g., gnomAD AF ~0.0001). The MLP especially needs this.
    scaled_vector = gene_data["scaler"].transform(raw_vector)

    # STEP 3: RUN THE ENSEMBLE (XGBoost + MLP)
    # Our ensemble combines two different ML models:
    #   - XGBoost: gradient-boosted decision trees (good at tabular data, interpretable via SHAP)
    #   - MLP: multi-layer perceptron neural network (captures nonlinear interactions)
    # Each model outputs a probability [0, 1], then we blend them using gene-specific weights.
    universal_calibrator = gene_data.get("calibrator")
    nn_model = gene_data.get("ensemble_model")   # The MLP (Keras neural network)
    xgb_model = gene_data.get("booster")          # The XGBoost model

    if universal_calibrator and nn_model and xgb_model:
        # BEST CASE: Full ensemble with both models available
        nn_p = nn_model.predict(scaled_vector, verbose=0).flatten()[0]   # MLP probability
        xgb_p = xgb_model.predict(xgb.DMatrix(scaled_vector))[0]        # XGBoost probability

        # Gene-adaptive ensemble weights: each gene has its own optimal blend ratio.
        # Example: BRCA1 uses MLP-only (100% MLP) because the MLP performs better on BRCA1,
        # while RAD51C uses 80% XGBoost + 20% MLP because XGBoost dominates for that gene.
        gene_weights = _GENE_ENSEMBLE_WEIGHTS.get(mutation_data.gene_name.upper())
        if gene_weights:
            _xgb_w = gene_weights["xgb_weight"]
            _mlp_w = gene_weights["mlp_weight"]
        else:
            _xgb_w = XGB_WEIGHT  # Default weights if gene-specific not available
            _mlp_w = NN_WEIGHT

        # Blend the two model outputs into a single probability
        blended = (_xgb_w * xgb_p) + (_mlp_w * nn_p)

        # Track how much the two models disagree (high disagreement = less confidence)
        model_disagreement = float(abs(xgb_p - nn_p))

        # STEP 4: CALIBRATE THE PROBABILITY
        # Isotonic calibration maps raw model scores to well-calibrated probabilities.
        # Without calibration, a model might output 0.7 when the true rate is really 0.85.
        # We prefer gene-specific calibrators (trained on that gene's data) over the
        # universal calibrator (trained on all genes pooled together).
        gene_cal = _GENE_CALIBRATORS.get(mutation_data.gene_name.upper())
        active_calibrator = gene_cal if gene_cal is not None else universal_calibrator
        calibrator_type = "gene_specific" if gene_cal is not None else "universal"

        try:
            calibrated = active_calibrator.predict([blended])[0]
            probability = float(calibrated)
        except Exception:
            # If calibration fails, fall back to the raw blended score
            probability = float(blended)
            calibrator_type = "raw_fallback"
            _warnings.append("Calibrator failed — using raw model score. Probability may be less well-calibrated.")
    elif xgb_model and universal_calibrator:
        # FALLBACK: XGBoost-only (MLP unavailable, e.g., due to Keras version mismatch)
        xgb_p = xgb_model.predict(xgb.DMatrix(scaled_vector))[0]
        blended = float(xgb_p)
        model_disagreement = 0.0  # No second model to compare against
        _xgb_w = 1.0
        _mlp_w = 0.0

        gene_cal = _GENE_CALIBRATORS.get(mutation_data.gene_name.upper())
        active_calibrator = gene_cal if gene_cal is not None else universal_calibrator
        calibrator_type = ("gene_specific" if gene_cal is not None else "universal") + "_xgb_only"

        try:
            calibrated = active_calibrator.predict([blended])[0]
            probability = float(calibrated)
        except Exception:
            probability = float(blended)
            calibrator_type = "raw_fallback_xgb_only"
        _warnings.append("MLP model unavailable (Keras version mismatch). Using XGBoost-only predictions.")
    else:
        # No models loaded at all -- server failed to start properly
        return JSONResponse(status_code=503, content={
            "error": "ML models could not be loaded. Prediction unavailable.",
            "detail": "The server failed to initialize required model artifacts. This is a server-side issue, not a problem with your input."
        })

    # STEP 5: COMPUTE CONFIDENCE INTERVALS
    # Bootstrap CI: we trained multiple XGBoost models on resampled data.
    # By running the variant through all of them, we get a distribution of
    # predictions. The spread of this distribution tells us how confident
    # the model is. Narrow spread = high confidence; wide spread = uncertain.
    feature_names_list = gene_data.get("feature_names", [])
    bootstrap_ci = compute_bootstrap_ci(scaled_vector, feature_names_list)

    # ─── Warning Generation ────────────────────────────────────────────────
    # Detect if ESM-2 protein language model features are all zeros.
    # This happens for novel variants without precomputed embeddings.
    # The prediction is still valid but relies on the other ~100 features.
    esm2_indices = [i for i, n in enumerate(feature_names_list) if n.startswith("esm2_")]
    if esm2_indices and raw_vector.shape[1] > 0:
        esm2_vals = [raw_vector[0][i] for i in esm2_indices if i < raw_vector.shape[1]]
        if all(v == 0.0 for v in esm2_vals):
            _warnings.append("ESM-2 protein language model features are zero for this variant (no precomputed embedding). Prediction relies on other 81 features.")

    # Warn if the probability is exactly 0.5 (likely a fallback, not a real prediction)
    if probability == 0.5 and not (universal_calibrator and nn_model and xgb_model):
        _warnings.append("ML models could not be loaded. Probability is a default fallback value (0.5) and should not be trusted.")

    # Out-of-distribution warning: variants near the very end of the protein
    # have fewer training examples, so predictions may be less reliable
    try:
        max_aa_ood = GENE_MAX_AA.get(mutation_data.gene_name.upper(), 5000)
        if aa_pos > max_aa_ood * 0.98:
            _warnings.append(f"AA position {aa_pos} is near the C-terminal end of {mutation_data.gene_name} (max: {max_aa_ood}). Model has fewer training examples in terminal regions.")
    except Exception:
        pass  # OOD check is non-critical

    # Clip probability to [1%, 99%] -- no model should claim absolute certainty.
    # Even the best model can be wrong, so we never output exactly 0.0 or 1.0.
    probability = float(np.clip(probability, 0.01, 0.99))

    # ─── Confidence Interval Estimation ──────────────────────────────────────
    # Two methods, in order of preference:
    #
    # 1. Bootstrap CI (preferred): Uses multiple resampled XGBoost models to get
    #    an empirical distribution. The 5th-95th percentile range gives a 90% CI.
    #
    # 2. Beta approximation (fallback): When bootstrap models are not available
    #    (e.g., LOW_MEMORY mode), we approximate the CI using a Beta distribution
    #    parameterized by the probability and the effective sample size.
    if bootstrap_ci is not None:
        ci_lower = bootstrap_ci["ci_lower"]
        ci_upper = bootstrap_ci["ci_upper"]
        ci_width = bootstrap_ci["ci_width"]
        ci_std = round(ci_width / 3.29, 4)  # Approximate std from 90% CI width
        ci_method = "bootstrap"
    else:
        N_eff = N_EFFECTIVE  # Effective sample size (from training set)
        alpha_param = probability * N_eff + 1
        beta_param = (1 - probability) * N_eff + 1
        ci_lower = float(beta_dist.ppf(0.025, alpha_param, beta_param))
        ci_upper = float(beta_dist.ppf(0.975, alpha_param, beta_param))
        ci_std = float(beta_dist.std(alpha_param, beta_param))
        ci_width = ci_upper - ci_lower
        ci_method = "beta_approximation"

    # ─── Population-aware CI widening ────────────────────────────────────
    # For underrepresented populations, widen the CI to honestly reflect
    # higher uncertainty. The model was trained on predominantly European
    # data, so predictions for Central Asian patients carry more epistemic
    # uncertainty than the bootstrap spread alone captures.
    _pop_lower_ci = mutation_data.population.lower() if mutation_data.population else None
    _ci_widened = False
    if _pop_lower_ci in UNDERREPRESENTED_POPULATIONS:
        UNDERREP_CI_FACTOR = 1.5  # Widen CI by 50%
        ci_center = (ci_lower + ci_upper) / 2
        half_width = (ci_upper - ci_lower) / 2
        ci_lower = float(np.clip(ci_center - half_width * UNDERREP_CI_FACTOR, 0.0, 1.0))
        ci_upper = float(np.clip(ci_center + half_width * UNDERREP_CI_FACTOR, 0.0, 1.0))
        ci_width = ci_upper - ci_lower
        ci_std = round(ci_width / 3.29, 4)
        _ci_widened = True
        _warnings.append(
            f"Confidence interval widened (×{UNDERREP_CI_FACTOR}) for underrepresented population "
            f"({mutation_data.population}). Training data has limited Central Asian representation."
        )

    # Translate CI width into a human-readable confidence label
    # Narrow CI = model is confident; wide CI = model is uncertain
    if ci_width < 0.10:
        uncertainty_label = "High Confidence"
    elif ci_width < 0.25:
        uncertainty_label = "Moderate Confidence"
    else:
        uncertainty_label = "Low Confidence"

    # STEP 6: APPLY THRESHOLD TO GET BINARY CLASSIFICATION
    # The threshold (e.g., 0.2998) was optimized on the calibration set to maximize
    # balanced accuracy. If probability >= threshold, the variant is called Pathogenic.
    threshold = gene_data.get("threshold", 0.5)
    label = "Pathogenic" if probability >= threshold else "Benign"
    risk = _compute_risk_tier(probability)  # Maps probability to "high"/"low"/"uncertain"

    # STEP 7: COMPUTE SHAP EXPLANATIONS
    # SHAP (SHapley Additive exPlanations) values explain WHY the model made its
    # prediction. Each feature gets a SHAP value:
    #   - Positive SHAP = this feature pushed toward "Pathogenic"
    #   - Negative SHAP = this feature pushed toward "Benign"
    #   - Large absolute value = this feature was very influential
    #
    # Example: "PhyloP Conservation Score: +1.23 (pathogenic)" means the high
    # conservation at this position strongly suggests the variant is damaging.
    #
    # XGBoost has built-in SHAP via pred_contribs=True (no separate SHAP library needed).
    booster = gene_data.get("booster")
    if booster:
        # Get SHAP values for all 120 features (+ 1 bias term which we discard)
        shap_vals_matrix = booster.predict(xgb.DMatrix(scaled_vector), pred_contribs=True)
        sv = shap_vals_matrix[0, :-1]  # Drop the bias term (last column)

        # Pair each feature name with its SHAP value, sort by importance
        shap_pairs = list(zip(gene_data.get("feature_names", []), sv))
        shap_pairs.sort(key=lambda x: (-abs(x[1]), x[0]))  # Descending by |SHAP|, then alphabetical

        # Top 8 features for the summary card in the frontend
        top_shap = []
        for name, val in shap_pairs[:8]:
            nice = NICE_NAMES.get(name, name)  # Convert internal names to human-readable (e.g., "phylop_score" -> "PhyloP Conservation")
            top_shap.append({
                "feature": nice,
                "value": round(float(val), 4),
                "direction": "pathogenic" if val > 0 else "benign"
            })
        # Full SHAP breakdown for the "View All Features" toggle in the frontend
        all_shap = []
        for name, val in shap_pairs:
            if abs(val) < 0.0001:
                continue  # Skip near-zero features for cleanliness
            nice = NICE_NAMES.get(name, name)
            all_shap.append({
                "feature": nice,
                "value": round(float(val), 4),
                "direction": "pathogenic" if val > 0 else "benign"
            })
    else:
        top_shap = [{"feature": "SHAP Disabled", "value": 0, "direction": "neutral"}]
        all_shap = top_shap

    # STEP 8: LOOK UP DATA SOURCE SCORES
    # These are external evidence scores displayed alongside the ML prediction
    # to give the user additional context. They are NOT model inputs (those are
    # in the feature vector), but they help the user understand the variant.

    # PhyloP: evolutionary conservation score. High = position is conserved across
    # species (mutations here are more likely damaging).
    phylop = gene_data["phylop"].get(int(mutation_data.cDNA_pos), 0.0)

    # MAVE/DMS: experimental functional assay scores (e.g., BRCA1 Findlay SGE).
    # These measure the actual effect of each mutation in a lab experiment.
    mave_key = f"{mutation_data.AA_ref}{aa_pos}{mutation_data.AA_alt}"
    mave_val = gene_data["mave_v"].get(mave_key, gene_data["mave_p"].get(int(mutation_data.cDNA_pos), None))

    # AlphaMissense: DeepMind's pathogenicity predictor (shown for reference only,
    # NOT used as a model input because it causes label leakage).
    am_key = f"{mutation_data.AA_ref}{aa_pos}{mutation_data.AA_alt}"
    am_val = gene_data["am_v"].get(am_key, gene_data["am_p"].get(int(aa_pos), None))

    # AlphaFold structural features at this amino acid position
    sf = gene_data["struct"].get(int(aa_pos), {})

    # gnomAD allele frequency: how common is this variant in the general population?
    # Very common variants (high AF) are almost certainly benign.
    gnomad_val = gene_data["gnomad_v"].get(f"{mutation_data.AA_ref}{aa_pos}{mutation_data.AA_alt}", gene_data["gnomad_p"].get(int(mutation_data.cDNA_pos), 0.0))
    gnomad_af_raw = gnomad_val.get("af", 0.0) if isinstance(gnomad_val, dict) else float(gnomad_val)

    # EVE: evolutionary model of variant effect (unsupervised, no label leakage)
    eve_key = f"{mutation_data.AA_ref}{aa_pos}{mutation_data.AA_alt}"
    eve_val = gene_data["eve_v"].get(eve_key, gene_data["eve_p"].get(int(aa_pos), None))

    # Warn if the variant has a suspiciously high gnomAD allele frequency.
    # Truly pathogenic variants for rare cancer genes should be rare in the
    # general population. High AF strongly suggests the variant is benign.
    # Thresholds are gene-specific (aligned with ACMG BA1 stand-alone benign rule).
    _GENE_OOD_THRESHOLDS = {
        "BRCA1": 0.001, "BRCA2": 0.001,
        "PALB2": 0.002, "RAD51C": 0.005, "RAD51D": 0.005,
    }
    if gnomad_af_raw > _GENE_OOD_THRESHOLDS.get(mutation_data.gene_name, 0.01):
        _warnings.append(f"gnomAD allele frequency ({gnomad_af_raw:.4f}) is unusually high for a rare-disease variant. This variant may be benign or a common polymorphism.")

    # STEP 9: ACMG EVIDENCE EVALUATION
    # PM5: check if a different pathogenic missense exists at the same AA position.
    # If yes, this supports pathogenicity (same position, different substitution).
    pm5_positions = gene_data.get("pm5_positions", set())
    known_path_at_pos = int(aa_pos) in pm5_positions if pm5_positions else False

    # Data Scarcity Quantification: how many similar training examples does the
    # model have? Fewer examples = less reliable prediction.
    sf.get('domain', 'uncharacterized')
    # Map structural domain to feature column name for matching
    _domain_for_scarcity = "uncharacterized"
    if _safe_critical_domain(raw_vector, gene_data, aa_pos):
        # Determine which domain feature column is active
        feature_names_list_ds = gene_data.get("feature_names", [])
        domain_feature_cols = ["in_critical_repeat_region", "in_DNA_binding", "in_OB_folds", "in_NLS", "in_primary_interaction"]
        for dc in domain_feature_cols:
            if dc in feature_names_list_ds:
                idx_dc = feature_names_list_ds.index(dc)
                if idx_dc < raw_vector.shape[1] and raw_vector[0][idx_dc] == 1:
                    _domain_for_scarcity = dc
                    break
    _data_support = compute_data_support(
        mutation_data.gene_name, aa_pos, mutation_data.AA_ref, mutation_data.AA_alt,
        in_domain_name=_domain_for_scarcity,
    )

    # Extract population-specific gnomAD allele frequencies for ACMG rules.
    # Different populations have different carrier frequencies for these genes,
    # so ACMG thresholds are population-aware (e.g., BRCA1 is more common in
    # Ashkenazi Jewish populations).
    _gnomad_pop_afs = {}
    if isinstance(gnomad_val, dict):
        _gnomad_pop_afs = {
            'gnomad_af_afr': float(gnomad_val.get('afr', 0.0)),
            'gnomad_af_amr': float(gnomad_val.get('amr', 0.0)),
            'gnomad_af_asj': float(gnomad_val.get('asj', 0.0)),
            'gnomad_af_eas': float(gnomad_val.get('eas', 0.0)),
            'gnomad_af_fin': float(gnomad_val.get('fin', 0.0)),
            'gnomad_af_nfe': float(gnomad_val.get('nfe', 0.0)),
            'gnomad_af_sas': float(gnomad_val.get('sas', 0.0)),
        }
    # Build the feature dictionary that the ACMG rule engine needs.
    # This is different from the ML feature vector -- ACMG rules use a smaller
    # set of interpretable features (domain, distances, AF, etc.)
    features_dict = {
        'domain': sf.get('domain', 'uncharacterized'),
        'dist_dna': sf.get('dist_dna', 999.0),
        'dist_palb2': sf.get('dist_palb2', 999.0),
        'gnomad_af': gnomad_af_raw,
        'in_critical_domain': _safe_critical_domain(raw_vector, gene_data, aa_pos),
        'known_pathogenic_at_pos': known_path_at_pos,
        **_gnomad_pop_afs,  # Population-specific AFs for population-aware ACMG
    }

    # Evaluate ACMG rules: returns a dict of triggered rules (e.g., PP3, PM1, BA1)
    # with human-readable explanations for each
    acmg_eval = evaluate_acmg_rules(features_dict, probability, gene_name=mutation_data.gene_name, population=mutation_data.population, founder_mutation=_founder_info)
    # Combine triggered rules into an overall classification
    # (Pathogenic / Likely Pathogenic / VUS / Likely Benign / Benign)
    acmg_classification = combine_acmg_evidence(acmg_eval)

    # When a specific population is selected, also compute global (all-populations)
    # ACMG so the frontend can show a side-by-side comparison
    _acmg_global = None
    if mutation_data.population:
        _acmg_global = evaluate_acmg_rules(features_dict, probability, gene_name=mutation_data.gene_name, population=None)

    # STEP 10: ADDITIONAL EXPLAINABILITY FEATURES

    # Contrastive explanation: finds the nearest training example with the opposite
    # label and shows which features differ. Answers "what would need to change
    # for this variant to be classified differently?"
    _contrastive = find_contrastive_explanation(
        mutation_data.gene_name, scaled_vector[0], probability, feature_names_list,
        nice_names=NICE_NAMES,
    )

    # Conformal prediction: provides prediction sets with coverage guarantees.
    # E.g., "with 90% probability, the true class is in {Pathogenic}"
    _conformal = compute_conformal_set(probability, mutation_data.gene_name)

    latency_ms = (time.perf_counter() - t_start) * 1000  # Total prediction latency

    # Build the standard HGVS cDNA notation (e.g., "NM_000059.4:c.8023A>G")
    # by prepending the transcript ID for this gene
    _transcript = GENE_TRANSCRIPTS.get(mutation_data.gene_name.upper(), "")
    _hgvs_c = f"{_transcript}:c.{mutation_data.cDNA_pos}{mutation_data.Mutation}" if _transcript else f"c.{mutation_data.cDNA_pos}{mutation_data.Mutation}"

    # ─── ASSEMBLE THE FINAL RESPONSE ──────────────────────────────────────
    # This is the rich JSON object returned to the frontend. It contains
    # everything needed to display the prediction result page.
    result = {
        "prediction": label,
        "probability": round(probability, 4),
        "risk_tier": risk,
        "threshold": round(float(threshold), 3),
        "aa_pos": aa_pos,
        "hgvs_c": _hgvs_c,
        "genomic_pos": gene_data["cdna_to_genomic"].get(mutation_data.cDNA_pos),
        "acmg_evidence": acmg_eval,
        "acmg_classification": acmg_classification,
        "classification_method": "ml_ensemble",
        "confidence": {
            "probability": round(probability, 4),
            "label": uncertainty_label,
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "ci_width": round(ci_width, 4),
            "std": round(ci_std, 4),
            "method": ci_method,
            "ci_widened_for_population": _ci_widened,
            "ci_note": ("CI widened ×1.5 for underrepresented population; " if _ci_widened else "") + ("CI reflects XGBoost model variance only; MLP component uncertainty not captured" if ci_method == "bootstrap" else ""),
        },
        "features_used": {
            "blosum62_score": get_blosum62(mutation_data.AA_ref, mutation_data.AA_alt),
            "volume_diff": float(abs(AA_VOLUME.get(mutation_data.AA_ref, 0) - AA_VOLUME.get(mutation_data.AA_alt, 0))),
            "hydro_diff": round(abs(AA_HYDROPHOBICITY.get(mutation_data.AA_ref, 0) - AA_HYDROPHOBICITY.get(mutation_data.AA_alt, 0)), 2),
            "charge_changed": get_charge(mutation_data.AA_ref) != get_charge(mutation_data.AA_alt),
            "is_nonsense": mutation_data.AA_alt == "Ter",
            "in_critical_domain": _safe_critical_domain(raw_vector, gene_data, aa_pos),
            "aa_position": aa_pos,
        },
        "data_sources": {
            "phylop": {"score": round(phylop, 3), "label": "Ultra-conserved" if phylop > 7.0 else ("Conserved" if phylop > 4.0 else "Variable")},
            "mave": {"score": round(mave_val, 3) if mave_val is not None else None,
                     "label": "Abnormal" if mave_val is not None and mave_val < 1.49 else ("Normal" if mave_val is not None and mave_val > 2.50 else ("Intermediate" if mave_val is not None else "No data"))},
            "alphamissense": {"score": round(am_val, 3) if am_val is not None else None,
                              "label": "Pathogenic" if am_val is not None and am_val > 0.564 else ("Benign" if am_val is not None and am_val < 0.340 else ("Ambiguous" if am_val is not None else "No data"))},
            "eve": {"score": round(eve_val, 3) if eve_val is not None else None,
                    "label": "Pathogenic" if eve_val is not None and eve_val > 0.5 else ("Benign" if eve_val is not None and eve_val <= 0.5 else "No data")},
            "structure": {
                "domain": sf.get("domain", "Unknown"),
                "rsa": round(sf.get("rsa", 0.4), 3) if sf else None,
                "secondary_structure": {"H": "Helix", "E": "Sheet", "C": "Coil"}.get(sf.get("ss", "C"), "Coil") if sf else None,
                "is_buried": sf.get("is_buried", False) if sf else None,
                "is_dna_contact": sf.get("is_dna_contact", False) if sf else None,
            },
        },
        "shap_explanation": top_shap,
        "shap_all": all_shap,
        "gene_reliability": GENE_RELIABILITY.get(mutation_data.gene_name.upper(), {"auc": None, "tier": "unknown", "note": ""}),
        "model_disagreement": round(model_disagreement, 4) if model_disagreement is not None else None,
        "ensemble_weights": {"xgb": round(_xgb_w, 2), "mlp": round(_mlp_w, 2)} if model_disagreement is not None else None,
        "feature_coverage": _feature_coverage,
        "data_support": _data_support,
        "contrastive_explanation": _contrastive,
        "conformal_prediction": _conformal,
        "calibrator_type": calibrator_type,
        "founder_mutation": _founder_info,
        "acmg_global": _acmg_global,
        "population_used": mutation_data.population if mutation_data.population else None,
        "population_equity": {
            "frequency_independent_pct": get_frequency_independence_stats()["pct_independent"],
            "pm2_adjusted": "PM2_insufficient" in acmg_eval,
            "founder_mutation_evidence": "PS4" in acmg_eval,
            "thresholds_relaxed": mutation_data.population and mutation_data.population.lower() in UNDERREPRESENTED_POPULATIONS,
        } if mutation_data.population and mutation_data.population.lower() in UNDERREPRESENTED_POPULATIONS else None,
        "warnings": _warnings if _warnings else None,
    }
    # Cache the result so identical future requests are instant
    _pred_cache_set(pred_cache_key, result)

    # Update server-level metrics (thread-safe)
    with _metrics_lock:
        _metrics["predictions"] += 1
        _metrics["total_predict_ms"] += latency_ms
    return result


# ============================================================
# Utility Endpoints (kept in main.py because they are small
# and closely tied to the app object)
# ============================================================

@app.get("/")
async def root():
    """Root endpoint: serves the frontend HTML if it exists, otherwise returns
    a JSON summary of the API status and model performance."""
    _frontend_index = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.isfile(_frontend_index):
        return FileResponse(_frontend_index)
    # No frontend found -- return a JSON API info page instead
    uni = _get_universal_models()
    per_gene_aucs = [0.994, 0.824, 0.785, 0.747, 0.605]
    return {
        "status": "SteppeDNA API running",
        "version": STEPPEDNA_VERSION,
        "models": "Universal (Multi-Gene HR)",
        "model_type": "Ensemble (XGBoost + MLP, gene-adaptive weights)",
        "macro_avg_auc": round(float(np.mean(per_gene_aucs)), 3),
        "per_gene_auc": {
            "BRCA2": 0.994, "RAD51D": 0.824, "RAD51C": 0.785,
            "BRCA1": 0.747, "PALB2": 0.605
        },
        "shap": uni.get("booster") is not None,
        "vcf": True
    }

# Note: startup logic is handled by the lifespan context manager (defined at app creation)


@app.get("/model_metrics", tags=["System"], summary="Model evaluation metrics",
         description="Returns trained model ROC-AUC, MCC, balanced accuracy, confusion matrix, and per-gene performance from the test set.")
async def model_metrics():
    metrics_path = os.path.join(DATA_DIR, "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return JSONResponse(content=json.load(f))
    return JSONResponse(status_code=404, content={"error": "Metrics not found. Retrain model."})


@app.get("/health", tags=["System"], summary="Health check",
         description="Quick liveness check. Returns 200 immediately. Use /health/ready for full model readiness.")
async def health():
    return {"status": "ok"}


@app.get("/health/ready", tags=["System"], summary="Readiness check",
         description="Validates all critical ML model artifacts are loaded and ready.")
async def health_ready():
    uni = _get_universal_models()
    checks = {
        "universal_models": uni.get("ensemble_model") is not None or uni.get("booster") is not None,
        "universal_scaler": uni.get("scaler") is not None,
        "universal_calibrator": uni.get("calibrator") is not None,
        "feature_names": len(uni.get("feature_names", [])),
        "shap_booster": uni.get("booster") is not None,
    }
    all_ok = all(checks.values())
    status_code = 200 if all_ok else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "healthy" if all_ok else "degraded", "checks": checks},
    )

# ─── Safe File Serving ────────────────────────────────────────────────────────
# Prevents path traversal attacks (e.g., "../../etc/passwd") by verifying
# that the resolved file path stays within the project directory.
_PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))


def _safe_file_response(path: str, **kwargs):
    """Serve a file only if its resolved path stays within the project directory.
    This prevents directory traversal attacks."""
    resolved = os.path.realpath(path)
    if not resolved.startswith(_PROJECT_ROOT):
        return JSONResponse(status_code=403, content={"error": "Access denied"})
    if not os.path.isfile(resolved):
        return JSONResponse(status_code=404, content={"error": "File not found"})
    return FileResponse(resolved, **kwargs)


@app.get("/docs/validation-report", tags=["System"], summary="Serve VALIDATION_REPORT.md")
async def validation_report():
    path = os.path.join(os.path.dirname(__file__), "..", "VALIDATION_REPORT.md")
    return _safe_file_response(path, media_type="text/markdown", filename="VALIDATION_REPORT.md")


@app.get("/structure/brca2/{fragment}", tags=["System"], summary="Serve BRCA2 AlphaFold fragment PDB")
async def brca2_fragment(fragment: int):
    """Serve local AlphaFold BRCA2 fragment PDB files (F1-F12).
    BRCA2 is too large for single AlphaFold prediction, split into 12 overlapping fragments."""
    if fragment < 1 or fragment > 12:
        return JSONResponse(status_code=404, content={"error": "Fragment must be 1-12"})
    path = os.path.join(os.path.dirname(__file__), "..", "data", "brca2_fragments",
                        f"AF-P51587-F{fragment}-model_v6.pdb")
    return _safe_file_response(path, media_type="chemical/x-pdb",
                               filename=f"AF-P51587-F{fragment}-model_v6.pdb")


@app.get("/structure/{gene}", tags=["System"], summary="Serve AlphaFold PDB for any gene")
async def gene_structure(gene: str):
    """Serve local AlphaFold PDB files for non-BRCA2 genes."""
    gene_lower = gene.lower()
    allowed = {"brca1", "palb2", "rad51c", "rad51d"}
    if gene_lower not in allowed:
        return JSONResponse(status_code=404, content={"error": f"No structure for {gene}"})
    path = os.path.join(os.path.dirname(__file__), "..", "data", "archived_raw_pdb",
                        f"alphafold_{gene_lower}.pdb")
    return _safe_file_response(path, media_type="chemical/x-pdb",
                               filename=f"alphafold_{gene_lower}.pdb")


# ─── Static File Serving ──────────────────────────────────────────────────────
# Mount the frontend directory (HTML/JS/CSS) as static files at the root path.
# This means visiting "/" in a browser will serve the frontend's index.html.
# The html=True flag enables serving index.html for directory URLs.
_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")


# ─── Direct Execution ────────────────────────────────────────────────────────
# When running this file directly (python backend/main.py), start a local
# development server on port 8000. In production, Render runs the app via
# gunicorn/uvicorn with different settings.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
