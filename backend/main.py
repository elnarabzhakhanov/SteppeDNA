"""
SteppeDNA - FastAPI Backend
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
import numpy as np
import sys
import os
import re

# Ensure the root project directory is in the Python path for direct execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import time
import hashlib
import asyncio
import threading
import xgboost as xgb
from scipy.stats import beta as beta_dist

_LOG_FORMAT = os.getenv("LOG_FORMAT", "text")  # "text" or "json"
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

_server_start_time = time.monotonic()
_metrics = {"predictions": 0, "vcf_uploads": 0, "total_predict_ms": 0.0}

# ─── Imports from refactored modules ─────────────────────────────────────────

from backend.models import (
    DATA_DIR, SUPPORTED_GENES, MAX_CDNA_LENGTHS, GENE_MAX_AA,
    GENE_RELIABILITY, XGB_WEIGHT, NN_WEIGHT,
    N_EFFECTIVE, VALID_AA_CODES,
    _GENE_CALIBRATORS,
    _load_gene_calibrators, _GENE_ENSEMBLE_WEIGHTS,
    _load_gene_ensemble_weights, _load_bootstrap_models, _load_active_learning_priorities,
    _load_conformal_thresholds, _get_universal_models,
)

from backend.explanations import (
    _build_training_index, compute_data_support, _build_contrastive_index,
    find_contrastive_explanation, compute_bootstrap_ci, compute_conformal_set,
)

from backend.features import (
    get_gene_data, build_feature_vector,
    NICE_NAMES, _safe_critical_domain,
)

from backend.external_api import (
    _pred_cache_get, _pred_cache_set,
)

from backend.vcf import (
    _compute_risk_tier,
    GENE_TRANSCRIPTS,
)

from backend.feature_engineering import (
    AA_HYDROPHOBICITY, AA_VOLUME, get_blosum62,
    get_charge,
)
from backend.acmg_rules import evaluate_acmg_rules, combine_acmg_evidence
from backend import __version__ as STEPPEDNA_VERSION
# Database storage disabled — privacy: no variant data is stored server-side
# from backend.database import init_db, record_analysis


# ─── Gene Config Validation ──────────────────────────────────────────────────
_GENE_CONFIG_REQUIRED_KEYS = {"cds_length", "aa_length", "domains", "chromosome", "strand"}
_unavailable_genes: set = set()


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
    """Verify SHA256 checksums of model files against data/checksums.json."""
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
_FOUNDER_MUTATIONS = {}


def _load_founder_mutations():
    """Load Kazakh/Central Asian founder mutation data from JSON."""
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
    """Application lifespan: pre-load models on startup."""
    logger.info("Initializing universal ML models...")
    _get_universal_models()
    _validate_gene_configs()
    _verify_model_checksums()
    _build_training_index()
    _build_contrastive_index()
    _load_gene_calibrators()
    _load_gene_ensemble_weights()
    _load_bootstrap_models()
    _load_active_learning_priorities()
    _load_founder_mutations()
    _load_conformal_thresholds()
    logger.info("Server ready.")

    # Background task: periodic rate limiter cleanup every 5 minutes
    async def _periodic_rate_cleanup():
        while True:
            await asyncio.sleep(300)  # 5 minutes
            now = time.monotonic()
            with _rate_lock:
                stale = [k for k, v in _rate_counts.items()
                         if not v or (now - max(v)) > RATE_WINDOW * 2]
                for k in stale:
                    del _rate_counts[k]
                if stale:
                    logger.info(f"[RATE-CLEANUP] Purged {len(stale)} stale IPs")
    cleanup_task = asyncio.create_task(_periodic_rate_cleanup())

    yield  # Application runs here

    cleanup_task.cancel()
    logger.info("Server shutting down.")

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
    lifespan=lifespan,
)

# ─── CORS ────────────────────────────────────────────────────────────────────
_ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost,http://127.0.0.1,http://localhost:8000,http://127.0.0.1:8000,http://localhost:8080,http://127.0.0.1:8080,http://localhost:5500,http://127.0.0.1:5500,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _ALLOWED_ORIGINS],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# ─── Middleware ────────────────────────────────────────────────────────────────
from backend.middleware import register_middleware, _rate_lock, _rate_counts, RATE_WINDOW
register_middleware(app)

_metrics_lock = threading.Lock()

# ─── Include Routers from refactored modules ────────────────────────────────
from backend.external_api import router as external_api_router
from backend.vcf import router as vcf_router
from backend.cohort import router as cohort_router

app.include_router(external_api_router)
app.include_router(vcf_router)
app.include_router(cohort_router)


# ============================================================
# API - Prediction Endpoint
# ============================================================

class MutationInput(BaseModel):
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
        if v.upper() not in SUPPORTED_GENES:
            raise ValueError(f'Unsupported gene: {v}. Supported: {", ".join(sorted(SUPPORTED_GENES))}')
        return v.upper()

    @field_validator('cDNA_pos')
    @classmethod
    def validate_cdna_pos(cls, v):
        if v < 1:
            raise ValueError('cDNA_pos must be positive')
        return v

    @field_validator('AA_ref', 'AA_alt')
    @classmethod
    def validate_aa(cls, v):
        # Normalize capitalization: 'asp' -> 'Asp', 'ALA' -> 'Ala'
        if len(v) >= 2:
            v = v[0].upper() + v[1:].lower()
        if v not in VALID_AA_CODES:
            raise ValueError(f'Invalid amino acid code: {v}. Must be a 3-letter code (e.g. Ala, Val, Gly)')
        return v

    @field_validator('Mutation')
    @classmethod
    def validate_mutation(cls, v):
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
    t_start = time.perf_counter()
    max_len = MAX_CDNA_LENGTHS.get(mutation_data.gene_name.upper(), 15000)
    if mutation_data.cDNA_pos > max_len:
        return JSONResponse(status_code=400, content={"error": f"cDNA position cannot exceed {max_len} for {mutation_data.gene_name}"})

    # ─── Prediction cache (avoids re-computing identical variants) ──────────
    pred_cache_key = f"pred_{mutation_data.gene_name}_{mutation_data.cDNA_pos}_{mutation_data.AA_ref}_{mutation_data.AA_alt}_{mutation_data.Mutation}"
    cached_pred = _pred_cache_get(pred_cache_key)
    if cached_pred is not None:
        return cached_pred

    aa_pos = mutation_data.AA_pos if mutation_data.AA_pos > 0 else (mutation_data.cDNA_pos - 1) // 3 + 1

    # Validate AA position against protein length
    max_aa = GENE_MAX_AA.get(mutation_data.gene_name.upper(), 5000)
    if aa_pos > max_aa:
        return JSONResponse(status_code=400, content={
            "error": f"AA position {aa_pos} exceeds {mutation_data.gene_name} protein length ({max_aa} AA)"
        })

    gene_data = get_gene_data(mutation_data.gene_name)

    # ─── TIER 1: RULE INTERCEPTOR ───────────────────────────────────────────────
    is_truncating = (
        mutation_data.AA_alt in ["Ter", "*", "Fs"] or
        any(x in mutation_data.Mutation.lower() for x in ["fs", "del", "ins", "dup"])
    )
    is_start_loss = (
        aa_pos == 1
        and mutation_data.AA_ref in ("Met", "M")
        and mutation_data.AA_alt not in ("Met", "M")
    )

    if is_truncating or is_start_loss:
        label = "Pathogenic"
        probability = 0.99
        # NOTE: Tier 1 truncating variants use "high (Truncating)" to distinguish from
        # Tier 2 missense "high"/"low"/"uncertain". Frontend handles both via .includes("high").
        risk = "high (Truncating)"
        if is_start_loss:
            acmg_eval = {"PVS1": "Start-loss variant (p.Met1?) \u2014 loss of initiator methionine abolishes translation initiation."}
        else:
            acmg_eval = {"PVS1": "Pathogenic truncating null variant (Nonsense/Frameshift) in a recognized tumor suppressor."}

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
        }
    # ─── TIER 2: MACHINE LEARNING ENGINE ──────────────────────────────────────
    # Track warnings for response (initialized early so calibrator fallback can append)
    _warnings = []

    if gene_data.get("scaler") is None:
        return JSONResponse(status_code=500, content={"error": f"ML Model artifacts not available for gene: {mutation_data.gene_name}"})

    raw_vector = build_feature_vector(mutation_data.cDNA_pos, mutation_data.AA_ref, mutation_data.AA_alt, mutation_data.Mutation, aa_pos, mutation_data.gene_name)
    if raw_vector.shape[1] == 0:
        return JSONResponse(status_code=500, content={"error": f"Feature names not available for gene: {mutation_data.gene_name}"})

    # Feature coverage: count non-zero features for transparency
    _total_features = raw_vector.shape[1]
    _nonzero_features = int(np.count_nonzero(raw_vector[0]))
    _feature_coverage = {
        "nonzero": _nonzero_features,
        "total": _total_features,
        "percentage": round((_nonzero_features / _total_features) * 100, 1) if _total_features > 0 else 0.0,
    }

    scaled_vector = gene_data["scaler"].transform(raw_vector)

    universal_calibrator = gene_data.get("calibrator")
    nn_model = gene_data.get("ensemble_model")
    xgb_model = gene_data.get("booster")

    if universal_calibrator and nn_model and xgb_model:
        # Full ensemble: XGBoost + MLP
        nn_p = nn_model.predict(scaled_vector, verbose=0).flatten()[0]
        xgb_p = xgb_model.predict(xgb.DMatrix(scaled_vector))[0]

        # Gene-adaptive ensemble weights (Item 38)
        gene_weights = _GENE_ENSEMBLE_WEIGHTS.get(mutation_data.gene_name.upper())
        if gene_weights:
            _xgb_w = gene_weights["xgb_weight"]
            _mlp_w = gene_weights["mlp_weight"]
        else:
            _xgb_w = XGB_WEIGHT
            _mlp_w = NN_WEIGHT
        blended = (_xgb_w * xgb_p) + (_mlp_w * nn_p)
        model_disagreement = float(abs(xgb_p - nn_p))

        # Per-gene calibrator (Item 45): use gene-specific if available, else universal
        gene_cal = _GENE_CALIBRATORS.get(mutation_data.gene_name.upper())
        active_calibrator = gene_cal if gene_cal is not None else universal_calibrator
        calibrator_type = "gene_specific" if gene_cal is not None else "universal"

        try:
            calibrated = active_calibrator.predict([blended])[0]
            probability = float(calibrated)
        except Exception:
            probability = float(blended)
            calibrator_type = "raw_fallback"
            _warnings.append("Calibrator failed — using raw model score. Probability may be less well-calibrated.")
    elif xgb_model and universal_calibrator:
        # XGBoost-only fallback (MLP unavailable due to Keras version mismatch)
        xgb_p = xgb_model.predict(xgb.DMatrix(scaled_vector))[0]
        blended = float(xgb_p)
        model_disagreement = 0.0  # no second model to compare
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
        return JSONResponse(status_code=503, content={
            "error": "ML models could not be loaded. Prediction unavailable.",
            "detail": "The server failed to initialize required model artifacts. This is a server-side issue, not a problem with your input."
        })

    # Bootstrap CI computation (Item 39)
    feature_names_list = gene_data.get("feature_names", [])
    bootstrap_ci = compute_bootstrap_ci(scaled_vector, feature_names_list)

    # Detect if ESM-2 features are all zeros (novel variant with no precomputed embedding)
    esm2_indices = [i for i, n in enumerate(feature_names_list) if n.startswith("esm2_")]
    if esm2_indices and raw_vector.shape[1] > 0:
        esm2_vals = [raw_vector[0][i] for i in esm2_indices if i < raw_vector.shape[1]]
        if all(v == 0.0 for v in esm2_vals):
            _warnings.append("ESM-2 protein language model features are zero for this variant (no precomputed embedding). Prediction relies on other 81 features.")

    # Warn if we fell back to p=0.5 due to missing models
    if probability == 0.5 and not (universal_calibrator and nn_model and xgb_model):
        _warnings.append("ML models could not be loaded. Probability is a default fallback value (0.5) and should not be trusted.")

    # Out-of-distribution feature warnings (AA position)
    try:
        max_aa_ood = GENE_MAX_AA.get(mutation_data.gene_name.upper(), 5000)
        if aa_pos > max_aa_ood * 0.98:
            _warnings.append(f"AA position {aa_pos} is near the C-terminal end of {mutation_data.gene_name} (max: {max_aa_ood}). Model has fewer training examples in terminal regions.")
    except Exception:
        pass  # OOD check is non-critical

    # Clip to [0.5%, 99.5%] -- no model should claim absolute certainty
    probability = float(np.clip(probability, 0.01, 0.99))

    # Confidence estimation: prefer bootstrap CI (empirical) over Beta approximation
    if bootstrap_ci is not None:
        # Use bootstrap-derived CI (90% CI: 5th-95th percentile)
        ci_lower = bootstrap_ci["ci_lower"]
        ci_upper = bootstrap_ci["ci_upper"]
        ci_width = bootstrap_ci["ci_width"]
        ci_std = round(ci_width / 3.29, 4)  # approximate std from 90% CI width
        ci_method = "bootstrap"
    else:
        # Fallback: Beta-distribution approximation
        N_eff = N_EFFECTIVE
        alpha_param = probability * N_eff + 1
        beta_param = (1 - probability) * N_eff + 1
        ci_lower = float(beta_dist.ppf(0.025, alpha_param, beta_param))
        ci_upper = float(beta_dist.ppf(0.975, alpha_param, beta_param))
        ci_std = float(beta_dist.std(alpha_param, beta_param))
        ci_width = ci_upper - ci_lower
        ci_method = "beta_approximation"

    # Confidence label derived from CI width (not distance-from-0.5)
    if ci_width < 0.10:
        uncertainty_label = "High Confidence"
    elif ci_width < 0.25:
        uncertainty_label = "Moderate Confidence"
    else:
        uncertainty_label = "Low Confidence"

    threshold = gene_data.get("threshold", 0.5)
    # Use the production threshold
    label = "Pathogenic" if probability >= threshold else "Benign"
    risk = _compute_risk_tier(probability)

    booster = gene_data.get("booster")
    if booster:
        shap_vals_matrix = booster.predict(xgb.DMatrix(scaled_vector), pred_contribs=True)
        sv = shap_vals_matrix[0, :-1]

        shap_pairs = list(zip(gene_data.get("feature_names", []), sv))
        # Stable sort: by absolute SHAP value descending, then by feature name for determinism
        shap_pairs.sort(key=lambda x: (-abs(x[1]), x[0]))
        top_shap = []
        for name, val in shap_pairs[:8]:
            nice = NICE_NAMES.get(name, name)
            top_shap.append({
                "feature": nice,
                "value": round(float(val), 4),
                "direction": "pathogenic" if val > 0 else "benign"
            })
        # Full SHAP breakdown for "View All Features" toggle
        all_shap = []
        for name, val in shap_pairs:
            if abs(val) < 0.0001:
                continue  # skip near-zero features for cleanliness
            nice = NICE_NAMES.get(name, name)
            all_shap.append({
                "feature": nice,
                "value": round(float(val), 4),
                "direction": "pathogenic" if val > 0 else "benign"
            })
    else:
        top_shap = [{"feature": "SHAP Disabled", "value": 0, "direction": "neutral"}]
        all_shap = top_shap

    phylop = gene_data["phylop"].get(int(mutation_data.cDNA_pos), 0.0)
    mave_key = f"{mutation_data.AA_ref}{aa_pos}{mutation_data.AA_alt}"
    mave_val = gene_data["mave_v"].get(mave_key, gene_data["mave_p"].get(int(mutation_data.cDNA_pos), None))
    am_key = f"{mutation_data.AA_ref}{aa_pos}{mutation_data.AA_alt}"
    am_val = gene_data["am_v"].get(am_key, gene_data["am_p"].get(int(aa_pos), None))

    sf = gene_data["struct"].get(int(aa_pos), {})

    gnomad_val = gene_data["gnomad_v"].get(f"{mutation_data.AA_ref}{aa_pos}{mutation_data.AA_alt}", gene_data["gnomad_p"].get(int(mutation_data.cDNA_pos), 0.0))
    gnomad_af_raw = gnomad_val.get("af", 0.0) if isinstance(gnomad_val, dict) else float(gnomad_val)
    eve_key = f"{mutation_data.AA_ref}{aa_pos}{mutation_data.AA_alt}"
    eve_val = gene_data["eve_v"].get(eve_key, gene_data["eve_p"].get(int(aa_pos), None))

    # OOD warning for gnomAD allele frequency (aligned with BA1 thresholds)
    _GENE_OOD_THRESHOLDS = {
        "BRCA1": 0.001, "BRCA2": 0.001,
        "PALB2": 0.002, "RAD51C": 0.005, "RAD51D": 0.005,
    }
    if gnomad_af_raw > _GENE_OOD_THRESHOLDS.get(mutation_data.gene_name, 0.01):
        _warnings.append(f"gnomAD allele frequency ({gnomad_af_raw:.4f}) is unusually high for a rare-disease variant. This variant may be benign or a common polymorphism.")

    # PM5: check if a different pathogenic missense exists at this AA position
    pm5_positions = gene_data.get("pm5_positions", set())
    known_path_at_pos = int(aa_pos) in pm5_positions if pm5_positions else False

    # Data Scarcity Quantification (Item 41)
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

    # Extract population-specific gnomAD AFs for ACMG rules
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
    features_dict = {
        'domain': sf.get('domain', 'uncharacterized'),
        'dist_dna': sf.get('dist_dna', 999.0),
        'dist_palb2': sf.get('dist_palb2', 999.0),
        'gnomad_af': gnomad_af_raw,
        'in_critical_domain': _safe_critical_domain(raw_vector, gene_data, aa_pos),
        'known_pathogenic_at_pos': known_path_at_pos,
        **_gnomad_pop_afs,  # population-specific AFs for population-aware ACMG
    }
    acmg_eval = evaluate_acmg_rules(features_dict, probability, gene_name=mutation_data.gene_name, population=mutation_data.population)
    acmg_classification = combine_acmg_evidence(acmg_eval)
    # When population is set, also compute global ACMG for side-by-side comparison
    _acmg_global = None
    if mutation_data.population:
        _acmg_global = evaluate_acmg_rules(features_dict, probability, gene_name=mutation_data.gene_name, population=None)

    # Contrastive Explanation Pairs (Item 43)
    _contrastive = find_contrastive_explanation(
        mutation_data.gene_name, scaled_vector[0], probability, feature_names_list,
        nice_names=NICE_NAMES,
    )

    # Split Conformal Prediction (Item 5.1)
    _conformal = compute_conformal_set(probability, mutation_data.gene_name)

    latency_ms = (time.perf_counter() - t_start) * 1000

    # Kazakh/Central Asian founder mutation check
    _founder_info = None
    hgvs_c = f"c.{mutation_data.cDNA_pos}{mutation_data.Mutation}"
    gene_founders = _FOUNDER_MUTATIONS.get(mutation_data.gene_name.upper(), {})
    if hgvs_c in gene_founders:
        _founder_info = gene_founders[hgvs_c]
        _founder_info["is_founder"] = True
    elif not _founder_info:
        # Try matching by cDNA position alone (some notations may differ)
        for fhgvs, finfo in gene_founders.items():
            if str(mutation_data.cDNA_pos) in fhgvs:
                _founder_info = finfo
                _founder_info["is_founder"] = True
                _founder_info["match_type"] = "position_approximate"
                break

    _transcript = GENE_TRANSCRIPTS.get(mutation_data.gene_name.upper(), "")
    _hgvs_c = f"{_transcript}:c.{mutation_data.cDNA_pos}{mutation_data.Mutation}" if _transcript else f"c.{mutation_data.cDNA_pos}{mutation_data.Mutation}"

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
            "ci_note": "CI reflects XGBoost model variance only; MLP component uncertainty not captured" if ci_method == "bootstrap" else None,
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
        "warnings": _warnings if _warnings else None,
    }
    _pred_cache_set(pred_cache_key, result)
    with _metrics_lock:
        _metrics["predictions"] += 1
        _metrics["total_predict_ms"] += latency_ms
    return result


# ============================================================
# Utility Endpoints (kept in main.py)
# ============================================================

@app.get("/")
async def root():
    _frontend_index = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.isfile(_frontend_index):
        return FileResponse(_frontend_index)
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

_PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))


def _safe_file_response(path: str, **kwargs):
    """Serve a file only if its resolved path stays within the project directory."""
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


_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
