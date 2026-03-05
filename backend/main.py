"""
SteppeDNA - FastAPI Backend
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, field_validator
import numpy as np
import pickle
import sys
import os

# Ensure the root project directory is in the Python path for direct execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
import logging
import logging.handlers
import time
import uuid
import hashlib
import asyncio
import xgboost as xgb
from scipy.stats import beta as beta_dist
from scipy.spatial import KDTree

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
        logger.warning("[CHECKSUM] data/checksums.json not found — skipping verification. Run scripts/generate_checksums.py to create it.")
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
            logger.warning(f"[CHECKSUM] Mismatch for {filename}: expected {expected_hash[:12]}... got {actual_hash[:12]}...")
        else:
            logger.info(f"[CHECKSUM] {filename} OK")

# ─── Prediction Audit Logger ─────────────────────────────────────────────────
_audit_logger = logging.getLogger("steppedna.audit")
_audit_logger.setLevel(logging.INFO)
_audit_logger.propagate = False  # Don't duplicate to root logger
_LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(_LOGS_DIR, exist_ok=True)
_audit_handler = logging.handlers.RotatingFileHandler(
    os.path.join(_LOGS_DIR, "predictions.jsonl"),
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=5,
    encoding="utf-8",
)
_audit_handler.setFormatter(logging.Formatter("%(message)s"))
_audit_logger.addHandler(_audit_handler)

def _log_prediction_audit(*, gene: str, hgvs_c: str, hgvs_p: str,
                          probability: float, risk_tier: str,
                          request_id: str, response_time_ms: float):
    """Write a structured JSON audit record for a prediction."""
    import datetime
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "gene": gene,
        "hgvs_c": hgvs_c,
        "hgvs_p": hgvs_p,
        "probability": round(probability, 4),
        "risk_tier": risk_tier,
        "request_id": request_id,
        "response_time_ms": round(response_time_ms, 1),
    }
    _audit_logger.info(json.dumps(record))

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
    _load_conformal_thresholds()
    logger.info("Server ready.")
    try:
        init_db()
    except Exception as e:
        logger.warning(f"[DB] Database init failed (non-fatal): {e}")

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

app = FastAPI(
    title="SteppeDNA API",
    description="Pan-gene variant pathogenicity classifier for Homologous Recombination DNA repair genes (BRCA1, BRCA2, PALB2, RAD51C, RAD51D). "
                "Ensemble of XGBoost + MLP with isotonic calibration, trained on 19,000+ ClinVar & gnomAD variants across 103 engineered features.",
    version="5.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={"name": "SteppeDNA Team"},
    license_info={"name": "Research Use Only"},
    lifespan=lifespan,
)

# ─── CORS ────────────────────────────────────────────────────────────────────
# In production, set ALLOWED_ORIGINS env var to a comma-separated list of allowed origins.
# Defaults to permissive localhost origins for development.
_ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost,http://127.0.0.1,http://localhost:8000,http://127.0.0.1:8000,null").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _ALLOWED_ORIGINS],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# ─── API Key ──────────────────────────────────────────────────────────────────
# Set STEPPEDNA_API_KEY env var to enable key-based access control.
# API key is ONLY enforced when ENVIRONMENT=production.
# In development, all endpoints are open (no key needed) to simplify local testing.
# Rate limiting always applies regardless of environment.
_API_KEY = os.getenv("STEPPEDNA_API_KEY")
if os.getenv("ENVIRONMENT", "").lower() == "production" and not _API_KEY:
    raise RuntimeError("STEPPEDNA_API_KEY is required when ENVIRONMENT=production")

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if _API_KEY:
        public_paths = {"/", "/health", "/docs", "/openapi.json", "/redoc"}
        if request.url.path not in public_paths:
            if request.headers.get("X-API-Key") != _API_KEY:
                return JSONResponse(status_code=401, content={"error": "Invalid or missing API key"})
    return await call_next(request)

# ─── Simple In-Memory Rate Limiter ───────────────────────────────────────────
# Allows up to RATE_LIMIT requests per RATE_WINDOW seconds per IP.
import collections, threading, csv

RATE_LIMIT  = int(os.getenv("RATE_LIMIT", "60"))     # requests per window
RATE_WINDOW = int(os.getenv("RATE_WINDOW", "60"))     # seconds

_rate_lock   = threading.Lock()
_rate_counts: dict = collections.defaultdict(list)   # ip -> [timestamps]

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    else:
        ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _rate_lock:
        # Drop timestamps outside the current window
        _rate_counts[ip] = [t for t in _rate_counts[ip] if now - t < RATE_WINDOW]
        if len(_rate_counts[ip]) >= RATE_LIMIT:
            return JSONResponse(
                status_code=429,
                content={"error": f"Rate limit exceeded: max {RATE_LIMIT} requests per {RATE_WINDOW}s"},
                headers={"Retry-After": str(RATE_WINDOW)},
            )
        _rate_counts[ip].append(now)
        # Periodic cleanup: purge stale IPs to prevent unbounded memory growth
        if len(_rate_counts) > 200:
            stale = [k for k, v in _rate_counts.items()
                     if not v or (now - max(v)) > RATE_WINDOW * 2]
            for k in stale:
                del _rate_counts[k]
    return await call_next(request)

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attach a unique request ID to each request for tracing."""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Add security headers to every response."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return response

from backend.feature_engineering import (
    AA_HYDROPHOBICITY, AA_VOLUME, BLOSUM62,
    ALL_AMINO_ACIDS, ALL_MUTATIONS,
    get_blosum62, get_charge,
)
from backend.acmg_rules import evaluate_acmg_rules
from backend.database import init_db, record_analysis, record_vcf_upload, get_recent_analyses, get_analysis_stats

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Whitelist of supported genes — used to prevent path traversal via gene_name
SUPPORTED_GENES = {"BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D"}

# Maximum cDNA lengths per gene
MAX_CDNA_LENGTHS = {'BRCA1': 5592, 'BRCA2': 10257, 'PALB2': 3561, 'RAD51C': 1131, 'RAD51D': 987}

# Maximum AA positions per gene (protein length)
GENE_MAX_AA = {'BRCA1': 1863, 'BRCA2': 3418, 'PALB2': 1186, 'RAD51C': 376, 'RAD51D': 328}

# Per-gene model reliability metadata from test set evaluation
GENE_RELIABILITY = {
    "BRCA2": {"auc": 0.983, "tier": "high", "note": ""},
    "RAD51D": {"auc": 0.804, "tier": "moderate", "note": "Limited training data (82 test variants)"},
    "RAD51C": {"auc": 0.743, "tier": "moderate", "note": "Limited training data (135 test variants)"},
    "BRCA1": {"auc": 0.706, "tier": "low", "note": "Extreme class imbalance in ClinVar (96.6% pathogenic)"},
    "PALB2": {"auc": 0.641, "tier": "low", "note": "Smallest training set, lowest model confidence"},
}

# BRCA2 CDS boundaries (used for relative position features)
MAX_CDNA_POS = 10257
MAX_AA_POS = 3418

# ─── Ensemble Hyperparameters ────────────────────────────────────────────────
XGB_WEIGHT = 0.6          # XGBoost contribution to ensemble blend
NN_WEIGHT = 0.4           # Neural network contribution to ensemble blend
N_EFFECTIVE = 200          # Pseudo-observations for Beta CI estimation (tuned to ensemble variance)
ESM2_WINDOW = 50           # Residues on each side for local ESM-2 windowed inference
ESM2_PCA_COMPONENTS = 20   # Number of PCA components from ESM-2 embeddings
MAX_VCF_SIZE = 50 * 1024 * 1024  # 50 MB

from backend.constants import CODON_TABLE, COMPLEMENT

def _load_pickle(filename, required=False):
    """Load a pickle file from DATA_DIR, returning None if optional and missing.

    Security note: Only loads files from the trusted DATA_DIR directory.
    Path traversal is blocked by os.path.basename normalization.
    """
    # Prevent path traversal — strip directory components from filename
    safe_name = os.path.basename(filename)
    path = os.path.join(DATA_DIR, safe_name)
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        logger.warning(f"[WARN] {safe_name} not found")
        return None
    file_size = os.path.getsize(path)
    if file_size == 0:
        logger.error(f"[ERROR] {safe_name} is empty (0 bytes) — likely corrupted")
        if required:
            raise ValueError(f"Model file is empty: {path}")
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info(f"[OK] {safe_name} loaded ({file_size:,} bytes)")
    return data

def _load_variant_dict(filename):
    """Load a pickle containing by_variant/by_position dicts."""
    raw = _load_pickle(filename)
    if raw is None:
        return {}, {}
    return raw.get("by_variant", {}), raw.get("by_position", {})

calibrator = None
scaler = None
feature_names = []
threshold = 0.5
ensemble_model = None
booster = None
logger.info("ML Models will be lazy-loaded dynamically via get_gene_data()")

# ─── Load BRCA2 Data (module-level for /predict data_sources & /health) ──────
phylop_scores = _load_pickle("phylop_scores.pkl") or {}
mave_by_variant, mave_by_position = _load_variant_dict("mave_scores.pkl")
am_by_variant, am_by_position = _load_variant_dict("alphamissense_scores.pkl")
structural_features = _load_pickle("structural_features.pkl") or {}
gnomad_by_variant, gnomad_by_position = _load_variant_dict("gnomad_frequencies.pkl")

_cdna_genomic = _load_pickle("cdna_to_genomic.pkl") or {}
cdna_to_genomic = _cdna_genomic
genomic_to_cdna = {v: k for k, v in _cdna_genomic.items()}
logger.info(f"[OK] Genomic mapping: {len(genomic_to_cdna)} positions")

# ─── Training Data Index for Data Scarcity Quantification (Item 41) ──────────
# Pre-loads training variant positions per gene for fast neighborhood lookups.
import pandas as pd

_TRAINING_INDEX: dict = {}  # gene -> {"aa_positions": np.array, "ref_aas": list, "alt_aas": list, "labels": list, "domains": list}
DATA_SCARCITY_WINDOW = 50   # +/- AA positions for neighborhood count

def _build_training_index():
    """Build per-gene training data index from master dataset for data scarcity computation."""
    global _TRAINING_INDEX
    csv_path = os.path.join(DATA_DIR, "master_training_dataset.csv")
    if not os.path.exists(csv_path):
        logger.warning("[DATA-SCARCITY] master_training_dataset.csv not found — data scarcity disabled")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.warning(f"[DATA-SCARCITY] Failed to load training CSV: {e}")
        return

    if "Gene" not in df.columns or "relative_aa_pos" not in df.columns:
        logger.warning("[DATA-SCARCITY] Required columns missing from training CSV")
        return

    # Reconstruct AA position from relative_aa_pos * aa_length
    aa_ref_cols = [c for c in df.columns if c.startswith("AA_ref_")]
    aa_alt_cols = [c for c in df.columns if c.startswith("AA_alt_")]

    for gene in SUPPORTED_GENES:
        gene_mask = df["Gene"] == gene
        gene_df = df[gene_mask]
        if len(gene_df) == 0:
            continue

        aa_length = GENE_MAX_AA.get(gene, 3418)
        aa_positions = np.round(gene_df["relative_aa_pos"].values * aa_length).astype(int)

        # Decode one-hot AA ref/alt back to amino acid names
        ref_aas = []
        alt_aas = []
        for _, row in gene_df.iterrows():
            ref_aa = "Unk"
            alt_aa = "Unk"
            for col in aa_ref_cols:
                if row[col] == 1:
                    ref_aa = col.replace("AA_ref_", "")
                    break
            for col in aa_alt_cols:
                if row[col] == 1:
                    alt_aa = col.replace("AA_alt_", "")
                    break
            ref_aas.append(ref_aa)
            alt_aas.append(alt_aa)

        # Determine domain membership from domain feature columns
        domain_cols = ["in_critical_repeat_region", "in_DNA_binding", "in_OB_folds", "in_NLS", "in_primary_interaction"]
        domains = []
        for _, row in gene_df.iterrows():
            domain = "uncharacterized"
            for dc in domain_cols:
                if dc in row and row[dc] == 1:
                    domain = dc
                    break
            domains.append(domain)

        _TRAINING_INDEX[gene] = {
            "aa_positions": aa_positions,
            "ref_aas": ref_aas,
            "alt_aas": alt_aas,
            "labels": gene_df["Label"].values.tolist(),
            "domains": domains,
        }
        logger.info(f"[DATA-SCARCITY] {gene}: indexed {len(aa_positions)} training variants")

    logger.info(f"[DATA-SCARCITY] Training index built for {len(_TRAINING_INDEX)} genes")


def compute_data_support(gene_name: str, aa_pos: int, aa_ref: str, aa_alt: str,
                         in_domain_name: str = "uncharacterized") -> dict:
    """Compute data scarcity quantification for a prediction.

    Returns dict with nearby_variants, same_substitution_type, in_known_domain, level.
    """
    gene = gene_name.upper()
    if gene not in _TRAINING_INDEX:
        return {"nearby_variants": 0, "same_substitution_type": 0,
                "in_known_domain": False, "level": "LOW"}

    idx = _TRAINING_INDEX[gene]
    positions = idx["aa_positions"]

    # Count training variants within +/-50 AA positions
    nearby_mask = np.abs(positions - aa_pos) <= DATA_SCARCITY_WINDOW
    nearby_count = int(nearby_mask.sum())

    # Count same substitution type (same ref+alt AA)
    same_subst_count = 0
    for i in range(len(idx["ref_aas"])):
        if idx["ref_aas"][i] == aa_ref and idx["alt_aas"][i] == aa_alt:
            same_subst_count += 1

    # Check if the variant's domain has training data
    in_known_domain = in_domain_name != "uncharacterized"

    # Compute support level
    if nearby_count > 100:
        level = "HIGH"
    elif nearby_count >= 10:
        level = "MODERATE"
    else:
        level = "LOW"

    return {
        "nearby_variants": nearby_count,
        "same_substitution_type": same_subst_count,
        "in_known_domain": in_known_domain,
        "level": level,
    }


# ─── Contrastive Explanation Pairs (Item 43) ─────────────────────────────────
# Pre-builds KD-trees of scaled training feature vectors per gene for nearest
# opposite-class lookup. At prediction time, finds the nearest training variant
# with the opposite predicted class and reports key feature differences.
_CONTRASTIVE_INDEX: dict = {}  # gene -> {"tree_path": KDTree, "tree_benign": KDTree, ...}

def _build_contrastive_index():
    """Build per-gene KD-trees on scaled training features for contrastive explanations."""
    global _CONTRASTIVE_INDEX
    csv_path = os.path.join(DATA_DIR, "master_training_dataset.csv")
    if not os.path.exists(csv_path):
        logger.warning("[CONTRASTIVE] master_training_dataset.csv not found -- contrastive explanations disabled")
        return

    models = _get_universal_models()
    model_scaler = models.get("scaler")
    model_feature_names = models.get("feature_names")
    if model_scaler is None or not model_feature_names:
        logger.warning("[CONTRASTIVE] Scaler or feature names not available -- contrastive explanations disabled")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.warning("[CONTRASTIVE] Failed to load training CSV: %s" % str(e))
        return

    if "Gene" not in df.columns or "Label" not in df.columns:
        logger.warning("[CONTRASTIVE] Required columns (Gene, Label) missing from training CSV")
        return

    # Decode one-hot AA columns for variant identifiers
    aa_ref_cols = [c for c in df.columns if c.startswith("AA_ref_")]
    aa_alt_cols = [c for c in df.columns if c.startswith("AA_alt_")]

    for gene in SUPPORTED_GENES:
        gene_mask = df["Gene"] == gene
        gene_df = df[gene_mask].copy()
        if len(gene_df) < 10:
            logger.info("[CONTRASTIVE] %s: too few variants (%d) -- skipping" % (gene, len(gene_df)))
            continue

        # Build feature matrix in model feature order
        feat_matrix = np.zeros((len(gene_df), len(model_feature_names)), dtype=np.float32)
        for j, fname in enumerate(model_feature_names):
            if fname in gene_df.columns:
                feat_matrix[:, j] = gene_df[fname].values

        # Scale using the production scaler
        try:
            scaled_matrix = model_scaler.transform(feat_matrix)
        except Exception as e:
            logger.warning("[CONTRASTIVE] %s: scaler.transform failed: %s" % (gene, str(e)))
            continue

        labels = gene_df["Label"].values.astype(int)

        # Reconstruct AA position and amino acid identifiers
        aa_length = GENE_MAX_AA.get(gene, 3418)
        aa_positions = np.round(gene_df["relative_aa_pos"].values * aa_length).astype(int)

        ref_aas = []
        alt_aas = []
        for _, row in gene_df.iterrows():
            ref_aa = "Unk"
            alt_aa = "Unk"
            for col in aa_ref_cols:
                if row.get(col, 0) == 1:
                    ref_aa = col.replace("AA_ref_", "")
                    break
            for col in aa_alt_cols:
                if row.get(col, 0) == 1:
                    alt_aa = col.replace("AA_alt_", "")
                    break
            ref_aas.append(ref_aa)
            alt_aas.append(alt_aa)

        # Build separate KD-trees for pathogenic (label=1) and benign (label=0)
        path_mask = labels == 1
        ben_mask = labels == 0

        gene_entry = {
            "scaled_features": scaled_matrix,
            "labels": labels,
            "aa_positions": aa_positions,
            "ref_aas": ref_aas,
            "alt_aas": alt_aas,
            "tree_pathogenic": None,
            "tree_benign": None,
            "path_indices": np.where(path_mask)[0],
            "ben_indices": np.where(ben_mask)[0],
        }

        if path_mask.sum() >= 1:
            gene_entry["tree_pathogenic"] = KDTree(scaled_matrix[path_mask])
        if ben_mask.sum() >= 1:
            gene_entry["tree_benign"] = KDTree(scaled_matrix[ben_mask])

        _CONTRASTIVE_INDEX[gene] = gene_entry
        logger.info("[CONTRASTIVE] %s: indexed %d variants (path=%d, ben=%d)" % (
            gene, len(gene_df), int(path_mask.sum()), int(ben_mask.sum())))

    logger.info("[CONTRASTIVE] Contrastive index built for %d genes" % len(_CONTRASTIVE_INDEX))


def find_contrastive_explanation(gene_name: str, scaled_vector: np.ndarray,
                                 probability: float, feature_names_list: list) -> dict:
    """Find the nearest opposite-class training variant and compute key feature differences.

    Returns dict with contrast_variant, contrast_class, contrast_distance,
    key_differences (top 5 features by difference magnitude), or None if unavailable.
    """
    gene = gene_name.upper()
    if gene not in _CONTRASTIVE_INDEX:
        return None

    idx = _CONTRASTIVE_INDEX[gene]
    query_class = "Pathogenic" if probability > 0.5 else "Benign"

    # Find nearest variant of the OPPOSITE class
    if query_class == "Pathogenic":
        # Query is pathogenic -> find nearest benign
        tree = idx["tree_benign"]
        opposite_indices = idx["ben_indices"]
        contrast_class = "Benign"
    else:
        # Query is benign -> find nearest pathogenic
        tree = idx["tree_pathogenic"]
        opposite_indices = idx["path_indices"]
        contrast_class = "Pathogenic"

    if tree is None or len(opposite_indices) == 0:
        return None

    # Query the KD-tree for the nearest neighbor
    query_vec = scaled_vector.reshape(1, -1) if scaled_vector.ndim == 1 else scaled_vector
    try:
        dist, local_idx = tree.query(query_vec[0], k=1)
    except Exception:
        return None

    # Map local tree index back to global index in the gene's dataset
    global_idx = opposite_indices[local_idx]

    # Get the contrast variant's info
    contrast_ref = idx["ref_aas"][global_idx]
    contrast_alt = idx["alt_aas"][global_idx]
    contrast_pos = int(idx["aa_positions"][global_idx])
    contrast_variant = "%s%d%s" % (contrast_ref, contrast_pos, contrast_alt)

    # Compute feature-wise differences (on scaled features)
    contrast_features = idx["scaled_features"][global_idx]
    query_features = query_vec[0]
    feature_diffs = np.abs(query_features - contrast_features)

    # Rank by difference magnitude, take top 5
    top_indices = np.argsort(feature_diffs)[::-1][:5]

    key_differences = []
    for fi in top_indices:
        if fi >= len(feature_names_list):
            continue
        fname = feature_names_list[fi]
        nice_name = NICE_NAMES.get(fname, fname)
        diff_val = float(feature_diffs[fi])
        # Skip near-zero differences
        if diff_val < 0.01:
            continue
        # Determine importance level based on scaled difference magnitude
        if diff_val > 2.0:
            importance = "high"
        elif diff_val > 1.0:
            importance = "moderate"
        else:
            importance = "low"

        key_differences.append({
            "feature": nice_name,
            "feature_key": fname,
            "query_value": round(float(query_features[fi]), 3),
            "contrast_value": round(float(contrast_features[fi]), 3),
            "difference": round(diff_val, 3),
            "importance": importance,
        })

    if not key_differences:
        return None

    return {
        "contrast_variant": contrast_variant,
        "contrast_class": contrast_class,
        "contrast_distance": round(float(dist), 4),
        "key_differences": key_differences,
    }


# ─── Per-Gene Calibrators (Item 45) ─────────────────────────────────────────
_GENE_CALIBRATORS: dict = {}  # gene -> IsotonicRegression or None

def _load_gene_calibrators():
    """Load per-gene calibrators from data/calibrator_<gene>.pkl files."""
    global _GENE_CALIBRATORS
    for gene in SUPPORTED_GENES:
        cal_path = os.path.join(DATA_DIR, f"calibrator_{gene.lower()}.pkl")
        if os.path.exists(cal_path):
            try:
                with open(cal_path, "rb") as f:
                    cal = pickle.load(f)
                _GENE_CALIBRATORS[gene] = cal
                logger.info(f"[CALIBRATOR] Loaded per-gene calibrator for {gene}")
            except Exception as e:
                logger.warning(f"[CALIBRATOR] Failed to load calibrator for {gene}: {e}")
        else:
            logger.info(f"[CALIBRATOR] No per-gene calibrator for {gene} — will use universal")


# ─── Gene-Adaptive Ensemble Weights (Item 38) ──────────────────────────────
_GENE_ENSEMBLE_WEIGHTS: dict = {}  # gene -> {"xgb_weight": float, "mlp_weight": float}

def _load_gene_ensemble_weights():
    """Load per-gene optimized ensemble weights from data/gene_ensemble_weights.json."""
    global _GENE_ENSEMBLE_WEIGHTS
    weights_path = os.path.join(DATA_DIR, "gene_ensemble_weights.json")
    if not os.path.exists(weights_path):
        logger.info("[WEIGHTS] gene_ensemble_weights.json not found — using default 0.6/0.4 for all genes")
        return
    try:
        with open(weights_path, "r") as f:
            raw = json.load(f)
        for gene, info in raw.items():
            gene_upper = gene.upper()
            xgb_w = float(info.get("xgb_weight", XGB_WEIGHT))
            mlp_w = float(info.get("mlp_weight", NN_WEIGHT))
            _GENE_ENSEMBLE_WEIGHTS[gene_upper] = {"xgb_weight": xgb_w, "mlp_weight": mlp_w}
            logger.info(f"[WEIGHTS] {gene_upper}: XGB={xgb_w:.2f} / MLP={mlp_w:.2f}")
    except Exception as e:
        logger.warning(f"[WEIGHTS] Failed to load gene ensemble weights: {e}")


# ─── Bootstrap Models for Confidence Intervals (Item 39) ─────────────────────
_BOOTSTRAP_MODELS: list = []  # list of xgb.Booster objects
N_BOOTSTRAP = 50

def _load_bootstrap_models():
    """Load pre-trained bootstrap XGBoost models for confidence interval estimation."""
    global _BOOTSTRAP_MODELS
    bootstrap_dir = os.path.join(DATA_DIR, "bootstrap_models")
    if not os.path.isdir(bootstrap_dir):
        logger.info("[BOOTSTRAP] data/bootstrap_models/ not found — bootstrap CIs disabled. Run scripts/generate_bootstrap_models.py to enable.")
        return
    loaded = 0
    for i in range(N_BOOTSTRAP):
        model_path = os.path.join(bootstrap_dir, f"bootstrap_{i}.json")
        if os.path.exists(model_path):
            try:
                booster = xgb.Booster()
                booster.load_model(model_path)
                _BOOTSTRAP_MODELS.append(booster)
                loaded += 1
            except Exception as e:
                logger.warning(f"[BOOTSTRAP] Failed to load bootstrap_{i}.json: {e}")
        else:
            break
    if loaded > 0:
        logger.info(f"[BOOTSTRAP] Loaded {loaded} bootstrap models for CI estimation")
    else:
        logger.info("[BOOTSTRAP] No bootstrap models found — CIs will use Beta approximation only")


def compute_bootstrap_ci(scaled_vector: np.ndarray, feature_names_list: list) -> dict:
    """Compute bootstrap confidence interval from pre-trained models.

    Returns dict with ci_lower, ci_upper, ci_width, n_models, or None if unavailable.
    """
    if not _BOOTSTRAP_MODELS:
        return None
    dmat = xgb.DMatrix(scaled_vector, feature_names=feature_names_list)
    preds = np.array([m.predict(dmat)[0] for m in _BOOTSTRAP_MODELS])
    ci_lower = float(np.percentile(preds, 5))
    ci_upper = float(np.percentile(preds, 95))
    return {
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "ci_width": round(ci_upper - ci_lower, 4),
        "n_models": len(_BOOTSTRAP_MODELS),
    }


# ─── Active Learning Priorities (Item 42) ────────────────────────────────────
_ACTIVE_LEARNING: dict = {}  # loaded from data/active_learning_priorities.json

def _load_active_learning_priorities():
    """Load pre-computed active learning priority rankings."""
    global _ACTIVE_LEARNING
    al_path = os.path.join(DATA_DIR, "active_learning_priorities.json")
    if not os.path.exists(al_path):
        logger.info("[ACTIVE-LEARNING] active_learning_priorities.json not found — run scripts/active_learning_ranker.py to generate")
        return
    try:
        with open(al_path, "r") as f:
            _ACTIVE_LEARNING = json.load(f)
        n_total = sum(len(v) for v in _ACTIVE_LEARNING.get("priorities", {}).values())
        logger.info(f"[ACTIVE-LEARNING] Loaded {n_total} priority variants across {len(_ACTIVE_LEARNING.get('priorities', {}))} genes")
    except Exception as e:
        logger.warning(f"[ACTIVE-LEARNING] Failed to load priorities: {e}")


# ─── Split Conformal Prediction (Item 5.1) ────────────────────────────────────
_CONFORMAL_THRESHOLDS: dict = {}  # gene -> {"quantile": float, "alpha": float, ...}

def _load_conformal_thresholds():
    """Load per-gene conformal prediction quantile thresholds."""
    global _CONFORMAL_THRESHOLDS
    ct_path = os.path.join(DATA_DIR, "conformal_thresholds.json")
    if not os.path.exists(ct_path):
        logger.info("[CONFORMAL] conformal_thresholds.json not found — run scripts/train_conformal.py to generate")
        return
    try:
        with open(ct_path, "r") as f:
            _CONFORMAL_THRESHOLDS = json.load(f)
        n_genes = len([k for k in _CONFORMAL_THRESHOLDS if k != "_global"])
        logger.info(f"[CONFORMAL] Loaded conformal thresholds for {n_genes} genes (alpha={_CONFORMAL_THRESHOLDS.get('_global', {}).get('alpha', 0.10)})")
    except Exception as e:
        logger.warning(f"[CONFORMAL] Failed to load conformal thresholds: {e}")


def compute_conformal_set(probability: float, gene_name: str) -> dict:
    """Compute the conformal prediction set for a given probability and gene.

    Returns dict with:
        conformal_set: list of class labels included at the coverage level
        conformal_coverage: the target coverage (e.g. 0.90)
        conformal_alpha: the alpha level (e.g. 0.10)
        set_size: number of classes in the set (1 or 2)
    Returns None if conformal thresholds are not loaded.
    """
    if not _CONFORMAL_THRESHOLDS:
        return None

    gene_upper = gene_name.upper()
    gene_info = _CONFORMAL_THRESHOLDS.get(gene_upper, _CONFORMAL_THRESHOLDS.get("_global"))
    if not gene_info:
        return None

    q = gene_info["quantile"]
    alpha = gene_info.get("alpha", 0.10)

    p_pathogenic = probability
    p_benign = 1.0 - probability

    pred_set = []
    if p_pathogenic >= 1.0 - q:
        pred_set.append("Pathogenic")
    if p_benign >= 1.0 - q:
        pred_set.append("Benign")

    # Safety: if empty set (shouldn't happen), include highest-probability class
    if not pred_set:
        pred_set.append("Pathogenic" if probability >= 0.5 else "Benign")

    return {
        "conformal_set": pred_set,
        "conformal_coverage": round(1 - alpha, 2),
        "conformal_alpha": alpha,
        "set_size": len(pred_set),
    }


# ─── ESM-2 Runtime Inference Defaults ─────────────────────────────────────────
# These are only populated if ESM-2 model is explicitly loaded (optional).
esm_model = None
esm_batch_converter = None
DEVICE = "cpu"

# Global Universal ML Model Architecture cache
_UNIVERSAL_MODELS = None

def _get_universal_models():
    """Load universal base ML artifacts only once across the entire application lifetime."""
    global _UNIVERSAL_MODELS
    if _UNIVERSAL_MODELS is not None:
        return _UNIVERSAL_MODELS
        
    from tensorflow.keras.models import load_model
    import xgboost as xgb
    
    _UNIVERSAL_MODELS = {}
    _UNIVERSAL_MODELS["calibrator"] = _load_pickle("universal_calibrator_ensemble.pkl")
    _UNIVERSAL_MODELS["scaler"] = _load_pickle("universal_scaler_ensemble.pkl")
    _UNIVERSAL_MODELS["feature_names"] = _load_pickle("universal_feature_names.pkl")
    _UNIVERSAL_MODELS["threshold"] = _load_pickle("universal_threshold_ensemble.pkl") or 0.5
    
    _UNIVERSAL_MODELS["ensemble_model"] = load_model(f"{DATA_DIR}/universal_nn.h5") if os.path.exists(f"{DATA_DIR}/universal_nn.h5") else None
        
    xgb_path = f"{DATA_DIR}/universal_xgboost_final.json"
    if os.path.exists(xgb_path):
        booster = xgb.Booster()
        booster.load_model(xgb_path)
        _UNIVERSAL_MODELS["booster"] = booster
    else:
        _UNIVERSAL_MODELS["booster"] = None

    return _UNIVERSAL_MODELS

def _derive_exon_boundaries(cdna_to_genomic: dict, strand: str) -> dict:
    """Derive exon-intron boundaries from the cDNA-to-genomic position mapping.

    Within an exon, consecutive cDNA positions map to consecutive genomic positions.
    When there's a jump > 1 in genomic position, that's an intron (= exon boundary).

    Returns dict with:
        exon_boundaries: list of (last_exon_gpos, first_next_exon_gpos) tuples
        canonical_splice: set of genomic positions ±1-2bp from each boundary
        near_splice: set of genomic positions ±3-8bp from each boundary (excluding canonical)
        splice_info: dict mapping gpos -> {"zone": "canonical"|"near", "distance": int}
    """
    if not cdna_to_genomic:
        return {"exon_boundaries": [], "canonical_splice": set(), "near_splice": set(), "splice_info": {}}

    sorted_cdna = sorted(cdna_to_genomic.keys())
    boundaries = []

    for i in range(len(sorted_cdna) - 1):
        gpos_curr = cdna_to_genomic[sorted_cdna[i]]
        gpos_next = cdna_to_genomic[sorted_cdna[i + 1]]

        if strand == "+":
            gap = gpos_next - gpos_curr
        else:  # reverse strand: genomic positions decrease as cDNA increases
            gap = gpos_curr - gpos_next

        if gap > 1:
            # Found an intron — record the boundary positions
            boundaries.append((gpos_curr, gpos_next))

    canonical_splice = set()
    near_splice = set()
    splice_info = {}

    for exon_end, next_exon_start in boundaries:
        # Generate positions around each boundary
        # For both the exon-end side and the next-exon-start side
        for anchor in [exon_end, next_exon_start]:
            for offset in range(-8, 9):
                if offset == 0:
                    continue
                gpos = anchor + offset
                dist = abs(offset)

                if dist <= 2:
                    canonical_splice.add(gpos)
                    splice_info[gpos] = {"zone": "canonical", "distance": dist}
                elif dist <= 8:
                    if gpos not in canonical_splice:
                        near_splice.add(gpos)
                        splice_info[gpos] = {"zone": "near", "distance": dist}

    # Also add the boundary positions themselves (they are intronic positions)
    for exon_end, next_exon_start in boundaries:
        # Positions between boundaries are intronic — add ±1,2 from boundary edges
        if strand == "+":
            # Intron spans from exon_end+1 to next_exon_start-1
            for d in [1, 2]:
                pos = exon_end + d
                canonical_splice.add(pos)
                splice_info[pos] = {"zone": "canonical", "distance": d}
                pos = next_exon_start - d
                canonical_splice.add(pos)
                splice_info[pos] = {"zone": "canonical", "distance": d}
            for d in [3, 4, 5, 6, 7, 8]:
                pos = exon_end + d
                if pos not in canonical_splice:
                    near_splice.add(pos)
                    splice_info[pos] = {"zone": "near", "distance": d}
                pos = next_exon_start - d
                if pos not in canonical_splice:
                    near_splice.add(pos)
                    splice_info[pos] = {"zone": "near", "distance": d}
        else:
            # Reverse strand: intron spans from next_exon_start+1 to exon_end-1
            # (genomic positions: next_exon_start < exon_end for reverse strand)
            low = min(exon_end, next_exon_start)
            high = max(exon_end, next_exon_start)
            for d in [1, 2]:
                pos = low + d
                canonical_splice.add(pos)
                splice_info[pos] = {"zone": "canonical", "distance": d}
                pos = high - d
                canonical_splice.add(pos)
                splice_info[pos] = {"zone": "canonical", "distance": d}
            for d in [3, 4, 5, 6, 7, 8]:
                pos = low + d
                if pos not in canonical_splice:
                    near_splice.add(pos)
                    splice_info[pos] = {"zone": "near", "distance": d}
                pos = high - d
                if pos not in canonical_splice:
                    near_splice.add(pos)
                    splice_info[pos] = {"zone": "near", "distance": d}

    # Remove any splice zone positions that are actually in the CDS
    cds_positions = set(cdna_to_genomic.values())
    canonical_splice -= cds_positions
    near_splice -= cds_positions
    splice_info = {k: v for k, v in splice_info.items() if k not in cds_positions}

    logger.info(f"[EXON] Derived {len(boundaries)} exon boundaries, "
                f"{len(canonical_splice)} canonical splice positions, "
                f"{len(near_splice)} near-splice positions")

    return {
        "exon_boundaries": boundaries,
        "canonical_splice": canonical_splice,
        "near_splice": near_splice,
        "splice_info": splice_info,
    }


_gene_cache: dict = {}
_gene_cache_lock = threading.Lock()

def get_gene_data(gene_name: str) -> dict:
    key = gene_name.upper()
    with _gene_cache_lock:
        if key in _gene_cache:
            return _gene_cache[key]

    prefix = gene_name.lower()

    def load_with_fallback(suffix, loader_func, *args, **kwargs):
        res = loader_func(f"{prefix}_{suffix}", *args, **kwargs)
        if hasattr(res, "__len__") and not res:
            res = loader_func(suffix, *args, **kwargs)
        elif res is None:
            res = loader_func(suffix, *args, **kwargs)
        return res

    phylop = load_with_fallback("phylop_scores.pkl", _load_pickle) or {}
    mave_v, mave_p = load_with_fallback("mave_scores.pkl", _load_variant_dict)
    am_v, am_p = load_with_fallback("alphamissense_scores.pkl", _load_variant_dict)
    struct = load_with_fallback("structural_features.pkl", _load_pickle) or {}
    gnomad_v, gnomad_p = load_with_fallback("gnomad_frequencies.pkl", _load_variant_dict)

    esm2 = load_with_fallback("esm2_embeddings.pkl", _load_pickle) or {}
    esm2_dict_local = esm2.get("embeddings", {})
    esm2_pca_model_local = esm2.get("pca_model", None)

    cds_path = f"{DATA_DIR}/{prefix}_cds.txt"
    cds = None
    if os.path.exists(cds_path):
        with open(cds_path, "r") as f:
            cds = f.read().strip()

    cdna_genomic = load_with_fallback("cdna_to_genomic.pkl", _load_pickle) or {}
    genomic_to_cdna = {v: k for k, v in cdna_genomic.items()}

    # Load gene config JSON (cached here to avoid redundant reads in VCF processing)
    gene_config = {}
    config_path = os.path.join(os.path.dirname(__file__), "gene_configs", f"{prefix}.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            gene_config = json.load(f)

    # Derive exon boundaries for splice-site variant detection
    strand = gene_config.get("strand", "+")
    exon_data = _derive_exon_boundaries(cdna_genomic, strand)

    # Load PM5 pathogenic position lookup (AA positions with known pathogenic missense)
    pm5_positions = set()
    pm5_path = os.path.join(DATA_DIR, "pathogenic_positions.json")
    if os.path.exists(pm5_path):
        with open(pm5_path, "r") as f:
            pm5_data = json.load(f)
            # Keys in JSON are gene names, values are lists of AA positions
            pm5_positions = set(pm5_data.get(prefix.upper(), []))

    # Utilize the static memory singleton to pull our massive Universal ML models
    # instead of exhausting RAM loading them per gene dropdown invocation.
    uni_models = _get_universal_models()

    result = {
        "phylop": phylop,
        "mave_v": mave_v, "mave_p": mave_p,
        "am_v": am_v, "am_p": am_p,
        "struct": struct,
        "gnomad_v": gnomad_v, "gnomad_p": gnomad_p,
        "esm2_dict": esm2_dict_local,
        "esm2_pca": esm2_pca_model_local,
        "cds": cds,
        "cdna_to_genomic": cdna_genomic,
        "genomic_to_cdna": genomic_to_cdna,
        "gene_config": gene_config,
        "exon_data": exon_data,
        "pm5_positions": pm5_positions,
        "calibrator": uni_models.get("calibrator"),
        "scaler": uni_models.get("scaler"),
        "feature_names": uni_models.get("feature_names"),
        "threshold": uni_models.get("threshold", 0.5),
        "ensemble_model": uni_models.get("ensemble_model"),
        "booster": uni_models.get("booster")
    }
    with _gene_cache_lock:
        _gene_cache[key] = result
    return result

def build_feature_vector(cDNA_pos, AA_ref, AA_alt, Mutation, AA_pos, gene_name="BRCA2"):
    gene_data = get_gene_data(gene_name)
    features = {}

    # Early guard: reject clearly invalid positions
    if cDNA_pos < 1 or AA_pos < 1:
        logger.warning(f"[FEATURE] Invalid positions: cDNA={cDNA_pos}, AA={AA_pos} for {gene_name}")
        return np.zeros((1, len(gene_data.get("feature_names", []) or range(103))), dtype=np.float32)

    config_path = os.path.join(os.path.dirname(__file__), "gene_configs", f"{gene_name.lower()}.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            gene_config = json.load(f)
    else:
        logger.error(f"[CRITICAL] Gene config missing for {gene_name}: {config_path}")
        raise FileNotFoundError(f"Gene configuration not found for {gene_name}. Expected at: {config_path}")

    features["cDNA_pos"] = cDNA_pos
    features["AA_pos"] = AA_pos
    features["relative_cdna_pos"] = cDNA_pos / gene_config["cds_length"]
    features["relative_aa_pos"] = AA_pos / gene_config["aa_length"]
    features["blosum62_score"] = get_blosum62(AA_ref, AA_alt)
    features["volume_diff"] = abs(AA_VOLUME.get(AA_ref, 0) - AA_VOLUME.get(AA_alt, 0))
    ref_hydro = AA_HYDROPHOBICITY.get(AA_ref, 0)
    alt_hydro = AA_HYDROPHOBICITY.get(AA_alt, 0)
    features["hydro_diff"] = abs(ref_hydro - alt_hydro)
    features["ref_hydro"] = ref_hydro
    features["alt_hydro"] = alt_hydro
    features["hydro_delta"] = alt_hydro - ref_hydro
    
    domains = gene_config.get("domains", {})
    def in_domain(pos, d_name):
        rng = domains.get(d_name)
        return int(rng[0] <= pos <= rng[1]) if rng else 0
        
    def check_domains(*d_names):
        for d in d_names:
            if domains.get(d):
                return in_domain(AA_pos, d)
        logger.debug("No matching domain found in gene config for candidates: %s (available: %s)", d_names, list(domains.keys()))
        return 0
        
    # Domain feature names must match feature_engineering.py (training) exactly
    features["in_critical_repeat_region"] = check_domains("BRC_repeats", "WD40_repeats", "BRCT1", "BRCT2", "SCD")
    features["in_DNA_binding"] = check_domains("DNA_binding", "ChAM_DNA_binding", "ssDNA_binding")
    features["in_OB_folds"] = check_domains("OB_folds", "RING", "Walker_A", "Walker_B")
    features["in_NLS"] = check_domains("NLS_nuclear_localization", "N_terminal_domain")
    features["in_primary_interaction"] = check_domains("PALB2_interaction", "BRCA1_interaction", "BARD1_interaction", "RAD51B_RAD51D_XRCC3_interaction", "Holliday_junction_resolution")
    
    features["is_nonsense"] = int(AA_alt == "Ter")
    transitions = {"A>G", "G>A", "C>T", "T>C"}
    features["is_transition"] = int(Mutation in transitions)
    features["is_transversion"] = int(Mutation not in transitions)
    ref_charge = get_charge(AA_ref)
    alt_charge = get_charge(AA_alt)
    features["charge_change"] = int(ref_charge != alt_charge)
    features["nonpolar_to_charged"] = int(ref_charge == "nonpolar" and alt_charge in ["positive", "negative"])

    # PhyloP conservation features
    phylop = gene_data["phylop"].get(int(cDNA_pos), 0.0)
    features["phylop_score"] = phylop
    features["high_conservation"] = int(phylop > 4.0)
    features["ultra_conservation"] = int(phylop > 7.0)
    features["conserv_x_blosum"] = phylop * features["blosum62_score"]

    # MAVE HDR functional features (Hu C et al. 2024)
    mave_key = f"{AA_ref}{AA_pos}{AA_alt}"
    mave = gene_data["mave_v"].get(mave_key, gene_data["mave_p"].get(int(cDNA_pos), 0.0))
    features["mave_score"] = mave
    features["has_mave"] = int(mave != 0.0)
    features["mave_abnormal"] = int(0.01 <= mave <= 1.49)
    features["mave_x_blosum"] = mave * features["blosum62_score"]

    # AlphaMissense features (Cheng et al. 2023)
    am_key = f"{AA_ref}{AA_pos}{AA_alt}"
    am = gene_data["am_v"].get(am_key, gene_data["am_p"].get(int(AA_pos), 0.0))
    features["am_score"] = am
    features["am_pathogenic"] = int(am > 0.564)
    features["am_x_phylop"] = am * features["phylop_score"]

    # 3D Structural Features
    sf = gene_data["struct"].get(int(AA_pos), {})
    features["rsa"] = sf.get("rsa", 0.4)
    features["is_buried"] = int(features["rsa"] < 0.25)
    features["bfactor"] = sf.get("bfactor", 50.0)
    features["dist_dna"] = sf.get("dist_dna", 999.0)
    features["dist_palb2"] = sf.get("dist_palb2", 999.0)
    features["is_dna_contact"] = int(sf.get("is_dna_contact", False))
    ss = sf.get("ss", "C")
    features["ss_helix"] = int(ss == "H")
    features["ss_sheet"] = int(ss == "E")
    features["buried_x_blosum"] = features["is_buried"] * features["blosum62_score"]
    features["dna_contact_x_blosum"] = features["is_dna_contact"] * features["blosum62_score"]

    # gnomAD Allele Frequency Features
    gnomad_key = f"{AA_ref}{AA_pos}{AA_alt}"
    gnomad_af_val = gene_data["gnomad_v"].get(gnomad_key, gene_data["gnomad_p"].get(int(cDNA_pos), 0.0))
    
    # Normalize gnomAD value — it can be a dict (new format) or float (legacy)
    if isinstance(gnomad_af_val, dict):
        gnomad_af = float(gnomad_af_val.get("af", 0.0))
        gnomad_popmax = float(gnomad_af_val.get("popmax", gnomad_af))
        gnomad_afr = float(gnomad_af_val.get("afr", 0.0))
        gnomad_amr = float(gnomad_af_val.get("amr", 0.0))
        gnomad_eas = float(gnomad_af_val.get("eas", 0.0))
        gnomad_nfe = float(gnomad_af_val.get("nfe", 0.0))
    else:
        gnomad_af = float(gnomad_af_val) if gnomad_af_val is not None else 0.0
        gnomad_popmax = gnomad_af
        gnomad_afr = gnomad_amr = gnomad_eas = gnomad_nfe = 0.0

    features["gnomad_af"] = gnomad_af
    # Feature names must match feature_engineering.py exactly for universal model compatibility
    features["gnomad_popmax_af"] = gnomad_popmax
    features["gnomad_af_afr"] = gnomad_afr
    features["gnomad_af_amr"] = gnomad_amr
    features["gnomad_af_eas"] = gnomad_eas
    features["gnomad_af_nfe"] = gnomad_nfe
    # Legacy aliases for backward-compat with old models
    features["gnomad_popmax"] = gnomad_popmax
    features["gnomad_afr"] = gnomad_afr
    features["gnomad_amr"] = gnomad_amr
    features["gnomad_eas"] = gnomad_eas
    features["gnomad_nfe"] = gnomad_nfe

    features["gnomad_af_log"] = np.log10(gnomad_af + 1e-8)
    features["gnomad_popmax_log"] = np.log10(gnomad_popmax + 1e-8)
    features["is_rare"] = int(gnomad_af < 0.001)
    features["is_popmax_rare"] = int(gnomad_popmax < 0.001)
    features["af_x_blosum"] = gnomad_af * features["blosum62_score"]

    # ESM-2 Structural Features
    AA3_TO_1 = {"Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E",
                "Gly":"G","His":"H","Ile":"I","Leu":"L","Lys":"K","Met":"M","Phe":"F",
                "Pro":"P","Ser":"S","Thr":"T","Trp":"W","Tyr":"Y","Val":"V","Ter":"*"}

    esm_k = f"{AA_ref}{AA_pos}{AA_alt}"
    if esm_k in gene_data["esm2_dict"]:
        esm_v = gene_data["esm2_dict"][esm_k]
        features["esm2_cosine_sim"] = esm_v.get("cosine_similarity", 0.0)
        features["esm2_l2_shift"] = esm_v.get("l2_shift", 0.0)
        pca_arr = esm_v.get("pca_components", [])
        for i in range(ESM2_PCA_COMPONENTS):
            features[f"esm2_pca_{i}"] = float(pca_arr[i]) if i < len(pca_arr) else 0.0
    elif hasattr(sys.modules[__name__], 'esm_model') and esm_model is not None and gene_data["esm2_pca"] is not None and gene_data["cds"] is not None and AA_alt != "Ter":
        # Novel mutation: Dynamic real-time PyTorch sequence embedding
        try:
            from Bio.Seq import Seq
            protein_seq = str(Seq(gene_data["cds"]).translate()).rstrip("*")
            # Bounds check: AA_pos must be within protein length
            if AA_pos < 1 or AA_pos > len(protein_seq):
                logger.warning(f"[ESM-2] AA_pos {AA_pos} out of protein range (len={len(protein_seq)}) for {gene_name}")
                features["esm2_cosine_sim"] = 0.0
                features["esm2_l2_shift"] = 0.0
                for i in range(ESM2_PCA_COMPONENTS):
                    features[f"esm2_pca_{i}"] = 0.0
                raise ValueError("AA_pos out of range")  # caught by outer except
            WINDOW = ESM2_WINDOW
            win_start = max(0, AA_pos - 1 - WINDOW)
            win_end   = min(len(protein_seq), AA_pos - 1 + WINDOW + 1)
            local_pos = AA_pos - 1 - win_start

            wt_window  = protein_seq[win_start:win_end]
            mut_window = list(wt_window)
            m_aa = AA3_TO_1.get(AA_alt, "A")
            if m_aa != "*":
                mut_window[local_pos] = m_aa
            mut_window = "".join(mut_window)
            
            data = [("wt", wt_window), ("mut", mut_window)]
            _, _, batch_tokens = esm_batch_converter(data)
            import torch
            batch_tokens = batch_tokens.to(DEVICE)
            with torch.no_grad():
                results = esm_model(batch_tokens, repr_layers=[6], return_contacts=False)
            
            wt_emb  = results["representations"][6][0, local_pos + 1].cpu().numpy()
            mut_emb = results["representations"][6][1, local_pos + 1].cpu().numpy()
            
            diff = mut_emb - wt_emb
            cos_sim = float(np.dot(wt_emb, mut_emb) / (np.linalg.norm(wt_emb) * np.linalg.norm(mut_emb) + 1e-8))
            l2_shift = float(np.linalg.norm(diff))
            pca_arr = gene_data["esm2_pca"].transform([diff])[0]
            
            features["esm2_cosine_sim"] = cos_sim
            features["esm2_l2_shift"] = l2_shift
            for i in range(20):
                features[f"esm2_pca_{i}"] = float(pca_arr[i]) if i < len(pca_arr) else 0.0
        except (ImportError, ValueError, IndexError, KeyError, AttributeError, RuntimeError, OSError) as e:
            logger.warning(f"ESM-2 Dynamic compilation failed: {type(e).__name__}: {e}")
            features["esm2_cosine_sim"] = 0.0
            features["esm2_l2_shift"] = 0.0
            for i in range(20):
                features[f"esm2_pca_{i}"] = 0.0
    else:
        features["esm2_cosine_sim"] = 0.0
        features["esm2_l2_shift"] = 0.0
        for i in range(ESM2_PCA_COMPONENTS):
            features[f"esm2_pca_{i}"] = 0.0

    for m in ALL_MUTATIONS:
        features[f"Mutation_{m}"] = int(Mutation == m)
    for aa in ALL_AMINO_ACIDS:
        features[f"AA_ref_{aa}"] = int(AA_ref == aa)
    for aa in ALL_AMINO_ACIDS:
        features[f"AA_alt_{aa}"] = int(AA_alt == aa)

    vector = []
    feature_names = gene_data.get("feature_names", [])
    for name in feature_names:
        vector.append(features.get(name, 0))
    if not vector:
        return np.array([[]], dtype=np.float32)
    return np.array([vector], dtype=np.float32)

# Human-readable labels for the feature names shown in the frontend
NICE_NAMES = {
    "blosum62_score": "BLOSUM62 Score",
    "volume_diff": "Volume Difference",
    "hydro_diff": "Hydrophobicity Diff",
    "hydro_delta": "Hydro Delta",
    "ref_hydro": "Ref Hydrophobicity",
    "alt_hydro": "Alt Hydrophobicity",
    "charge_change": "Charge Change",
    "nonpolar_to_charged": "Nonpolar to Charged",
    "is_nonsense": "Nonsense Mutation",
    "is_transition": "Transition",
    "is_transversion": "Transversion",
    "in_critical_repeat_region": "In Critical Repeat Region",
    "in_DNA_binding": "In DNA Binding Domain",
    "in_OB_folds": "In OB Folds",
    "in_NLS": "In NLS Region",
    "in_primary_interaction": "In Primary Interaction Site",
    "cDNA_pos": "cDNA Position",
    "AA_pos": "AA Position",
    "relative_cdna_pos": "Relative cDNA Pos",
    "relative_aa_pos": "Relative AA Pos",
    "phylop_score": "PhyloP Conservation",
    "high_conservation": "Highly Conserved Site",
    "ultra_conservation": "Ultra-Conserved Site",
    "conserv_x_blosum": "Conservation × BLOSUM62",
    "mave_score": "MAVE HDR Score",
    "has_mave": "Has MAVE Data",
    "mave_abnormal": "MAVE Abnormal",
    "mave_x_blosum": "MAVE × BLOSUM62",
    "am_score": "AlphaMissense Score",
    "am_pathogenic": "AM Pathogenic",
    "am_x_phylop": "AM × PhyloP",
    "rsa": "Solvent Accessibility",
    "is_buried": "Buried Residue",
    "bfactor": "Structural Confidence",
    "dist_dna": "Distance to DNA Site",
    "dist_palb2": "Distance to PALB2 Site",
    "is_dna_contact": "DNA Contact Residue",
    "ss_helix": "Alpha Helix",
    "ss_sheet": "Beta Sheet",
    "buried_x_blosum": "Buried × BLOSUM62",
    "dna_contact_x_blosum": "DNA Contact × BLOSUM62",
    "gnomad_af": "gnomAD Frequency",
    "gnomad_af_log": "gnomAD AF (log scale)",
    "is_rare": "Rare Variant",
    "af_x_blosum": "Frequency × BLOSUM62",
    "gnomad_popmax_af": "gnomAD PopMax Frequency",
    "gnomad_af_afr": "gnomAD African AF",
    "gnomad_af_amr": "gnomAD American AF",
    "gnomad_af_eas": "gnomAD East Asian AF",
    "gnomad_af_nfe": "gnomAD Non-Finnish European AF",
    # ESM-2 embedding features
    "esm2_cosine_sim": "ESM-2 Cosine Similarity",
    "esm2_l2_shift": "ESM-2 Embedding Shift",
    # SpliceAI features
    "spliceai_score": "SpliceAI Score",
    "splice_pathogenic": "Splice Pathogenic",
    **{f"esm2_pca_{i}": f"ESM-2 PCA Component {i}" for i in range(20)},
}

# ============================================================
# API
# ============================================================
VALID_AA_CODES = {'Ala','Arg','Asn','Asp','Cys','Gln','Glu','Gly','His','Ile',
                   'Leu','Lys','Met','Phe','Pro','Ser','Ter','Thr','Trp','Tyr','Val',
                   'Fs','Del','Ins','Dup','*'}

class MutationInput(BaseModel):
    cDNA_pos: int
    AA_ref: str
    AA_alt: str
    Mutation: str = "Unknown"
    AA_pos: int = 0

    gene_name: str = "BRCA2"

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
        if v != "Unknown":
            v = v.upper()
            if not re.match(r'^[ACGT]>[ACGT]$', v):
                raise ValueError('Mutation must be in format X>Y where X,Y are A/C/G/T (e.g. A>G)')
        return v

def _safe_critical_domain(raw_vector, gene_data, aa_pos) -> bool:
    """Safely check if a variant is in a critical domain without risking IndexError."""
    feature_names = gene_data.get("feature_names", [])
    try:
        if "in_critical_repeat_region" in feature_names and "in_DNA_binding" in feature_names:
            idx_repeat = feature_names.index("in_critical_repeat_region")
            idx_dna = feature_names.index("in_DNA_binding")
            if idx_repeat < raw_vector.shape[1] and idx_dna < raw_vector.shape[1]:
                return bool(raw_vector[0][idx_repeat] or raw_vector[0][idx_dna])
    except (ValueError, IndexError):
        pass
    # Fallback to BRCA2 domain ranges
    return (1009 <= aa_pos <= 2083) or (2402 <= aa_pos <= 3190)

def _compute_risk_tier(probability: float) -> str:
    """Unified risk tier logic used by both /predict and /predict/vcf."""
    if probability > 0.7:
        return "high"
    elif probability < 0.3:
        return "low"
    return "uncertain"

@app.post("/predict", tags=["Prediction"], summary="Predict variant pathogenicity",
          description="Accepts a single missense variant and returns pathogenicity probability, SHAP explanations, ACMG evidence, and data source scores.")
async def predict(mutation_data: MutationInput, request: Request):
    t_start = time.perf_counter()
    max_len = MAX_CDNA_LENGTHS.get(mutation_data.gene_name.upper(), 15000)
    if mutation_data.cDNA_pos > max_len:
         return JSONResponse(status_code=400, content={"error": f"cDNA position cannot exceed {max_len} for {mutation_data.gene_name}"})

    # ─── Prediction cache (avoids re-computing identical variants) ──────────
    pred_cache_key = f"pred_{mutation_data.gene_name}_{mutation_data.cDNA_pos}_{mutation_data.AA_ref}_{mutation_data.AA_alt}_{mutation_data.Mutation}"
    cached_pred = _pred_cache_get(pred_cache_key)
    if cached_pred is not None:
        logger.info(f"[PREDICT-CACHE] Cache hit for {pred_cache_key}")
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
        mutation_data.AA_alt in ["Ter", "*", "fs"] or
        any(x in mutation_data.Mutation.lower() for x in ["fs", "del", "ins", "dup"])
    )
    
    if is_truncating:
        label = "Pathogenic"
        probability = 0.9999
        risk = "high (Truncating)"
        acmg_eval = {"PVS1": "Pathogenic truncating null variant (Nonsense/Frameshift) in a recognized tumor suppressor."}
        
        latency_ms = (time.perf_counter() - t_start) * 1000
        logger.info(f"[PREDICT-TIER1] {mutation_data.AA_ref}{aa_pos}{mutation_data.AA_alt} (cDNA:{mutation_data.cDNA_pos}) -> {label} (Tier 1 Rule) ({latency_ms:.0f}ms)")
        
        return {
            "prediction": label,
            "probability": round(probability, 4),
            "risk_tier": risk,
            "threshold": round(float(gene_data.get("threshold", 0.5)), 3),
            "aa_pos": aa_pos,
            "genomic_pos": gene_data.get("cdna_to_genomic", {}).get(mutation_data.cDNA_pos),
            "acmg_evidence": acmg_eval,
            "confidence": {
                "probability": round(probability, 4),
                "label": "High Confidence",
                "ci_lower": 0.9900,
                "ci_upper": 1.0000,
                "std": 0.0001,
            },
            "features_used": {
                "is_nonsense": "Ter" in mutation_data.AA_alt or "*" in mutation_data.AA_alt,
                "aa_position": aa_pos,
            },
            "data_sources": {},
            "shap_explanation": [{"feature": "Tier 1 Protocol (PVS1 Null Variant)", "value": 9.99, "direction": "pathogenic"}],
        }
    # ─── TIER 2: MACHINE LEARNING ENGINE ──────────────────────────────────────
    
    if gene_data.get("ensemble_model") is None or gene_data.get("scaler") is None:
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
        # Native model pass
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
            # Fallback to raw blended if calibrator fails
            probability = float(blended)
            calibrator_type = "raw_fallback"
    else:
        # Fallback to pure dummy inference if loading failed
        probability = 0.5
        model_disagreement = None
        _xgb_w = XGB_WEIGHT
        _mlp_w = NN_WEIGHT
        calibrator_type = "none"

    # Bootstrap CI computation (Item 39)
    feature_names_list = gene_data.get("feature_names", [])
    bootstrap_ci = compute_bootstrap_ci(scaled_vector, feature_names_list)

    # Track warnings for response
    _warnings = []

    # Detect if ESM-2 features are all zeros (novel variant with no precomputed embedding)
    esm2_indices = [i for i, n in enumerate(feature_names_list) if n.startswith("esm2_")]
    if esm2_indices and raw_vector.shape[1] > 0:
        esm2_vals = [raw_vector[0][i] for i in esm2_indices if i < raw_vector.shape[1]]
        if all(v == 0.0 for v in esm2_vals):
            _warnings.append("ESM-2 protein language model features are zero for this variant (no precomputed embedding). Prediction relies on other 81 features.")

    # Warn if we fell back to p=0.5 due to missing models
    if probability == 0.5 and not (calibrator and nn_model and xgb_model):
        _warnings.append("ML models could not be loaded. Probability is a default fallback value (0.5) and should not be trusted.")

    # Out-of-distribution feature warnings (AA position)
    try:
        max_aa = GENE_MAX_AA.get(mutation_data.gene_name.upper(), 5000)
        if aa_pos > max_aa * 0.98:
            _warnings.append(f"AA position {aa_pos} is near the C-terminal end of {mutation_data.gene_name} (max: {max_aa}). Model has fewer training examples in terminal regions.")
    except Exception:
        pass  # OOD check is non-critical

    # Clip to [0.5%, 99.5%] — no model should claim absolute certainty
    probability = float(np.clip(probability, 0.005, 0.995))

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
        beta_param  = (1 - probability) * N_eff + 1
        ci_lower = float(beta_dist.ppf(0.025, alpha_param, beta_param))
        ci_upper = float(beta_dist.ppf(0.975, alpha_param, beta_param))
        ci_std   = float(beta_dist.std(alpha_param, beta_param))
        ci_width = ci_upper - ci_lower
        ci_method = "beta_approximation"

    if abs(probability - 0.5) > 0.4:
         uncertainty_label = "High Confidence"
    elif abs(probability - 0.5) > 0.2:
         uncertainty_label = "Moderate Confidence"
    else:
         uncertainty_label = "Low Confidence"

    threshold = gene_data.get("threshold", 0.5)
    # Use the production threshold
    label = "Pathogenic" if probability >= threshold else "Benign"
    risk = _compute_risk_tier(probability)

    # Log variants that may need wet-lab follow-up
    if label == "Pathogenic" or uncertainty_label == "Low Confidence":
        triage_file = os.path.join(DATA_DIR, "needs_wetlab_assay.csv")
        os.makedirs(os.path.dirname(triage_file) or ".", exist_ok=True)
        reason = "Low Confidence" if uncertainty_label == "Low Confidence" else "Predicted Pathogenic"
        with _rate_lock:  # Reuse existing threading lock for atomic file writes
            with open(triage_file, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(["cDNA_pos", "Mutation", "Probability", "Confidence", "Triage_Reason"])
                writer.writerow([
                    mutation_data.cDNA_pos,
                    f"{mutation_data.AA_ref}{aa_pos}{mutation_data.AA_alt}",
                    f"{probability:.4f}",
                    uncertainty_label,
                    reason
                ])

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

    # OOD warning for gnomAD allele frequency
    if gnomad_af_raw > 0.01:
        _warnings.append(f"gnomAD allele frequency ({gnomad_af_raw:.4f}) is unusually high for a rare-disease variant. This variant may be benign or a common polymorphism.")

    # PM5: check if a different pathogenic missense exists at this AA position
    pm5_positions = gene_data.get("pm5_positions", set())
    known_path_at_pos = int(aa_pos) in pm5_positions if pm5_positions else False

    # Data Scarcity Quantification (Item 41)
    _variant_domain = sf.get('domain', 'uncharacterized')
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

    features_dict = {
        'domain': sf.get('domain', 'uncharacterized'),
        'dist_dna': sf.get('dist_dna', 999.0),
        'dist_palb2': sf.get('dist_palb2', 999.0),
        'gnomad_af': gnomad_af_raw,
        'in_critical_domain': _safe_critical_domain(raw_vector, gene_data, aa_pos),
        'known_pathogenic_at_pos': known_path_at_pos,
    }
    acmg_eval = evaluate_acmg_rules(features_dict, probability, gene_name=mutation_data.gene_name)

    # Contrastive Explanation Pairs (Item 43)
    _contrastive = find_contrastive_explanation(
        mutation_data.gene_name, scaled_vector[0], probability, feature_names_list,
    )

    # Split Conformal Prediction (Item 5.1)
    _conformal = compute_conformal_set(probability, mutation_data.gene_name)

    latency_ms = (time.perf_counter() - t_start) * 1000
    logger.info(f"[PREDICT] {mutation_data.AA_ref}{aa_pos}{mutation_data.AA_alt} (cDNA:{mutation_data.cDNA_pos}) -> {label} p={probability:.4f} ({latency_ms:.0f}ms)")

    result = {
        "prediction": label,
        "probability": round(probability, 4),
        "risk_tier": risk,
        "threshold": round(float(threshold), 3),
        "aa_pos": aa_pos,
        "genomic_pos": gene_data["cdna_to_genomic"].get(mutation_data.cDNA_pos),
        "acmg_evidence": acmg_eval,
        "confidence": {
            "probability": round(probability, 4),
            "label": uncertainty_label,
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "ci_width": round(ci_width, 4),
            "std": round(ci_std, 4),
            "method": ci_method,
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
        "warnings": _warnings if _warnings else None,
    }
    _pred_cache_set(pred_cache_key, result)
    # Record to SQLite database (non-blocking, fire-and-forget)
    try:
        record_analysis(
            gene=mutation_data.gene_name, cdna_pos=mutation_data.cDNA_pos,
            aa_ref=mutation_data.AA_ref, aa_alt=mutation_data.AA_alt,
            aa_pos=aa_pos, mutation=mutation_data.Mutation,
            prediction=label, probability=probability,
            risk_tier=risk, confidence_label=uncertainty_label,
            ci_lower=ci_lower, ci_upper=ci_upper, latency_ms=latency_ms
        )
    except Exception as e:
        logger.warning(f"[DB] Failed to record analysis: {type(e).__name__}: {e}")
    # Structured audit log (no PII)
    _log_prediction_audit(
        gene=mutation_data.gene_name,
        hgvs_c=f"c.{mutation_data.cDNA_pos}{mutation_data.Mutation}",
        hgvs_p=f"p.{mutation_data.AA_ref}{aa_pos}{mutation_data.AA_alt}",
        probability=probability,
        risk_tier=risk,
        request_id=getattr(request.state, "request_id", str(uuid.uuid4())[:8]),
        response_time_ms=latency_ms,
    )
    with _rate_lock:
        _metrics["predictions"] += 1
        _metrics["total_predict_ms"] += latency_ms
    return result

@app.get("/")
async def root():
    uni = _get_universal_models()
    return {
        "status": "SteppeDNA API running",
        "models": "Universal (Pan-Gene)",
        "model_type": "Ensemble (XGBoost + MLP, gene-adaptive weights)",
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
         description="Validates all critical ML model artifacts, data files, and mappings are loaded and ready.")
async def health():

    uni = _get_universal_models()
    checks = {
        "universal_models": uni.get("ensemble_model") is not None,
        "universal_scaler": uni.get("scaler") is not None,
        "universal_calibrator": uni.get("calibrator") is not None,
        "feature_names": len(uni.get("feature_names", [])),
        "phylop_scores": len(phylop_scores) > 0,
        "mave_scores": len(mave_by_variant) > 0,
        "alphamissense_scores": len(am_by_variant) > 0,
        "shap_booster": uni.get("booster") is not None,
        "genomic_mapping": len(genomic_to_cdna) > 0,
    }
    all_ok = all(checks.values())
    status_code = 200 if all_ok else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "healthy" if all_ok else "degraded", "checks": checks},
    )

@app.get("/docs/validation-report", tags=["System"], summary="Serve VALIDATION_REPORT.md")
async def validation_report():
    path = os.path.join(os.path.dirname(__file__), "..", "VALIDATION_REPORT.md")
    return FileResponse(path, media_type="text/markdown", filename="VALIDATION_REPORT.md")

@app.get("/structure/brca2/{fragment}", tags=["System"], summary="Serve BRCA2 AlphaFold fragment PDB")
async def brca2_fragment(fragment: int):
    """Serve local AlphaFold BRCA2 fragment PDB files (F1-F12).
    BRCA2 is too large for single AlphaFold prediction, split into 12 overlapping fragments."""
    if fragment < 1 or fragment > 12:
        return JSONResponse(status_code=404, content={"error": "Fragment must be 1-12"})
    path = os.path.join(os.path.dirname(__file__), "..", "data", "brca2_fragments",
                        f"AF-P51587-F{fragment}-model_v6.pdb")
    if not os.path.isfile(path):
        return JSONResponse(status_code=404, content={"error": f"Fragment F{fragment} not found"})
    return FileResponse(path, media_type="chemical/x-pdb",
                        filename=f"AF-P51587-F{fragment}-model_v6.pdb")

# ============================================================
# VCF PARSING ENDPOINT
# ============================================================
def parse_vcf_line(line):
    """Parse a single VCF data line into (chrom, pos, ref, alt, genotype).

    Design: Returns the raw ALT field which may contain comma-separated
    multi-allelic values (e.g. "T,G"). Splitting into individual alleles
    is handled by the /predict/vcf endpoint caller, so each allele gets
    its own prediction while sharing the same genotype context.

    Genotype is extracted from FORMAT+SAMPLE columns if present (cols 8+9),
    otherwise None. Used for compound heterozygosity detection.
    """
    parts = line.strip().split("\t")
    if len(parts) < 5:
        return None
    chrom = parts[0].replace("chr", "")
    # Validate chromosome format
    if not re.match(r'^([1-9]|1[0-9]|2[0-2]|X|Y|MT?)$', chrom, re.IGNORECASE):
        return None
    try:
        pos = int(parts[1])
    except ValueError:
        return None
    ref = parts[3].upper()
    alt = parts[4].upper()

    # Parse genotype from FORMAT + SAMPLE columns if present
    genotype = None
    if len(parts) >= 10:
        format_field = parts[8]
        sample_field = parts[9]
        format_keys = format_field.split(":")
        sample_vals = sample_field.split(":")
        if "GT" in format_keys:
            gt_idx = format_keys.index("GT")
            if gt_idx < len(sample_vals):
                genotype = sample_vals[gt_idx]

    return chrom, pos, ref, alt, genotype

def vcf_variant_to_prediction(chrom, genomic_pos, ref_allele, alt_allele, gene_name="BRCA2", gene_data=None):
    """
    Convert a VCF variant to a prediction.

    Returns:
        tuple: (result_dict, error_message)
            - On success: (dict with prediction/probability/variant_type/..., None)
            - On skip: (None, str reason why variant was skipped)

    Handles 6 variant types:
      - Frameshift (Tier 1 rule): p=0.9999, PVS1
      - In-frame indel (Tier 1 rule): domain-aware PM4
      - Splice-site (Tier 1 rule): canonical ±1-2bp / near ±3-8bp
      - Nonsense (Tier 1 rule): p=0.9999, PVS1
      - Synonymous (Tier 1 rule): splice-proximity aware, BP7
      - Missense (Tier 2 ML): ensemble prediction
    """
    if gene_data is None:
        gene_data = get_gene_data(gene_name)

    gene_config = gene_data.get("gene_config", {})
    if not gene_config:
        return None, f"Gene configuration not found for {gene_name}"
    target_chrom = str(gene_config.get("chromosome", "13"))
    target_strand = gene_config.get("strand", "+")

    if chrom != target_chrom:
        return None, f"Not on chr{target_chrom}"

    genomic_to_cdna = gene_data.get("genomic_to_cdna", {})
    if not genomic_to_cdna:
        return None, "Genomic mapping not loaded"

    exon_data = gene_data.get("exon_data", {})

    # ─── INDEL CHECK ───────────────────────────────────────────────────────────
    if len(ref_allele) != 1 or len(alt_allele) != 1:
        len_diff = abs(len(ref_allele) - len(alt_allele))
        if len_diff % 3 != 0:
            # Frameshift — highly pathogenic
            return {
                "cdna_pos": genomic_to_cdna.get(genomic_pos, 0),
                "aa_pos": 0,
                "aa_ref": "Indel",
                "aa_alt": "fs",
                "mutation": "delins",
                "hgvs_p": "p.fs",
                "prediction": "Pathogenic",
                "probability": 0.9999,
                "risk_tier": "high (Frameshift Truncation)",
                "genomic_pos": genomic_pos,
                "variant_type": "frameshift",
                "acmg_evidence": {"PVS1": "Frameshift null variant causing premature truncation"},
                "comments": "Frameshift mutation resulting in likely premature truncation."
            }, None
        else:
            # ─── IN-FRAME INDEL: domain-aware rule-based classification ────────
            cdna_pos_approx = genomic_to_cdna.get(genomic_pos, 0)
            aa_pos_approx = (cdna_pos_approx - 1) // 3 + 1 if cdna_pos_approx > 0 else 0
            aa_change = len_diff // 3

            in_domain = False
            domain_name = None
            if aa_pos_approx > 0:
                for dname, drange in gene_config.get("domains", {}).items():
                    if drange[0] <= aa_pos_approx <= drange[1]:
                        in_domain = True
                        domain_name = dname
                        break

            if in_domain:
                return {
                    "cdna_pos": cdna_pos_approx,
                    "aa_pos": aa_pos_approx,
                    "aa_ref": "Indel",
                    "aa_alt": f"inframe({aa_change}aa)",
                    "mutation": "delins",
                    "hgvs_p": f"p.inframe_indel_{aa_pos_approx}",
                    "prediction": "Likely Pathogenic",
                    "probability": 0.80,
                    "risk_tier": "high (In-Frame Indel in Functional Domain)",
                    "genomic_pos": genomic_pos,
                    "variant_type": "inframe_indel",
                    "acmg_evidence": {"PM4": f"In-frame deletion/insertion in {domain_name} domain"},
                    "comments": f"In-frame indel ({aa_change} AA) in {domain_name} domain"
                }, None
            else:
                return {
                    "cdna_pos": cdna_pos_approx,
                    "aa_pos": aa_pos_approx,
                    "aa_ref": "Indel",
                    "aa_alt": f"inframe({aa_change}aa)",
                    "mutation": "delins",
                    "hgvs_p": f"p.inframe_indel_{aa_pos_approx}",
                    "prediction": "VUS",
                    "probability": 0.50,
                    "risk_tier": "uncertain (In-Frame Indel)",
                    "genomic_pos": genomic_pos,
                    "variant_type": "inframe_indel",
                    "acmg_evidence": {"PM4": "In-frame deletion/insertion outside known functional domain"},
                    "comments": f"In-frame indel ({aa_change} AA) outside known functional domains"
                }, None

    if ref_allele not in "ACGT" or alt_allele not in "ACGT":
        return None, "Invalid alleles"

    # ─── SPLICE-SITE CHECK (before CDS check) ─────────────────────────────────
    if genomic_pos not in genomic_to_cdna:
        splice_info = exon_data.get("splice_info", {}).get(genomic_pos)
        if splice_info and splice_info["zone"] == "canonical":
            return {
                "cdna_pos": 0,
                "aa_pos": 0,
                "aa_ref": ref_allele,
                "aa_alt": alt_allele,
                "mutation": f"{ref_allele}>{alt_allele}",
                "hgvs_p": f"splice_site_g.{genomic_pos}",
                "prediction": "Likely Pathogenic",
                "probability": 0.95,
                "risk_tier": "high (Canonical Splice Site)",
                "genomic_pos": genomic_pos,
                "variant_type": "splice_canonical",
                "acmg_evidence": {"PVS1": f"Canonical splice site variant ({splice_info['distance']}bp from exon boundary) disrupting mRNA splicing"},
                "comments": f"Canonical splice site variant at {splice_info['distance']}bp from exon boundary"
            }, None
        elif splice_info and splice_info["zone"] == "near":
            return {
                "cdna_pos": 0,
                "aa_pos": 0,
                "aa_ref": ref_allele,
                "aa_alt": alt_allele,
                "mutation": f"{ref_allele}>{alt_allele}",
                "hgvs_p": f"near_splice_g.{genomic_pos}",
                "prediction": "VUS (Splice Proximity)",
                "probability": 0.70,
                "risk_tier": "moderate (Near Splice Site)",
                "genomic_pos": genomic_pos,
                "variant_type": "splice_near",
                "acmg_evidence": {"PP3_splice": f"Near splice site variant ({splice_info['distance']}bp from exon boundary) with potential splicing impact"},
                "comments": f"Near splice site variant at {splice_info['distance']}bp from exon boundary"
            }, None
        else:
            return None, f"Not in {gene_name} CDS"

    cdna_pos = genomic_to_cdna[genomic_pos]

    if target_strand == "-":
        cds_ref = COMPLEMENT.get(ref_allele, ref_allele)
        cds_alt = COMPLEMENT.get(alt_allele, alt_allele)
    else:
        cds_ref = ref_allele
        cds_alt = alt_allele

    # Build the mutation string (CDS-strand)
    mutation_str = f"{cds_ref}>{cds_alt}"

    # Determine amino acid change using the CDS
    cds_seq = gene_data.get("cds")
    if cds_seq is None:
        return None, "CDS not loaded"

    # cDNA is 1-based, Python string is 0-based
    codon_index = (cdna_pos - 1) // 3  # 0-based codon number
    pos_in_codon = (cdna_pos - 1) % 3  # 0, 1, or 2
    codon_start = codon_index * 3

    if codon_start + 3 > len(cds_seq):
        return None, "Codon out of CDS range"

    ref_codon = cds_seq[codon_start:codon_start + 3]

    # Verify the CDS matches what the VCF says (after complement)
    if ref_codon[pos_in_codon] != cds_ref:
        # Try without complementing (in case mapping is already on CDS strand)
        if ref_codon[pos_in_codon] == ref_allele:
            cds_ref = ref_allele
            cds_alt = alt_allele
            mutation_str = f"{cds_ref}>{cds_alt}"
        else:
            return None, f"Ref mismatch: CDS has {ref_codon[pos_in_codon]} at cDNA {cdna_pos}, VCF says {ref_allele}"

    # Build the mutant codon
    alt_codon = list(ref_codon)
    alt_codon[pos_in_codon] = cds_alt
    alt_codon = "".join(alt_codon)

    ref_aa = CODON_TABLE.get(ref_codon.upper(), "Unknown")
    alt_aa = CODON_TABLE.get(alt_codon.upper(), "Unknown")
    aa_pos = codon_index + 1  # 1-based

    if ref_aa == "Unknown" or alt_aa == "Unknown":
        return None, "Unknown codon"

    # ─── SYNONYMOUS VARIANT HANDLING ──────────────────────────────────────────
    if ref_aa == alt_aa:
        # Check if near an exon boundary (splice-proximity)
        cdna_to_genomic = gene_data.get("cdna_to_genomic", {})
        near_boundary = False
        for delta in [-3, -2, -1, 1, 2, 3]:
            adj_cdna = cdna_pos + delta
            adj_gpos = cdna_to_genomic.get(adj_cdna)
            curr_gpos = cdna_to_genomic.get(cdna_pos)
            if adj_gpos is not None and curr_gpos is not None:
                if target_strand == "+":
                    if abs(adj_gpos - curr_gpos) > abs(delta) + 1:
                        near_boundary = True
                        break
                else:
                    if abs(adj_gpos - curr_gpos) > abs(delta) + 1:
                        near_boundary = True
                        break

        if near_boundary:
            return {
                "cdna_pos": cdna_pos,
                "aa_pos": aa_pos,
                "aa_ref": ref_aa,
                "aa_alt": ref_aa,
                "mutation": mutation_str,
                "hgvs_p": f"p.{ref_aa}{aa_pos}= (synonymous)",
                "prediction": "VUS (Synonymous, Near Splice)",
                "probability": 0.40,
                "risk_tier": "uncertain (Synonymous Near Splice)",
                "genomic_pos": genomic_pos,
                "variant_type": "synonymous",
                "acmg_evidence": {},
                "comments": "Synonymous variant near exon boundary; may affect splicing"
            }, None
        else:
            return {
                "cdna_pos": cdna_pos,
                "aa_pos": aa_pos,
                "aa_ref": ref_aa,
                "aa_alt": ref_aa,
                "mutation": mutation_str,
                "hgvs_p": f"p.{ref_aa}{aa_pos}= (synonymous)",
                "prediction": "Likely Benign",
                "probability": 0.05,
                "risk_tier": "low (Synonymous)",
                "genomic_pos": genomic_pos,
                "variant_type": "synonymous",
                "acmg_evidence": {"BP7": "Synonymous variant with no predicted splice impact"},
                "comments": "Synonymous variant far from splice site"
            }, None

    # ─── TIER 1: RULE INTERCEPTOR ───────────────────────────────────────────────
    if alt_aa in ["Ter", "*", "fs"]:
        return {
            "cdna_pos": cdna_pos,
            "aa_pos": aa_pos,
            "aa_ref": ref_aa,
            "aa_alt": alt_aa,
            "mutation": mutation_str,
            "hgvs_p": f"p.{ref_aa}{aa_pos}{alt_aa}",
            "prediction": "Pathogenic",
            "probability": 0.9999,
            "risk_tier": "high (Truncating)",
            "genomic_pos": genomic_pos,
            "variant_type": "nonsense",
            "acmg_evidence": {"PVS1": "Pathogenic truncating null variant (Nonsense)"},
            "comments": "Tier 1 Protocol: Pathogenic truncating null variant (Nonsense)"
        }, None

    # ─── TIER 2: MACHINE LEARNING ENGINE ──────────────────────────────────────

    # Build feature vector and predict
    if gene_data.get("ensemble_model") is None or gene_data.get("scaler") is None:
        return None, "ML models missing for gene"

    raw_vector = build_feature_vector(cdna_pos, ref_aa, alt_aa, mutation_str, aa_pos, gene_name=gene_name)
    if raw_vector.shape[1] == 0:
        return None, "Feature names not available for gene"

    scaled_vector = gene_data["scaler"].transform(raw_vector)

    calibrator = gene_data.get("calibrator")
    nn_model = gene_data.get("ensemble_model")
    xgb_model = gene_data.get("booster")

    if calibrator is not None and nn_model is not None and xgb_model is not None:
        nn_p = nn_model.predict(scaled_vector, verbose=0).flatten()[0]
        xgb_p = xgb_model.predict(xgb.DMatrix(scaled_vector))[0]
        blended = (XGB_WEIGHT * xgb_p) + (NN_WEIGHT * nn_p)
        probability = float(calibrator.predict([blended])[0])
        vcf_disagreement = round(float(abs(xgb_p - nn_p)), 4)
    else:
        probability = 0.5
        vcf_disagreement = None

    # Bootstrap CI for VCF variants (Item 39)
    vcf_feature_names = gene_data.get("feature_names", [])
    vcf_bootstrap_ci = compute_bootstrap_ci(scaled_vector, vcf_feature_names)

    # Clip to [0.5%, 99.5%] — no model should claim absolute certainty
    probability = float(np.clip(probability, 0.005, 0.995))

    gene_threshold = gene_data.get("threshold", 0.5)
    label = "Pathogenic" if probability >= gene_threshold else "Benign"
    risk = _compute_risk_tier(probability)

    vcf_result = {
        "cdna_pos": cdna_pos,
        "aa_pos": aa_pos,
        "aa_ref": ref_aa,
        "aa_alt": alt_aa,
        "mutation": mutation_str,
        "hgvs_p": f"p.{ref_aa}{aa_pos}{alt_aa}",
        "prediction": label,
        "probability": round(probability, 4),
        "risk_tier": risk,
        "genomic_pos": genomic_pos,
        "variant_type": "missense",
        "model_disagreement": vcf_disagreement,
    }
    if vcf_bootstrap_ci:
        vcf_result["bootstrap_ci"] = vcf_bootstrap_ci
    return vcf_result, None

@app.post("/predict/vcf", tags=["Prediction"], summary="Batch VCF prediction",
          description="Upload a VCF file to predict pathogenicity for all missense variants in the targeted gene. Returns per-variant predictions with risk tiers.")
async def predict_vcf(file: UploadFile = File(...), gene: str = Form("BRCA2")):
    t_start = time.perf_counter()
    if gene.upper() not in SUPPORTED_GENES:
        return {"error": f"Unsupported gene: {gene}. Supported: {', '.join(sorted(SUPPORTED_GENES))}"}
    gene = gene.upper()
    gene_data = get_gene_data(gene)
    if not gene_data.get("genomic_to_cdna"):
        return {"error": f"VCF parsing not available — {gene} genomic mapping not loaded"}

    content = await file.read()
    if len(content) > MAX_VCF_SIZE:
        return {"error": f"File too large ({len(content) // (1024*1024)} MB). Maximum is 50 MB."}
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return {"error": "Could not read file as UTF-8"}
    
    lines = text.strip().split("\n")

    # Validate VCF format — check for header or at least tab-separated data
    has_vcf_header = any(line.startswith("##fileformat=VCF") for line in lines[:50])
    has_chrom_header = any(line.startswith("#CHROM") for line in lines[:100])
    if not has_vcf_header and not has_chrom_header:
        # Check if it looks like tab-separated variant data at all
        data_lines = [l for l in lines if l.strip() and not l.startswith("#")]
        if data_lines and len(data_lines[0].split("\t")) < 5:
            return {"error": "File does not appear to be in VCF format. Expected tab-separated columns with CHROM, POS, ID, REF, ALT."}

    results = []
    skipped = []
    genotypes = []  # track genotypes for compound het detection
    seen_variants = set()  # track (chrom, pos, ref, alt) for duplicate detection
    total_data_lines = 0

    for line in lines:
        line = line.strip()
        if not line or line.startswith("##"):
            continue
        if line.startswith("#CHROM") or line.startswith("#"):
            continue

        total_data_lines += 1
        parsed = parse_vcf_line(line)
        if parsed is None:
            skipped.append({"line": total_data_lines, "reason": "Parse error"})
            continue

        chrom, pos, ref, alt, gt = parsed

        # Handle multi-allelic VCF: parse_vcf_line returns the raw ALT field
        # (may be "T,G" for multi-allelic sites). Split here so each allele
        # gets its own prediction while sharing the same genotype context.
        for single_alt in alt.split(","):
            variant_key = (chrom, pos, ref, single_alt.strip())
            if variant_key in seen_variants:
                skipped.append({"line": total_data_lines, "reason": "Duplicate variant", "pos": pos})
                continue
            seen_variants.add(variant_key)

            result, reason = vcf_variant_to_prediction(chrom, pos, ref, single_alt.strip(), gene_name=gene, gene_data=gene_data)
            if result:
                result.setdefault("variant_type", "unknown")  # defensive fallback
                results.append(result)
                genotypes.append(gt)
            elif reason:
                skipped.append({"line": total_data_lines, "reason": reason, "pos": pos})

    # ─── COMPOUND HETEROZYGOSITY DETECTION ─────────────────────────────────────
    compound_het_warning = None
    if len(results) >= 2:
        het_variants = []
        has_gt_data = False
        for i, r in enumerate(results):
            gt = genotypes[i] if i < len(genotypes) else None
            if gt is not None:
                has_gt_data = True
            # Consider heterozygous if GT is 0/1, 0|1, 1|0, 1/0, or if GT is absent
            is_het = (gt is None or gt in ("0/1", "0|1", "1|0", "1/0"))
            if is_het:
                het_variants.append(r)

        if len(het_variants) >= 2:
            variant_descs = [
                f"{r.get('hgvs_p', '?')} ({r.get('variant_type', 'unknown')}, p={r.get('probability', 0):.2f})"
                for r in het_variants[:5]
            ]
            compound_het_warning = {
                "gene": gene,
                "num_variants": len(het_variants),
                "variants": variant_descs,
                "has_gt_data": has_gt_data,
                "message": (
                    f"{len(het_variants)} heterozygous variants detected in {gene}. "
                    f"Compound heterozygosity may affect pathogenicity -- "
                    f"consider phasing analysis to determine if variants are in cis (same allele) or trans (different alleles)."
                ),
            }

    latency_ms = (time.perf_counter() - t_start) * 1000
    n_classified = len(results)
    type_counts = {}
    for r in results:
        vt = r.get("variant_type", "unknown")
        type_counts[vt] = type_counts.get(vt, 0) + 1
    logger.info(f"[VCF] {file.filename}: {total_data_lines} total, {n_classified} classified {type_counts}, {len(skipped)} skipped ({latency_ms:.0f}ms)")

    with _rate_lock:
        _metrics["vcf_uploads"] += 1
    try:
        nPath = len([r for r in results if r.get("prediction") in ("Pathogenic", "Likely Pathogenic")])
        nBen = len([r for r in results if "Benign" in r.get("prediction", "")])
        record_vcf_upload(
            filename=file.filename, gene=gene,
            total_variants=total_data_lines, missense_found=n_classified,
            pathogenic_count=nPath, benign_count=nBen, latency_ms=latency_ms
        )
    except Exception as e:
        logger.warning(f"[DB] Failed to record VCF upload: {type(e).__name__}: {e}")

    response = {
        "total_variants_in_file": total_data_lines,
        "brca2_missense_found": n_classified,  # kept identical key for front-end backwards compatibility
        "variants_classified": n_classified,
        "gene_analyzed": gene,
        "skipped_count": len(skipped),
        "predictions": results,
        "skipped_reasons": skipped[:50],  # first 50 skipped for debugging
        "variant_type_counts": type_counts,
    }
    if compound_het_warning:
        response["compound_het_warning"] = compound_het_warning

    return response

# ─── Live ClinVar / gnomAD Lookup ────────────────────────────────────────────
# Real-time cross-referencing against external databases.
# These are optional enrichment endpoints — the core prediction works offline.

import httpx
import urllib.parse

# Per-IP rate limiter for external API proxy endpoints (ClinVar/gnomAD)
EXTERNAL_RATE_LIMIT = int(os.getenv("EXTERNAL_RATE_LIMIT", "20"))  # per window
_external_rate: dict = collections.defaultdict(list)

def _check_external_rate(ip: str) -> bool:
    """Returns True if request is allowed, False if rate-limited."""
    now = time.monotonic()
    with _rate_lock:
        _external_rate[ip] = [t for t in _external_rate[ip] if now - t < RATE_WINDOW]
        if len(_external_rate[ip]) >= EXTERNAL_RATE_LIMIT:
            return False
        _external_rate[ip].append(now)
    return True

# ─── Separate Caches ──────────────────────────────────────────────────────────
# Prediction cache (short TTL, keyed by variant): avoids re-computing identical variants
_pred_cache = collections.OrderedDict()
MAX_PRED_CACHE = 500
PRED_CACHE_TTL = 1800  # 30 minutes

# External API cache (longer TTL): ClinVar / gnomAD lookups
_api_cache = collections.OrderedDict()
MAX_CACHE_SIZE = 1000
CACHE_TTL = 3600  # 1 hour

def _pred_cache_set(key, value):
    while len(_pred_cache) >= MAX_PRED_CACHE:
        _pred_cache.popitem(last=False)
    _pred_cache[key] = (time.time(), value)

def _pred_cache_get(key):
    if key in _pred_cache:
        cached_time, cached_result = _pred_cache[key]
        if time.time() - cached_time < PRED_CACHE_TTL:
            _pred_cache.move_to_end(key)
            return cached_result
        else:
            del _pred_cache[key]
    return None

def _cache_set(key, value):
    # Evict oldest entries when full (LRU)
    while len(_api_cache) >= MAX_CACHE_SIZE:
        _api_cache.popitem(last=False)
    _api_cache[key] = (time.time(), value)

def _cache_get(key):
    """Return cached value if present and not expired, else None."""
    if key in _api_cache:
        cached_time, cached_result = _api_cache[key]
        if time.time() - cached_time < CACHE_TTL:
            _api_cache.move_to_end(key)  # Mark as recently used
            return cached_result
        else:
            del _api_cache[key]  # Expired
    return None

@app.get("/lookup/clinvar/{variant}", tags=["External Lookups"], summary="ClinVar lookup",
         description="Look up a variant in NCBI ClinVar. Accepts format: p.Thr2722Arg or Thr2722Arg. Returns clinical significance and review status.")
async def lookup_clinvar(variant: str, request: Request):
    cache_key = f"clinvar_{variant}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # Rate-limit external API calls separately
    ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or (request.client.host if request.client else "unknown")
    if not _check_external_rate(ip):
        return JSONResponse(status_code=429, content={"error": "Too many external lookups. Please wait before trying again."})

    t_start = time.perf_counter()
    clean = variant.replace("p.", "").strip()

    # Validate variant format to prevent query injection (e.g. Thr2722Arg)
    if not re.match(r'^[A-Za-z]{2,4}\d{1,5}[A-Za-z]{2,4}$', clean):
        return {"variant": clean, "error": "Invalid variant format. Expected e.g. Thr2722Arg"}

    gene_upper = request.query_params.get("gene", "BRCA2").upper()
    if gene_upper not in SUPPORTED_GENES:
        gene_upper = "BRCA2"
    query = f'{gene_upper}[gene] AND "{clean}"[variant name] AND "homo sapiens"[organism]'
    encoded = urllib.parse.urlencode({
        "db": "clinvar", "term": query, "retmax": 5, "retmode": "json"
    })
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{encoded}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Simple retry: 1 attempt + 1 retry with 2s delay
            for _attempt in range(2):
                try:
                    resp = await client.get(url, headers={"User-Agent": "SteppeDNA/1.0"})
                    resp.raise_for_status()
                    break
                except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException):
                    if _attempt == 0:
                        import asyncio; await asyncio.sleep(2)
                    else:
                        raise
            data = resp.json()

            ids = data.get("esearchresult", {}).get("idlist", [])
            if not ids:
                res = {"variant": clean, "clinvar": "Not found", "ids": []}
                _cache_set(cache_key, res)
                return res

            # Fetch summary for first match
            summary_url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                f"?db=clinvar&id={ids[0]}&retmode=json"
            )
            resp2 = await client.get(summary_url, headers={"User-Agent": "SteppeDNA/1.0"})
            resp2.raise_for_status()
            sdata = resp2.json()

            rec = sdata.get("result", {}).get(ids[0], {})
            latency_ms = (time.perf_counter() - t_start) * 1000
            logger.info(f"[ClinVar] {clean}: {len(ids)} hits ({latency_ms:.0f}ms)")

            res = {
                "variant": clean,
                "clinvar_id": ids[0],
                "clinical_significance": rec.get("clinical_significance", {}).get("description", "Unknown"),
                "title": rec.get("title", ""),
                "review_status": rec.get("clinical_significance", {}).get("review_status", ""),
                "n_results": len(ids),
            }
            _cache_set(cache_key, res)
            return res
    except Exception as e:
        logger.warning(f"[ClinVar] Lookup failed for {clean}: {type(e).__name__}: {e}")
        return {"variant": clean, "error": "ClinVar lookup failed. Please try again later."}


@app.get("/lookup/gnomad/{variant}", tags=["External Lookups"], summary="gnomAD lookup",
         description="Look up a variant in gnomAD v4. Accepts format: 13-32316461-A-G (chr-pos-ref-alt). Returns genome/exome allele frequencies and homozygote counts.")
async def lookup_gnomad(variant: str, request: Request):
    cache_key = f"gnomad_{variant}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # Rate-limit external API calls separately
    ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or (request.client.host if request.client else "unknown")
    if not _check_external_rate(ip):
        return JSONResponse(status_code=429, content={"error": "Too many external lookups. Please wait before trying again."})

    t_start = time.perf_counter()

    # If protein notation provided, try to map to genomic via our lookup
    if not variant[0].isdigit():
        return {
            "variant": variant,
            "note": "Provide genomic coordinates: chr-pos-ref-alt (e.g. 13-32316461-A-G)",
            "tip": "Use /predict first to get the genomic coordinates for your variant.",
        }

    parts = variant.split("-")
    if len(parts) != 4:
        return {"error": "Expected format: chr-pos-ref-alt (e.g. 13-32316461-A-G)"}

    chrom, pos, ref, alt = parts

    # Validate parts to prevent injection — only allow alphanumeric and expected chars
    if not re.match(r'^([1-9]|1[0-9]|2[0-2]|X|Y|MT?)$', chrom, re.IGNORECASE):
        return {"error": "Invalid chromosome in variant"}
    if not re.match(r'^\d+$', pos):
        return {"error": "Invalid position in variant"}
    if not re.match(r'^[ACGT]+$', ref):
        return {"error": "Invalid ref allele in variant"}
    if not re.match(r'^[ACGT]+$', alt):
        return {"error": "Invalid alt allele in variant"}

    variant_id = f"{chrom}-{pos}-{ref}-{alt}"
    query = """
    query GnomadVariant($variantId: String!) {
      variant(dataset: gnomad_r4, variantId: $variantId) {
        variant_id
        genome {
          ac
          an
          af
          homozygote_count
        }
        exome {
          ac
          an
          af
          homozygote_count
        }
      }
    }
    """

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Simple retry: 1 attempt + 1 retry with 2s delay
            for _attempt in range(2):
                try:
                    resp = await client.post(
                        "https://gnomad.broadinstitute.org/api",
                        json={"query": query, "variables": {"variantId": variant_id}},
                        headers={"Content-Type": "application/json", "User-Agent": "SteppeDNA/1.0"}
                    )
                    resp.raise_for_status()
                    break
                except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException):
                    if _attempt == 0:
                        import asyncio; await asyncio.sleep(2)
                    else:
                        raise
            data = resp.json()

            # Check for GraphQL-level errors
            if "errors" in data:
                logger.warning(f"[gnomAD] GraphQL errors for {variant}: {data['errors']}")
                res = {"variant": variant, "error": "gnomAD returned an error. Try again later."}
                _cache_set(cache_key, res)
                return res

            v = data.get("data", {}).get("variant")
            latency_ms = (time.perf_counter() - t_start) * 1000
            logger.info(f"[gnomAD] {variant}: {'found' if v else 'not found'} ({latency_ms:.0f}ms)")

            if not v:
                res = {"variant": variant, "gnomad": "Not found in gnomAD v4"}
                _cache_set(cache_key, res)
                return res

            genome = v.get("genome") or {}
            exome  = v.get("exome") or {}
            res = {
                "variant": variant,
                "variant_id": v.get("variant_id"),
                "genome_af": genome.get("af"),
                "genome_ac": genome.get("ac"),
                "genome_an": genome.get("an"),
                "genome_hom": genome.get("homozygote_count"),
                "exome_af": exome.get("af"),
                "exome_ac": exome.get("ac"),
                "exome_an": exome.get("an"),
                "exome_hom": exome.get("homozygote_count"),
            }
            _cache_set(cache_key, res)
            return res
    except Exception as e:
        logger.warning(f"[gnomAD] Lookup failed for {variant}: {type(e).__name__}: {e}")
        return {"variant": variant, "error": "gnomAD lookup failed. Please try again later."}


@app.get("/umap", tags=["Visualization"], summary="UMAP variant landscape",
         description="Returns precomputed UMAP 2D coordinates for up to 5,000 training variants, used by the frontend variant landscape visualization.")
async def get_umap():
    umap_path = os.path.join(DATA_DIR, "umap_coordinates.json")
    if os.path.exists(umap_path):
        with open(umap_path) as f:
            return JSONResponse(content=json.load(f))
    return JSONResponse(status_code=404, content={"error": "UMAP coordinates not precomputed. Run: python scripts/precompute_umap.py"})

# ============================================================
# SERVER MONITORING
# ============================================================
@app.get("/metrics", tags=["System"], summary="Server metrics",
         description="Returns server uptime, request counts, latency stats, and memory usage for monitoring dashboards.")
async def server_metrics():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        memory_rss_mb = round(mem.rss / (1024 * 1024), 1)
        memory_vms_mb = round(mem.vms / (1024 * 1024), 1)
        cpu_percent = process.cpu_percent(interval=0.1)
    except ImportError:
        memory_rss_mb = None
        memory_vms_mb = None
        cpu_percent = None

    with _rate_lock:
        predictions = _metrics["predictions"]
        vcf_uploads = _metrics["vcf_uploads"]
        total_predict_ms = _metrics["total_predict_ms"]

    return {
        "uptime_seconds": round(time.monotonic() - _server_start_time, 1),
        "memory_rss_mb": memory_rss_mb,
        "memory_vms_mb": memory_vms_mb,
        "cpu_percent": cpu_percent,
        "total_predictions": predictions,
        "total_vcf_uploads": vcf_uploads,
        "avg_predict_latency_ms": round(total_predict_ms / max(predictions, 1), 1),
        "cache_size": len(_api_cache),
        "rate_limit_active_ips": len(_rate_counts),
        "python_version": sys.version.split()[0],
    }


# ============================================================
# PATIENT COHORT TRACKING
# ============================================================
@app.post("/cohort/submit", tags=["Cohort"], summary="Submit anonymized variant for cohort tracking",
          description="Allows hospitals/clinics to submit anonymized variant observations for population-level tracking. No patient identifiers are stored.")
async def cohort_submit(request: Request):
    body = await request.json()

    # Validate required fields
    required = ["gene", "aa_ref", "aa_pos", "aa_alt", "prediction", "probability"]
    for field in required:
        if field not in body:
            return JSONResponse(status_code=400, content={"error": f"Missing required field: {field}"})

    gene = str(body["gene"]).upper()
    if gene not in SUPPORTED_GENES:
        return JSONResponse(status_code=400, content={"error": f"Unsupported gene: {gene}"})

    # Sanitize free-text fields: strip non-alphanumeric (except spaces, hyphens, dots), limit length
    def _sanitize_text(value: str, max_length: int = 100) -> str:
        """Strip special characters and limit length for free-text cohort fields."""
        cleaned = re.sub(r'[^\w\s\-\.]', '', str(value))
        return cleaned.strip()[:max_length]

    # Store in a simple append-only CSV (anonymized - no patient IDs)
    cohort_file = os.path.join(DATA_DIR, "cohort_observations.csv")
    import datetime
    fieldnames = ["timestamp", "gene", "variant", "prediction", "probability", "institution", "country"]
    row = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "gene": gene,
        "variant": f"{body['aa_ref']}{body['aa_pos']}{body['aa_alt']}",
        "prediction": str(body["prediction"])[:20],
        "probability": round(float(body["probability"]), 4),
        "institution": _sanitize_text(body.get("institution", "anonymous")),
        "country": _sanitize_text(body.get("country", "unknown"), max_length=60),
    }

    with _rate_lock:
        write_header = not os.path.exists(cohort_file) or os.path.getsize(cohort_file) == 0
        with open(cohort_file, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    logger.info(f"[COHORT] {row['gene']} {row['variant']} from {row['institution']}")
    return {"status": "recorded", "variant": row["variant"], "gene": row["gene"]}


@app.get("/cohort/stats", tags=["Cohort"], summary="Cohort aggregate statistics",
         description="Returns aggregate variant observation counts by gene and prediction class. No individual-level data is exposed.")
async def cohort_stats():
    cohort_file = os.path.join(DATA_DIR, "cohort_observations.csv")
    if not os.path.exists(cohort_file):
        return {"total_observations": 0, "by_gene": {}, "by_prediction": {}}

    with open(cohort_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    by_gene = {}
    by_pred = {"Pathogenic": 0, "Benign": 0}
    for row in rows:
        g = row.get("gene", "Unknown")
        by_gene[g] = by_gene.get(g, 0) + 1
        p = row.get("prediction", "Unknown")
        if p in by_pred:
            by_pred[p] += 1

    return {
        "total_observations": len(rows),
        "by_gene": by_gene,
        "by_prediction": by_pred,
        "unique_variants": len(set(f"{r.get('gene','')}-{r.get('variant','')}" for r in rows)),
    }


@app.get("/history", tags=["System"], summary="Recent analysis history",
         description="Returns recent variant analyses stored in the server-side database.")
async def analysis_history(limit: int = 50):
    try:
        return {"analyses": get_recent_analyses(min(limit, 200))}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Database error: {str(e)}"})

@app.get("/stats", tags=["System"], summary="Analysis statistics",
         description="Returns aggregate analysis statistics from the server-side database.")
async def analysis_stats():
    try:
        return get_analysis_stats()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Database error: {str(e)}"})


@app.get("/research/priorities", tags=["Research"], summary="Active learning priority variants",
         description="Returns VUS variants ranked by their value for functional validation, "
                     "computed via query-by-committee (model disagreement), gene scarcity weighting, "
                     "and positional novelty scoring.")
async def research_priorities(gene: str = None, limit: int = 10):
    # Validate gene first (even if no data loaded)
    if gene:
        gene_upper = gene.upper()
        if gene_upper not in SUPPORTED_GENES:
            return JSONResponse(status_code=400, content={"error": f"Unsupported gene: {gene}. Supported: {', '.join(sorted(SUPPORTED_GENES))}"})
    if not _ACTIVE_LEARNING:
        return {"error": "Active learning priorities not available. Run scripts/active_learning_ranker.py to generate."}
    priorities = _ACTIVE_LEARNING.get("priorities", {})
    metadata = _ACTIVE_LEARNING.get("metadata", {})
    if gene:
        entries = priorities.get(gene_upper, [])[:min(limit, 50)]
        return {
            "gene": gene_upper,
            "priorities": entries,
            "metadata": metadata,
        }
    # Return top N per gene
    result = {}
    for g in sorted(SUPPORTED_GENES):
        result[g] = priorities.get(g, [])[:min(limit, 50)]
    return {
        "priorities": result,
        "metadata": metadata,
        "gene_training_sizes": _ACTIVE_LEARNING.get("gene_training_sizes", {}),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
