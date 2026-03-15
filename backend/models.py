"""
SteppeDNA - Model Loading & Data Infrastructure

Handles loading of ML model artifacts, pickle data files,
per-gene calibrators, ensemble weights, bootstrap models,
active learning priorities, conformal thresholds, and ESM-2 globals.
"""

import os
import json
import pickle
import logging

import xgboost as xgb

logger = logging.getLogger("steppedna")

# ─── Paths & Constants ────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Whitelist of supported genes -- used to prevent path traversal via gene_name
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
N_EFFECTIVE = 200          # Pseudo-observations for Beta CI estimation
ESM2_WINDOW = 50           # Residues on each side for local ESM-2 windowed inference
ESM2_PCA_COMPONENTS = 20   # Number of PCA components from ESM-2 embeddings
MAX_VCF_SIZE = 50 * 1024 * 1024  # 50 MB

# Valid amino acid codes for input validation
VALID_AA_CODES = {'Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly', 'His', 'Ile',
                  'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser', 'Ter', 'Thr', 'Trp', 'Tyr', 'Val',
                  'Fs', 'Del', 'Ins', 'Dup', '*'}


# ─── Pickle Loading ──────────────────────────────────────────────────────────

def _load_pickle(filename, required=False):
    """Load a pickle file from DATA_DIR, returning None if optional and missing.

    Security note: Only loads files from the trusted DATA_DIR directory.
    Path traversal is blocked by os.path.basename normalization.
    """
    # Prevent path traversal -- strip directory components from filename
    safe_name = os.path.basename(filename)
    path = os.path.join(DATA_DIR, safe_name)
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        logger.warning(f"[WARN] {safe_name} not found")
        return None
    file_size = os.path.getsize(path)
    if file_size == 0:
        logger.error(f"[ERROR] {safe_name} is empty (0 bytes) -- likely corrupted")
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


# ─── Module-Level Data Loading ────────────────────────────────────────────────
# These are legacy BRCA2 data loaded at module level for /health endpoint.
calibrator = None
scaler = None
feature_names = []
threshold = 0.5
ensemble_model = None
booster = None
logger.info("ML Models will be lazy-loaded dynamically via get_gene_data()")

# Load shared data files (module-level for /predict data_sources & /health)
phylop_scores = _load_pickle("phylop_scores.pkl") or {}
mave_by_variant, mave_by_position = _load_variant_dict("mave_scores.pkl")
am_by_variant, am_by_position = _load_variant_dict("alphamissense_scores.pkl")
structural_features = _load_pickle("structural_features.pkl") or {}
gnomad_by_variant, gnomad_by_position = _load_variant_dict("gnomad_frequencies.pkl")

_cdna_genomic = _load_pickle("cdna_to_genomic.pkl") or {}
cdna_to_genomic = _cdna_genomic
genomic_to_cdna = {v: k for k, v in _cdna_genomic.items()}
logger.info(f"[OK] Genomic mapping: {len(genomic_to_cdna)} positions")


# ─── Per-Gene Calibrators (Item 45) ─────────────────────────────────────────
_GENE_CALIBRATORS: dict = {}  # gene -> IsotonicRegression or None


def _load_gene_calibrators():
    """Load per-gene calibrators from data/calibrator_<gene>.pkl files."""
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
            logger.info(f"[CALIBRATOR] No per-gene calibrator for {gene} -- will use universal")


# ─── Gene-Adaptive Ensemble Weights (Item 38) ──────────────────────────────
_GENE_ENSEMBLE_WEIGHTS: dict = {}  # gene -> {"xgb_weight": float, "mlp_weight": float}


def _load_gene_ensemble_weights():
    """Load per-gene optimized ensemble weights from data/gene_ensemble_weights.json."""
    weights_path = os.path.join(DATA_DIR, "gene_ensemble_weights.json")
    if not os.path.exists(weights_path):
        logger.info("[WEIGHTS] gene_ensemble_weights.json not found -- using default 0.6/0.4 for all genes")
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
    bootstrap_dir = os.path.join(DATA_DIR, "bootstrap_models")
    if not os.path.isdir(bootstrap_dir):
        logger.info("[BOOTSTRAP] data/bootstrap_models/ not found -- bootstrap CIs disabled. Run scripts/generate_bootstrap_models.py to enable.")
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
        logger.info("[BOOTSTRAP] No bootstrap models found -- CIs will use Beta approximation only")


# ─── Active Learning Priorities (Item 42) ────────────────────────────────────
_ACTIVE_LEARNING: dict = {}  # loaded from data/active_learning_priorities.json


def _get_active_learning() -> dict:
    """Getter to avoid stale module-level import references."""
    return _ACTIVE_LEARNING


def _load_active_learning_priorities():
    """Load pre-computed active learning priority rankings."""
    global _ACTIVE_LEARNING
    al_path = os.path.join(DATA_DIR, "active_learning_priorities.json")
    if not os.path.exists(al_path):
        logger.info("[ACTIVE-LEARNING] active_learning_priorities.json not found -- run scripts/active_learning_ranker.py to generate")
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
        logger.info("[CONFORMAL] conformal_thresholds.json not found -- run scripts/train_conformal.py to generate")
        return
    try:
        with open(ct_path, "r") as f:
            _CONFORMAL_THRESHOLDS = json.load(f)
        n_genes = len([k for k in _CONFORMAL_THRESHOLDS if k != "_global"])
        logger.info(f"[CONFORMAL] Loaded conformal thresholds for {n_genes} genes (alpha={_CONFORMAL_THRESHOLDS.get('_global', {}).get('alpha', 0.10)})")
    except Exception as e:
        logger.warning(f"[CONFORMAL] Failed to load conformal thresholds: {e}")


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
