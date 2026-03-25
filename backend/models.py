"""
SteppeDNA - Model Loading & Data Infrastructure
=================================================

This file is the "warehouse manager" for all machine-learning artifacts that
SteppeDNA needs at runtime.  It loads everything from disk into memory so the
rest of the backend can make predictions without touching the filesystem again.

What gets loaded (and why):
    1. **XGBoost model** -- a gradient-boosted tree model (one of two ensemble members).
    2. **MLP (neural network)** -- the second ensemble member.  We ship a pure-numpy
       reimplementation (NumpyMLP) so the server doesn't need TensorFlow installed.
    3. **StandardScaler** -- normalizes the 120 input features to zero-mean / unit-
       variance so the MLP sees numbers in a friendly range.
    4. **Isotonic calibrator** -- maps raw ensemble scores to well-calibrated
       probabilities.  "Calibrated" means that when the model says 0.80, roughly
       80 % of those variants really are pathogenic.
    5. **Per-gene calibrators** -- optional gene-specific isotonic calibrators that
       can improve calibration for genes with different score distributions.
    6. **Per-gene ensemble weights** -- each gene gets its own XGBoost-vs-MLP mix
       (e.g. BRCA1 is MLP-only, RAD51C is 80 % XGBoost) because the two models
       have different strengths on different genes.
    7. **Bootstrap models** -- 50 XGBoost models each trained on a resampled dataset.
       Running a variant through all 50 gives a distribution of scores, which we
       use to build confidence intervals.
    8. **Conformal thresholds** -- thresholds derived from a held-out calibration set
       that let us say "this prediction set covers the true label with ≥90 %
       probability" (split conformal prediction).
    9. **Lookup tables** -- PhyloP conservation scores, gnomAD allele frequencies,
       AlphaMissense scores, AlphaFold structural features, MAVE functional assay
       scores, and cDNA-to-genomic coordinate mappings.
   10. **Active learning priorities** -- a ranked list of VUS (variants of uncertain
       significance) that would be most valuable to experimentally characterize.

Design decisions:
    - The core ML models are **lazy-loaded** via ``_get_universal_models()``.  This
      means they are NOT loaded when the module is first imported -- only when the
      first prediction request arrives.  This keeps server startup fast (important
      for Render's 60-second deploy timeout) and avoids wasting memory if only the
      /health endpoint is hit.
    - Lookup tables (PhyloP, gnomAD, etc.) ARE loaded eagerly at module level because
      the /health endpoint reports their status and they are small.
    - All pickle loading goes through ``_load_pickle()`` which (a) sanitizes the
      filename to prevent path-traversal attacks and (b) gracefully handles missing
      or empty files with clear logging.
"""

import os
import json
import pickle
import logging

import xgboost as xgb  # Gradient-boosted tree library (one half of our ensemble)

logger = logging.getLogger("steppedna")  # All log messages go under the "steppedna" namespace

# ─── Paths & Constants ────────────────────────────────────────────────────────

# DATA_DIR points to the top-level "data/" folder where all model artifacts and
# lookup tables live.  We build the path relative to THIS file's location:
#   this file  ->  backend/models.py
#   parent     ->  backend/
#   grandparent ->  project root (SteppeDNA/)
#   + "data"   ->  SteppeDNA/data/
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Whitelist of supported genes.  Any gene name coming from user input is checked
# against this set.  This prevents path-traversal attacks where someone sends
# gene_name="../../etc/passwd" to trick us into loading arbitrary files.
SUPPORTED_GENES = {"BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D"}

# Maximum cDNA (coding DNA) lengths per gene, in nucleotide bases.
# Used to compute relative position features (e.g. "this variant is at 45 % of
# the coding sequence") so the model can learn that certain regions are more
# functionally important.
MAX_CDNA_LENGTHS = {'BRCA1': 5592, 'BRCA2': 10257, 'PALB2': 3561, 'RAD51C': 1131, 'RAD51D': 987}

# Maximum amino-acid (protein) positions per gene.
# Same idea as above but at the protein level rather than DNA level.
GENE_MAX_AA = {'BRCA1': 1863, 'BRCA2': 3418, 'PALB2': 1186, 'RAD51C': 376, 'RAD51D': 328}

# Per-gene model reliability metadata, measured on the held-out test set.
# "auc" = Area Under the ROC Curve (1.0 is perfect, 0.5 is random guessing).
# "tier" = human-friendly label shown in the UI to set user expectations.
# "note" = extra context explaining WHY performance differs across genes.
GENE_RELIABILITY = {
    "BRCA2": {"auc": 0.994, "tier": "high", "note": ""},
    "RAD51D": {"auc": 0.824, "tier": "moderate", "note": "Limited training data (82 test variants)"},
    "RAD51C": {"auc": 0.785, "tier": "moderate", "note": "Limited training data (135 test variants)"},
    "BRCA1": {"auc": 0.747, "tier": "moderate", "note": "Extreme class imbalance in ClinVar (96.6% pathogenic)"},
    "PALB2": {"auc": 0.605, "tier": "low", "note": "Smallest training set, lowest model confidence"},
}

# Legacy BRCA2 CDS (coding sequence) boundaries -- kept for backward compatibility
# with older code paths that assumed a single-gene (BRCA2-only) model.
MAX_CDNA_POS = 10257
MAX_AA_POS = 3418

# ─── Ensemble Hyperparameters ────────────────────────────────────────────────
# The ensemble prediction is:  score = XGB_WEIGHT * xgb_score + NN_WEIGHT * mlp_score
# These are the DEFAULT weights; per-gene weights (loaded below) override them.
XGB_WEIGHT = 0.6          # Default XGBoost contribution to the ensemble blend
NN_WEIGHT = 0.4           # Default MLP (neural network) contribution

# N_EFFECTIVE controls how wide our Beta-distribution confidence intervals are.
# Think of it as "how many imaginary coin flips does the model's probability
# correspond to?"  Higher = narrower CIs = more confident.
N_EFFECTIVE = 200          # Pseudo-observations for Beta CI estimation

ESM2_WINDOW = 50  # AAs on each side of variant for ESM-2 embeddings
ESM2_PCA_COMPONENTS = 20  # Reduce ESM-2 output to 20 dims via PCA

MAX_VCF_SIZE = 50 * 1024 * 1024  # 50 MB -- reject VCF uploads larger than this

# Valid amino acid codes accepted by the /predict endpoint.  Includes standard
# 3-letter amino acid codes plus special codes for frameshifts (Fs), deletions
# (Del), insertions (Ins), duplications (Dup), and stop codons (Ter / *).
VALID_AA_CODES = {'Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly', 'His', 'Ile',
                  'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser', 'Ter', 'Thr', 'Trp', 'Tyr', 'Val',
                  'Fs', 'Del', 'Ins', 'Dup', '*'}


# ─── Pickle Loading ──────────────────────────────────────────────────────────
# Pickle is Python's built-in serialization format.  We save trained models,
# scalers, and lookup tables as .pkl files during training, then load them here
# at server startup or on first prediction.

def _load_pickle(filename, required=False):
    """Safely load a pickle (.pkl) file from the data/ directory.

    This is the single gateway for all pickle loading in the project.  It adds
    two safety layers:
      1. **Path sanitization** -- strips any directory components from the
         filename so an attacker can't pass "../../etc/passwd" and read
         arbitrary files.
      2. **Graceful degradation** -- if a file is missing or empty and is NOT
         marked as required, we log a warning and return None instead of
         crashing the whole server.

    Args:
        filename: Name of the file inside data/ (e.g. "phylop_scores.pkl").
        required: If True, raise an error when the file is missing or empty.
                  If False (default), just return None.

    Returns:
        The deserialized Python object, or None if the file is missing/empty
        and required=False.
    """
    # Prevent path traversal -- os.path.basename("../../secret.pkl") → "secret.pkl"
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
    """Load a pickle that contains variant-level lookup data.

    Many of our data files (gnomAD frequencies, MAVE scores, AlphaMissense
    scores) are stored as dictionaries with two access patterns:
      - by_variant: keyed by a unique variant identifier string
                    (e.g. "BRCA2:p.Ala2345Thr")
      - by_position: keyed by amino-acid position number (e.g. 2345)

    This lets us look up data by exact variant match first, and fall back to
    positional lookup if the exact variant isn't in the database.

    Returns:
        A tuple of (by_variant_dict, by_position_dict), both empty dicts if
        the file is missing.
    """
    raw = _load_pickle(filename)
    if raw is None:
        return {}, {}
    return raw.get("by_variant", {}), raw.get("by_position", {})


# ─── Module-Level Data Loading ────────────────────────────────────────────────
# These module-level variables are legacy placeholders from the original single-
# gene (BRCA2-only) design.  They are set to None/empty here and only populated
# later via _get_universal_models() on first prediction.  The /health endpoint
# checks whether they are None to report model load status.
calibrator = None
scaler = None
feature_names = []
threshold = 0.5          # Default classification threshold (overridden once models load)
ensemble_model = None    # Will hold the MLP (NumpyMLP or Keras model)
booster = None           # Will hold the XGBoost Booster object
logger.info("ML Models will be lazy-loaded dynamically via get_gene_data()")

# --- Shared lookup tables ---
# These are loaded EAGERLY (at import time) because:
#   1. They are small enough to load quickly (won't cause deploy timeouts).
#   2. The /health endpoint reports how many entries each table has.
#   3. Every single /predict call needs them, so lazy-loading would just add
#      latency to the first request without saving anything.

# PhyloP conservation scores: higher = more conserved across species = more
# likely that a mutation here will be damaging.
phylop_scores = _load_pickle("phylop_scores.pkl") or {}

# MAVE (Multiplexed Assay of Variant Effect) scores from functional experiments
# (e.g. Findlay et al. saturation genome editing for BRCA1).
mave_by_variant, mave_by_position = _load_variant_dict("mave_scores.pkl")

# AlphaMissense scores from DeepMind's protein-language-model-based predictor.
# NOTE: These are loaded for lookup/display but NOT used as training features
# (removed in v5.4 due to label leakage concerns).
am_by_variant, am_by_position = _load_variant_dict("alphamissense_scores.pkl")

# AlphaFold-derived structural features: secondary structure, solvent
# accessibility, pLDDT confidence, etc., for each amino-acid position.
structural_features = _load_pickle("structural_features.pkl") or {}

# gnomAD population allele frequencies: how common each variant is in the
# general population.  Rare = more suspicious; common = likely benign.
gnomad_by_variant, gnomad_by_position = _load_variant_dict("gnomad_frequencies.pkl")

# Coordinate mapping between cDNA positions (e.g. c.1234) and genomic
# coordinates (e.g. chr13:32340300).  We build both directions so we can
# convert in either direction during VCF parsing or ClinVar lookups.
_cdna_genomic = _load_pickle("cdna_to_genomic.pkl") or {}
cdna_to_genomic = _cdna_genomic
genomic_to_cdna = {v: k for k, v in _cdna_genomic.items()}  # Reverse mapping
logger.info(f"[OK] Genomic mapping: {len(genomic_to_cdna)} positions")


# ─── Per-Gene Calibrators ────────────────────────────────────────────────────
# WHY per-gene calibrators?
# The universal calibrator maps raw ensemble scores → probabilities using a
# single isotonic regression curve fitted on ALL genes combined.  But different
# genes have different score distributions (e.g. BRCA2 scores cluster near 0
# and 1, while PALB2 scores are more spread out).  Per-gene calibrators fit a
# separate curve for each gene, which can produce better-calibrated probabilities.
# If a per-gene calibrator exists for a gene, we use it; otherwise we fall back
# to the universal one.

_GENE_CALIBRATORS: dict = {}  # Maps gene name (str) -> sklearn IsotonicRegression object


def _load_gene_calibrators():
    """Load per-gene isotonic calibrators from data/calibrator_<gene>.pkl files.

    Isotonic regression is a simple, non-parametric calibration method: it fits
    a monotonically increasing step function that maps raw scores to calibrated
    probabilities.  "Isotonic" just means "order-preserving" -- if variant A
    has a higher raw score than variant B, it will also have a higher calibrated
    probability.
    """
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


# ─── Gene-Adaptive Ensemble Weights ──────────────────────────────────────────
# WHY per-gene weights?
# Our ensemble blends XGBoost and MLP predictions:
#     final_score = xgb_weight * xgb_prediction + mlp_weight * mlp_prediction
#
# But each model has different strengths per gene.  For example:
#   - BRCA1: MLP alone performs best (xgb_weight=0, mlp_weight=1)
#   - RAD51C: XGBoost dominates (xgb_weight=0.8, mlp_weight=0.2)
#   - BRCA2: Balanced mix (xgb_weight=0.6, mlp_weight=0.4)
#
# These weights were optimized on the calibration set to maximize per-gene AUC.

_GENE_ENSEMBLE_WEIGHTS: dict = {}  # Maps gene name -> {"xgb_weight": float, "mlp_weight": float}


def _load_gene_ensemble_weights():
    """Load per-gene optimized ensemble weights from data/gene_ensemble_weights.json.

    The JSON file maps each gene to its optimal XGBoost/MLP blend.  If the file
    is missing, all genes default to the global 60/40 split (XGB_WEIGHT/NN_WEIGHT).

    We support two key formats in the JSON for backward compatibility:
      - {"xgb_weight": 0.6, "mlp_weight": 0.4}  (verbose format)
      - {"xgb": 0.6, "mlp": 0.4}                 (short format)
    """
    weights_path = os.path.join(DATA_DIR, "gene_ensemble_weights.json")
    if not os.path.exists(weights_path):
        logger.info("[WEIGHTS] gene_ensemble_weights.json not found -- using default 0.6/0.4 for all genes")
        return
    try:
        with open(weights_path, "r") as f:
            raw = json.load(f)
        for gene, info in raw.items():
            gene_upper = gene.upper()
            # Support both key formats: {"xgb": 0.6} and {"xgb_weight": 0.6}
            xgb_w = float(info.get("xgb_weight", info.get("xgb", XGB_WEIGHT)))
            mlp_w = float(info.get("mlp_weight", info.get("mlp", NN_WEIGHT)))
            _GENE_ENSEMBLE_WEIGHTS[gene_upper] = {"xgb_weight": xgb_w, "mlp_weight": mlp_w}
            logger.info(f"[WEIGHTS] {gene_upper}: XGB={xgb_w:.2f} / MLP={mlp_w:.2f}")
    except Exception as e:
        logger.warning(f"[WEIGHTS] Failed to load gene ensemble weights: {e}")


# ─── Bootstrap Models for Confidence Intervals ──────────────────────────────
# WHY bootstrap?
# A single model gives you a point estimate ("this variant is 73 % likely to be
# pathogenic") but doesn't tell you how uncertain that estimate is.  Bootstrap
# resampling trains N separate models, each on a slightly different random
# subsample of the training data.  When we predict with all N models, the spread
# of their predictions tells us how stable the estimate is:
#   - If all 50 models say ~73 %, we are confident.
#   - If they range from 40 % to 90 %, there is high uncertainty.
#
# We report this spread as a confidence interval (e.g. "73 % [65 %-81 %]").
# If bootstrap models aren't available, we fall back to a simpler Beta-
# distribution approximation (less accurate but always available).

_BOOTSTRAP_MODELS: list = []  # List of xgb.Booster objects, one per bootstrap resample
N_BOOTSTRAP = 50              # We train 50 bootstrap models (a common choice)


def _load_bootstrap_models():
    """Load pre-trained bootstrap XGBoost models for confidence interval estimation.

    Each model is stored as a separate JSON file (bootstrap_0.json through
    bootstrap_49.json) in data/bootstrap_models/.  These are generated offline
    by scripts/generate_bootstrap_models.py -- they are NOT created at runtime.
    """
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
            break  # Files are numbered sequentially, so a gap means we're done
    if loaded > 0:
        logger.info(f"[BOOTSTRAP] Loaded {loaded} bootstrap models for CI estimation")
    else:
        logger.info("[BOOTSTRAP] No bootstrap models found -- CIs will use Beta approximation only")


# ─── Active Learning Priorities ───────────────────────────────────────────────
# Active learning identifies which unclassified variants (VUS = Variants of
# Uncertain Significance) would be most valuable to experimentally test next.
# The idea: if the model is very uncertain about a variant AND that variant is
# clinically common, then resolving it would improve the model the most.
# This data is pre-computed offline and served via an API endpoint.

_ACTIVE_LEARNING: dict = {}  # Loaded from data/active_learning_priorities.json


def _get_active_learning() -> dict:
    """Return the active learning priorities dict.

    WHY a getter function instead of importing the variable directly?
    Because Python caches module-level variable references at import time.
    If another module does ``from models import _ACTIVE_LEARNING``, it gets
    the initial empty dict and never sees the data loaded later by
    ``_load_active_learning_priorities()``.  A getter function always returns
    the current value.
    """
    return _ACTIVE_LEARNING


def _load_active_learning_priorities():
    """Load pre-computed active learning priority rankings.

    The JSON file contains a ranked list of VUS per gene, scored by expected
    information gain (how much the model would improve if we knew the true
    label for that variant).  Generated offline by
    scripts/active_learning_ranker.py.
    """
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


# ─── Split Conformal Prediction ─────────────────────────────────────────────
# WHY conformal prediction?
# Standard ML models give a probability, but they don't come with a formal
# guarantee like "this prediction set contains the true label at least 90 % of
# the time."  Conformal prediction provides exactly that guarantee by using a
# held-out calibration set to compute score thresholds.
#
# HOW it works (simplified):
#   1. On a calibration set, compute the model's "nonconformity score" for each
#      example (how badly the model got it wrong).
#   2. Find the score at the (1-alpha) quantile (e.g. 90th percentile for
#      alpha=0.10).
#   3. At prediction time, include a label in the prediction set if the model's
#      score for that label is below the threshold.
#
# The result is a prediction SET (could be {pathogenic}, {benign}, or
# {pathogenic, benign}) with a coverage guarantee.

_CONFORMAL_THRESHOLDS: dict = {}  # Maps gene name -> {"quantile": float, "alpha": float, ...}


def _load_conformal_thresholds():
    """Load per-gene conformal prediction quantile thresholds.

    The thresholds are pre-computed offline by scripts/train_conformal.py on a
    held-out calibration set.  The JSON file also contains a "_global" key with
    the alpha level (significance level, typically 0.10 for 90 % coverage).
    """
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
# ESM-2 is Meta's protein language model.  It reads an amino acid sequence and
# produces a high-dimensional "embedding" vector for each residue that captures
# evolutionary and structural information.  We use these embeddings as features.
#
# ESM-2 is OPTIONAL -- it requires a large download (~30 MB for the small model)
# and is slow on CPU.  If not loaded, we fall back to pre-computed embeddings
# stored in per-gene pickle files.  These globals are only populated if someone
# explicitly loads the ESM-2 model (e.g. for batch re-embedding).
esm_model = None              # The ESM-2 PyTorch model (None = not loaded)
esm_batch_converter = None    # Tokenizer that converts AA sequences to model input
DEVICE = "cpu"                # PyTorch device; "cuda" would use GPU if available


# ─── Pure-Numpy MLP (replaces TensorFlow for inference) ─────────────────────
class NumpyMLP:
    """A pure-numpy reimplementation of our Keras MLP neural network.

    WHY do we need this?
    Our MLP was originally trained in TensorFlow/Keras, which produces a .h5
    model file.  But TensorFlow is HUGE (~500 MB) and causes dependency
    conflicts on some platforms.  Since we only need forward-pass inference
    (no training), we extracted the trained weights into a small pickle file
    and reimplemented the math in pure numpy.  This gives identical results
    with zero extra dependencies.

    HOW it works:
    The MLP is a simple feedforward neural network with this architecture:

        Input (120 features)
          |
        Dense(128) --> BatchNorm --> ReLU    (Layer 0)
          |
        Dense(64)  --> BatchNorm --> ReLU    (Layer 1)
          |
        Dense(32)  --> ReLU                  (Layer 2)
          |
        Dense(1)   --> Sigmoid               (Layer 3: output)
          |
        Output: single probability (0 = benign, 1 = pathogenic)

    Key concepts for beginners:
      - Dense layer: multiplies input by a weight matrix and adds a bias.
        Like y = Wx + b from algebra, but with matrices.
      - BatchNorm: normalizes the values to have mean=0 and std=1, then
        applies learnable scale (gamma) and shift (beta).  This helps the
        network train faster and more stably.
      - ReLU: activation function that replaces negative values with 0.
        f(x) = max(0, x).  Adds non-linearity so the network can learn
        complex patterns.
      - Sigmoid: squashes the final output to the range [0, 1] so it can
        be interpreted as a probability.  f(x) = 1 / (1 + e^(-x)).
    """

    def __init__(self, weights_path):
        """Load pre-extracted weight matrices from a pickle file.

        The pickle contains a dict with keys like "dense_0_kernel",
        "bn_0_gamma", etc. -- one entry per learnable parameter in the
        network.  These were extracted from the trained Keras model using
        model.get_weights().
        """
        import pickle as _pkl
        with open(weights_path, "rb") as f:
            self.w = _pkl.load(f)

    def predict(self, x, verbose=0):  # noqa: ARG002
        """Run the forward pass: input features in, probability out.

        This does exactly what Keras's model.predict() does, but using only
        numpy operations.  The ``verbose`` parameter is accepted (to match
        the Keras API signature) but ignored.

        Args:
            x: Input feature array of shape (n_samples, 120).
            verbose: Ignored.  Kept for API compatibility with Keras.

        Returns:
            Array of shape (n_samples, 1) with pathogenicity probabilities.
        """
        import numpy as _np
        w = self.w
        eps = 1e-5  # Small constant to prevent division by zero in BatchNorm
        h = _np.asarray(x, dtype=_np.float32)  # Ensure consistent dtype

        # --- Layer 0: Dense(128) + BatchNorm + ReLU ---
        # Dense: multiply input (120 features) by a 120x128 weight matrix, add bias
        h = h @ w["dense_0_kernel"] + w["dense_0_bias"]
        # BatchNorm: normalize to zero mean and unit variance using statistics
        # computed during training (moving_mean / moving_variance), then scale
        # and shift with learned parameters (gamma / beta)
        h = (h - w["bn_0_moving_mean"]) / _np.sqrt(w["bn_0_moving_variance"] + eps)
        h = h * w["bn_0_gamma"] + w["bn_0_beta"]
        # ReLU: zero out negative values (introduces non-linearity)
        h = _np.maximum(h, 0)

        # --- Layer 1: Dense(64) + BatchNorm + ReLU ---
        # Same pattern: linear transform → normalize → activate
        h = h @ w["dense_1_kernel"] + w["dense_1_bias"]
        h = (h - w["bn_1_moving_mean"]) / _np.sqrt(w["bn_1_moving_variance"] + eps)
        h = h * w["bn_1_gamma"] + w["bn_1_beta"]
        h = _np.maximum(h, 0)

        # --- Layer 2: Dense(32) + ReLU (no BatchNorm here) ---
        h = h @ w["dense_2_kernel"] + w["dense_2_bias"]
        h = _np.maximum(h, 0)

        # --- Layer 3: Dense(1) + Sigmoid (output layer) ---
        # Produces a single raw score (logit), then sigmoid squashes it to [0, 1]
        h = h @ w["dense_3_kernel"] + w["dense_3_bias"]
        h = 1.0 / (1.0 + _np.exp(-h))  # Sigmoid activation

        return h


# ─── Lazy-Loaded Universal Model Cache ───────────────────────────────────────
# _UNIVERSAL_MODELS is a dict that holds ALL the core ML artifacts needed for
# prediction.  It starts as None and is populated on the first call to
# _get_universal_models().
#
# WHY lazy loading?
# Loading XGBoost + MLP + scaler + calibrator takes several seconds and uses
# significant memory.  If we loaded everything at module import time (i.e. when
# the server starts), two bad things would happen:
#   1. Render's deploy health check has a 60-second timeout.  Slow loading could
#      cause the deploy to fail.
#   2. If someone only hits /health or /docs, we'd waste memory on models that
#      are never used.
# Lazy loading means: the first /predict request pays the loading cost, but
# subsequent requests are instant because the models are cached in this variable.
_UNIVERSAL_MODELS = None


def _get_universal_models():
    """Load the core ML model artifacts, but only on the first call.

    This is a classic "lazy singleton" pattern.  On the first call, it loads
    everything from disk and stores it in the module-level _UNIVERSAL_MODELS
    dict.  On subsequent calls, it just returns the cached dict immediately.

    The dict contains:
      - "booster":        XGBoost Booster object (gradient-boosted trees)
      - "ensemble_model": MLP neural network (NumpyMLP or Keras model)
      - "scaler":         StandardScaler that normalizes the 120 input features
      - "calibrator":     IsotonicRegression that calibrates raw scores to
                          well-calibrated probabilities
      - "feature_names":  List of 120 feature name strings (must match the
                          order used during training)
      - "threshold":      Optimal classification threshold (e.g. 0.2998) --
                          scores above this are classified as pathogenic

    Returns:
        dict with all the artifacts described above.
    """
    global _UNIVERSAL_MODELS
    if _UNIVERSAL_MODELS is not None:
        return _UNIVERSAL_MODELS  # Already loaded -- return cached version

    import xgboost as xgb

    _UNIVERSAL_MODELS = {}

    # --- Load calibrator, scaler, feature names, and threshold ---
    # These are all small pickle files that were saved during training.
    _UNIVERSAL_MODELS["calibrator"] = _load_pickle("universal_calibrator_ensemble.pkl")
    _UNIVERSAL_MODELS["scaler"] = _load_pickle("universal_scaler_ensemble.pkl")
    _UNIVERSAL_MODELS["feature_names"] = _load_pickle("universal_feature_names.pkl")
    _UNIVERSAL_MODELS["threshold"] = _load_pickle("universal_threshold_ensemble.pkl") or 0.5

    # --- Load the MLP (neural network) ---
    # We try three strategies in order of preference:
    #   1. Pure-numpy weights (fastest, no dependencies beyond numpy)
    #   2. TensorFlow/Keras .h5 file (works but requires TF installed)
    #   3. Give up on MLP and run XGBoost-only (still works, slightly worse)
    _UNIVERSAL_MODELS["ensemble_model"] = None
    numpy_weights_path = os.path.join(DATA_DIR, "mlp_weights_numpy.pkl")
    if os.path.exists(numpy_weights_path):
        try:
            _UNIVERSAL_MODELS["ensemble_model"] = NumpyMLP(numpy_weights_path)
            logger.info("[MLP] Loaded pure-numpy MLP (no TensorFlow required)")
        except Exception as e:
            logger.warning(f"[WARN] Numpy MLP failed: {e}. Trying TensorFlow...")
    if _UNIVERSAL_MODELS["ensemble_model"] is None:
        try:
            from tensorflow.keras.models import load_model
            h5_path = os.path.join(DATA_DIR, "universal_nn.h5")
            _UNIVERSAL_MODELS["ensemble_model"] = load_model(h5_path) if os.path.exists(h5_path) else None
            if _UNIVERSAL_MODELS["ensemble_model"]:
                logger.info("[MLP] Loaded TensorFlow MLP")
        except Exception as e:
            logger.warning(f"[WARN] Could not load MLP model: {e}. Using XGBoost-only mode.")

    # --- Load the XGBoost model ---
    # XGBoost stores its model as a JSON file containing all the decision trees.
    # We create an empty Booster object and then load the trained trees into it.
    xgb_path = f"{DATA_DIR}/universal_xgboost_final.json"
    if os.path.exists(xgb_path):
        booster = xgb.Booster()
        booster.load_model(xgb_path)
        _UNIVERSAL_MODELS["booster"] = booster
    else:
        _UNIVERSAL_MODELS["booster"] = None

    return _UNIVERSAL_MODELS
