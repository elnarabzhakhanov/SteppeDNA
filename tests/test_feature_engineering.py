"""
Unit tests for the build_feature_vector() function in backend/main.py.

Verifies that:
- Feature vector has exactly the right shape (1, N_FEATURES)
- Domain flags are set correctly for known positions
- Boundary positions behave correctly
- Physicochemical features have expected sign/magnitude
"""

import os
import sys
import pytest
import numpy as np
import pickle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
from backend.main import build_feature_vector

# Load feature names directly from the serialized configuration.
# This prevents Pytest from instantiating Keras/XGBoost C-extensions 
# into memory during the module discovery phase, which triggers segfaults.
feature_names_path = os.path.join(ROOT, "data", "universal_feature_names.pkl")
with open(feature_names_path, "rb") as f:
    feature_names = pickle.load(f)

N_FEATURES = len(feature_names)


def _vec(cdna, aa_ref, aa_alt, mutation, aa_pos):
    """Helper: return the raw numpy array from build_feature_vector."""
    return build_feature_vector(cdna, aa_ref, aa_alt, mutation, aa_pos)


def _feat(cdna, aa_ref, aa_alt, mutation, aa_pos, name):
    """Helper: return value of a single named feature."""
    vec = _vec(cdna, aa_ref, aa_alt, mutation, aa_pos)
    idx = feature_names.index(name)
    return vec[0, idx]


# ─── Shape ───────────────────────────────────────────────────────────────────

def test_feature_vector_shape():
    """Output must be exactly (1, N_FEATURES)."""
    vec = _vec(8165, "Thr", "Arg", "c.8165A>G", 2722)
    assert vec.shape == (1, N_FEATURES), (
        f"Expected (1, {N_FEATURES}), got {vec.shape}"
    )


def test_feature_count_matches_scaler():
    """Feature names list length must match the saved production scaler's expected input."""
    import pickle
    # Try ensemble scaler first (current production), fall back to legacy
    scaler_path = os.path.join(ROOT, "data", "universal_scaler_ensemble.pkl")
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(ROOT, "data", "scaler.pkl")
    if not os.path.exists(scaler_path):
        pytest.skip("No scaler found — run training script first")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    assert scaler.n_features_in_ == N_FEATURES


# ─── Domain Flags ─────────────────────────────────────────────────────────────

def test_brc_repeat_flag_on():
    """AA position 1500 is inside BRC repeats (1009-2083)."""
    assert _feat(4500, "Ala", "Val", "c.4500C>T", 1500, "in_critical_repeat_region") == 1


def test_brc_repeat_flag_off():
    """AA position 100 is outside BRC repeats."""
    assert _feat(300, "Ala", "Val", "c.300C>T", 100, "in_critical_repeat_region") == 0


def test_dna_binding_flag_on():
    """AA position 2700 is inside DNA binding (2402-3190)."""
    assert _feat(8100, "Ala", "Val", "c.8100C>T", 2700, "in_DNA_binding") == 1


def test_dna_binding_flag_off():
    """AA position 100 is outside DNA binding."""
    assert _feat(300, "Ala", "Val", "c.300C>T", 100, "in_DNA_binding") == 0


def test_palb2_bind_flag_on():
    """AA position 25 is inside PALB2 binding region (10-40)."""
    assert _feat(75, "Ala", "Val", "c.75C>T", 25, "in_primary_interaction") == 1


# ─── Relative Position ────────────────────────────────────────────────────────

def test_relative_cdna_pos_first():
    """cDNA pos 1 → relative_cdna_pos ≈ 1/10257 ≈ very small."""
    val = _feat(1, "Met", "Val", "A>G", 1, "relative_cdna_pos")
    assert val == pytest.approx(1 / 10257, abs=1e-6)


def test_relative_aa_pos_last():
    """AA pos 3418 (C-terminus) → relative_aa_pos = 1.0."""
    val = _feat(10257, "Lys", "Arg", "A>G", 3418, "relative_aa_pos")
    assert val == pytest.approx(1.0, abs=1e-6)


# ─── Physicochemical Features ─────────────────────────────────────────────────

def test_nonsense_flag():
    """Terminator alt AA sets is_nonsense = 1."""
    val = _feat(6174, "Ser", "Ter", "C>A", 2058, "is_nonsense")
    assert val == 1


def test_missense_not_nonsense():
    """Standard missense sets is_nonsense = 0."""
    val = _feat(8165, "Thr", "Arg", "A>G", 2722, "is_nonsense")
    assert val == 0


def test_blosum62_score_is_finite():
    """BLOSUM62 score should always be a finite number."""
    vec = _vec(8165, "Thr", "Arg", "A>G", 2722)
    idx = feature_names.index("blosum62_score")
    assert np.isfinite(vec[0, idx])


def test_volume_diff_nonnegative():
    """Volume difference is always >= 0 (absolute value)."""
    vec = _vec(8165, "Thr", "Arg", "A>G", 2722)
    idx = feature_names.index("volume_diff")
    assert vec[0, idx] >= 0


def test_hydro_diff_nonnegative():
    """Hydrophobicity difference is always >= 0."""
    vec = _vec(8165, "Thr", "Arg", "A>G", 2722)
    idx = feature_names.index("hydro_diff")
    assert vec[0, idx] >= 0


# ─── All Features Are Finite ─────────────────────────────────────────────────

def test_all_features_finite():
    """No feature value should be NaN or infinite."""
    vec = _vec(8165, "Thr", "Arg", "A>G", 2722)
    assert np.all(np.isfinite(vec)), (
        f"Non-finite values at indices: {np.where(~np.isfinite(vec))[1].tolist()}"
    )


def test_all_features_finite_boundary_low():
    """Boundary cDNA position 1 should produce all-finite features."""
    vec = _vec(1, "Met", "Val", "A>G", 1)
    assert np.all(np.isfinite(vec))


def test_all_features_finite_boundary_high():
    """Boundary cDNA position 10257 should produce all-finite features."""
    vec = _vec(10257, "Lys", "Arg", "A>G", 3418)
    assert np.all(np.isfinite(vec))
