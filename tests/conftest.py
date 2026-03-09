"""
Shared fixtures for the SteppeDNA pytest suite.

Run from project root:
  pytest tests/ -v

The TestClient fixture spins up the FastAPI app in-process — no live server needed.
"""

import os
import sys
import pytest

# Make sure the project root is on the path so backend.* imports work
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def client():
    """In-process FastAPI test client (loads model once per test session)."""
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Clear the rate limiter between tests so the suite doesn't hit 429s."""
    from backend.middleware import _rate_counts, _rate_lock
    with _rate_lock:
        _rate_counts.clear()


# ─── Known test variants ──────────────────────────────────────────────────────
# These are real BRCA2 variants with clinically established classification.

KNOWN_PATHOGENIC = {
    "cDNA_pos": 6174,
    "AA_ref":   "Ser",
    "AA_alt":   "Ter",
    "Mutation": "C>A",  # nonsense variant (Ser→Ter)
    "AA_pos":   2058,
}

# Safe single-amino-acid missense known to be benign (common polymorphism)
KNOWN_BENIGN = {
    "cDNA_pos": 1114,
    "AA_ref":   "Asn",
    "AA_alt":   "Ser",
    "Mutation": "A>G",
    "AA_pos":   372,
}

# A random missense for SHAP stability testing
DETERMINISTIC_VARIANT = {
    "cDNA_pos": 8165,
    "AA_ref":   "Thr",
    "AA_alt":   "Arg",
    "Mutation": "A>G",
    "AA_pos":   2722,
}


# ─── Multi-gene known variants ──────────────────────────────────────────────

# BRCA2 variants (reuse existing KNOWN_PATHOGENIC and KNOWN_BENIGN)

KNOWN_BRCA1_PATHOGENIC = {
    "gene_name": "BRCA1",
    "cDNA_pos": 181,
    "AA_ref": "Cys",
    "AA_alt": "Ter",
    "Mutation": "C>A",
    "AA_pos": 61,
}

KNOWN_BRCA1_BENIGN = {
    "gene_name": "BRCA1",
    "cDNA_pos": 2612,
    "AA_ref": "Pro",
    "AA_alt": "Leu",
    "Mutation": "C>T",
    "AA_pos": 871,
}

KNOWN_PALB2_PATHOGENIC = {
    "gene_name": "PALB2",
    "cDNA_pos": 3113,
    "AA_ref": "Gln",
    "AA_alt": "Ter",
    "Mutation": "C>T",
    "AA_pos": 1038,
}

KNOWN_PALB2_BENIGN = {
    "gene_name": "PALB2",
    "cDNA_pos": 1676,
    "AA_ref": "Gln",
    "AA_alt": "Arg",
    "Mutation": "A>G",
    "AA_pos": 559,
}

KNOWN_RAD51C_MISSENSE = {
    "gene_name": "RAD51C",
    "cDNA_pos": 376,
    "AA_ref": "Leu",
    "AA_alt": "Phe",
    "Mutation": "C>T",
    "AA_pos": 126,
}

KNOWN_RAD51D_MISSENSE = {
    "gene_name": "RAD51D",
    "cDNA_pos": 271,
    "AA_ref": "Gly",
    "AA_alt": "Arg",
    "Mutation": "G>A",
    "AA_pos": 91,
}
