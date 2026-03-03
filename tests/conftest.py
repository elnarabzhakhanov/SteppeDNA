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
