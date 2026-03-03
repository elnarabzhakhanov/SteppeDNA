#!/usr/bin/env python3
"""Generate SHA256 checksums for all model files and save to data/checksums.json.

Usage:
    python scripts/generate_checksums.py

Run this after retraining models or updating data files to keep checksums current.
"""
import hashlib
import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

MODEL_FILES = [
    "universal_calibrator_ensemble.pkl",
    "universal_scaler_ensemble.pkl",
    "universal_feature_names.pkl",
    "universal_threshold_ensemble.pkl",
    "universal_ensemble_final.pkl",
    "universal_xgboost_final.json",
    "universal_nn.h5",
]


def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    checksums = {}
    for filename in MODEL_FILES:
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  SKIP  {filename} (not found)")
            continue
        digest = sha256_file(filepath)
        checksums[filename] = digest
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  OK    {filename} ({size_mb:.1f} MB) -> {digest[:16]}...")

    out_path = os.path.join(DATA_DIR, "checksums.json")
    with open(out_path, "w") as f:
        json.dump(checksums, f, indent=2)
    print(f"\nSaved {len(checksums)} checksums to {out_path}")


if __name__ == "__main__":
    main()
