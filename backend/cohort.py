"""
SteppeDNA - Cohort, Metrics, History, and Research Endpoints

Handles:
- /umap visualization
- /metrics server monitoring
- /cohort/submit and /cohort/stats
- /history and /stats
- /research/priorities active learning
"""

import os
import re
import sys
import csv
import json
import time
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from backend.models import (
    DATA_DIR, SUPPORTED_GENES,
    _get_active_learning,
)
from backend.external_api import _api_cache
from backend.middleware import _rate_counts, _API_KEY
# Database storage disabled — privacy: no variant data is stored server-side
# from backend.database import get_recent_analyses, get_analysis_stats

logger = logging.getLogger("steppedna")

router = APIRouter()

# Maximum cohort CSV file size (10 MB) to prevent unbounded disk growth
MAX_COHORT_FILE_SIZE = 10 * 1024 * 1024


# ─── UMAP ────────────────────────────────────────────────────────────────────

@router.get("/umap", tags=["Visualization"], summary="UMAP variant landscape",
            description="Returns precomputed UMAP 2D coordinates for up to 5,000 training variants, used by the frontend variant landscape visualization.")
async def get_umap():
    umap_path = os.path.join(DATA_DIR, "umap_coordinates.json")
    if os.path.exists(umap_path):
        with open(umap_path) as f:
            return JSONResponse(content=json.load(f))
    return JSONResponse(status_code=404, content={"error": "UMAP coordinates not precomputed. Run: python scripts/precompute_umap.py"})


# ─── Server Monitoring ────────────────────────────────────────────────────────

@router.get("/metrics", tags=["System"], summary="Server metrics",
            description="Returns server uptime, request counts, latency stats, and memory usage for monitoring dashboards.")
async def server_metrics():
    # Import lazily to avoid circular dependency
    from backend import main as _main_mod

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

    with _main_mod._metrics_lock:
        predictions = _main_mod._metrics["predictions"]
        vcf_uploads = _main_mod._metrics["vcf_uploads"]
        total_predict_ms = _main_mod._metrics["total_predict_ms"]

    return {
        "uptime_seconds": round(time.monotonic() - _main_mod._server_start_time, 1),
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


# ─── Patient Cohort Tracking ─────────────────────────────────────────────────

@router.post("/cohort/submit", tags=["Cohort"], summary="Submit anonymized variant for cohort tracking",
             description="Allows hospitals/clinics to submit anonymized variant observations for population-level tracking. No patient identifiers are stored.")
async def cohort_submit(request: Request):
    # Require API key for write operations
    if _API_KEY and request.headers.get("X-API-Key") != _API_KEY:
        return JSONResponse(status_code=401, content={"error": "API key required for cohort submission"})

    # Import lazily to avoid circular dependency
    from backend import main as _main_mod

    body = await request.json()

    # Validate required fields
    required = ["gene", "aa_ref", "aa_pos", "aa_alt", "prediction", "probability"]
    for field in required:
        if field not in body:
            return JSONResponse(status_code=400, content={"error": f"Missing required field: {field}"})

    gene = str(body["gene"]).upper()
    if gene not in SUPPORTED_GENES:
        return JSONResponse(status_code=400, content={"error": f"Unsupported gene: {gene}"})

    # Sanitize free-text fields
    def _sanitize_text(value: str, max_length: int = 100) -> str:
        """Strip special characters and limit length for free-text cohort fields."""
        cleaned = re.sub(r'[^\w\s\-\.]', '', str(value))
        return cleaned.strip()[:max_length]

    # Store in a simple append-only CSV (anonymized - no patient IDs)
    cohort_file = os.path.join(DATA_DIR, "cohort_observations.csv")

    # Prevent unbounded disk growth
    if os.path.exists(cohort_file) and os.path.getsize(cohort_file) > MAX_COHORT_FILE_SIZE:
        return JSONResponse(status_code=507, content={"error": "Cohort storage full"})

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

    with _main_mod._file_lock:
        write_header = not os.path.exists(cohort_file) or os.path.getsize(cohort_file) == 0
        with open(cohort_file, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    logger.info(f"[COHORT] {row['gene']} {row['variant']} from {row['institution']}")
    return {"status": "recorded", "variant": row["variant"], "gene": row["gene"]}


@router.get("/cohort/stats", tags=["Cohort"], summary="Cohort aggregate statistics",
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


# ─── History & Stats ─────────────────────────────────────────────────────────

@router.get("/history", tags=["System"], summary="Recent analysis history",
            description="Server-side history disabled for privacy. Use client-side localStorage history.")
async def analysis_history(limit: int = 50):
    return {"analyses": [], "note": "Server-side history disabled for privacy."}


@router.get("/stats", tags=["System"], summary="Analysis statistics",
            description="Server-side stats disabled for privacy.")
async def analysis_stats():
    return {"total_analyses": 0, "total_vcf_uploads": 0, "by_gene": {}, "by_prediction": {}, "avg_latency_ms": 0, "note": "Server-side stats disabled for privacy."}


# ─── Research Priorities / Active Learning ────────────────────────────────────

@router.get("/research/priorities", tags=["Research"], summary="Active learning priority variants",
            description="Returns VUS variants ranked by their value for functional validation, "
            "computed via query-by-committee (model disagreement), gene scarcity weighting, "
            "and positional novelty scoring.")
async def research_priorities(gene: str = None, limit: int = 10):
    # Validate gene first (even if no data loaded)
    if gene:
        gene_upper = gene.upper()
        if gene_upper not in SUPPORTED_GENES:
            return JSONResponse(status_code=400, content={"error": f"Unsupported gene: {gene}. Supported: {', '.join(sorted(SUPPORTED_GENES))}"})
    al = _get_active_learning()
    if not al:
        return {"error": "Active learning priorities not available. Run scripts/active_learning_ranker.py to generate."}
    priorities = al.get("priorities", {})
    metadata = al.get("metadata", {})
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
        "gene_training_sizes": al.get("gene_training_sizes", {}),
    }
