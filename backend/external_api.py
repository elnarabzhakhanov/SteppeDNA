"""
SteppeDNA - External API Lookups (ClinVar & gnomAD)

Real-time cross-referencing against external databases.
These are optional enrichment endpoints -- the core prediction works offline.
"""

import os
import re
import time
import logging
import asyncio
import collections

import httpx
import urllib.parse
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from backend.models import SUPPORTED_GENES
from backend.middleware import _rate_lock, RATE_WINDOW

logger = logging.getLogger("steppedna")

router = APIRouter(tags=["External Lookups"])

# ─── Per-IP Rate Limiter for External API Proxy ──────────────────────────────
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


# ─── ClinVar Lookup ──────────────────────────────────────────────────────────

@router.get("/lookup/clinvar/{variant}", summary="ClinVar lookup",
            description="Look up a variant in NCBI ClinVar. Accepts format: p.Thr2722Arg or Thr2722Arg. Returns clinical significance and review status.")
async def lookup_clinvar(variant: str, request: Request):
    cache_key = f"clinvar_{variant}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # Rate-limit external API calls separately
    # Use rightmost XFF IP — the one added by the trusted proxy (Render/Vercel)
    ip = request.headers.get("X-Forwarded-For", "").split(",")[-1].strip() or (request.client.host if request.client else "unknown")
    if not _check_external_rate(ip):
        return JSONResponse(status_code=429, content={"error": "Too many external lookups. Please wait before trying again."})

    t_start = time.perf_counter()
    clean = variant.replace("p.", "").strip()

    # Validate variant format to prevent query injection (e.g. Thr2722Arg)
    if not re.match(r'^[A-Za-z]{2,4}\d{1,5}[A-Za-z]{2,4}$', clean):
        return {"variant": clean, "error": "Invalid variant format. Expected e.g. Thr2722Arg"}

    gene_param = request.query_params.get("gene")
    if gene_param is None:
        logger.warning("[ClinVar] No gene parameter provided for variant %s, defaulting to BRCA2. "
                       "Pass ?gene=GENENAME for accurate results.", clean)
        gene_upper = "BRCA2"
    else:
        gene_upper = gene_param.upper()
    if gene_upper not in SUPPORTED_GENES:
        logger.warning("[ClinVar] Unsupported gene '%s' for variant %s, defaulting to BRCA2.",
                       gene_upper, clean)
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
                        await asyncio.sleep(2)
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


# ─── gnomAD Lookup ────────────────────────────────────────────────────────────

@router.get("/lookup/gnomad/{variant}", summary="gnomAD lookup",
            description="Look up a variant in gnomAD v4. Accepts format: 13-32316461-A-G (chr-pos-ref-alt). Returns genome/exome allele frequencies and homozygote counts.")
async def lookup_gnomad(variant: str, request: Request):  # noqa: C901
    cache_key = f"gnomad_{variant}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # Use rightmost XFF IP — the one added by the trusted proxy (Render/Vercel)
    ip = request.headers.get("X-Forwarded-For", "").split(",")[-1].strip() or (request.client.host if request.client else "unknown")
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

    # Validate parts to prevent injection
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
                        await asyncio.sleep(2)
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
            exome = v.get("exome") or {}
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
