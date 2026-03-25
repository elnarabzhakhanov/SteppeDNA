"""
SteppeDNA - External API Lookups (ClinVar & gnomAD)

This module provides real-time lookups against two major external genetics databases:

  1. ClinVar (NCBI) — the gold-standard public archive of clinically interpreted
     genetic variants. We query it to show users whether a variant has already been
     reviewed by clinical labs and what classification it received (e.g. "Pathogenic",
     "Benign", "Uncertain significance").

  2. gnomAD (Genome Aggregation Database) — a massive collection of sequencing data
     from ~800k individuals. We query it to retrieve population allele frequencies,
     which tell us how common a variant is in the general population. Very rare
     variants are more likely to be disease-causing; very common ones are almost
     certainly benign.

IMPORTANT: These are *enrichment* endpoints only. SteppeDNA's core ML prediction
works entirely offline using pre-trained models. These lookups simply give the user
extra context ("ClinVar already classifies this as Pathogenic", "this variant is
seen in 1 in 10,000 people in gnomAD", etc.).

Architecture — Two Separate Caches:
  - Prediction cache (_pred_cache): Short TTL (30 min), small (500 entries).
    Caches the results of /predict so that re-submitting the same variant doesn't
    re-run the ML model. Short TTL because model results don't change often, but
    we still want to pick up any hot-reload changes relatively quickly.

  - API cache (_api_cache): Longer TTL (1 hour), larger (1000 entries).
    Caches ClinVar/gnomAD responses. Longer TTL because external database content
    changes very rarely (maybe weekly), and we want to minimize calls to external
    services to stay within rate limits and reduce latency.

  Both caches use an LRU (Least Recently Used) eviction strategy built on Python's
  OrderedDict: when the cache is full, the oldest (least-recently-accessed) entry
  is removed first.

Security:
  - Input validation prevents query injection into NCBI/gnomAD queries
  - Per-IP rate limiting prevents abuse of these endpoints as a free proxy
  - X-Forwarded-For header is used to identify the real client IP behind Render/Vercel
"""

import os
import re
import time
import logging
import asyncio
import collections

import httpx                              # Async HTTP client (like requests, but async-friendly)
import urllib.parse                       # URL encoding for safe query string construction
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from backend.models import SUPPORTED_GENES   # The 5 HR genes: BRCA1, BRCA2, PALB2, RAD51C, RAD51D
from backend.middleware import _rate_lock, RATE_WINDOW  # Thread lock + window size for rate limiting

logger = logging.getLogger("steppedna")

# FastAPI router — all endpoints defined here get mounted under the main app
router = APIRouter(tags=["External Lookups"])

# ─── Per-IP Rate Limiter for External API Proxy ──────────────────────────────
# WHY: Without rate limiting, someone could use our /lookup/* endpoints as a free
# proxy to hammer ClinVar/gnomAD, which would get our server IP blocked by those
# services. This limits each user (by IP) to a fixed number of lookups per time window.
EXTERNAL_RATE_LIMIT = int(os.getenv("EXTERNAL_RATE_LIMIT", "20"))  # max requests per window
# Maps each IP address -> list of timestamps of their recent requests
_external_rate: dict = collections.defaultdict(list)


def _check_external_rate(ip: str) -> bool:
    """Returns True if request is allowed, False if rate-limited.

    HOW it works (sliding window algorithm):
      1. Get the current time
      2. Remove all timestamps older than the rate window (e.g. 60 seconds)
      3. If the remaining count >= limit, reject the request (return False)
      4. Otherwise, record this request's timestamp and allow it (return True)

    The _rate_lock mutex ensures thread safety — multiple concurrent requests
    from the same IP won't cause a race condition in the counter.
    """
    now = time.monotonic()  # monotonic() never goes backwards, unlike time.time()
    with _rate_lock:
        # Keep only timestamps within the current sliding window
        _external_rate[ip] = [t for t in _external_rate[ip] if now - t < RATE_WINDOW]
        if len(_external_rate[ip]) >= EXTERNAL_RATE_LIMIT:
            return False  # Over the limit — reject
        _external_rate[ip].append(now)  # Record this request
    return True  # Under the limit — allow


# ─── Separate Caches ──────────────────────────────────────────────────────────
# WHY two caches? They serve different purposes and have different freshness needs:
#
# 1. Prediction cache: Stores ML model outputs for recently-queried variants.
#    Short TTL (30 min) because if we hot-reload a model, we want predictions to
#    update relatively quickly. Small size (500) since each prediction is cheap-ish
#    to recompute if evicted.
#
# 2. API cache: Stores ClinVar/gnomAD responses. Longer TTL (1 hour) because
#    external database content changes very slowly (new ClinVar submissions are
#    reviewed weekly at best). Larger size (1000) since external calls are slow
#    (network latency) and we want to avoid redundant calls.
#
# Both use OrderedDict as an LRU cache: when full, the least-recently-used entry
# is evicted (popitem(last=False) removes the oldest item). When an entry is
# accessed, move_to_end() moves it to the back of the queue so it's evicted last.

# --- Prediction cache (short TTL, keyed by variant string) ---
_pred_cache = collections.OrderedDict()  # key -> (timestamp, result)
MAX_PRED_CACHE = 500
PRED_CACHE_TTL = 1800  # 30 minutes

# --- External API cache (longer TTL, keyed by "clinvar_{variant}" or "gnomad_{variant}") ---
_api_cache = collections.OrderedDict()   # key -> (timestamp, result)
MAX_CACHE_SIZE = 1000
CACHE_TTL = 3600  # 1 hour


def _pred_cache_set(key, value):
    """Store a prediction result in the short-TTL prediction cache.

    If the cache is full, evict the oldest entry first (LRU eviction).
    Each entry is stored as a tuple of (timestamp, value) so we can check expiry later.
    """
    while len(_pred_cache) >= MAX_PRED_CACHE:
        _pred_cache.popitem(last=False)  # Remove the oldest (least recently used) entry
    _pred_cache[key] = (time.time(), value)


def _pred_cache_get(key):
    """Retrieve a prediction from the cache, or None if missing/expired.

    If found and still fresh (within TTL), move it to the end of the OrderedDict
    so it won't be evicted soon (this is the "recently used" part of LRU).
    If found but expired, delete it and return None.
    """
    if key in _pred_cache:
        cached_time, cached_result = _pred_cache[key]
        if time.time() - cached_time < PRED_CACHE_TTL:
            _pred_cache.move_to_end(key)  # Mark as recently used
            return cached_result
        else:
            del _pred_cache[key]  # Expired — remove stale entry
    return None


def _cache_set(key, value):
    """Store an external API response in the long-TTL API cache.

    Same LRU eviction logic as the prediction cache, just with a bigger capacity
    and longer TTL since external API data changes very infrequently.
    """
    while len(_api_cache) >= MAX_CACHE_SIZE:
        _api_cache.popitem(last=False)  # Evict oldest entries when full (LRU)
    _api_cache[key] = (time.time(), value)


def _cache_get(key):
    """Return cached API response if present and not expired, else None.

    Same pattern as _pred_cache_get: check existence -> check TTL -> move to end if fresh.
    """
    if key in _api_cache:
        cached_time, cached_result = _api_cache[key]
        if time.time() - cached_time < CACHE_TTL:
            _api_cache.move_to_end(key)  # Mark as recently used so it's evicted last
            return cached_result
        else:
            del _api_cache[key]  # Expired — discard
    return None


# ─── ClinVar Lookup ──────────────────────────────────────────────────────────
#
# ClinVar is NCBI's public archive of human genetic variants and their clinical
# interpretations. Labs and researchers submit variant classifications like
# "Pathogenic", "Likely Benign", "Uncertain Significance (VUS)", etc.
#
# HOW the lookup works (two-step process using NCBI's E-utilities REST API):
#   Step 1 — esearch: Search ClinVar by gene name + variant name.
#            Returns a list of ClinVar record IDs (e.g. ["12345", "67890"]).
#   Step 2 — esummary: Fetch the detailed summary for the first matching ID.
#            Returns the clinical significance, review status, title, etc.
#
# WHY two API calls? NCBI's E-utilities are designed as a search-then-fetch
# pipeline. You can't get full record details in a single search call.

@router.get("/lookup/clinvar/{variant}", summary="ClinVar lookup",
            description="Look up a variant in NCBI ClinVar. Accepts format: p.Thr2722Arg or Thr2722Arg. Returns clinical significance and review status.")
async def lookup_clinvar(variant: str, request: Request):
    # --- Step 0: Check the cache first to avoid unnecessary API calls ---
    cache_key = f"clinvar_{variant}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached  # Cache hit — return immediately without calling NCBI

    # --- Step 0b: Rate limiting ---
    # Extract the real client IP from the X-Forwarded-For header.
    # We take the RIGHTMOST entry because that's the one added by our trusted
    # reverse proxy (Render/Vercel). Earlier entries could be spoofed by the client.
    ip = request.headers.get("X-Forwarded-For", "").split(",")[-1].strip() or (request.client.host if request.client else "unknown")
    if not _check_external_rate(ip):
        return JSONResponse(status_code=429, content={"error": "Too many external lookups. Please wait before trying again."})

    t_start = time.perf_counter()  # Start timing for latency logging

    # Strip the "p." prefix if present (protein notation like "p.Thr2722Arg" -> "Thr2722Arg")
    clean = variant.replace("p.", "").strip()

    # --- Input validation ---
    # The regex ensures the variant looks like AminoAcid + Position + AminoAcid
    # (e.g. "Thr2722Arg"). This prevents query injection — without this check,
    # a malicious user could craft a variant string that manipulates the NCBI search query.
    if not re.match(r'^[A-Za-z]{2,4}\d{1,5}[A-Za-z]{2,4}$', clean):
        return {"variant": clean, "error": "Invalid variant format. Expected e.g. Thr2722Arg"}

    # Gene is required so we can narrow the ClinVar search to the correct gene
    gene_param = request.query_params.get("gene")
    if gene_param is None:
        return {"variant": clean, "error": "Missing required parameter: gene. Pass ?gene=BRCA1|BRCA2|PALB2|RAD51C|RAD51D"}
    gene_upper = gene_param.upper()
    if gene_upper not in SUPPORTED_GENES:
        return {"variant": clean, "error": f"Unsupported gene '{gene_param}'. Supported: BRCA1, BRCA2, PALB2, RAD51C, RAD51D"}

    # --- Step 1: Build the NCBI E-utilities search query ---
    # This uses NCBI's field-tagged search syntax:
    #   - BRCA2[gene]                    → restrict to gene
    #   - "Thr2722Arg"[variant name]     → match the variant name
    #   - "homo sapiens"[organism]       → restrict to human variants
    query = f'{gene_upper}[gene] AND "{clean}"[variant name] AND "homo sapiens"[organism]'
    encoded = urllib.parse.urlencode({
        "db": "clinvar",     # Search the ClinVar database
        "term": query,       # The search query we built above
        "retmax": 5,         # Return at most 5 matching IDs
        "retmode": "json"    # Return results as JSON (not XML)
    })
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{encoded}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Simple retry logic: try once, and if it fails, wait 2 seconds and retry.
            # This handles transient network blips or brief NCBI outages gracefully.
            for _attempt in range(2):
                try:
                    resp = await client.get(url, headers={"User-Agent": "SteppeDNA/1.0"})
                    resp.raise_for_status()  # Raise exception if HTTP status >= 400
                    break
                except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException):
                    if _attempt == 0:
                        await asyncio.sleep(2)  # Wait before retry
                    else:
                        raise  # Second attempt failed — give up
            data = resp.json()

            # Extract the list of ClinVar record IDs from the search results
            ids = data.get("esearchresult", {}).get("idlist", [])
            if not ids:
                # No matches found — cache this "not found" result too, so we don't
                # keep querying NCBI for the same non-existent variant
                res = {"variant": clean, "clinvar": "Not found", "ids": []}
                _cache_set(cache_key, res)
                return res

            # --- Step 2: Fetch the detailed summary for the first (best) match ---
            # esummary returns rich metadata: clinical significance, review stars, etc.
            summary_url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                f"?db=clinvar&id={ids[0]}&retmode=json"
            )
            resp2 = await client.get(summary_url, headers={"User-Agent": "SteppeDNA/1.0"})
            resp2.raise_for_status()
            sdata = resp2.json()

            # Extract the record for our specific ClinVar ID
            rec = sdata.get("result", {}).get(ids[0], {})
            latency_ms = (time.perf_counter() - t_start) * 1000
            logger.info(f"[ClinVar] {clean}: {len(ids)} hits ({latency_ms:.0f}ms)")

            # Build a clean response with the fields the frontend needs
            res = {
                "variant": clean,
                "clinvar_id": ids[0],                      # ClinVar accession number
                "clinical_significance": rec.get(          # e.g. "Pathogenic", "Benign", "VUS"
                    "clinical_significance", {}
                ).get("description", "Unknown"),
                "title": rec.get("title", ""),  # Human-readable variant title
                "review_status": rec.get(  # e.g. "criteria provided, multiple submitters"
                    "clinical_significance", {}  # (more stars = more confidence)
                ).get("review_status", ""),
                "n_results": len(ids),  # How many ClinVar entries matched
            }
            _cache_set(cache_key, res)  # Cache for future lookups
            return res
    except Exception as e:
        # Don't crash the server if NCBI is down — just return a friendly error
        logger.warning(f"[ClinVar] Lookup failed for {clean}: {type(e).__name__}: {e}")
        return {"variant": clean, "error": "ClinVar lookup failed. Please try again later."}


# ─── gnomAD Lookup ────────────────────────────────────────────────────────────
#
# gnomAD (Genome Aggregation Database) contains allele frequency data from
# ~800,000 sequenced individuals. It answers the question: "How common is this
# variant in the general population?"
#
# WHY this matters for pathogenicity prediction:
#   - If a variant is very common (e.g. seen in 5% of people), it's almost
#     certainly benign — a disease-causing variant can't be that frequent.
#   - If a variant is extremely rare or absent from gnomAD, it *could* be
#     pathogenic (but rarity alone doesn't prove pathogenicity).
#   - ACMG guidelines use specific AF thresholds (BA1, BS1) to classify variants.
#
# KEY POPULATION GENETICS TERMS returned by this endpoint:
#   - AC (Allele Count): How many times this variant allele was observed.
#   - AN (Allele Number): Total number of alleles examined at this position.
#     (Each person has 2 alleles, so AN ~ 2 * number_of_people_sequenced)
#   - AF (Allele Frequency): AC / AN — the fraction of alleles carrying this variant.
#     Example: AF=0.001 means 1 in 1000 alleles carry this variant.
#   - Homozygote Count: Number of individuals who have the variant on BOTH copies
#     of the chromosome. Relevant because homozygous pathogenic variants cause
#     recessive disease.
#
# gnomAD separates data by sequencing technology:
#   - Genome: Whole-genome sequencing (covers all DNA, fewer samples)
#   - Exome:  Exome sequencing (covers only protein-coding regions, more samples)
#
# HOW: gnomAD uses a GraphQL API (not REST). We send a structured query that
# specifies exactly which fields we want, and gnomAD returns only those fields.
# This is more efficient than REST since we don't get unnecessary data.

@router.get("/lookup/gnomad/{variant}", summary="gnomAD lookup",
            description="Look up a variant in gnomAD v4. Accepts format: 13-32316461-A-G (chr-pos-ref-alt). Returns genome/exome allele frequencies and homozygote counts.")
async def lookup_gnomad(variant: str, request: Request):  # noqa: C901
    # --- Check cache first ---
    cache_key = f"gnomad_{variant}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached  # Cache hit — skip the network call

    # --- Rate limiting (same logic as ClinVar above) ---
    ip = request.headers.get("X-Forwarded-For", "").split(",")[-1].strip() or (request.client.host if request.client else "unknown")
    if not _check_external_rate(ip):
        return JSONResponse(status_code=429, content={"error": "Too many external lookups. Please wait before trying again."})

    t_start = time.perf_counter()

    # gnomAD requires genomic coordinates (chr-pos-ref-alt), not protein notation.
    # If the user passed something like "Thr2722Arg", tell them to use /predict first
    # which can return the genomic coordinates for their variant.
    if not variant[0].isdigit():
        return {
            "variant": variant,
            "note": "Provide genomic coordinates: chr-pos-ref-alt (e.g. 13-32316461-A-G)",
            "tip": "Use /predict first to get the genomic coordinates for your variant.",
        }

    # Parse the "chr-pos-ref-alt" format (e.g. "13-32316461-A-G")
    parts = variant.split("-")
    if len(parts) != 4:
        return {"error": "Expected format: chr-pos-ref-alt (e.g. 13-32316461-A-G)"}

    chrom, pos, ref, alt = parts

    # --- Input validation to prevent GraphQL injection ---
    # Each component is validated against strict patterns:
    #   chrom: 1-22, X, Y, or MT (human chromosomes only)
    #   pos:   digits only (genomic position)
    #   ref/alt: DNA bases only (A, C, G, T)
    if not re.match(r'^([1-9]|1[0-9]|2[0-2]|X|Y|MT?)$', chrom, re.IGNORECASE):
        return {"error": "Invalid chromosome in variant"}
    if not re.match(r'^\d+$', pos):
        return {"error": "Invalid position in variant"}
    if not re.match(r'^[ACGT]+$', ref):
        return {"error": "Invalid ref allele in variant"}
    if not re.match(r'^[ACGT]+$', alt):
        return {"error": "Invalid alt allele in variant"}

    # Reconstruct the validated variant ID (safe to embed in the query now)
    variant_id = f"{chrom}-{pos}-{ref}-{alt}"

    # --- GraphQL query for gnomAD v4 ---
    # GraphQL is a query language where the client specifies exactly which fields
    # it wants. The structure below says: "For this variant in gnomAD release 4,
    # give me the AC/AN/AF/homozygote_count from both genome and exome datasets."
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
        # gnomAD can be slow, so we allow a slightly longer timeout (15s vs 10s for ClinVar)
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Same retry pattern as ClinVar: try once, retry once after 2s delay
            for _attempt in range(2):
                try:
                    # GraphQL uses POST (not GET) with the query in the JSON body.
                    # The $variantId variable is passed separately in "variables" —
                    # this is GraphQL's built-in parameterization (like SQL prepared statements).
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

            # GraphQL can return HTTP 200 but still include errors in the response body
            # (unlike REST where errors are signaled via HTTP status codes)
            if "errors" in data:
                logger.warning(f"[gnomAD] GraphQL errors for {variant}: {data['errors']}")
                res = {"variant": variant, "error": "gnomAD returned an error. Try again later."}
                _cache_set(cache_key, res)
                return res

            # Extract the variant data from the GraphQL response structure
            v = data.get("data", {}).get("variant")
            latency_ms = (time.perf_counter() - t_start) * 1000
            logger.info(f"[gnomAD] {variant}: {'found' if v else 'not found'} ({latency_ms:.0f}ms)")

            if not v:
                # Variant not in gnomAD — this is common for very rare variants.
                # Cache this "not found" result too, to avoid re-querying.
                res = {"variant": variant, "gnomad": "Not found in gnomAD v4"}
                _cache_set(cache_key, res)
                return res

            # Extract genome (whole-genome sequencing) and exome data separately.
            # Either can be None if gnomAD only has data from one sequencing type.
            genome = v.get("genome") or {}
            exome = v.get("exome") or {}

            # Build a flat response dict with all the population frequency data.
            # The frontend uses these to display allele frequency badges and to
            # feed into ACMG criteria (e.g. BA1 = AF > 5% = almost certainly benign).
            res = {
                "variant": variant,
                "variant_id": v.get("variant_id"),     # gnomAD's canonical variant ID
                "genome_af": genome.get("af"),          # Allele frequency in genomes
                "genome_ac": genome.get("ac"),          # Allele count in genomes
                "genome_an": genome.get("an"),          # Total alleles examined in genomes
                "genome_hom": genome.get("homozygote_count"),  # Homozygous individuals (genomes)
                "exome_af": exome.get("af"),            # Allele frequency in exomes
                "exome_ac": exome.get("ac"),            # Allele count in exomes
                "exome_an": exome.get("an"),            # Total alleles examined in exomes
                "exome_hom": exome.get("homozygote_count"),    # Homozygous individuals (exomes)
            }
            _cache_set(cache_key, res)  # Cache for future lookups
            return res
    except Exception as e:
        # Graceful degradation — if gnomAD is down, the prediction still works
        logger.warning(f"[gnomAD] Lookup failed for {variant}: {type(e).__name__}: {e}")
        return {"variant": variant, "error": "gnomAD lookup failed. Please try again later."}
