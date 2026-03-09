"""
SteppeDNA Middleware

Rate limiting, API key enforcement, request ID generation,
body size limits, and security headers.
"""

import os
import time
import uuid
import collections
import threading
import logging

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("steppedna")

# ─── Rate Limiter State ─────────────────────────────────────────────────────
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "60"))
RATE_WINDOW = int(os.getenv("RATE_WINDOW", "60"))

_rate_lock = threading.Lock()
_rate_counts: dict = collections.defaultdict(list)

# ─── API Key ─────────────────────────────────────────────────────────────────
_API_KEY = os.getenv("STEPPEDNA_API_KEY")
if os.getenv("ENVIRONMENT", "").lower() == "production" and not _API_KEY:
    raise RuntimeError("STEPPEDNA_API_KEY is required when ENVIRONMENT=production")

# ─── Body Size Limit ─────────────────────────────────────────────────────────
MAX_JSON_BODY = 1 * 1024 * 1024  # 1 MB limit for JSON endpoints


def register_middleware(app):
    """Register all middleware on the FastAPI app."""

    @app.middleware("http")
    async def body_size_limit(request: Request, call_next):
        """Reject oversized request bodies (excludes VCF multipart uploads)."""
        if request.url.path != "/predict/vcf":
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > MAX_JSON_BODY:
                return JSONResponse(status_code=413, content={"error": "Request body too large"})
        return await call_next(request)

    @app.middleware("http")
    async def api_key_middleware(request: Request, call_next):
        if _API_KEY:
            # Public paths accessible without API key
            public_paths = {"/", "/health", "/docs", "/openapi.json", "/redoc",
                            "/model_metrics", "/umap", "/stats",
                            "/history", "/research/priorities",
                            "/docs/validation-report"}
            if request.url.path in public_paths:
                return await call_next(request)
            # Allow GET requests to lookup/structure/cohort-stats without key
            if request.method == "GET" and (
                request.url.path.startswith("/lookup/")
                or request.url.path.startswith("/structure/")
                or request.url.path.startswith("/cohort/stats")
            ):
                return await call_next(request)
            # All other endpoints require API key
            if request.headers.get("X-API-Key") != _API_KEY:
                return JSONResponse(status_code=401, content={"error": "Invalid or missing API key"})
        return await call_next(request)

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Use rightmost IP — the one added by the trusted proxy (Render/Vercel)
            ip = forwarded.split(",")[-1].strip()
        else:
            ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with _rate_lock:
            _rate_counts[ip] = [t for t in _rate_counts[ip] if now - t < RATE_WINDOW]
            if len(_rate_counts[ip]) >= RATE_LIMIT:
                return JSONResponse(
                    status_code=429,
                    content={"error": f"Rate limit exceeded: max {RATE_LIMIT} requests per {RATE_WINDOW}s"},
                    headers={"Retry-After": str(RATE_WINDOW)},
                )
            _rate_counts[ip].append(now)
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
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response
