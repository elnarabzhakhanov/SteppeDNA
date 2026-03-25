"""
SteppeDNA Middleware Stack
==========================

This module registers 5 middleware layers that wrap EVERY incoming HTTP request
to the SteppeDNA API. Think of middleware as a series of security checkpoints
that each request must pass through before reaching the actual endpoint handler.

The 5 layers (in the order they EXECUTE, not the order they are registered):

    Request arrives
      |
      v
    1. Security Headers   -- adds protective HTTP headers to every response
    2. Request ID         -- stamps each request with a unique ID for log tracing
    3. Rate Limiter       -- blocks IPs that send too many requests (anti-abuse)
    4. API Key Check      -- authenticates private endpoints via X-API-Key header
    5. Body Size Limit    -- rejects oversized payloads (anti-DoS)
      |
      v
    Endpoint handler runs, response travels back through the stack

IMPORTANT: FastAPI/Starlette middleware executes in REVERSE order of registration.
The LAST middleware registered with @app.middleware("http") is the FIRST to run.
That is why security_headers_middleware is registered last but executes first --
it wraps everything so headers appear on ALL responses, even error responses
from other middleware.

Deployment context:
  - Backend runs on Render (steppedna-api.onrender.com)
  - Frontend runs on Vercel (steppedna.vercel.app), which proxies API calls
  - Requests arrive through Render's reverse proxy, so the real client IP
    is in the X-Forwarded-For header, not in request.client.host
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
# How many requests a single IP can make within the time window.
# Defaults to 60 requests per 60 seconds. Override via environment variables
# (e.g., set RATE_LIMIT=9999 in tests to effectively disable rate limiting).
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "60"))
RATE_WINDOW = int(os.getenv("RATE_WINDOW", "60"))  # Window size in seconds

# Thread lock to prevent race conditions when multiple requests update the
# rate-limit dictionary at the same time. FastAPI is async, but the rate
# counter is shared mutable state, so we need synchronization.
_rate_lock = threading.Lock()

# Dictionary mapping IP address -> list of request timestamps.
# This is the core data structure for the "sliding window" algorithm:
# each IP has a list of the timestamps of its recent requests, and we
# evict entries older than RATE_WINDOW seconds on every new request.
_rate_counts: dict = collections.defaultdict(list)

# ─── API Key ─────────────────────────────────────────────────────────────────
# Optional server-side secret. When set (via STEPPEDNA_API_KEY env var),
# private endpoints require clients to send this value in the X-API-Key header.
# When NOT set, all endpoints are public -- this is useful for local development.
_API_KEY = os.getenv("STEPPEDNA_API_KEY")

# ─── Body Size Limit ─────────────────────────────────────────────────────────
# Maximum allowed size for JSON request bodies (1 MB). This prevents attackers
# from sending huge payloads that could exhaust server memory (a type of
# Denial-of-Service attack). VCF file uploads are exempt because they can
# legitimately be larger than 1 MB.
MAX_JSON_BODY = 1 * 1024 * 1024  # 1 MB = 1,048,576 bytes


def register_middleware(app):  # noqa: C901
    """
    Register all 5 middleware layers on the FastAPI app.

    Middleware is registered inside-out: the FIRST one registered here
    (body_size_limit) is the LAST to execute on an incoming request,
    and the LAST one registered (security_headers) is the FIRST to execute.

    Execution order for an incoming request:
        security_headers -> request_id -> rate_limit -> api_key -> body_size_limit -> endpoint
    """

    # ── Layer 5 (innermost): Body Size Limit ────────────────────────────────
    # WHY: Prevents Denial-of-Service (DoS) attacks where an attacker sends
    #       a massive JSON payload to exhaust server memory or CPU.
    # HOW: Checks the Content-Length header BEFORE reading the body. If it
    #       exceeds 1 MB, immediately returns HTTP 413 (Payload Too Large).
    # NOTE: VCF file uploads (/predict/vcf) are exempted because they use
    #       multipart form encoding and can legitimately be several MB.
    @app.middleware("http")
    async def body_size_limit(request: Request, call_next):
        """Reject oversized request bodies (excludes VCF multipart uploads)."""
        if request.url.path != "/predict/vcf":
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > MAX_JSON_BODY:
                return JSONResponse(status_code=413, content={"error": "Request body too large"})
        # If the body is within limits (or it's a VCF upload), pass to the next layer.
        return await call_next(request)

    # ── Layer 4: API Key Authentication ────────────────────────────────────
    # WHY: Protects sensitive endpoints (like /predict) from unauthorized use.
    #       Without this, anyone could call our prediction API and consume
    #       server resources. Public endpoints (health checks, docs, stats)
    #       are exempted so the frontend and monitoring tools can access them.
    # HOW: If STEPPEDNA_API_KEY is set in the environment, the middleware
    #       checks each request's X-API-Key header against it. If the key
    #       doesn't match, the request is rejected with HTTP 401 (Unauthorized).
    #       If STEPPEDNA_API_KEY is NOT set, all endpoints are public (dev mode).
    @app.middleware("http")
    async def api_key_middleware(request: Request, call_next):
        if _API_KEY:
            # These paths are intentionally public -- no API key needed.
            # Includes the frontend (/), health checks, API docs, and read-only
            # informational endpoints that the frontend needs without auth.
            # NOTE: /model_metrics is intentionally public (no API key required)
            # to allow monitoring dashboards. This is a deliberate access control decision.
            # If metrics should be private, remove them from this set.
            public_paths = {"/", "/health", "/docs", "/openapi.json", "/redoc",
                            "/model_metrics", "/umap", "/stats",
                            "/history", "/research/priorities",
                            "/docs/validation-report"}
            if request.url.path in public_paths:
                return await call_next(request)
            # Also allow read-only GET requests to lookup, structure, and cohort
            # endpoints without a key -- these are used by the frontend UI.
            if request.method == "GET" and (
                request.url.path.startswith("/lookup/")
                or request.url.path.startswith("/structure/")
                or request.url.path.startswith("/cohort/stats")
            ):
                return await call_next(request)
            # Everything else (especially POST /predict) requires a valid API key.
            if request.headers.get("X-API-Key") != _API_KEY:
                return JSONResponse(status_code=401, content={"error": "Invalid or missing API key"})
        # If _API_KEY is None (not configured), skip auth entirely (dev mode).
        return await call_next(request)

    # ── Layer 3: Rate Limiter (Sliding Window Algorithm) ───────────────────
    # WHY: Prevents abuse and brute-force attacks. Without this, a single
    #       client could flood the server with thousands of requests per second,
    #       starving other users and potentially crashing the service.
    # HOW: Uses a "sliding window" approach:
    #       1. For each IP, we keep a list of timestamps of recent requests.
    #       2. On every new request, we remove ("evict") timestamps older than
    #          RATE_WINDOW seconds (default 60s).
    #       3. If the remaining count >= RATE_LIMIT (default 60), reject with
    #          HTTP 429 (Too Many Requests).
    #       4. Otherwise, record the current timestamp and let the request through.
    #       This is called "sliding window" because the 60-second window slides
    #       forward in real time, as opposed to a "fixed window" that resets at
    #       fixed intervals (which would allow burst abuse at window boundaries).
    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        # --- Step 1: Determine the client's real IP address ---
        # When behind a reverse proxy (Render/Vercel), the client's real IP
        # is in the X-Forwarded-For header. We take the RIGHTMOST value because
        # that's the one added by our trusted proxy -- earlier values can be
        # spoofed by the client.
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[-1].strip()
        else:
            # Direct connection (no proxy) -- use the socket-level IP.
            ip = request.client.host if request.client else "unknown"

        # time.monotonic() gives a clock that never goes backward (unlike
        # time.time() which can jump due to NTP adjustments).
        now = time.monotonic()

        # --- Step 2: Check and update the sliding window (thread-safe) ---
        with _rate_lock:
            # Evict timestamps older than the window. After this, only
            # requests from the last RATE_WINDOW seconds remain in the list.
            _rate_counts[ip] = [t for t in _rate_counts[ip] if now - t < RATE_WINDOW]

            # If this IP already used up all its allowed requests, reject.
            if len(_rate_counts[ip]) >= RATE_LIMIT:
                return JSONResponse(
                    status_code=429,
                    content={"error": f"Rate limit exceeded: max {RATE_LIMIT} requests per {RATE_WINDOW}s"},
                    # Retry-After tells the client how long to wait before retrying.
                    headers={"Retry-After": str(RATE_WINDOW)},
                )

            # Record this request's timestamp for future rate-limit checks.
            _rate_counts[ip].append(now)

            # --- Step 3: Memory cleanup ---
            # If we're tracking more than 200 distinct IPs, remove stale entries
            # (IPs that haven't made any requests in 2x the window). This prevents
            # the dictionary from growing unboundedly in a long-running server.
            if len(_rate_counts) > 200:
                stale = [k for k, v in _rate_counts.items()
                         if not v or (now - max(v)) > RATE_WINDOW * 2]
                for k in stale:
                    del _rate_counts[k]

        return await call_next(request)

    # ── Layer 2: Request ID ────────────────────────────────────────────────
    # WHY: When something goes wrong, we need to trace a specific request
    #       through logs. A unique ID lets us correlate a user's error report
    #       ("I got error X with request ID abc12345") with the exact server
    #       log entry. Without this, debugging production issues is very hard.
    # HOW: Generates a short (8-character) random UUID for each request,
    #       stores it on request.state (so endpoint handlers can access it),
    #       and also returns it in the X-Request-ID response header.
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        """Attach a unique request ID to each request for tracing."""
        request_id = str(uuid.uuid4())[:8]  # e.g., "a3f1b2c4"
        request.state.request_id = request_id  # Available to endpoint handlers
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id  # Visible to the client
        return response

    # ── Layer 1 (outermost): Security Headers ──────────────────────────────
    # WHY: These HTTP headers tell browsers to enforce security policies that
    #       prevent common web attacks (XSS, clickjacking, MIME sniffing, etc.).
    #       This layer runs FIRST (outermost) so that even error responses from
    #       other middleware layers still get these protective headers.
    @app.middleware("http")
    async def security_headers_middleware(request: Request, call_next):
        """Add security headers to every response."""
        response = await call_next(request)

        # -- X-Content-Type-Options: nosniff --
        # Prevents browsers from "guessing" a file's type (MIME sniffing).
        # Without this, a browser might treat a text file as JavaScript and
        # execute it, which is a common XSS attack vector.
        response.headers["X-Content-Type-Options"] = "nosniff"

        # -- X-Frame-Options: DENY --
        # Prevents any website from embedding our site in an <iframe>.
        # This blocks "clickjacking" attacks where a malicious site overlays
        # invisible iframes to trick users into clicking hidden buttons.
        response.headers["X-Frame-Options"] = "DENY"

        # -- Referrer-Policy: strict-origin-when-cross-origin --
        # Controls how much URL information is sent in the Referer header when
        # navigating to other sites. "strict-origin-when-cross-origin" sends
        # the full URL for same-origin requests but only the origin (domain)
        # for cross-origin requests, preventing URL path leakage.
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # -- Permissions-Policy --
        # Explicitly disables browser features we don't need (camera, mic, GPS).
        # Even if an XSS attacker injects code, they can't access these APIs.
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"

        # -- Strict-Transport-Security (HSTS) --
        # Tells browsers to ONLY connect via HTTPS for the next year (31536000s).
        # After a user visits once, their browser will refuse plain HTTP
        # connections, preventing man-in-the-middle downgrade attacks.
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # -- Content-Security-Policy (CSP) --
        # The most powerful security header. It tells the browser exactly which
        # sources of content (scripts, styles, images, etc.) are allowed to load.
        # If an attacker injects a <script> tag pointing to evil.com, the browser
        # will block it because evil.com is not in our allow-list.
        #
        # Breakdown of each directive:
        #   default-src 'self'        -> By default, only load resources from our own domain.
        #   script-src 'self' cdn...  -> JavaScript can ONLY come from our domain,
        #                                cdn.jsdelivr.net, or unpkg.com (used for
        #                                frontend libraries like Chart.js, UMAP, etc.).
        #   style-src 'self' 'unsafe-inline' -> CSS from our domain + inline <style> tags.
        #                                       'unsafe-inline' is needed because many JS
        #                                       libraries inject inline styles dynamically.
        #   img-src 'self' data: blob: -> Images from our domain + data: URIs (base64
        #                                encoded images) + blob: URIs (dynamically
        #                                generated images, e.g., UMAP plots).
        #   connect-src 'self'        -> AJAX/fetch requests can ONLY go to our own domain.
        #                                This prevents stolen data from being exfiltrated
        #                                to an attacker's server via JavaScript.
        #   frame-src 'none'          -> No iframes allowed (reinforces X-Frame-Options).
        #   object-src 'none'         -> Blocks <object>, <embed>, <applet> tags (legacy
        #                                plugin attack vectors like Flash).
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' https://cdn.jsdelivr.net https://unpkg.com; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "connect-src 'self'; "
            "frame-src 'none'; "
            "object-src 'none'"
        )
        return response
