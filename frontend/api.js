/* SteppeDNA — API Configuration
 * ================================
 * Single source of truth for all backend URLs.
 * Override: set window.STEPPEDNA_API_BASE before this script loads,
 * or pass ?api=https://your-backend.onrender.com in the URL.
 */
"use strict";

const API_BASE = (() => {
    // 1. Explicit override via global variable
    if (window.STEPPEDNA_API_BASE) return window.STEPPEDNA_API_BASE;
    // 2. URL parameter override (e.g. ?api=https://steppedna-api.onrender.com)
    const params = new URLSearchParams(window.location.search);
    if (params.get('api')) return params.get('api');
    // 3. Local development
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || window.location.protocol === 'file:')
        return 'http://127.0.0.1:8000';
    // 4. Same-origin production
    return window.location.origin;
})();

const API_URL = API_BASE + '/predict';
const VCF_API = API_BASE + '/predict/vcf';
const CLINVAR_API = API_BASE + '/lookup/clinvar/';
const GNOMAD_API = API_BASE + '/lookup/gnomad/';
const METRICS_API = API_BASE + '/model_metrics';
const UMAP_API = API_BASE + '/umap';

/* ─── Backend Health Check & Offline Banner ─────────────────────────── */
let _backendOnline = false;
const _isLocalDev = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || window.location.protocol === 'file:';

async function checkBackendHealth() {
    const banner = document.getElementById('offlineBanner');
    const text   = document.getElementById('offlineText');
    if (!banner) return;
    try {
        const resp = await fetch(API_BASE + '/health', { signal: AbortSignal.timeout(5000) });
        if (resp.ok) {
            const wasOffline = !_backendOnline;
            _backendOnline = true;
            // Only briefly show "connected" if recovering from offline state
            if (wasOffline && banner.style.display === 'flex') {
                banner.className = 'offline-banner connected';
                text.textContent = _isLocalDev ? 'Backend connected' : 'Connected';
                setTimeout(() => { banner.style.display = 'none'; }, 1500);
            } else {
                // First load, backend is fine — show nothing
                banner.style.display = 'none';
            }
            return;
        }
        throw new Error('Not OK');
    } catch (_e) {
        _backendOnline = false;
        banner.className = 'offline-banner';
        if (_isLocalDev) {
            text.innerHTML = 'Server offline &mdash; start the backend to enable predictions.';
        } else {
            text.innerHTML = 'Server is starting up &mdash; predictions will be available shortly.';
        }
        banner.style.display = 'flex';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    checkBackendHealth();
    // Retry every 15 s while offline (faster recovery for cold starts)
    setInterval(() => { if (!_backendOnline) checkBackendHealth(); }, 15000);
});
