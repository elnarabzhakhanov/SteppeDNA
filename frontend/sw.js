/* SteppeDNA Service Worker — Offline-first caching */
const CACHE_NAME = 'steppedna-v5.4.1';
const STATIC_ASSETS = [
    '/',
    '/index.html',
    '/styles.css',
    '/api.js',
    '/lang.js',
    '/app.js',
    '/sw-register.js',
    '/manifest.json',
];

// Content types safe to cache dynamically
const CACHEABLE_TYPES = ['text/html', 'text/css', 'application/javascript',
                         'image/', 'font/', 'application/json'];

// Install: cache static shell
self.addEventListener('install', (e) => {
    e.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => cache.addAll(STATIC_ASSETS))
            .then(() => self.skipWaiting())
    );
});

// Activate: clean up old caches
self.addEventListener('activate', (e) => {
    e.waitUntil(
        caches.keys().then(keys =>
            Promise.all(keys
                .filter(k => k !== CACHE_NAME)
                .map(k => caches.delete(k))
            )
        ).then(() => self.clients.claim())
    );
});

// Fetch: network-first for API, cache-first for static assets
self.addEventListener('fetch', (e) => {
    const url = new URL(e.request.url);

    // API calls: always go to network (don't cache predictions)
    if (url.pathname.startsWith('/predict') ||
        url.pathname.startsWith('/lookup') ||
        url.pathname.startsWith('/health') ||
        url.pathname.startsWith('/model_metrics') ||
        url.pathname.startsWith('/umap') ||
        url.pathname.startsWith('/cohort') ||
        url.pathname.startsWith('/metrics') ||
        url.pathname.startsWith('/history') ||
        url.pathname.startsWith('/stats')) {
        e.respondWith(fetch(e.request));
        return;
    }

    // Static assets: cache-first, fall back to network
    e.respondWith(
        caches.match(e.request).then(cached => {
            if (cached) return cached;
            return fetch(e.request).then(resp => {
                // Only cache responses with safe content types
                if (resp.ok && e.request.method === 'GET') {
                    const ct = resp.headers.get('Content-Type') || '';
                    if (CACHEABLE_TYPES.some(t => ct.includes(t))) {
                        const clone = resp.clone();
                        caches.open(CACHE_NAME).then(c => c.put(e.request, clone));
                    }
                }
                return resp;
            });
        }).catch(() => {
            // Offline fallback for navigation
            if (e.request.mode === 'navigate') {
                return caches.match('/index.html');
            }
        })
    );
});
