/* SteppeDNA: Self-destructing service worker.
   Clears all caches and unregisters itself so updates are always instant. */
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => {
    e.waitUntil(
        caches.keys()
            .then(keys => Promise.all(keys.map(k => caches.delete(k))))
            .then(() => self.clients.claim())
            .then(() => self.registration.unregister())
    );
});
