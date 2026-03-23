/* Unregister any existing service worker and clear caches */
"use strict";
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.getRegistrations().then(function(regs) {
        regs.forEach(function(reg) { reg.unregister(); });
    });
    if ('caches' in window) {
        caches.keys().then(function(keys) {
            keys.forEach(function(k) { caches.delete(k); });
        });
    }
}
