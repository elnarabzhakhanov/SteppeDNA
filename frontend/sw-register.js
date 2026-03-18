/* Service Worker Registration with Update Detection */
"use strict";
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('sw.js').then(function(reg) {
        reg.addEventListener('updatefound', function() {
            var newWorker = reg.installing;
            newWorker.addEventListener('statechange', function() {
                if (newWorker.state === 'activated' && navigator.serviceWorker.controller) {
                    // Show update toast to user
                    var toast = document.createElement('div');
                    toast.className = 'sw-update-toast';
                    toast.setAttribute('role', 'alert');
                    toast.innerHTML = '<span>Update available</span><button onclick="window.location.reload()">Refresh</button><button onclick="this.parentElement.remove()" aria-label="Dismiss">&times;</button>';
                    toast.style.cssText = 'position:fixed;bottom:20px;right:20px;background:#1a1a2e;color:#fff;padding:12px 18px;border-radius:8px;z-index:10000;display:flex;align-items:center;gap:10px;box-shadow:0 4px 12px rgba(0,0,0,0.3);font-family:inherit;';
                    var btns = toast.querySelectorAll('button');
                    btns[0].style.cssText = 'background:#4cc9f0;color:#000;border:none;padding:6px 14px;border-radius:4px;cursor:pointer;font-weight:600;';
                    btns[1].style.cssText = 'background:none;color:#888;border:none;font-size:18px;cursor:pointer;padding:0 4px;';
                    document.body.appendChild(toast);
                }
            });
        });
    }).catch(function(err) {
        console.warn('[SW] Registration failed:', err);
    });
}
