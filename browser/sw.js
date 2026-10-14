// Cache only public install assets. Private pages and Python state stay outside
// this service-worker cache; the normal network handles app/runtime updates.
const scope = self.registration.scope;
const cacheName = `dailydigest-pages-install:${scope}:v1`;
const assets = ['manifest.webmanifest', 'app-icon.svg'].map(path => new URL(path, scope).href);
self.addEventListener('install', event => {
  event.waitUntil(caches.open(cacheName).then(cache => cache.addAll(assets)));
  self.skipWaiting();
});
self.addEventListener('activate', event => event.waitUntil(self.clients.claim()));
self.addEventListener('fetch', event => {
  if (event.request.method !== 'GET' || !assets.includes(event.request.url)) return;
  event.respondWith(caches.open(cacheName).then(async cache => {
    try {
      const response = await fetch(event.request);
      if (response.ok) await cache.put(event.request, response.clone());
      return response;
    } catch (error) {
      const saved = await cache.match(event.request);
      if (saved) return saved;
      throw error;
    }
  }));
});
