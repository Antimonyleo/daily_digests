// Cache only public install assets. Private pages and Python state stay outside
// this service-worker cache; the normal network handles app/runtime updates.
const scope = self.registration.scope;
const isolate = new URL(self.location.href).searchParams.get('isolate') === '1';
const cacheName = `dailydigest-pages-install:${scope}:v1`;
const assets = ['manifest.webmanifest', 'app-icon.svg'].map(path => new URL(path, scope).href);
self.addEventListener('install', event => {
  event.waitUntil(caches.open(cacheName).then(cache => cache.addAll(assets)));
  self.skipWaiting();
});
self.addEventListener('activate', event => event.waitUntil(self.clients.claim()));
self.addEventListener('fetch', event => {
  // Isolation is enabled only by an explicit Compute choice. Keep the handler
  // inside this project and leave provider requests to normal browser CORS.
  if (isolate && event.request.method === 'GET' && event.request.url.startsWith(scope)) {
    event.respondWith(fetch(event.request).then(response => {
      if (response.type === 'opaque' || response.status === 0) return response;
      const headers = new Headers(response.headers);
      headers.set('Cross-Origin-Opener-Policy', 'same-origin');
      headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
      return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
    }));
    return;
  }
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
