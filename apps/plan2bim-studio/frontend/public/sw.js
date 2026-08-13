const CACHE = "dajoong-shell-v8";
const SCOPE = self.registration.scope;
const scoped = (path) => new URL(path, SCOPE).pathname;
const SHELL = [
  scoped("./"),
  scoped("./studio"),
  scoped("./manifest.webmanifest"),
  scoped("./brand/dajoong-logo-mark.svg"),
];
const NETWORK_FIRST = new Set([
  scoped("./sample/sample-manifest.json"),
]);

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((key) => key !== CACHE).map((key) => caches.delete(key))))
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  const url = new URL(request.url);
  if (request.method !== "GET" || url.origin !== self.location.origin || url.pathname.includes("/api/")) return;
  if (request.mode === "navigate") {
    event.respondWith(fetch(request).catch(() => caches.match(scoped("./")).then((response) => response || Response.error())));
    return;
  }
  if (NETWORK_FIRST.has(url.pathname)) {
    event.respondWith(
      fetch(request).then((response) => {
        if (!response.ok) return response;
        const copy = response.clone();
        caches.open(CACHE).then((cache) => cache.put(request, copy));
        return response;
      }).catch(() => caches.match(request).then((response) => response || Response.error()))
    );
    return;
  }
  event.respondWith(
    caches.match(request).then((cached) => cached || fetch(request).then((response) => {
      if (!response.ok) return response;
      const copy = response.clone();
      caches.open(CACHE).then((cache) => cache.put(request, copy));
      return response;
    }))
  );
});
