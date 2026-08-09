const CACHE = "dajoong-shell-v3";
const SCOPE = self.registration.scope;
const scoped = (path) => new URL(path, SCOPE).pathname;
const SHELL = [
  scoped("./"),
  scoped("./studio"),
  scoped("./manifest.webmanifest"),
  scoped("./brand/dajoong-logo-mark-512.png"),
  scoped("./sample/source.png"),
  scoped("./sample/03-plan-graph.json"),
];

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
  event.respondWith(
    caches.match(request).then((cached) => cached || fetch(request).then((response) => {
      const copy = response.clone();
      caches.open(CACHE).then((cache) => cache.put(request, copy));
      return response;
    }))
  );
});
