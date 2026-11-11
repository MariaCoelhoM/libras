// Service worker do Libras: guarda só a "casca" do app para ele abrir rápido.
// O reconhecimento (/stream, /predict, /predict_palavra, /frase) sempre vai ao servidor.
const VERSAO = "libras-v1";
const CASCA = [
  "/",
  "/manifest.webmanifest",
  "/static/icons/icon-192.png",
  "/static/icons/icon-512.png",
  "/static/icons/apple-touch-icon.png",
];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(VERSAO).then((c) => c.addAll(CASCA)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((chaves) => Promise.all(chaves.filter((k) => k !== VERSAO).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;                 // POSTs vão direto ao servidor
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;  // VLibras e outros sites: sem cache

  if (req.mode === "navigate") {
    // página: rede primeiro (pega atualizações); sem rede, usa a cópia guardada
    e.respondWith(
      fetch(req)
        .then((resp) => {
          const copia = resp.clone();
          caches.open(VERSAO).then((c) => c.put("/", copia));
          return resp;
        })
        .catch(() => caches.match("/"))
    );
    return;
  }

  if (url.pathname.startsWith("/static/") || url.pathname === "/manifest.webmanifest") {
    e.respondWith(caches.match(req).then((guardado) => guardado || fetch(req)));
  }
});
