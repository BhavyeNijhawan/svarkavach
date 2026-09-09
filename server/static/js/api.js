/* Thin wrapper over the Flask API. Every call returns parsed JSON or throws
 * an Error carrying the server's message, so views can show something useful
 * instead of a blank panel. */

async function req(url, opts = {}) {
  const res = await fetch(url, opts);
  const ct = res.headers.get("content-type") || "";
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    if (ct.includes("application/json")) {
      try { msg = (await res.json()).error || msg; } catch { /* keep status */ }
    }
    throw new Error(msg);
  }
  return ct.includes("application/json") ? res.json() : res.text();
}

export const api = {
  health: () => req("/api/health"),

  corpus: () => req("/api/corpus"),
  call: (id) => req(`/api/call/${encodeURIComponent(id)}`),
  audioUrl: (id, codec, snr) => {
    const q = new URLSearchParams();
    if (codec && codec !== "clean") q.set("codec", codec);
    if (snr) q.set("snr", snr);
    const s = q.toString();
    return `/api/audio/${encodeURIComponent(id)}${s ? "?" + s : ""}`;
  },

  analyze: (body) => req("/api/analyze", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  }),

  simulate: (body) => req("/api/simulate", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  }),

  upload: (file, transcript) => {
    const fd = new FormData();
    fd.append("audio", file);
    if (transcript) fd.append("transcript", transcript);
    return req("/api/upload", { method: "POST", body: fd });
  },

  waveform: (id, codec, snr) => {
    const q = new URLSearchParams({ codec: codec || "clean" });
    if (snr) q.set("snr", snr);
    return req(`/api/waveform/${encodeURIComponent(id)}?${q}`);
  },

  spectrogram: (id, codec, snr) => {
    const q = new URLSearchParams({ codec: codec || "clean" });
    if (snr) q.set("snr", snr);
    return req(`/api/spectrogram/${encodeURIComponent(id)}?${q}`);
  },

  results: () => req("/api/results"),
  result: (name) => req(`/api/results/${encodeURIComponent(name)}`),

  meta: () => req("/api/meta"),

  /** Server-sent events for the streaming simulation.
   *  Returns a handle with .close(); callbacks fire per event. */
  stream(id, { codec, snr, speed }, handlers = {}) {
    const q = new URLSearchParams({ codec: codec || "clean", speed: String(speed || 1) });
    if (snr) q.set("snr", snr);
    const es = new EventSource(`/api/stream/${encodeURIComponent(id)}?${q}`);
    es.addEventListener("meta", (e) => handlers.onMeta?.(JSON.parse(e.data)));
    es.addEventListener("turn", (e) => handlers.onTurn?.(JSON.parse(e.data)));
    es.addEventListener("alert", (e) => handlers.onAlert?.(JSON.parse(e.data)));
    es.addEventListener("done", (e) => { handlers.onDone?.(JSON.parse(e.data)); es.close(); });
    es.addEventListener("failed", (e) => { handlers.onError?.(JSON.parse(e.data)); es.close(); });
    es.onerror = () => { handlers.onError?.({ error: "stream interrupted" }); es.close(); };
    return es;
  },
};

export default api;
