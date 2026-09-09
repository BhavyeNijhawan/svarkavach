/* Router, theme handling and boot for the SwarKavach console. */

import api from "./api.js";
import { redrawAll, clearResponsive } from "./charts.js";
import monitor from "./views/monitor.js";
import analysis from "./views/analysis.js";
import results from "./views/results.js";
import corpus from "./views/corpus.js";
import system from "./views/system.js";

const VIEWS = {
  monitor: { mod: monitor, title: "Live monitor", sub: "streaming risk, turn by turn" },
  analysis: { mod: analysis, title: "Call analysis", sub: "one call, full evidence" },
  results: { mod: results, title: "Results lab", sub: "measured on the held-out split" },
  corpus: { mod: corpus, title: "Corpus", sub: "what the models were trained on" },
  system: { mod: system, title: "System", sub: "architecture and design notes" },
};

/* shared state every view can read */
export const STATE = {
  calls: [],
  stats: null,
  meta: null,
  health: null,
  loaded: new Set(),
};

let current = null;

function setView(name) {
  if (!VIEWS[name]) name = "monitor";
  if (current === name) return;
  current = name;

  for (const btn of document.querySelectorAll(".nav-item")) {
    if (btn.dataset.view === name) btn.setAttribute("aria-current", "page");
    else btn.removeAttribute("aria-current");
  }
  for (const sec of document.querySelectorAll(".view")) {
    sec.classList.toggle("active", sec.id === `view-${name}`);
  }
  document.getElementById("view-title").textContent = VIEWS[name].title;
  document.getElementById("view-sub").innerHTML =
    `<span class="dot"></span>${VIEWS[name].sub}`;

  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);

  const v = VIEWS[name];
  if (!STATE.loaded.has(name)) {
    STATE.loaded.add(name);
    Promise.resolve(v.mod.init?.(STATE)).catch((e) => {
      console.error(`view ${name} failed to init`, e);
      showViewError(name, e);
    });
  } else {
    v.mod.refresh?.(STATE);
  }
  requestAnimationFrame(redrawAll);
}

function showViewError(name, err) {
  const sec = document.getElementById(`view-${name}`);
  if (!sec) return;
  const b = document.createElement("div");
  b.className = "banner crit";
  b.style.margin = "0 0 14px";
  b.innerHTML =
    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 7.5v5.5"/><circle cx="12" cy="16.4" r=".7" fill="currentColor"/></svg>` +
    `<div><b>This view could not load.</b><br><span class="mono xs">${String(err.message || err)}</span></div>`;
  sec.prepend(b);
}

/* ------------------------------------------------------------- theming */

function applyTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  try { localStorage.setItem("sk-theme", t); } catch { /* private mode */ }
  requestAnimationFrame(redrawAll);
}

function initTheme() {
  let t = "dark";
  try { t = localStorage.getItem("sk-theme") || "dark"; } catch { /* ignore */ }
  applyTheme(t);
  document.getElementById("theme-toggle").addEventListener("click", () => {
    const now = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    applyTheme(now);
  });
}

/* ---------------------------------------------------------------- boot */

async function boot() {
  initTheme();

  document.getElementById("nav").addEventListener("click", (e) => {
    const btn = e.target.closest(".nav-item");
    if (btn) setView(btn.dataset.view);
  });
  window.addEventListener("hashchange", () => setView(location.hash.slice(1)));

  const badge = document.getElementById("engine-status");
  try {
    STATE.health = await api.health();
    const ok = STATE.health.ready;
    badge.className = "badge " + (ok ? "low" : "elevated");
    badge.innerHTML = `<span class="dot"></span>${ok ? "ready" : "partial"}`;
    badge.title = ok
      ? "All branches loaded"
      : "Some models are missing. Train them with: swarkavach train";
    document.getElementById("build-info").textContent =
      `v${STATE.health.version} · ${STATE.health.fusion_model || "no fusion model"}`;
  } catch (e) {
    badge.className = "badge critical";
    badge.innerHTML = `<span class="dot"></span>offline`;
    badge.title = String(e.message || e);
  }

  try {
    const c = await api.corpus();
    STATE.calls = c.calls || [];
    STATE.stats = c.stats || null;
  } catch (e) {
    console.warn("corpus unavailable", e);
  }
  try { STATE.meta = await api.meta(); } catch { /* optional */ }

  setView(location.hash.slice(1) || "monitor");
}

window.addEventListener("beforeunload", clearResponsive);
boot();
