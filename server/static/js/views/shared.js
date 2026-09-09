/* Pieces that more than one view needs: transcript rendering with entity
 * spans, entity chips, and the plain-language reason rows. */

import { cssVar } from "../charts.js";

export const ENT_LABEL = {
  OTP: "OTP",
  BANK_ENTITY: "Bank",
  AUTHORITY_CLAIM: "Authority",
  THREAT_DEADLINE: "Threat",
  PAYMENT_HANDLE: "Payment",
  PERSONAL_INFO_REQ: "Personal info",
  MONEY_AMOUNT: "Amount",
};

export const ENT_COLOR = {
  OTP: "--series-8",
  PERSONAL_INFO_REQ: "--series-5",
  AUTHORITY_CLAIM: "--series-7",
  THREAT_DEADLINE: "--series-2",
  PAYMENT_HANDLE: "--series-4",
  BANK_ENTITY: "--series-1",
  MONEY_AMOUNT: "--series-3",
};

export function entColor(t) { return cssVar(ENT_COLOR[t] || "--series-1"); }

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/**
 * Rebuild a turn's text from its tokens, wrapping entity spans and lexicon
 * hits in marks. Working from tokens rather than the raw string keeps the
 * highlight aligned with the tagger, which is the whole point of sharing one
 * tokeniser across the project.
 */
export function highlightTokens(tokens, spans) {
  const open = new Map();     // token index -> span
  const close = new Set();
  for (const s of spans || []) {
    open.set(s.tok_start, s);
    close.add(s.tok_end - 1);
  }
  let html = "";
  let active = null;
  tokens.forEach((tok, i) => {
    const glue = i === 0 || /^[.,!?;:%)\]]$/.test(tok) ? "" : " ";
    if (open.has(i)) {
      active = open.get(i);
      const kind = active.source === "lexicon" ? "lex" : "ent";
      const title = active.source === "lexicon"
        ? `${active.type} cue, weight ${(+active.weight || 0).toFixed(2)}`
        : `${ENT_LABEL[active.type] || active.type} entity`;
      html += `${glue}<span class="${kind}" data-t="${esc(active.type)}" title="${esc(title)}">`;
      html += esc(tok);
    } else {
      html += glue + esc(tok);
    }
    if (active && close.has(i)) { html += "</span>"; active = null; }
  });
  if (active) html += "</span>";
  return html;
}

export function renderTurn(turn, spans, opts = {}) {
  const d = document.createElement("div");
  d.className = `turn ${turn.speaker}` +
    (opts.animate ? " enter" : "") + (opts.flagged ? " flagged" : "");
  d.dataset.index = turn.index;

  const t = Number(turn.t_start || 0).toFixed(1);
  const act = (turn.act || "").replace(/_/g, " ").toLowerCase();
  d.innerHTML =
    `<div class="meta">` +
      `<span class="who">${turn.speaker === "caller" ? "caller" : "callee"}</span>` +
      `<span class="t">${t}s</span>` +
      (opts.showAct !== false && act ? `<span class="act">${esc(act)}</span>` : "") +
    `</div>` +
    `<div class="txt">${highlightTokens(turn.tokens || [], spans)}</div>`;
  return d;
}

export function entityChip(type, count) {
  const c = document.createElement("span");
  c.className = "chip";
  c.innerHTML =
    `<span class="sw" style="background:${entColor(type)}"></span>` +
    `${ENT_LABEL[type] || type}` +
    (count !== undefined ? ` <span class="n">${count}</span>` : "");
  return c;
}

const ICONS = {
  voice: `<path d="M12 3a3 3 0 0 1 3 3v6a3 3 0 0 1-6 0V6a3 3 0 0 1 3-3z"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/>`,
  intent: `<path d="M4 5h16v11H8l-4 4z"/>`,
  cross: `<path d="M4 12h6M14 12h6M12 4v6M12 14v6"/>`,
  clear: `<path d="M20 6.5L9.6 17 4 11.5"/>`,
};

export function reasonRow(r) {
  const text = typeof r === "string" ? r : r.text;
  const group = typeof r === "string" ? "intent" : (r.group || "intent");
  const good = typeof r === "object" && r.direction === "clears";
  const color = good ? cssVar("--status-good")
    : group === "voice" ? cssVar("--series-2")
    : group === "cross" ? cssVar("--series-7")
    : cssVar("--series-1");
  const d = document.createElement("div");
  d.className = "reason";
  d.innerHTML =
    `<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="${color}" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">${good ? ICONS.clear : (ICONS[group] || ICONS.intent)}</svg>` +
    `<span>${esc(text)}</span>`;
  return d;
}

export function tile(host, { k, v, d, color, big }) {
  const el = document.createElement("div");
  el.className = "stat";
  el.innerHTML =
    `<span class="k">${esc(k)}</span>` +
    `<span class="v${big ? "" : " sm"}"${color ? ` style="color:${color}"` : ""}>${v}</span>` +
    (d ? `<span class="d">${esc(d)}</span>` : "");
  host.appendChild(el);
  return el;
}

export function table(node, cols, rows, opts = {}) {
  const thead = `<thead><tr>${cols.map((c) =>
    `<th class="${c.num ? "num" : ""}">${esc(c.label)}</th>`).join("")}</tr></thead>`;
  const tbody = `<tbody>${rows.map((r) =>
    `<tr>${cols.map((c) => {
      const raw = r[c.key];
      const txt = c.fmt ? c.fmt(raw, r) : (raw ?? "");
      const best = opts.bestKey && c.key === opts.bestKey && r.__best;
      return `<td class="${c.num ? "num" : ""}${best ? " best" : ""}">${txt}</td>`;
    }).join("")}</tr>`).join("")}</tbody>`;
  node.innerHTML = thead + tbody;
  return node;
}

export function fmtSeconds(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "n/a";
  return (+v).toFixed(1) + "s";
}

export function titleCase(s) {
  return String(s || "").replace(/_/g, " ").replace(/\b\w/g, (m) => m.toUpperCase());
}
