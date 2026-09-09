/* Corpus explorer: what the models were trained and tested on. */

import api from "../api.js";
import {
  heatmap, barChart, groupedBars, cssVar, fmt, pct, responsive, emptyState,
} from "../charts.js";
import { tile, table, titleCase, entityChip } from "./shared.js";

const $ = (id) => document.getElementById(id);

let STATE = null;
let rows = [];

function paintTiles(s) {
  const host = $("cor-tiles");
  host.innerHTML = "";
  tile(host, { k: "Calls", v: s.n_calls ?? "--", d: `${s.n_turns ?? "?"} turns`, big: true });
  tile(host, { k: "Tokens", v: (s.n_tokens ?? 0).toLocaleString(), d: "gold transcripts, no manual typing", big: true });
  tile(host, {
    k: "Annotated entities", v: (s.n_entities ?? 0).toLocaleString(),
    d: `${Object.keys(s.entities_per_type || {}).length} types, BIO tagged`, big: true,
  });
  tile(host, {
    k: "Mean code-mixing index", v: fmt(s.mean_cmi ?? 0, 3),
    d: "0 is monolingual, higher means more switching", big: true,
  });
}

function paintCells(s) {
  const cells = s.cells || {};
  const rowsL = ["human", "cloned"], colsL = ["benign", "scam"];
  responsive(() => {
    heatmap($("cor-cells"), {
      rows: rowsL.map(titleCase), cols: colsL.map(titleCase),
      values: rowsL.map((r) => colsL.map((c) => cells[`${r}x${c}`] ?? 0)),
      vFmt: (v) => String(v), metric: "calls", dp: 0, labelW: 76, cellH: 56,
    });
    const note = document.createElement("div");
    note.className = "xs muted mt";
    note.textContent =
      "The two off-diagonal cells carry the argument. A cloned voice reading a benign script " +
      "should not be flagged, and a human running a scam script should be.";
    $("cor-cells").appendChild(note);
  });
}

function paintEntities(s) {
  const e = s.entities_per_type || {};
  const keys = Object.keys(e).sort((a, b) => e[b] - e[a]);
  if (!keys.length) { emptyState($("cor-entities"), "No entities"); return; }
  responsive(() => {
    barChart($("cor-entities"), {
      height: 236, horizontal: true, labelWidth: 136, dp: 0,
      items: keys.map((k, i) => ({
        label: titleCase(k), value: e[k], color: cssVar(`--series-${(i % 8) + 1}`),
      })),
      vFmt: (v) => String(Math.round(v)), metric: "spans",
    });
  });
}

function paintScenarios(s) {
  const sc = s.scenarios || {};
  const scamSet = new Set(s.scam_scenarios || []);
  const keys = Object.keys(sc).sort((a, b) => sc[b] - sc[a]);
  if (!keys.length) { emptyState($("cor-scenarios"), "No scenarios"); return; }
  responsive(() => {
    barChart($("cor-scenarios"), {
      height: 300, horizontal: true, labelWidth: 158, dp: 0,
      items: keys.map((k) => ({
        label: titleCase(k), value: sc[k],
        color: scamSet.has(k) ? cssVar("--status-critical") : cssVar("--status-good"),
        note: scamSet.has(k) ? "fraudulent" : "legitimate",
      })),
      vFmt: (v) => String(Math.round(v)), metric: "calls",
    });
    const legend = document.createElement("div");
    legend.className = "legend";
    legend.innerHTML =
      `<span class="item"><span class="key" style="background:${cssVar("--status-critical")}"></span>Fraudulent</span>` +
      `<span class="item"><span class="key" style="background:${cssVar("--status-good")}"></span>Legitimate</span>`;
    $("cor-scenarios").appendChild(legend);
  });
}

function paintCmi(s) {
  const cm = s.cmi_by_class;
  if (!cm) { emptyState($("cor-cmi"), "No code-mixing statistics"); return; }
  responsive(() => {
    groupedBars($("cor-cmi"), {
      height: 300,
      groups: [
        { label: "Code-mixing index", values: [cm.benign?.cmi ?? 0, cm.scam?.cmi ?? 0] },
        { label: "Switch points / 100 tok", values: [cm.benign?.switch_rate ?? 0, cm.scam?.switch_rate ?? 0] },
        { label: "English share", values: [cm.benign?.en_ratio ?? 0, cm.scam?.en_ratio ?? 0] },
        { label: "Entities in English", values: [cm.benign?.ent_lang_align ?? 0, cm.scam?.ent_lang_align ?? 0] },
      ],
      series: [
        { name: "Legitimate", color: cssVar("--status-good") },
        { name: "Fraudulent", color: cssVar("--status-critical") },
      ],
      vFmt: (v) => fmt(v, 2), rotateLabels: true,
    });
    const note = document.createElement("div");
    note.className = "xs muted mt";
    note.textContent =
      "Fraud scripts keep the threat in Hindi and the financial vocabulary in English. " +
      "That split is what the entity-language alignment feature measures.";
    $("cor-cmi").appendChild(note);
  });
}

function paintTable(filter = "") {
  const f = filter.trim().toLowerCase();
  const show = rows.filter((r) => !f ||
    r.call_id.toLowerCase().includes(f) ||
    r.scenario.toLowerCase().includes(f) ||
    r.cell.toLowerCase().includes(f) ||
    r.split.toLowerCase().includes(f));
  table($("cor-table"), [
    { key: "call_id", label: "Call" },
    { key: "scenario", label: "Scenario", fmt: (v) => titleCase(v) },
    { key: "cell", label: "Cell", fmt: (v) => {
      const scam = v.includes("scam");
      return `<span class="badge ${scam ? "critical" : "low"}"><span class="dot"></span>${v.replace("x", " / ")}</span>`;
    } },
    { key: "split", label: "Split" },
    { key: "speaker_id", label: "Speaker" },
    { key: "n_turns", label: "Turns", num: true },
    { key: "n_entities", label: "Entities", num: true },
    { key: "duration", label: "Length", num: true, fmt: (v) => v ? fmt(v, 1) + "s" : "n/a" },
    { key: "cmi", label: "CMI", num: true, fmt: (v) => v === undefined ? "n/a" : fmt(v, 3) },
  ], show.slice(0, 400));

  const head = $("cor-table").closest(".card").querySelector(".card-head h2");
  head.textContent = `Call browser (${show.length} of ${rows.length})`;
}

export default {
  async init(state) {
    STATE = state;
    let s = state.stats;
    if (!s) {
      try { s = await api.result("corpus_stats.json"); } catch { s = null; }
    }
    if (!s) {
      $("cor-tiles").innerHTML =
        `<div class="banner warn" style="grid-column:1/-1">No corpus on disk. Generate one with <code>swarkavach gen-corpus</code>.</div>`;
      return;
    }
    rows = state.calls || [];
    paintTiles(s);
    paintCells(s);
    paintEntities(s);
    paintScenarios(s);
    paintCmi(s);
    paintTable();
    $("cor-filter").addEventListener("input", (e) => paintTable(e.target.value));
  },
};
