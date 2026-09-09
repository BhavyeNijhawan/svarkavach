/* Call analysis: one call, every piece of evidence behind the verdict. */

import api from "../api.js";
import {
  gauge, meter, waterfall, lineChart, radar, cssVar, fmt, pct, responsive, emptyState,
} from "../charts.js";
import { renderTurn, entityChip, reasonRow, tile, ENT_LABEL } from "./shared.js";

const $ = (id) => document.getElementById(id);

let STATE = null;
let verdict = null;
let call = null;
let uploaded = null;      // {call_id} returned by /api/upload

/* --------------------------------------------------------- spectrogram */

function drawSpectrogram(spec) {
  const cv = $("an-spec");
  const w = cv.parentElement.clientWidth, h = 180;
  const dpr = window.devicePixelRatio || 1;
  cv.width = w * dpr; cv.height = h * dpr;
  cv.style.width = w + "px"; cv.style.height = h + "px";
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!spec || !spec.db || !spec.db.length) return;

  const rows = spec.db.length, cols = spec.db[0].length;
  // sequential single hue, light to dark, taken from the project ramp
  const ramp = ["--seq-100", "--seq-250", "--seq-400", "--seq-550", "--seq-700"]
    .map(cssVar).map(hexToRgb);
  const lo = spec.vmin ?? -80, hi = spec.vmax ?? 0;

  const img = ctx.createImageData(cols, rows);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      let t = (spec.db[rows - 1 - r][c] - lo) / ((hi - lo) || 1);
      t = Math.max(0, Math.min(1, t));
      const p = t * (ramp.length - 1);
      const i0 = Math.floor(p), i1 = Math.min(ramp.length - 1, i0 + 1), f = p - i0;
      const px = (r * cols + c) * 4;
      img.data[px] = ramp[i0][0] + (ramp[i1][0] - ramp[i0][0]) * f;
      img.data[px + 1] = ramp[i0][1] + (ramp[i1][1] - ramp[i0][1]) * f;
      img.data[px + 2] = ramp[i0][2] + (ramp[i1][2] - ramp[i0][2]) * f;
      img.data[px + 3] = 255;
    }
  }
  const off = document.createElement("canvas");
  off.width = cols; off.height = rows;
  off.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, 0, 0, w, h);

  ctx.fillStyle = cssVar("--text-3");
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText("4 kHz", 5, 12);
  ctx.fillText("0", 5, h - 5);
  if (spec.duration) {
    const lbl = spec.duration.toFixed(1) + "s";
    ctx.fillText(lbl, w - ctx.measureText(lbl).width - 5, h - 5);
  }
}

function hexToRgb(hex) {
  const m = hex.replace("#", "");
  const n = m.length === 3
    ? m.split("").map((c) => parseInt(c + c, 16))
    : [0, 2, 4].map((i) => parseInt(m.slice(i, i + 2), 16));
  return n.map((v) => (Number.isFinite(v) ? v : 0));
}

/* ---------------------------------------------------------- rendering */

function paintTiles(v) {
  const host = $("an-tiles");
  host.innerHTML = "";
  const bandColorMap = {
    low: "--status-good", elevated: "--status-warning",
    high: "--status-serious", critical: "--status-critical",
  };
  tile(host, {
    k: "Fused fraud risk", v: (v.risk * 100).toFixed(0) + "<span class='muted' style='font-size:14px'>%</span>",
    d: v.band + " band", color: cssVar(bandColorMap[v.band] || "--text-1"), big: true,
  });
  tile(host, {
    k: "Ground truth", v: call ? (call.label_scam ? "Fraud" : "Legitimate") : "unknown",
    d: call ? `${call.label_voice} voice, ${call.scenario.replace(/_/g, " ")}` : "uploaded audio",
    color: call ? cssVar(call.label_scam ? "--status-critical" : "--status-good") : undefined,
  });
  tile(host, {
    k: "Alert latency", v: v.ttd !== null && v.ttd !== undefined ? v.ttd.toFixed(1) + "s" : "no alert",
    d: v.ttd_turns !== null && v.ttd_turns !== undefined
      ? `after turn ${v.ttd_turns + 1}` : "stayed under threshold",
  });
  const lat = Object.values(v.latency_ms || {}).reduce((a, b) => a + b, 0);
  tile(host, {
    k: "Analysis time", v: lat ? lat.toFixed(0) + "<span class='muted' style='font-size:14px'>ms</span>" : "n/a",
    d: "CPU only, no GPU in this path",
  });
}

function paintVerdict(v) {
  gauge($("an-gauge"), v.risk, v.band, { size: 150 });
  const m = $("an-meters");
  m.innerHTML = "";
  meter(m, { name: "Voice authenticity", value: v.authenticity, color: cssVar("--series-2"),
             sub: `${v.branches?.antispoof?.backend || "detector"} backend` });
  meter(m, { name: "Scam intent", value: v.intent, color: cssVar("--series-1"),
             sub: `${v.branches?.intent?.backend || "classifier"} backend` });
  meter(m, { name: "Prosody-intent mismatch", value: v.pim, color: cssVar("--series-7"),
             sub: "how far delivery sits from wording" });
}

function paintTranscript(v) {
  const host = $("an-transcript");
  host.innerHTML = "";
  const turns = v.turns || call?.turns || [];
  if (!turns.length) { emptyState(host, "No transcript available for this call"); return; }
  const byTurn = new Map();
  for (const s of v.evidence?.spans || []) {
    if (!byTurn.has(s.turn_index)) byTurn.set(s.turn_index, []);
    byTurn.get(s.turn_index).push(s);
  }
  const flagged = new Set((v.timeline || []).filter((t) => t.flagged).map((t) => t.turn_index));
  for (const t of turns) {
    host.appendChild(renderTurn(t, byTurn.get(t.index) || [], { flagged: flagged.has(t.index) }));
  }
  $("an-src").textContent = `transcript: ${v.transcript_source}`;
}

function paintReasons(v) {
  const host = $("an-reasons");
  host.innerHTML = "";
  const rs = v.evidence?.reasons || [];
  if (!rs.length) { host.innerHTML = `<div class="empty" style="padding:20px">Nothing stood out.</div>`; return; }
  for (const r of rs) host.appendChild(reasonRow(r));
}

function paintEntities(v) {
  const host = $("an-entities");
  host.innerHTML = "";
  const counts = {};
  for (const s of v.evidence?.spans || []) {
    if (s.source === "lexicon") continue;
    counts[s.type] = (counts[s.type] || 0) + 1;
  }
  const keys = Object.keys(counts);
  if (!keys.length) { host.innerHTML = `<span class="muted small">No fraud entities detected.</span>`; return; }
  for (const k of keys.sort((a, b) => counts[b] - counts[a])) {
    host.appendChild(entityChip(k, counts[k]));
  }
}

function paintShap(v) {
  responsive(() => {
    const items = (v.evidence?.contributions || [])
      .slice()
      .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
    waterfall($("an-shap"), { items, limit: 11, labelWidth: 176 });
  });
}

function paintPim(v) {
  responsive(() => {
    const per = v.branches?.pim?.detail?.per_turn || [];
    const host = $("an-pim");
    if (!per.length) { emptyState(host, "Cross-modal trace needs call audio"); return; }
    lineChart(host, {
      height: 250,
      series: [
        { name: "Lexical pressure", points: per.map((p, i) => [p.t ?? i, p.lex]), color: cssVar("--series-1") },
        { name: "Acoustic arousal", points: per.map((p, i) => [p.t ?? i, p.aco]), color: cssVar("--series-2") },
      ],
      yDomain: [0, 1], xTitle: "seconds into the call", xDp: 0,
      xFmt: (x) => x.toFixed(0) + "s", yFmt: (x) => x.toFixed(1),
      tipFmt: (x) => fmt(x, 3), xLabelPrefix: "at ",
    });
  });
}

function paintRadar(v) {
  responsive(() => {
    const f = v.features || {};
    const ref = STATE.meta?.feature_reference || {};
    const axes = ["Jitter", "Shimmer", "HNR", "Flatness var", "Spk drift", "Synth score"];
    const norm = (k, invert) => {
      const hi = ref[k]?.p95 ?? 1;
      const lo = ref[k]?.p05 ?? 0;
      let t = ((f[k] ?? 0) - lo) / ((hi - lo) || 1);
      t = Math.max(0, Math.min(1, t));
      return invert ? 1 - t : t;
    };
    radar($("an-radar"), {
      size: 292,
      axes,
      series: [{
        name: "This call",
        color: cssVar("--series-2"),
        values: [
          norm("jitter", true), norm("shimmer", true), norm("hnr", false),
          norm("spec_flatness_var", true), 1 - (f.spk_consistency ?? 0), f.as_score ?? 0,
        ],
      }],
    });
  });
}

/* ---------------------------------------------------------------- run */

async function run() {
  const btn = $("an-run");
  btn.disabled = true;
  const old = btn.textContent;
  btn.innerHTML = `<span class="spinner"></span>Analysing`;
  try {
    const id = uploaded ? uploaded.call_id : $("an-call").value;
    const snr = $("an-snr").value;
    const body = { call_id: id, codec: $("an-codec").value };
    if (snr) body.snr = +snr;
    verdict = await api.analyze(body);
    call = verdict.call || await api.call(id).catch(() => null);

    paintTiles(verdict);
    paintVerdict(verdict);
    paintTranscript(verdict);
    paintReasons(verdict);
    paintEntities(verdict);
    paintShap(verdict);
    paintPim(verdict);
    paintRadar(verdict);

    const spec = await api.spectrogram(id, $("an-codec").value, snr).catch(() => null);
    drawSpectrogram(spec);
    window.__lastSpec = spec;
  } catch (e) {
    $("an-transcript").innerHTML =
      `<div class="banner crit" style="margin:6px">Analysis failed: ${e.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = old;
  }
}

export default {
  async init(state) {
    STATE = state;
    const sel = $("an-call");
    sel.innerHTML = "";
    for (const c of state.calls) {
      const o = document.createElement("option");
      o.value = c.call_id;
      o.textContent = `${c.call_id}  ·  ${c.scenario.replace(/_/g, " ")}  ·  ${c.cell.replace("x", " / ")}`;
      sel.appendChild(o);
    }
    sel.addEventListener("change", () => { uploaded = null; });
    $("an-run").addEventListener("click", run);
    $("an-upload").addEventListener("change", async (e) => {
      const f = e.target.files?.[0];
      if (!f) return;
      try {
        uploaded = await api.upload(f);
        const o = document.createElement("option");
        o.value = uploaded.call_id;
        o.textContent = `${uploaded.call_id}  ·  uploaded  ·  ${f.name}`;
        sel.prepend(o);
        sel.value = uploaded.call_id;
        run();
      } catch (err) {
        alert("Upload failed: " + err.message);
      }
    });
    window.addEventListener("resize", () => {
      if (window.__lastSpec) drawSpectrogram(window.__lastSpec);
    });
    if (state.calls.length) run();
  },
  refresh() { if (window.__lastSpec) drawSpectrogram(window.__lastSpec); },
};
