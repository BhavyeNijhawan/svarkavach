/* Results lab: every measured number, with the table view alongside each
 * chart so nothing depends on reading a color. */

import api from "../api.js";
import {
  heatmap, groupedBars, barChart, lineChart, detCurve, histogram,
  cssVar, fmt, pct, responsive, emptyState,
} from "../charts.js";
import { tile, table, titleCase } from "./shared.js";

const $ = (id) => document.getElementById(id);

const CELL_LABEL = {
  humanxbenign: "Human / benign",
  humanxscam: "Human / scam",
  clonedxbenign: "Cloned / benign",
  clonedxscam: "Cloned / scam",
};
const ARM_LABEL = {
  audio_only: "Audio only",
  text_only: "Text only",
  late_fusion: "Late fusion",
  full: "Full (cross-modal)",
};

let R = {};

async function load() {
  let index = { available: [] };
  try { index = await api.results(); } catch { /* server may be down */ }
  const names = index.available || [];
  const out = {};
  await Promise.all(names.map(async (n) => {
    try { out[n.replace(/\.json$/, "")] = await api.result(n); } catch { /* skip */ }
  }));
  return out;
}

/* ------------------------------------------------------------- sections */

function paintTiles() {
  const host = $("res-tiles");
  host.innerHTML = "";
  const ab = R.ablation_results;
  const as = R.antispoof_results;
  const ner = R.ner_results;
  const ttd = R.ttd_results;

  const fullAuc = ab?.overall?.full?.auc;
  const bestSingle = ab ? Math.max(
    ab.overall?.audio_only?.auc ?? 0, ab.overall?.text_only?.auc ?? 0) : null;
  tile(host, {
    k: "Fused AUC", v: fullAuc !== undefined ? fmt(fullAuc, 3) : "--",
    d: bestSingle ? `best single branch ${fmt(bestSingle, 3)}` : "run the evaluation",
    color: cssVar("--status-good"), big: true,
  });

  let bestEer = null, bestName = "";
  if (as?.eer) {
    for (const [f, models] of Object.entries(as.eer)) {
      for (const [m, conds] of Object.entries(models)) {
        const v = conds.clean ?? conds[Object.keys(conds)[0]];
        if (v !== undefined && v !== null && (bestEer === null || v < bestEer)) {
          bestEer = v; bestName = `${f.toUpperCase()} + ${m.toUpperCase()}`;
        }
      }
    }
  }
  tile(host, {
    k: "Best anti-spoof EER", v: bestEer !== null ? pct(bestEer, 2) : "--",
    d: bestName || "no anti-spoof run yet", big: true,
  });

  const f1 = ner?.overall?.crf?.gold?.f1 ?? ner?.overall?.crf?.f1;
  tile(host, {
    k: "Entity F1 (CRF, gold)", v: f1 !== undefined ? fmt(f1, 3) : "--",
    d: ner?.overall?.bilstm?.gold?.f1 !== undefined
      ? `BiLSTM ${fmt(ner.overall.bilstm.gold.f1, 3)}` : "entity level, exact span",
    big: true,
  });

  tile(host, {
    k: "Median time to detection", v: ttd?.median !== undefined ? fmt(ttd.median, 1) + "s" : "--",
    d: ttd ? `${ttd.detected} of ${ttd.total} scam calls flagged` : "streaming metric",
    big: true,
  });
}

function paintAblation() {
  const ab = R.ablation_results;
  if (!ab) { emptyState($("res-ablation-heat"), "Ablation has not been run"); return; }
  const cells = ab.cells || Object.keys(ab.detection_rate?.full || {});
  const arms = ab.arms || Object.keys(ab.detection_rate || {});

  responsive(() => {
    heatmap($("res-ablation-heat"), {
      rows: arms.map((a) => ARM_LABEL[a] || titleCase(a)),
      cols: cells.map((c) => CELL_LABEL[c] || c),
      values: arms.map((a) => cells.map((c) => ab.detection_rate?.[a]?.[c] ?? null)),
      vFmt: (v) => pct(v, 0), metric: "correct decisions", labelW: 148, cellH: 44,
      min: 0, max: 1,
    });
  });

  const rows = arms.map((a) => {
    const o = ab.overall?.[a] || {};
    const r = { arm: ARM_LABEL[a] || titleCase(a), auc: o.auc, acc: o.accuracy ?? o.acc, f1: o.f1, eer: o.eer };
    for (const c of cells) r[c] = ab.detection_rate?.[a]?.[c];
    return r;
  });
  const best = Math.max(...rows.map((r) => r.auc ?? 0));
  rows.forEach((r) => { r.__best = r.auc === best; });

  table($("res-ablation-tbl"), [
    { key: "arm", label: "Arm" },
    ...cells.map((c) => ({ key: c, label: CELL_LABEL[c] || c, num: true, fmt: (v) => v === undefined ? "n/a" : pct(v, 0) })),
    { key: "auc", label: "AUC", num: true, fmt: (v) => fmt(v, 3) },
    { key: "acc", label: "Accuracy", num: true, fmt: (v) => v === undefined ? "n/a" : pct(v, 1) },
    { key: "f1", label: "F1", num: true, fmt: (v) => fmt(v, 3) },
    { key: "eer", label: "EER", num: true, fmt: (v) => v === undefined ? "n/a" : pct(v, 2) },
  ], rows, { bestKey: "auc" });
}

function paintAntispoof() {
  const as = R.antispoof_results;
  if (!as?.eer) { emptyState($("res-as-heat"), "No anti-spoof results"); emptyState($("res-det"), "No DET curves"); return; }
  const feats = as.feature_sets || Object.keys(as.eer);
  const models = as.models || Object.keys(as.eer[feats[0]] || {});
  const cond = as.default_condition || "clean";

  responsive(() => {
    heatmap($("res-as-heat"), {
      rows: feats.map((f) => f.toUpperCase()),
      cols: models.map((m) => m.toUpperCase()),
      values: feats.map((f) => models.map((m) => as.eer?.[f]?.[m]?.[cond] ?? null)),
      vFmt: (v) => pct(v, 1), metric: "EER", reverse: true, labelW: 74, cellH: 40,
    });
  });

  responsive(() => {
    const series = (as.det || []).slice(0, 5).map((d, i) => ({
      name: d.name, points: d.points, eer: d.eer,
    }));
    if (!series.length) { emptyState($("res-det"), "No DET curves stored"); return; }
    detCurve($("res-det"), {
      height: 280, series, maxX: 0.6, maxY: 0.6,
      xTitle: "false alarm rate", yTitle: "miss rate",
    });
  });
}

function paintNer() {
  const ner = R.ner_results;
  if (!ner?.overall) { emptyState($("res-ner"), "NER has not been evaluated"); return; }
  const models = Object.keys(ner.overall);
  const transcripts = ner.transcripts || ["gold", "asr"];

  responsive(() => {
    groupedBars($("res-ner"), {
      height: 250,
      groups: models.map((m) => ({
        label: m.toUpperCase(),
        values: transcripts.map((t) => ner.overall[m]?.[t]?.f1 ?? 0),
      })),
      series: transcripts.map((t) => ({ name: t === "gold" ? "Gold transcript" : "ASR transcript" })),
      max: 1, vFmt: (v) => fmt(v, 2), yTitle: "entity F1",
    });
  });

  const rows = [];
  for (const m of models) {
    for (const t of transcripts) {
      const o = ner.overall[m]?.[t];
      if (!o) continue;
      rows.push({ model: m.toUpperCase(), src: t, p: o.p ?? o.precision, r: o.r ?? o.recall, f1: o.f1 });
    }
  }
  table($("res-ner-tbl"), [
    { key: "model", label: "Tagger" },
    { key: "src", label: "Transcript" },
    { key: "p", label: "Precision", num: true, fmt: (v) => fmt(v, 3) },
    { key: "r", label: "Recall", num: true, fmt: (v) => fmt(v, 3) },
    { key: "f1", label: "F1", num: true, fmt: (v) => fmt(v, 3) },
  ], rows);
}

function paintIntent() {
  const it = R.intent_results;
  if (!it?.models) { emptyState($("res-intent"), "Intent models have not been evaluated"); return; }
  const names = Object.keys(it.models);
  responsive(() => {
    barChart($("res-intent"), {
      height: 250, horizontal: true, labelWidth: 128, max: 1,
      items: names.map((n, i) => ({
        label: titleCase(n), value: it.models[n].auc ?? it.models[n].accuracy ?? 0,
        color: cssVar(`--series-${(i % 8) + 1}`),
        note: `accuracy ${pct(it.models[n].accuracy ?? 0, 1)}`,
      })),
      vFmt: (v) => fmt(v, 2), metric: "AUC",
    });
  });
  table($("res-intent-tbl"), [
    { key: "m", label: "Model" },
    { key: "acc", label: "Accuracy", num: true, fmt: (v) => v === undefined ? "n/a" : pct(v, 1) },
    { key: "f1", label: "F1", num: true, fmt: (v) => fmt(v, 3) },
    { key: "auc", label: "AUC", num: true, fmt: (v) => fmt(v, 3) },
  ], names.map((n) => ({
    m: titleCase(n), acc: it.models[n].accuracy, f1: it.models[n].f1, auc: it.models[n].auc,
  })));
}

function paintRobustness() {
  const rb = R.robustness_results;
  if (!rb?.auc) { emptyState($("res-robust"), "Robustness sweep has not been run"); return; }
  const codecs = rb.codecs || Object.keys(rb.auc);
  responsive(() => {
    barChart($("res-robust"), {
      height: 260, horizontal: false, max: 1, rotateLabels: codecs.length > 4,
      items: codecs.map((c) => ({
        label: (rb.labels?.[c] || c),
        value: rb.auc[c],
        color: rb.real_codec?.[c] === false ? cssVar("--series-4") : cssVar("--series-1"),
        note: rb.real_codec?.[c] === false ? "numpy approximation, ffmpeg absent" : "real codec",
      })),
      vFmt: (v) => fmt(v, 2), metric: "end-to-end AUC", yTitle: "AUC",
    });
  });
}

function paintTtd() {
  const t = R.ttd_results;
  if (!t?.values?.length) { emptyState($("res-ttd"), "No streaming run recorded"); return; }
  responsive(() => {
    histogram($("res-ttd"), {
      height: 260, values: t.values, bins: 12, color: cssVar("--series-3"),
      xTitle: "seconds of call heard before the alert", yTitle: "calls",
      refs: t.median !== undefined
        ? [{ value: t.median, label: `median ${fmt(t.median, 1)}s`, color: cssVar("--status-warning") }] : [],
    });
  });
}

function paintCalibration() {
  const c = R.calibration;
  if (!c?.bins?.length) { emptyState($("res-calib"), "No calibration curve"); return; }
  responsive(() => {
    lineChart($("res-calib"), {
      height: 270, dots: true,
      series: [
        { name: "Observed", points: c.bins.map((b) => [b.p_pred, b.p_obs]), color: cssVar("--series-1") },
        { name: "Perfect", points: [[0, 0], [1, 1]], color: cssVar("--text-3"), dash: "4 4" },
      ],
      xDomain: [0, 1], yDomain: [0, 1],
      xTitle: "predicted fraud probability", yTitle: "observed fraud rate",
      xFmt: (v) => v.toFixed(1), yFmt: (v) => v.toFixed(1),
      tipFmt: (v) => fmt(v, 3),
    });
    const note = document.createElement("div");
    note.className = "xs muted mt";
    note.textContent =
      `Brier score ${fmt(c.brier, 4)}, expected calibration error ${fmt(c.ece, 4)}. ` +
      `Late fusion needs calibrated inputs, so this curve is a precondition for the ablation above, not a footnote.`;
    $("res-calib").appendChild(note);
  });
}

function paintProvenance() {
  const p = R.provenance;
  const rows = (p?.entries || []).map((e) => ({
    a: e.artifact, s: e.source, g: e.generated, n: e.notes || "",
  }));
  if (!rows.length) {
    $("res-prov").innerHTML =
      `<tbody><tr><td class="muted">No provenance file. Results were produced locally.</td></tr></tbody>`;
    return;
  }
  table($("res-prov"), [
    { key: "a", label: "Artifact" },
    { key: "s", label: "Produced on", fmt: (v) => v === "colab"
      ? `<span class="badge neutral"><span class="dot"></span>Colab GPU</span>`
      : `<span class="badge neutral"><span class="dot"></span>local CPU</span>` },
    { key: "g", label: "When" },
    { key: "n", label: "Notes" },
  ], rows);
}

export default {
  async init() {
    R = await load();
    const any = Object.keys(R).length > 0;
    $("results-empty").hidden = any;
    if (!any) return;
    paintTiles();
    paintAblation();
    paintAntispoof();
    paintNer();
    paintIntent();
    paintRobustness();
    paintTtd();
    paintCalibration();
    paintProvenance();
  },
};
