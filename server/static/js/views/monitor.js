/* Live monitor: replay a call and watch the fused risk score move.
 *
 * The server streams one incremental verdict per turn as soon as it has
 * computed it. The client buffers those events and renders each one when the
 * playback clock reaches that turn's end time, so the transcript, the gauge
 * and the audio stay together even when analysis runs faster than real time.
 */

import api from "../api.js";
import {
  gauge, meter, lineChart, cssVar, bandColor, fmt, responsive, sparkline,
} from "../charts.js";
import { renderTurn, entityChip, reasonRow, ENT_LABEL } from "./shared.js";

const $ = (id) => document.getElementById(id);

let STATE = null;
let stream = null;
let audio = null;
let clock = null;         // rAF handle
let queue = [];           // pending turn events
let shown = 0;
let traj = [];            // [[t, risk], ...]
let started = 0;
let alerted = null;
let duration = 0;
let waveData = null;
let currentCall = null;

/* ------------------------------------------------------------ waveform */

function drawWave(peaks) {
  const wrap = $("mon-wave");
  const cv = $("mon-wave-canvas");
  const w = wrap.clientWidth, h = 88;
  const dpr = window.devicePixelRatio || 1;
  cv.width = w * dpr; cv.height = h * dpr;
  cv.style.width = w + "px"; cv.style.height = h + "px";
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!peaks || !peaks.length) return;

  const mid = h / 2;
  const n = peaks.length;
  const bw = Math.max(1, w / n);
  ctx.fillStyle = cssVar("--series-1");
  ctx.globalAlpha = 0.55;
  for (let i = 0; i < n; i++) {
    const a = Math.max(0.02, peaks[i]) * (h / 2 - 3);
    ctx.fillRect(i * bw, mid - a, Math.max(1, bw - 0.6), a * 2);
  }
  ctx.globalAlpha = 1;
  ctx.strokeStyle = cssVar("--line-2");
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, mid + .5); ctx.lineTo(w, mid + .5); ctx.stroke();
}

function markAlert(tSec) {
  const wrap = $("mon-wave");
  wrap.querySelectorAll(".alert-marker").forEach((n) => n.remove());
  if (tSec === null || tSec === undefined || !duration) return;
  const m = document.createElement("div");
  m.className = "alert-marker";
  m.dataset.label = `alert ${tSec.toFixed(1)}s`;
  m.style.left = (100 * tSec / duration) + "%";
  wrap.appendChild(m);
}

/* --------------------------------------------------------------- panels */

function paintVerdict(v) {
  gauge($("mon-gauge"), v.risk, v.band, { size: 154 });

  const m = $("mon-meters");
  m.innerHTML = "";
  meter(m, {
    name: "Voice authenticity", value: v.authenticity,
    color: cssVar("--series-2"),
    sub: v.authenticity >= 0.5 ? "leans synthetic" : "leans human",
  });
  meter(m, {
    name: "Scam intent", value: v.intent,
    color: cssVar("--series-1"),
    sub: v.intent >= 0.5 ? "coercive script pattern" : "conversational",
  });
  meter(m, {
    name: "Prosody-intent mismatch", value: v.pim,
    color: cssVar("--series-7"),
    sub: v.pim >= 0.5 ? "flat delivery, high-pressure words" : "delivery matches content",
  });
}

function paintTrajectory() {
  responsive(() => {
    const host = $("mon-traj");
    if (!traj.length) { host.innerHTML = ""; return; }
    lineChart(host, {
      height: 190, dots: traj.length <= 24,
      series: [
        { name: "Fused risk", points: traj.map((p) => [p[0], p[1]]), color: cssVar("--series-8") },
        { name: "Scam intent", points: traj.map((p) => [p[0], p[2]]), color: cssVar("--series-1") },
        { name: "Synthetic voice", points: traj.map((p) => [p[0], p[3]]), color: cssVar("--series-2") },
      ],
      yDomain: [0, 1], xDomain: [0, Math.max(duration, 1)],
      xTitle: "seconds into the call", xDp: 0,
      yFmt: (v) => v.toFixed(1),
      xFmt: (v) => v.toFixed(0) + "s",
      tipFmt: (v) => fmt(v, 3),
      xLabelPrefix: "at ",
      refs: [{ value: 0.65, label: "alert", color: cssVar("--status-warning") }],
    });
  });
}

function paintReasons(reasons) {
  const host = $("mon-reasons");
  if (!reasons || !reasons.length) {
    host.innerHTML = `<div class="empty" style="padding:22px">Nothing flagged yet.</div>`;
    return;
  }
  host.innerHTML = "";
  for (const r of reasons.slice(0, 7)) host.appendChild(reasonRow(r));
}

/* ------------------------------------------------------------ playback */

function nowSeconds() {
  if (audio && !audio.paused && !audio.ended) return audio.currentTime;
  const speed = +$("mon-speed").value || 1;
  return ((performance.now() - started) / 1000) * speed;
}

function tick() {
  const t = nowSeconds();
  $("mon-clock").textContent = t.toFixed(1) + "s";
  const ph = $("mon-playhead");
  if (duration > 0) {
    ph.style.display = "block";
    ph.style.left = (100 * Math.min(1, t / duration)) + "%";
  }

  while (queue.length && queue[0].t_end <= t + 0.02) {
    renderEvent(queue.shift());
  }

  if (queue.length || (duration && t < duration)) {
    clock = requestAnimationFrame(tick);
  } else {
    finish();
  }
}

function renderEvent(ev) {
  shown++;
  const box = $("mon-transcript");
  if (shown === 1) box.innerHTML = "";
  const node = renderTurn(ev.turn, ev.spans || [], { animate: true, flagged: ev.flagged });
  box.appendChild(node);
  box.scrollTop = box.scrollHeight;

  traj.push([ev.t_end, ev.risk, ev.intent, ev.authenticity]);
  paintVerdict(ev);
  paintTrajectory();
  if (ev.reasons?.length) paintReasons(ev.reasons);

  if (ev.alert && alerted === null) {
    alerted = ev.t_end;
    $("mon-ttd").textContent = ev.t_end.toFixed(1) + "s";
    $("mon-ttd").style.color = cssVar("--status-critical");
    $("mon-ttd-turn").textContent = `after turn ${ev.turn.index + 1} of ${currentCall?.turns?.length ?? "?"}`;
    markAlert(ev.t_end);
    node.classList.add("flash");
    const live = $("mon-live");
    live.className = "badge critical";
    live.innerHTML = `<span class="dot"></span>alert raised`;
  }
}

function finish() {
  cancelAnimationFrame(clock);
  clock = null;
  const live = $("mon-live");
  if (alerted === null) {
    live.className = "badge low";
    live.innerHTML = `<span class="dot"></span>call cleared`;
    $("mon-ttd").textContent = "no alert";
    $("mon-ttd-turn").textContent = "risk stayed below threshold";
  }
  $("mon-run").disabled = false;
  $("mon-run").innerHTML =
    `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5.2v13.6L19 12z"/></svg>Start call`;
}

function stop() {
  if (stream) { stream.close(); stream = null; }
  if (clock) { cancelAnimationFrame(clock); clock = null; }
  if (audio) { audio.pause(); audio = null; }
}

/* ----------------------------------------------------------------- run */

async function run() {
  const id = $("mon-call").value;
  if (!id) return;
  stop();

  queue = []; shown = 0; traj = []; alerted = null;
  $("mon-transcript").innerHTML = `<div class="empty"><span class="spinner" style="margin:0 auto 10px"></span>Connecting to the analysis stream.</div>`;
  $("mon-reasons").innerHTML = `<div class="empty" style="padding:22px">Listening.</div>`;
  $("mon-ttd").textContent = "--";
  $("mon-ttd").style.color = "";
  $("mon-ttd-turn").textContent = "not yet triggered";
  $("mon-traj").innerHTML = "";
  markAlert(null);

  const live = $("mon-live");
  live.className = "badge elevated";
  live.innerHTML = `<span class="dot"></span>listening`;
  $("mon-run").disabled = true;
  $("mon-run").innerHTML = `<span class="spinner"></span>Analysing`;

  const codec = $("mon-codec").value;
  const speed = +$("mon-speed").value || 1;

  currentCall = await api.call(id).catch(() => null);
  const truth = currentCall
    ? (currentCall.label_scam ? "fraudulent" : "legitimate")
    : "unknown";
  $("mon-truth").textContent = truth;
  $("mon-truth").style.color = currentCall?.label_scam
    ? cssVar("--status-critical") : cssVar("--status-good");
  $("mon-truth-note").textContent = currentCall
    ? `${currentCall.label_voice} voice, ${currentCall.scenario.replace(/_/g, " ")}`
    : "";

  // audio drives the clock when the browser can play it
  try {
    audio = new Audio(api.audioUrl(id, codec));
    audio.playbackRate = speed;
    audio.preload = "auto";
    await audio.play();
  } catch {
    audio = null;              // autoplay blocked or no audio, fall back to a timer
  }
  started = performance.now();

  stream = api.stream(id, { codec, speed }, {
    onMeta: (m) => {
      duration = m.duration || 0;
      $("mon-duration").textContent = duration.toFixed(1) + "s";
      $("mon-channel-note").textContent = m.channel_note || "";
      if (!clock) clock = requestAnimationFrame(tick);
    },
    onTurn: (ev) => { queue.push(ev); if (!clock) clock = requestAnimationFrame(tick); },
    onDone: () => { /* the clock drains the queue and calls finish */ },
    onError: (e) => {
      $("mon-transcript").innerHTML =
        `<div class="banner crit" style="margin:8px">Stream failed: ${e.error || "unknown"}</div>`;
      finish();
    },
  });
}

/* -------------------------------------------------------------- wiring */

async function loadWave() {
  const id = $("mon-call").value;
  if (!id) return;
  try {
    waveData = await api.waveform(id, $("mon-codec").value);
    duration = waveData.duration || 0;
    $("mon-duration").textContent = duration.toFixed(1) + "s";
    drawWave(waveData.peaks);
  } catch {
    drawWave(null);
  }
}

function fillCalls(calls) {
  const sel = $("mon-call");
  sel.innerHTML = "";
  const order = { clonedxscam: 0, humanxscam: 1, clonedxbenign: 2, humanxbenign: 3 };
  const sorted = calls.slice().sort((a, b) =>
    (order[a.cell] ?? 9) - (order[b.cell] ?? 9) || a.call_id.localeCompare(b.call_id));
  for (const c of sorted) {
    const o = document.createElement("option");
    o.value = c.call_id;
    o.textContent = `${c.call_id}  ·  ${c.scenario.replace(/_/g, " ")}  ·  ${c.cell.replace("x", " / ")}`;
    sel.appendChild(o);
  }
}

export default {
  async init(state) {
    STATE = state;
    fillCalls(state.calls);
    $("mon-run").addEventListener("click", run);
    $("mon-call").addEventListener("change", loadWave);
    $("mon-codec").addEventListener("change", loadWave);
    $("mon-speed").addEventListener("change", () => {
      if (audio) audio.playbackRate = +$("mon-speed").value || 1;
    });
    $("mon-new").addEventListener("click", async () => {
      const btn = $("mon-new");
      btn.disabled = true;
      const old = btn.innerHTML;
      btn.innerHTML = `<span class="spinner"></span>Synthesising`;
      try {
        const r = await api.simulate({ audio: true });
        STATE.calls.unshift(r.summary);
        fillCalls(STATE.calls);
        $("mon-call").value = r.summary.call_id;
        await loadWave();
        run();
      } catch (e) {
        alert("Could not synthesise a call: " + e.message);
      } finally {
        btn.disabled = false;
        btn.innerHTML = old;
      }
    });
    window.addEventListener("resize", () => waveData && drawWave(waveData.peaks));
    await loadWave();
  },
  refresh() { if (waveData) drawWave(waveData.peaks); },
};
