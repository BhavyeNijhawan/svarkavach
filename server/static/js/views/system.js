/* System view: the architecture, what is new, the runtime, and the mapping
 * back to the course syllabus (which is what the viva asks about). */

import api from "../api.js";
import { cssVar, fmt } from "../charts.js";
import { table, titleCase } from "./shared.js";

const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------- architecture */

function archSvg() {
  const c = {
    line: cssVar("--line-2"),
    text: cssVar("--text-1"),
    dim: cssVar("--text-2"),
    surf: cssVar("--surface-2"),
    v: cssVar("--series-2"),
    t: cssVar("--series-1"),
    x: cssVar("--series-7"),
    out: cssVar("--status-critical"),
  };
  const box = (x, y, w, h, fill, stroke) =>
    `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="9" fill="${fill}" stroke="${stroke}" stroke-width="1.4"/>`;
  // Font sizes are deliberately large for the viewBox. The card is often
  // narrower than 880px, so the whole drawing gets scaled down; at 0.45 scale
  // a 10px label lands at 4px on screen, which nobody can read.
  const label = (x, y, s, size = 15, weight = 600, fill = c.text, anchor = "middle") =>
    `<text x="${x}" y="${y}" text-anchor="${anchor}" font-size="${size}" font-weight="${weight}" fill="${fill}" font-family="var(--sans)">${s}</text>`;
  const arrow = (x1, y1, x2, y2) =>
    `<path d="M${x1},${y1} L${x2},${y2}" stroke="${c.line}" stroke-width="1.8" marker-end="url(#ah)"/>`;
  const soft = (col) => `color-mix(in srgb, ${col} 12%, transparent)`;

  return `
<svg viewBox="0 0 880 430" width="100%" style="max-width:1100px;display:block;margin:0 auto" role="img"
     aria-label="Architecture: call audio splits into a voice branch and a text branch, which meet at a cross-modal fusion layer">
  <defs>
    <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill="${c.line}"/>
    </marker>
  </defs>

  ${box(18, 22, 190, 58, c.surf, c.line)}
  ${label(113, 47, "Call audio", 17)}
  ${label(113, 68, "8 kHz mono, VAD trimmed", 13, 400, c.dim)}

  ${arrow(210, 51, 256, 51)}
  ${box(256, 18, 190, 66, c.surf, c.line)}
  ${label(351, 44, "Channel model", 17)}
  ${label(351, 63, "G.711, GSM, AMR", 13, 400, c.dim)}
  ${label(351, 78, "noise at a set SNR", 13, 400, c.dim)}

  ${arrow(351, 84, 200, 132)}
  ${arrow(351, 84, 590, 132)}

  <!-- voice branch -->
  ${box(30, 132, 340, 132, soft(c.v), c.v)}
  ${label(200, 158, "Voice authenticity", 17, 650, c.v)}
  ${label(200, 182, "LFCC, GFCC, MFCC, CQCC, LPCC", 13.5, 400, c.dim)}
  ${label(200, 201, "filterbanks and DCT written by hand", 13.5, 400, c.dim)}
  ${label(200, 220, "GMM ratio, boosting, RawNet-lite", 13.5, 400, c.dim)}
  ${label(200, 250, "P(voice is synthetic)", 14.5, 700, c.text)}

  <!-- text branch -->
  ${box(420, 132, 340, 132, soft(c.t), c.t)}
  ${label(590, 158, "Scam intent", 17, 650, c.t)}
  ${label(590, 182, "transcript, per-token language ID", 13.5, 400, c.dim)}
  ${label(590, 201, "entity CRF written from scratch,", 13.5, 400, c.dim)}
  ${label(590, 220, "BiLSTM-CRF, TF-IDF, rule baseline", 13.5, 400, c.dim)}
  ${label(590, 250, "P(conversation is a scam)", 14.5, 700, c.text)}

  ${arrow(200, 264, 340, 300)}
  ${arrow(590, 264, 450, 300)}

  <!-- cross modal -->
  ${box(150, 300, 490, 86, soft(c.x), c.x)}
  ${label(395, 326, "Cross-modal layer", 17, 650, c.x)}
  ${label(395, 348, "prosody-intent mismatch, coercion trajectory,", 13.5, 400, c.dim)}
  ${label(395, 366, "code-switching profile, dialogue-act sequence LLR", 13.5, 400, c.dim)}
  ${label(395, 381, "needs both branches, so neither can produce these alone", 12, 400, c.dim)}

  ${arrow(395, 386, 395, 400)}
  ${box(150, 400, 490, 30, soft(c.out), c.out)}
  ${label(395, 420, "Calibrated risk score, band and evidence panel", 15, 700, c.text)}

  <!-- side note -->
  ${box(790, 132, 82, 132, c.surf, c.line)}
  ${label(831, 158, "Streaming", 14)}
  ${label(831, 182, "reruns", 12, 400, c.dim)}
  ${label(831, 198, "after every", 12, 400, c.dim)}
  ${label(831, 214, "turn: this", 12, 400, c.dim)}
  ${label(831, 230, "is where", 12, 400, c.dim)}
  ${label(831, 246, "TTD comes", 12, 400, c.dim)}
  ${label(831, 262, "from", 12, 400, c.dim)}
  <path d="M760,198 L790,198" stroke="${c.line}" stroke-width="1.6" stroke-dasharray="3 3"/>
</svg>`;
}

/* ------------------------------------------------------------ content */

const NOVELTY = [
  ["Prosody-intent mismatch",
   "A cloned voice reading a threat script says frightening things in a flat voice. The system measures lexical pressure per turn from the weighted fraud lexicon and acoustic arousal from pitch movement, energy dynamics and speaking-rate variation, then scores the gap between them. Neither branch can compute this on its own, which is what separates it from ordinary late fusion of two scores."],
  ["Coercion trajectory",
   "A scam call is a script with a shape: introduce an authority, state a problem, threaten, attach a deadline, isolate the victim, extract. Each turn is tagged with a dialogue act, each act carries a pressure rank, and the fitted slope over the call plus the act-sequence likelihood ratio under scam and benign Markov models become features. A bag of words cannot see order; this does."],
  ["Code-switching as a fraud signal",
   "Indian fraud scripts keep the emotional and threatening content in Hindi while the financial and technical vocabulary stays English. Per-token language identification gives a code-mixing index, switch-point entropy, and the share of fraud entities carried by English. That last quantity is the one that actually discriminates, and it is not a feature anyone has used for fraud detection before."],
  ["Time to detection",
   "Accuracy on a finished recording is the wrong question for a system meant to interrupt a live call. The pipeline re-scores after every turn and reports how many seconds of audio it needed before crossing the alert threshold. It is an operational capability and an evaluation metric at the same time."],
  ["CallForge corpus generator",
   "A parametric generator produces code-mixed scam and benign dialogues with gold BIO entities, gold dialogue acts and gold per-token language tags, all emitted by construction rather than annotated by hand. It also renders paired human-sounding and synthetic-sounding audio, so the four-cell evaluation grid can be populated on a laptop with no downloads."],
];

const ETHICS = `
<p>This is a detector. Nothing in the repository automates a scam, generates a
convincing clone of a real person on request, or helps anyone place calls.</p>
<h3>Voices</h3>
<ul>
<li>The offline audio is produced by a source-filter synthesiser in this repository. It is not a clone of anybody.</li>
<li>The Colab voice-cloning notebook is limited to a speaker's own recording with written consent, which is the condition stated in the project's research document.</li>
<li>Cloned audio never leaves the project and is deleted after the assessment.</li>
</ul>
<h3>Content</h3>
<ul>
<li>No real bank, no real UPI handle, no real phone number, no real case number appears anywhere in the generated corpus.</li>
<li>Scripts are fictional and written to be recognisable as fraud patterns, not to be usable as fraud scripts.</li>
</ul>
<h3>Where this fails</h3>
<ul>
<li>A false positive on a legitimate synthetic voice has a real cost. Automated bank IVR, accessibility voices and recorded reminders are all synthetic and all legitimate, which is exactly why the intent branch has to be there.</li>
<li>The offline results come from generated audio. They demonstrate that the method works; they are not a claim about performance on real calls. The Colab notebooks run the same code on ASVspoof and In-the-Wild for that.</li>
<li>Hinglish transcription is the weakest link. The transcript-in mode exists because ASR error, not model error, is what usually breaks this kind of pipeline.</li>
</ul>`;

const SYLLABUS = [
  ["M1", "Regex, POS, pattern rules", "Urgency and threat regex patterns in the rule baseline; the coarse POS tagger feeding CRF features"],
  ["M2", "N-grams, TF-IDF, embeddings", "Word 1-2 grams plus character 3-5 grams for the intent classifier, which is what handles romanised spelling variation"],
  ["M3", "NER, CRF, LSTM, sentiment", "Scam-entity tagset, a linear-chain CRF implemented from scratch with forward-backward and Viterbi, and a BiLSTM-CRF tagger for comparison"],
  ["M4", "Attention, encoder-decoder", "MuRIL and DistilBERT intent fine-tuning in the Colab notebook, with attention weights shown as suspicious spans"],
  ["M5", "Short-time analysis, energy, ZCR, STFT", "Framing, windowing, pre-emphasis, short-time energy and zero-crossing rate, and the energy-plus-ZCR voice activity detector"],
  ["M6", "MFCC, GFCC, LPCC, i-vector", "All five cepstral front ends written by hand, plus a TDNN speaker embedding standing in for the i-vector idea"],
  ["M7", "ASR, speaker recognition, DNN, metrics", "Whisper transcription with word error rate against gold scripts, RawNet-lite, EER and minimum t-DCF"],
];

function drawArch() {
  const host = $("sys-arch");
  if (host) host.innerHTML = archSvg();
}

export default {
  async init(state) {
    drawArch();

    $("sys-novelty").innerHTML = NOVELTY.map(([h, b]) =>
      `<h3>${h}</h3><p>${b}</p>`).join("");

    $("sys-ethics").innerHTML = ETHICS;

    const h = state.health || {};
    const meta = state.meta || {};
    const kv = $("sys-runtime");
    const entries = [
      ["Version", h.version || "n/a"],
      ["Python", h.python || "n/a"],
      ["Anti-spoof backend", h.antispoof_backend || "not loaded"],
      ["NER backend", h.ner_backend || "not loaded"],
      ["Intent backend", h.intent_backend || "not loaded"],
      ["Fusion model", h.fusion_model || "not trained"],
      ["ASR backend", h.asr_backend || "gold transcripts"],
      ["ffmpeg", h.ffmpeg ? "found, real GSM and AMR" : "absent, G.711 in numpy"],
      ["Torch", h.torch || "n/a"],
      ["Corpus calls", String(state.calls?.length ?? 0)],
      ["Alert threshold", fmt(h.alert_threshold ?? 0.65, 2)],
      ["Random seed", String(h.seed ?? "n/a")],
    ];
    kv.innerHTML = entries.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");

    const feats = meta.fusion_features || [];
    const groupColor = { voice: "--series-2", intent: "--series-1", cross: "--series-7" };
    table($("sys-features"), [
      { key: "group", label: "Branch", fmt: (v) =>
        `<span class="chip"><span class="sw" style="background:${cssVar(groupColor[v] || "--series-1")}"></span>${v}</span>` },
      { key: "name", label: "Feature", fmt: (v) => `<span class="mono xs">${v}</span>` },
      { key: "label", label: "Meaning" },
    ], feats);

    table($("sys-syllabus"), [
      { key: "m", label: "Module" },
      { key: "topic", label: "Topic" },
      { key: "where", label: "Where it is used" },
    ], SYLLABUS.map(([m, topic, where]) => ({ m, topic, where })));
  },

  // The diagram is an SVG string with the theme colours already resolved into
  // it, so it has to be rebuilt when the theme flips.
  refresh() { drawArch(); },
};
