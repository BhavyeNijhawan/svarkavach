/* SVG chart primitives for the SwarKavach console.
 *
 * No chart library and no CDN: the console has to run with the laptop offline.
 * Everything here draws into an SVG element sized to its container and
 * redraws on resize.
 *
 * House rules baked in, so callers cannot get them wrong:
 *   - 2px lines, >=8px markers with a 2px surface ring
 *   - bars capped at 24px with a 4px rounded data end
 *   - hairline solid gridlines, recessive
 *   - a legend whenever there are two or more series
 *   - text always uses text tokens, never the series color
 *   - one y scale per chart, never two
 */

const NS = "http://www.w3.org/2000/svg";

export const SERIES = ["--series-1", "--series-2", "--series-3", "--series-4",
                       "--series-5", "--series-6", "--series-7", "--series-8"];
export const STATUS = {
  low: "--status-good",
  elevated: "--status-warning",
  high: "--status-serious",
  critical: "--status-critical",
};

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
export function seriesColor(i) { return cssVar(SERIES[i % SERIES.length]); }
export function bandColor(band) { return cssVar(STATUS[band] || "--text-3"); }

function el(tag, attrs = {}, parent = null) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined) continue;
    n.setAttribute(k, v);
  }
  if (parent) parent.appendChild(n);
  return n;
}

function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

/* ------------------------------------------------------------- tooltip */

let TIP = null;
function tip() {
  if (!TIP) {
    TIP = document.createElement("div");
    TIP.className = "tip";
    document.body.appendChild(TIP);
  }
  return TIP;
}
export function showTip(html, ev) {
  const t = tip();
  t.innerHTML = html;
  t.classList.add("show");
  const pad = 14;
  const r = t.getBoundingClientRect();
  let x = ev.clientX + pad;
  let y = ev.clientY + pad;
  if (x + r.width > window.innerWidth - 8) x = ev.clientX - r.width - pad;
  if (y + r.height > window.innerHeight - 8) y = ev.clientY - r.height - pad;
  t.style.left = Math.max(8, x) + "px";
  t.style.top = Math.max(8, y) + "px";
}
export function hideTip() { if (TIP) TIP.classList.remove("show"); }

/* --------------------------------------------------------------- scales */

function linear(d0, d1, r0, r1) {
  const span = (d1 - d0) || 1;
  const f = (v) => r0 + ((v - d0) / span) * (r1 - r0);
  f.invert = (p) => d0 + ((p - r0) / ((r1 - r0) || 1)) * span;
  f.domain = [d0, d1];
  f.range = [r0, r1];
  return f;
}

function niceTicks(lo, hi, count = 5) {
  if (!isFinite(lo) || !isFinite(hi)) return [0, 1];
  if (lo === hi) { lo -= 0.5; hi += 0.5; }
  const span = hi - lo;
  const raw = span / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 7.5 ? 10 : norm >= 3.5 ? 5 : norm >= 1.5 ? 2 : 1) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
    out.push(Math.abs(v) < step * 1e-9 ? 0 : +v.toFixed(10));
  }
  return out;
}

export function fmt(v, dp = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return "n/a";
  if (Math.abs(v) >= 10000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return (+v).toFixed(dp);
}
export function pct(v, dp = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return "n/a";
  return (v * 100).toFixed(dp) + "%";
}

/* --------------------------------------------------------------- canvas */

function frame(host, opts = {}) {
  const width = host.clientWidth || opts.width || 480;
  const height = opts.height || 240;
  clear(host);
  const svg = el("svg", {
    class: "chart", width, height, viewBox: `0 0 ${width} ${height}`,
    role: "img", "aria-label": opts.label || "chart",
  }, host);
  const m = Object.assign({ t: 14, r: 16, b: 30, l: 44 }, opts.margin || {});
  const iw = Math.max(10, width - m.l - m.r);
  const ih = Math.max(10, height - m.t - m.b);
  const g = el("g", { transform: `translate(${m.l},${m.t})` }, svg);
  return { svg, g, width, height, m, iw, ih };
}

function axes(F, x, y, opts = {}) {
  const { g, iw, ih } = F;
  const yt = opts.yTicks || niceTicks(y.domain[0], y.domain[1], opts.yTickCount || 4);
  for (const v of yt) {
    const py = y(v);
    if (py < -1 || py > ih + 1) continue;
    el("line", { class: "grid-line", x1: 0, x2: iw, y1: py, y2: py }, g);
    el("text", {
      class: "tick-text", x: -8, y: py + 3.5, "text-anchor": "end",
    }, g).textContent = opts.yFmt ? opts.yFmt(v) : fmt(v, opts.yDp ?? 1);
  }
  el("line", { class: "axis-line", x1: 0, x2: 0, y1: 0, y2: ih }, g);
  el("line", { class: "axis-line", x1: 0, x2: iw, y1: ih, y2: ih }, g);

  if (opts.xTicks) {
    for (const t of opts.xTicks) {
      const px = x(t.v ?? t);
      el("text", {
        class: "tick-text", x: px, y: ih + 15, "text-anchor": "middle",
      }, g).textContent = t.label ?? (opts.xFmt ? opts.xFmt(t) : fmt(t, 0));
    }
  }
  if (opts.yTitle) {
    el("text", {
      class: "axis-title", transform: `translate(${-F.m.l + 11},${ih / 2}) rotate(-90)`,
      "text-anchor": "middle",
    }, g).textContent = opts.yTitle;
  }
  if (opts.xTitle) {
    el("text", {
      class: "axis-title", x: iw / 2, y: ih + F.m.b - 2, "text-anchor": "middle",
    }, g).textContent = opts.xTitle;
  }
  return yt;
}

/* ---------------------------------------------------------------- legend */

export function legend(host, items, kind = "key") {
  const box = document.createElement("div");
  box.className = "legend";
  for (const it of items) {
    const d = document.createElement("span");
    d.className = "item";
    const k = document.createElement("span");
    k.className = "key" + (kind === "line" ? " line" : "");
    k.style.background = it.color;
    d.appendChild(k);
    d.appendChild(document.createTextNode(it.label));
    box.appendChild(d);
  }
  host.appendChild(box);
  return box;
}

/* ------------------------------------------------------------ line chart */

/** series: [{name, color, points: [[x,y],...], dash}] */
export function lineChart(host, cfg) {
  const F = frame(host, cfg);
  const { g, iw, ih } = F;
  const all = cfg.series.flatMap((s) => s.points);
  if (!all.length) { emptyState(host, cfg.empty || "No data"); return; }

  const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
  const xd = cfg.xDomain || [Math.min(...xs), Math.max(...xs)];
  const yd = cfg.yDomain || [Math.min(0, Math.min(...ys)), Math.max(...ys)];
  const x = linear(xd[0], xd[1], 0, iw);
  const y = linear(yd[0], yd[1], ih, 0);

  const xt = niceTicks(xd[0], xd[1], cfg.xTickCount || 5)
    .map((v) => ({ v, label: cfg.xFmt ? cfg.xFmt(v) : fmt(v, cfg.xDp ?? 0) }));
  axes(F, x, y, { ...cfg, xTicks: xt });

  // reference bands (for example the alert threshold)
  for (const r of cfg.refs || []) {
    const py = y(r.value);
    el("line", { class: "ref-line", x1: 0, x2: iw, y1: py, y2: py,
                 stroke: r.color || undefined }, g);
    if (r.label) {
      el("text", { class: "tick-text", x: iw - 2, y: py - 5, "text-anchor": "end",
                   fill: r.color || undefined }, g).textContent = r.label;
    }
  }

  cfg.series.forEach((s, i) => {
    const color = s.color || seriesColor(i);
    const pts = s.points.slice().sort((a, b) => a[0] - b[0]);
    const d = pts.map((p, j) => `${j ? "L" : "M"}${x(p[0]).toFixed(2)},${y(p[1]).toFixed(2)}`).join("");
    if (cfg.area && pts.length) {
      const base = y(Math.max(yd[0], 0));
      el("path", {
        class: "series-area", fill: color,
        d: `${d}L${x(pts[pts.length - 1][0]).toFixed(2)},${base}L${x(pts[0][0]).toFixed(2)},${base}Z`,
      }, g);
    }
    el("path", {
      class: "series-line", d, stroke: color,
      "stroke-dasharray": s.dash || null,
    }, g);
    if (cfg.dots !== false && pts.length <= 40) {
      for (const p of pts) {
        el("circle", { class: "dot", cx: x(p[0]), cy: y(p[1]), r: 4, fill: color }, g);
      }
    }
    // direct label at the line end, which keeps identity off color alone
    if (cfg.endLabels && pts.length) {
      const last = pts[pts.length - 1];
      el("text", {
        class: "label-text", x: x(last[0]) + 7, y: y(last[1]) + 3.5,
      }, g).textContent = s.name;
    }
  });

  // crosshair
  const hl = el("line", { class: "hover-line", x1: 0, x2: 0, y1: 0, y2: ih, opacity: 0 }, g);
  const hitRect = el("rect", { x: 0, y: 0, width: iw, height: ih, fill: "transparent" }, g);
  hitRect.addEventListener("mousemove", (ev) => {
    const bb = F.svg.getBoundingClientRect();
    const px = ev.clientX - bb.left - F.m.l;
    const xv = x.invert(px);
    hl.setAttribute("x1", px); hl.setAttribute("x2", px); hl.setAttribute("opacity", 1);
    const rows = cfg.series.map((s, i) => {
      if (!s.points.length) return null;
      let best = s.points[0];
      for (const p of s.points) if (Math.abs(p[0] - xv) < Math.abs(best[0] - xv)) best = p;
      return { name: s.name, color: s.color || seriesColor(i), v: best[1], x: best[0] };
    }).filter(Boolean);
    const head = cfg.xFmt ? cfg.xFmt(rows[0]?.x ?? xv) : fmt(rows[0]?.x ?? xv, cfg.xDp ?? 1);
    showTip(
      `<div class="th">${cfg.xLabelPrefix || ""}${head}</div>` +
      rows.map((r) => `<div class="tr"><span class="k"><span class="sw" style="background:${r.color}"></span>${r.name}</span><span class="v">${cfg.tipFmt ? cfg.tipFmt(r.v) : fmt(r.v, 3)}</span></div>`).join(""),
      ev
    );
  });
  hitRect.addEventListener("mouseleave", () => { hl.setAttribute("opacity", 0); hideTip(); });

  if (cfg.series.length >= 2 && cfg.legend !== false) {
    legend(host, cfg.series.map((s, i) => ({ label: s.name, color: s.color || seriesColor(i) })), "line");
  }
  return F;
}

/* ------------------------------------------------------------ bar chart */

/** items: [{label, value, color, note}] ; grouped: series of items */
export function barChart(host, cfg) {
  const items = cfg.items || [];
  if (!items.length) { emptyState(host, cfg.empty || "No data"); return; }
  const horizontal = cfg.horizontal !== false;
  const F = frame(host, {
    ...cfg,
    margin: cfg.margin || (horizontal
      ? { t: 10, r: 52, b: 24, l: cfg.labelWidth || 120 }
      : { t: 14, r: 12, b: 46, l: 44 }),
  });
  const { g, iw, ih } = F;
  const vals = items.map((d) => d.value);
  const maxV = cfg.max ?? ((Math.max(0, ...vals) * 1.05) || 1);
  const minV = Math.min(0, ...vals);

  if (horizontal) {
    const x = linear(minV, maxV, 0, iw);
    const band = ih / items.length;
    const thick = Math.min(24, band * 0.62);
    for (const t of niceTicks(minV, maxV, 4)) {
      el("line", { class: "grid-line", x1: x(t), x2: x(t), y1: 0, y2: ih }, g);
      el("text", { class: "tick-text", x: x(t), y: ih + 15, "text-anchor": "middle" }, g)
        .textContent = cfg.vFmt ? cfg.vFmt(t) : fmt(t, cfg.dp ?? 1);
    }
    const zero = x(Math.max(minV, 0));
    el("line", { class: "axis-line", x1: zero, x2: zero, y1: 0, y2: ih }, g);
    items.forEach((d, i) => {
      const cy = i * band + band / 2;
      const color = d.color || seriesColor(cfg.colorIndex ?? 0);
      const w = Math.abs(x(d.value) - zero);
      const bx = d.value >= 0 ? zero : x(d.value);
      // 4px rounded data end, square at the baseline
      const r = Math.min(4, w);
      const path = d.value >= 0
        ? `M${bx},${cy - thick / 2}h${Math.max(0, w - r)}a${r},${r} 0 0 1 ${r},${r}v${thick - 2 * r}a${r},${r} 0 0 1 ${-r},${r}h${-Math.max(0, w - r)}z`
        : `M${bx + w},${cy - thick / 2}h${-Math.max(0, w - r)}a${r},${r} 0 0 0 ${-r},${r}v${thick - 2 * r}a${r},${r} 0 0 0 ${r},${r}h${Math.max(0, w - r)}z`;
      const bar = el("path", { class: "bar", d: path, fill: color }, g);
      el("text", { class: "tick-text", x: -10, y: cy + 3.5, "text-anchor": "end" }, g)
        .textContent = d.label;
      el("text", {
        class: "label-text", x: (d.value >= 0 ? bx + w + 6 : bx - 6),
        y: cy + 3.5, "text-anchor": d.value >= 0 ? "start" : "end",
      }, g).textContent = cfg.vFmt ? cfg.vFmt(d.value) : fmt(d.value, cfg.dp ?? 2);
      bar.addEventListener("mousemove", (ev) => showTip(
        `<div class="th">${d.label}</div><div class="tr"><span class="k">${cfg.metric || "value"}</span><span class="v">${cfg.vFmt ? cfg.vFmt(d.value) : fmt(d.value, cfg.dp ?? 3)}</span></div>` +
        (d.note ? `<div class="tr"><span class="k">${d.note}</span></div>` : ""), ev));
      bar.addEventListener("mouseleave", hideTip);
    });
  } else {
    const y = linear(minV, maxV, ih, 0);
    const band = iw / items.length;
    const thick = Math.min(24, band * 0.6);
    axes(F, linear(0, 1, 0, iw), y, { ...cfg, yFmt: cfg.vFmt });
    items.forEach((d, i) => {
      const cx = i * band + band / 2;
      const color = d.color || seriesColor(cfg.colorIndex ?? 0);
      const y0 = y(Math.max(minV, 0)), y1 = y(d.value);
      const h = Math.abs(y1 - y0);
      const r = Math.min(4, h);
      const top = Math.min(y0, y1);
      const path = `M${cx - thick / 2},${top + h}v${-(h - r)}a${r},${r} 0 0 1 ${r},${-r}h${thick - 2 * r}a${r},${r} 0 0 1 ${r},${r}v${h - r}z`;
      const bar = el("path", { class: "bar", d: path, fill: color }, g);
      el("text", {
        class: "tick-text", x: cx, y: ih + 15, "text-anchor": "middle",
        transform: cfg.rotateLabels ? `rotate(-32 ${cx} ${ih + 15})` : null,
      }, g).textContent = d.label;
      el("text", { class: "label-text", x: cx, y: top - 6, "text-anchor": "middle" }, g)
        .textContent = cfg.vFmt ? cfg.vFmt(d.value) : fmt(d.value, cfg.dp ?? 2);
      bar.addEventListener("mousemove", (ev) => showTip(
        `<div class="th">${d.label}</div><div class="tr"><span class="k">${cfg.metric || "value"}</span><span class="v">${cfg.vFmt ? cfg.vFmt(d.value) : fmt(d.value, cfg.dp ?? 3)}</span></div>`, ev));
      bar.addEventListener("mouseleave", hideTip);
    });
  }
  return F;
}

/* ------------------------------------------------------- grouped columns */

/** groups: [{label, values:[..]}], series: [{name,color}] */
export function groupedBars(host, cfg) {
  const F = frame(host, { ...cfg, margin: { t: 14, r: 12, b: 46, l: 46 } });
  const { g, iw, ih } = F;
  const flat = cfg.groups.flatMap((gr) => gr.values);
  if (!flat.length) { emptyState(host, "No data"); return; }
  const maxV = cfg.max ?? Math.max(...flat) * 1.12;
  const y = linear(0, maxV, ih, 0);
  axes(F, linear(0, 1, 0, iw), y, { ...cfg, yFmt: cfg.vFmt });

  const band = iw / cfg.groups.length;
  const n = cfg.series.length;
  // 2px surface gap between adjacent bars is what separates them
  const inner = Math.min(band * 0.78, n * 26);
  const bw = Math.min(24, (inner - (n - 1) * 2) / n);

  cfg.groups.forEach((gr, gi) => {
    const x0 = gi * band + (band - inner) / 2;
    gr.values.forEach((v, si) => {
      const color = cfg.series[si].color || seriesColor(si);
      const bx = x0 + si * (bw + 2);
      const h = Math.max(0, ih - y(v));
      const r = Math.min(4, h);
      const path = h < 0.5 ? "" :
        `M${bx},${ih}v${-(h - r)}a${r},${r} 0 0 1 ${r},${-r}h${bw - 2 * r}a${r},${r} 0 0 1 ${r},${r}v${h - r}z`;
      if (path) {
        const bar = el("path", { class: "bar", d: path, fill: color }, g);
        bar.addEventListener("mousemove", (ev) => showTip(
          `<div class="th">${gr.label}</div><div class="tr"><span class="k"><span class="sw" style="background:${color}"></span>${cfg.series[si].name}</span><span class="v">${cfg.vFmt ? cfg.vFmt(v) : fmt(v, 3)}</span></div>`, ev));
        bar.addEventListener("mouseleave", hideTip);
      }
    });
    el("text", {
      class: "tick-text", x: gi * band + band / 2, y: ih + 15, "text-anchor": "middle",
      transform: cfg.rotateLabels ? `rotate(-24 ${gi * band + band / 2} ${ih + 15})` : null,
    }, g).textContent = gr.label;
  });

  legend(host, cfg.series.map((s, i) => ({ label: s.name, color: s.color || seriesColor(i) })));
  return F;
}

/* ----------------------------------------------------------- risk gauge */

export function gauge(host, value, band, opts = {}) {
  const size = opts.size || 168;
  const stroke = opts.stroke || 13;
  clear(host);
  const wrap = document.createElement("div");
  wrap.className = "gauge";
  host.appendChild(wrap);
  const svg = el("svg", { width: size, height: size, viewBox: `0 0 ${size} ${size}` }, wrap);
  const cx = size / 2, cy = size / 2, r = (size - stroke) / 2 - 2;
  const START = 135, SWEEP = 270;

  const pol = (deg) => {
    const rad = (deg - 90) * Math.PI / 180;
    return [cx + r * Math.cos(rad), cy + r * Math.sin(rad)];
  };
  const arc = (a0, a1) => {
    const [x0, y0] = pol(a0), [x1, y1] = pol(a1);
    return `M${x0},${y0}A${r},${r} 0 ${a1 - a0 > 180 ? 1 : 0} 1 ${x1},${y1}`;
  };

  el("path", {
    d: arc(START, START + SWEEP), fill: "none",
    stroke: cssVar("--surface-3"), "stroke-width": stroke, "stroke-linecap": "round",
  }, svg);

  // band ticks so the number has context without a second scale
  for (const t of [0.35, 0.65, 0.85]) {
    const a = START + SWEEP * t;
    const [tx, ty] = pol(a);
    const [ix, iy] = [cx + (r - stroke / 2 - 3) * Math.cos((a - 90) * Math.PI / 180),
                      cy + (r - stroke / 2 - 3) * Math.sin((a - 90) * Math.PI / 180)];
    el("line", { x1: ix, y1: iy, x2: tx, y2: ty, stroke: cssVar("--text-3"),
                 "stroke-width": 1, opacity: .5 }, svg);
  }

  const v = Math.max(0, Math.min(1, value ?? 0));
  const color = bandColor(band);
  const p = el("path", {
    d: arc(START, START + SWEEP * Math.max(v, 0.001)), fill: "none",
    stroke: color, "stroke-width": stroke, "stroke-linecap": "round",
  }, svg);
  const len = p.getTotalLength();
  p.style.strokeDasharray = `${len} ${len}`;
  p.style.strokeDashoffset = String(len);
  p.getBoundingClientRect();
  p.style.transition = "stroke-dashoffset .8s cubic-bezier(.2,.7,.3,1)";
  p.style.strokeDashoffset = "0";

  const read = document.createElement("div");
  read.className = "readout";
  read.innerHTML =
    `<div class="num" style="color:${color}">${(v * 100).toFixed(0)}</div>` +
    `<div class="lbl">${opts.label || "fraud risk"}</div>`;
  wrap.appendChild(read);
  return wrap;
}

/* --------------------------------------------------------------- meters */

export function meter(host, { name, value, sub, color, max = 1 }) {
  const d = document.createElement("div");
  d.className = "meter";
  const c = color || cssVar("--series-1");
  d.innerHTML =
    `<div class="top"><span class="name">${name}</span>` +
    `<span class="val" style="color:${c}">${(value * 100).toFixed(0)}<span class="muted" style="font-size:10px">%</span></span></div>` +
    `<div class="track"><div class="fill" style="width:0%;background:${c}"></div></div>` +
    (sub ? `<div class="sub">${sub}</div>` : "");
  host.appendChild(d);
  requestAnimationFrame(() => {
    d.querySelector(".fill").style.width = (100 * Math.max(0, Math.min(1, value / max))) + "%";
  });
  return d;
}

/* -------------------------------------------------------------- heatmap */

/** rows: [label], cols: [label], values: [[..]] ; sequential single hue */
export function heatmap(host, cfg) {
  const rows = cfg.rows, cols = cfg.cols, V = cfg.values;
  const cellH = cfg.cellH || 42;
  const labelW = cfg.labelW || 118;
  const headH = 26;
  const width = host.clientWidth || 480;
  clear(host);
  const height = headH + rows.length * cellH + 8;
  const svg = el("svg", { class: "chart", width, height, viewBox: `0 0 ${width} ${height}` }, host);
  const gridW = Math.max(60, width - labelW - 8);
  const cellW = gridW / cols.length;

  const flat = V.flat().filter((v) => v !== null && v !== undefined && !Number.isNaN(v));
  const lo = cfg.min ?? Math.min(...flat), hi = cfg.max ?? Math.max(...flat);
  const ramp = ["--seq-100", "--seq-250", "--seq-400", "--seq-550", "--seq-700"].map(cssVar);
  const colorOf = (v) => {
    if (v === null || v === undefined || Number.isNaN(v)) return cssVar("--surface-2");
    let t = (v - lo) / ((hi - lo) || 1);
    if (cfg.reverse) t = 1 - t;
    t = Math.max(0, Math.min(1, t));
    const i = Math.min(ramp.length - 1, Math.floor(t * (ramp.length - 1)));
    return ramp[i];
  };

  cols.forEach((c, j) => {
    el("text", {
      class: "tick-text", x: labelW + j * cellW + cellW / 2, y: 16, "text-anchor": "middle",
    }, svg).textContent = c;
  });

  rows.forEach((rlab, i) => {
    el("text", {
      class: "tick-text", x: labelW - 10, y: headH + i * cellH + cellH / 2 + 4,
      "text-anchor": "end",
    }, svg).textContent = rlab;
    cols.forEach((clab, j) => {
      const v = V[i][j];
      // 2px surface gap does the separating, no strokes
      const rect = el("rect", {
        x: labelW + j * cellW + 1, y: headH + i * cellH + 1,
        width: cellW - 2, height: cellH - 2, rx: 5, fill: colorOf(v),
      }, svg);
      const t = (v - lo) / ((hi - lo) || 1);
      const dark = (cfg.reverse ? 1 - t : t) > 0.55;
      el("text", {
        class: "cell-label", x: labelW + j * cellW + cellW / 2,
        y: headH + i * cellH + cellH / 2 + 4, "text-anchor": "middle",
        fill: dark ? "#ffffff" : cssVar("--text-1"),
      }, svg).textContent = v === null || v === undefined || Number.isNaN(v)
        ? "n/a" : (cfg.vFmt ? cfg.vFmt(v) : fmt(v, cfg.dp ?? 2));
      rect.addEventListener("mousemove", (ev) => showTip(
        `<div class="th">${rlab} / ${clab}</div><div class="tr"><span class="k">${cfg.metric || "value"}</span><span class="v">${cfg.vFmt ? cfg.vFmt(v) : fmt(v, 3)}</span></div>`, ev));
      rect.addEventListener("mouseleave", hideTip);
    });
  });
  return svg;
}

/* ------------------------------------------------------------ waterfall */

/** contributions: [{label, contribution, value, group}] sorted by |contribution| */
export function waterfall(host, cfg) {
  const items = (cfg.items || []).slice(0, cfg.limit || 10);
  if (!items.length) { emptyState(host, "No contributions to show"); return; }
  const rowH = 26;
  const F = frame(host, {
    height: items.length * rowH + 34,
    margin: { t: 8, r: 58, b: 24, l: cfg.labelWidth || 168 },
  });
  const { g, iw, ih } = F;
  const mx = Math.max(...items.map((d) => Math.abs(d.contribution))) * 1.1 || 1;
  const x = linear(-mx, mx, 0, iw);
  const zero = x(0);

  for (const t of niceTicks(-mx, mx, 4)) {
    el("line", { class: "grid-line", x1: x(t), x2: x(t), y1: 0, y2: ih }, g);
    el("text", { class: "tick-text", x: x(t), y: ih + 15, "text-anchor": "middle" }, g)
      .textContent = fmt(t, 1);
  }
  el("line", { class: "axis-line", x1: zero, x2: zero, y1: 0, y2: ih }, g);

  const up = cssVar("--status-critical"), down = cssVar("--status-good");
  items.forEach((d, i) => {
    const cy = i * rowH + rowH / 2;
    const thick = Math.min(16, rowH - 9);
    const w = Math.abs(x(d.contribution) - zero);
    const bx = d.contribution >= 0 ? zero : x(d.contribution);
    const color = d.contribution >= 0 ? up : down;
    const r = Math.min(4, w);
    const path = d.contribution >= 0
      ? `M${bx},${cy - thick / 2}h${Math.max(0, w - r)}a${r},${r} 0 0 1 ${r},${r}v${thick - 2 * r}a${r},${r} 0 0 1 ${-r},${r}h${-Math.max(0, w - r)}z`
      : `M${bx + w},${cy - thick / 2}h${-Math.max(0, w - r)}a${r},${r} 0 0 0 ${-r},${r}v${thick - 2 * r}a${r},${r} 0 0 0 ${r},${r}h${Math.max(0, w - r)}z`;
    const bar = el("path", { d: path, fill: color }, g);
    el("text", { class: "tick-text", x: -10, y: cy + 3.5, "text-anchor": "end" }, g)
      .textContent = d.label;
    el("text", {
      class: "label-text", x: d.contribution >= 0 ? bx + w + 6 : bx - 6, y: cy + 3.5,
      "text-anchor": d.contribution >= 0 ? "start" : "end",
    }, g).textContent = (d.contribution >= 0 ? "+" : "") + fmt(d.contribution, 2);
    bar.addEventListener("mousemove", (ev) => showTip(
      `<div class="th">${d.label}</div>` +
      `<div class="tr"><span class="k">feature value</span><span class="v">${fmt(d.value, 3)}</span></div>` +
      `<div class="tr"><span class="k">log-odds push</span><span class="v">${(d.contribution >= 0 ? "+" : "") + fmt(d.contribution, 3)}</span></div>` +
      (d.group ? `<div class="tr"><span class="k">branch</span><span class="v">${d.group}</span></div>` : ""), ev));
    bar.addEventListener("mouseleave", hideTip);
  });

  legend(host, [
    { label: "pushes toward fraud", color: up },
    { label: "pushes toward legitimate", color: down },
  ]);
  return F;
}

/* ----------------------------------------------------------- DET / ROC */

export function detCurve(host, cfg) {
  const F = frame(host, { ...cfg, margin: { t: 12, r: 14, b: 44, l: 50 } });
  const { g, iw, ih } = F;
  const x = linear(0, cfg.maxX ?? 1, 0, iw);
  const y = linear(0, cfg.maxY ?? 1, ih, 0);
  axes(F, x, y, {
    ...cfg,
    xTicks: niceTicks(0, cfg.maxX ?? 1, 5).map((v) => ({ v, label: pct(v, 0) })),
    yFmt: (v) => pct(v, 0),
  });
  // chance line
  el("line", { class: "ref-line", x1: x(0), y1: y(0), x2: x(cfg.maxX ?? 1), y2: y(cfg.maxY ?? 1) }, g);

  (cfg.series || []).forEach((s, i) => {
    const color = s.color || seriesColor(i);
    const d = s.points.map((p, j) => `${j ? "L" : "M"}${x(p[0]).toFixed(2)},${y(p[1]).toFixed(2)}`).join("");
    el("path", { class: "series-line", d, stroke: color }, g);
    if (s.eer !== undefined && s.eer !== null) {
      el("circle", { class: "dot", cx: x(s.eer), cy: y(s.eer), r: 4.5, fill: color }, g);
    }
  });
  legend(host, (cfg.series || []).map((s, i) => ({
    label: s.eer !== undefined ? `${s.name} (EER ${pct(s.eer, 2)})` : s.name,
    color: s.color || seriesColor(i),
  })), "line");
  return F;
}

/* ------------------------------------------------------------- scatter */

export function scatter(host, cfg) {
  const F = frame(host, { ...cfg, margin: { t: 12, r: 16, b: 44, l: 48 } });
  const { g, iw, ih } = F;
  const all = cfg.groups.flatMap((s) => s.points);
  if (!all.length) { emptyState(host, "No data"); return; }
  const xd = cfg.xDomain || [Math.min(...all.map(p => p[0])), Math.max(...all.map(p => p[0]))];
  const yd = cfg.yDomain || [Math.min(...all.map(p => p[1])), Math.max(...all.map(p => p[1]))];
  const x = linear(xd[0], xd[1], 0, iw);
  const y = linear(yd[0], yd[1], ih, 0);
  axes(F, x, y, {
    ...cfg,
    xTicks: niceTicks(xd[0], xd[1], 5).map((v) => ({ v, label: fmt(v, cfg.xDp ?? 1) })),
  });
  if (cfg.diagonal) {
    el("line", { class: "ref-line", x1: x(xd[0]), y1: y(xd[0]), x2: x(xd[1]), y2: y(xd[1]) }, g);
  }
  cfg.groups.forEach((s, i) => {
    const color = s.color || seriesColor(i);
    for (const p of s.points) {
      const c = el("circle", {
        class: "dot", cx: x(p[0]), cy: y(p[1]), r: 4, fill: color, "fill-opacity": .85,
      }, g);
      c.addEventListener("mousemove", (ev) => showTip(
        `<div class="th">${p[2] || s.name}</div>` +
        `<div class="tr"><span class="k">${cfg.xTitle || "x"}</span><span class="v">${fmt(p[0], 3)}</span></div>` +
        `<div class="tr"><span class="k">${cfg.yTitle || "y"}</span><span class="v">${fmt(p[1], 3)}</span></div>`, ev));
      c.addEventListener("mouseleave", hideTip);
    }
  });
  if (cfg.groups.length >= 2) {
    legend(host, cfg.groups.map((s, i) => ({ label: s.name, color: s.color || seriesColor(i) })));
  }
  return F;
}

/* ------------------------------------------------------------ histogram */

export function histogram(host, cfg) {
  const values = cfg.values || [];
  if (!values.length) { emptyState(host, cfg.empty || "No data"); return; }
  const nb = cfg.bins || 14;
  const lo = cfg.min ?? Math.min(...values), hi = cfg.max ?? Math.max(...values);
  const w = (hi - lo) / nb || 1;
  const counts = new Array(nb).fill(0);
  for (const v of values) {
    counts[Math.max(0, Math.min(nb - 1, Math.floor((v - lo) / w)))]++;
  }
  const F = frame(host, { ...cfg, margin: { t: 12, r: 14, b: 42, l: 44 } });
  const { g, iw, ih } = F;
  const x = linear(lo, hi, 0, iw);
  const y = linear(0, Math.max(...counts) * 1.1 || 1, ih, 0);
  axes(F, x, y, {
    ...cfg, yDp: 0,
    xTicks: niceTicks(lo, hi, 5).map((v) => ({ v, label: fmt(v, cfg.xDp ?? 1) })),
  });
  const color = cfg.color || seriesColor(0);
  counts.forEach((c, i) => {
    if (!c) return;
    const bx = x(lo + i * w) + 1;           // the 2px surface gap between bars
    const bw = Math.max(1, (iw / nb) - 2);
    const h = ih - y(c);
    const r = Math.min(4, h, bw / 2);
    const bar = el("path", {
      d: `M${bx},${ih}v${-(h - r)}a${r},${r} 0 0 1 ${r},${-r}h${bw - 2 * r}a${r},${r} 0 0 1 ${r},${r}v${h - r}z`,
      fill: color,
    }, g);
    bar.addEventListener("mousemove", (ev) => showTip(
      `<div class="th">${fmt(lo + i * w, 1)} to ${fmt(lo + (i + 1) * w, 1)}</div>` +
      `<div class="tr"><span class="k">calls</span><span class="v">${c}</span></div>`, ev));
    bar.addEventListener("mouseleave", hideTip);
  });
  for (const r of cfg.refs || []) {
    el("line", { class: "ref-line", x1: x(r.value), x2: x(r.value), y1: 0, y2: ih,
                 stroke: r.color || undefined }, g);
    if (r.label) {
      el("text", { class: "tick-text", x: x(r.value) + 4, y: 10, fill: r.color || undefined }, g)
        .textContent = r.label;
    }
  }
  return F;
}

/* ---------------------------------------------------------------- radar */

export function radar(host, cfg) {
  const size = cfg.size || Math.min(host.clientWidth || 300, 320);
  clear(host);
  const svg = el("svg", { class: "chart", width: size, height: size, viewBox: `0 0 ${size} ${size}` }, host);
  const cx = size / 2, cy = size / 2, R = size / 2 - 46;
  const axesN = cfg.axes.length;
  const ang = (i) => (i / axesN) * 2 * Math.PI - Math.PI / 2;

  for (const t of [0.25, 0.5, 0.75, 1]) {
    const pts = cfg.axes.map((_, i) =>
      `${cx + R * t * Math.cos(ang(i))},${cy + R * t * Math.sin(ang(i))}`).join(" ");
    el("polygon", { points: pts, fill: "none", class: "grid-line" }, svg);
  }
  cfg.axes.forEach((a, i) => {
    const ex = cx + R * Math.cos(ang(i)), ey = cy + R * Math.sin(ang(i));
    el("line", { class: "grid-line", x1: cx, y1: cy, x2: ex, y2: ey }, svg);
    const lx = cx + (R + 16) * Math.cos(ang(i)), ly = cy + (R + 16) * Math.sin(ang(i));
    el("text", {
      class: "tick-text", x: lx, y: ly + 3,
      "text-anchor": Math.abs(lx - cx) < 6 ? "middle" : (lx > cx ? "start" : "end"),
    }, svg).textContent = a;
  });

  cfg.series.forEach((s, si) => {
    const color = s.color || seriesColor(si);
    const pts = s.values.map((v, i) => {
      const t = Math.max(0, Math.min(1, v));
      return `${cx + R * t * Math.cos(ang(i))},${cy + R * t * Math.sin(ang(i))}`;
    }).join(" ");
    el("polygon", { points: pts, fill: color, "fill-opacity": .10, stroke: color,
                    "stroke-width": 2, "stroke-linejoin": "round" }, svg);
    s.values.forEach((v, i) => {
      const t = Math.max(0, Math.min(1, v));
      const px = cx + R * t * Math.cos(ang(i)), py = cy + R * t * Math.sin(ang(i));
      const c = el("circle", { class: "dot", cx: px, cy: py, r: 4, fill: color }, svg);
      c.addEventListener("mousemove", (ev) => showTip(
        `<div class="th">${cfg.axes[i]}</div><div class="tr"><span class="k"><span class="sw" style="background:${color}"></span>${s.name}</span><span class="v">${fmt(v, 3)}</span></div>`, ev));
      c.addEventListener("mouseleave", hideTip);
    });
  });
  if (cfg.series.length >= 2) {
    legend(host, cfg.series.map((s, i) => ({ label: s.name, color: s.color || seriesColor(i) })));
  }
  return svg;
}

/* ------------------------------------------------------------ sparkline */

export function sparkline(host, values, opts = {}) {
  const w = opts.width || host.clientWidth || 120;
  const h = opts.height || 28;
  clear(host);
  if (!values.length) return;
  const svg = el("svg", { class: "chart", width: w, height: h, viewBox: `0 0 ${w} ${h}` }, host);
  const lo = opts.min ?? Math.min(...values), hi = opts.max ?? Math.max(...values);
  const x = linear(0, Math.max(1, values.length - 1), 1, w - 1);
  const y = linear(lo, hi, h - 2, 2);
  const d = values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const color = opts.color || seriesColor(0);
  el("path", { d: `${d}L${x(values.length - 1)},${h}L${x(0)},${h}Z`, fill: color, opacity: .10 }, svg);
  el("path", { class: "series-line", d, stroke: color }, svg);
  el("circle", {
    class: "dot", cx: x(values.length - 1), cy: y(values[values.length - 1]), r: 3.5, fill: color,
  }, svg);
  return svg;
}

/* ------------------------------------------------------------- helpers */

export function emptyState(host, msg) {
  clear(host);
  const d = document.createElement("div");
  d.className = "empty";
  d.innerHTML =
    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M3 3v18h18"/><path d="M7 15l4-5 3 3 5-7"/></svg><div>${msg}</div>`;
  host.appendChild(d);
}

/** Redraw registered charts when the window resizes or the theme flips. */
const REDRAW = new Set();
export function responsive(fn) { REDRAW.add(fn); fn(); return fn; }
export function redrawAll() { for (const f of REDRAW) { try { f(); } catch (e) { console.warn(e); } } }
export function clearResponsive() { REDRAW.clear(); }

let rt = null;
window.addEventListener("resize", () => {
  clearTimeout(rt);
  rt = setTimeout(redrawAll, 140);
});
