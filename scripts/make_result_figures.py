"""Figures for the results chapter: console screenshots cropped per panel, and
one chart per result file drawn from data/results/*.json.

Formal print style: one serif-friendly sans face, two categorical hues at
most (navy, orange: the colour-vision-safe pair), a single-hue ramp for
magnitude, thin marks, direct labels, no dual axes.
"""
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "results"
FIG = ROOT / "docs" / "figures"
SHOTS = FIG / "shots"
FIG.mkdir(exist_ok=True)

NAVY, ORANGE, GREY, INK = "#1F3A5F", "#C8772E", "#9A9A9A", "#222222"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#444444",
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.grid": False,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def load(name):
    return json.load(io.open(RES / name, encoding="utf-8"))


# ------------------------------------------------------------- screenshots
def crop(src, box, out):
    im = Image.open(SHOTS / src)
    im.crop(box).save(FIG / out, optimize=True)
    print("crop", out, im.crop(box).size)


# results_light.png is 1440 x 2400; the sidebar ends at x = 232
crop("results_light.png", (240, 60, 1430, 675), "shot_ablation.png")
crop("results_light.png", (240, 680, 1430, 1035), "shot_antispoof.png")
crop("results_light.png", (240, 1040, 1430, 1490), "shot_ner_intent.png")
crop("results_light.png", (240, 1500, 1430, 1850), "shot_robust_ttd.png")
crop("results_light.png", (240, 1860, 1430, 2390), "shot_calibration.png")
crop("analysis_light.png", (240, 60, 1430, 1890), "shot_analysis.png")
crop("monitor_light.png", (240, 60, 1430, 1390), "shot_monitor.png")
crop("corpus_light.png", (240, 60, 1430, 1390), "shot_corpus.png")
crop("system_light.png", (240, 60, 1430, 1390), "shot_system.png")


# ------------------------------------------------------------------ charts
def label_bars(ax, bars, fmt="{:.3f}", dy=0.01):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=8, color=INK)


ab = load("ablation_results.json")
arms = ab["arms"]
names = {"audio_only": "Audio only", "text_only": "Text only",
         "late_fusion": "Late fusion", "full": "Full"}

# 1. ablation: AUC and F1 per arm
fig, ax = plt.subplots(figsize=(5.6, 2.8))
x = np.arange(len(arms)); w = 0.36
auc = [ab["overall"][a]["auc"] for a in arms]; f1 = [ab["overall"][a]["f1"] for a in arms]
b1 = ax.bar(x - w / 2, auc, w, color=NAVY, label="AUC")
b2 = ax.bar(x + w / 2, f1, w, color=ORANGE, label="F1")
label_bars(ax, b1); label_bars(ax, b2)
ax.set_xticks(x, [names[a] for a in arms]); ax.set_ylim(0, 1.12); ax.set_ylabel("score")
ax.set_title("Ablation on the test split (120 calls)"); ax.legend(frameon=False, loc="upper left", ncol=2)
fig.savefig(FIG / "chart_ablation.png"); plt.close(fig)

# 2. per-cell accuracy heatmap
cells = ab["cells"]
cell_names = {"humanxbenign": "Human\nbenign", "humanxscam": "Human\nscam",
              "clonedxbenign": "Cloned\nbenign", "clonedxscam": "Cloned\nscam"}
M = np.array([[ab["detection_rate"][a][c] for c in cells] for a in arms])
fig, ax = plt.subplots(figsize=(5.2, 2.6))
im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                color="white" if M[i, j] > 0.6 else INK)
ax.set_xticks(range(len(cells)), [cell_names[c] for c in cells]); ax.set_yticks(range(len(arms)), [names[a] for a in arms])
ax.set_title("Per-cell accuracy at threshold 0.65"); ax.spines[:].set_visible(False)
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="accuracy")
fig.savefig(FIG / "chart_percell.png"); plt.close(fig)

# 3. hard subset
hs = ab["hard_subset"]
fig, ax = plt.subplots(figsize=(5.2, 2.6))
vals = [hs[a]["auc"] for a in arms]
b = ax.bar([names[a] for a in arms], vals, color=[GREY, NAVY, NAVY, ORANGE], width=0.55)
label_bars(ax, b); ax.set_ylim(0, 1.12); ax.set_ylabel("AUC")
ax.set_title(f"Hard subset: {hs['n_scam']} mild scams against {hs['n_benign']} hard negatives")
fig.savefig(FIG / "chart_hard.png"); plt.close(fig)

# 4. NER per type
ner = load("ner_results.json")["per_type"]["crf"]["gold"]
types = sorted(ner, key=lambda t: ner[t]["f1"])
fig, ax = plt.subplots(figsize=(5.4, 2.8))
bars = ax.barh([t.replace("_", " ").title() for t in types], [ner[t]["f1"] for t in types], color=NAVY, height=0.6)
for bar, t in zip(bars, types):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{ner[t]['f1']:.3f}  (n={ner[t]['support']})", va="center", fontsize=8, color=INK)
ax.set_xlim(0, 1.25); ax.set_xlabel("F1, exact span match"); ax.set_title("Entity recognition by type (CRF, 324 spans)")
fig.savefig(FIG / "chart_ner.png"); plt.close(fig)

# 5. intent: rules vs tfidf
it = load("intent_results.json")["models"]
fig, ax = plt.subplots(figsize=(5.2, 2.6))
metrics = ["auc", "f1", "recall", "precision"]; x = np.arange(len(metrics)); w = 0.36
b1 = ax.bar(x - w / 2, [it["rules"][m] for m in metrics], w, color=GREY, label="Lexicon rules")
b2 = ax.bar(x + w / 2, [it["tfidf"][m] for m in metrics], w, color=NAVY, label="TF-IDF intent")
label_bars(ax, b1); label_bars(ax, b2)
ax.set_xticks(x, [m.upper() if m == "auc" else m.title() for m in metrics]); ax.set_ylim(0, 1.15)
ax.set_title("Scam intent: rule baseline against TF-IDF (test split)"); ax.legend(frameon=False, ncol=2, loc="upper left")
fig.savefig(FIG / "chart_intent.png"); plt.close(fig)

# 6. calibration reliability diagram
cal = load("calibration.json")
fig, ax = plt.subplots(figsize=(3.6, 3.4))
ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1, label="perfect")
xs = [b["p_pred"] for b in cal["bins"]]; ys = [b["p_obs"] for b in cal["bins"]]; ns = [b["n"] for b in cal["bins"]]
ax.plot(xs, ys, "-o", color=NAVY, lw=1.4, ms=5, label="observed")
for xv, yv, n in zip(xs, ys, ns):
    ax.annotate(f"n={n}", (xv, yv), textcoords="offset points", xytext=(6, -10), fontsize=7, color=INK)
ax.set_xlabel("predicted fraud probability"); ax.set_ylabel("observed fraud rate")
ax.set_title(f"Calibration: Brier {cal['brier']:.4f}, ECE {cal['ece']:.4f}"); ax.legend(frameon=False, loc="upper left")
fig.savefig(FIG / "chart_calibration.png"); plt.close(fig)

# 7. anti-spoofing EER heatmaps
asr = load("antispoof_results.json"); eer = asr["eer"]; conds = asr["conditions"]; fes = list(eer)
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
for ax, m in zip(axes, ("gmm", "gbm")):
    M = np.array([[eer[f][m][c] for c in conds] for f in fes]) * 100
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=5, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] > 3 else INK)
    ax.set_xticks(range(len(conds)), [c.upper() if c != "clean" else "Clean" for c in conds])
    ax.set_yticks(range(len(fes)), [f.upper() for f in fes]); ax.set_title(f"{m.upper()} back end"); ax.spines[:].set_visible(False)
fig.suptitle("Anti-spoofing EER (percent) on 310 real Hindi test clips", fontsize=10, fontweight="bold")
fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label="EER, percent")
fig.savefig(FIG / "chart_antispoof.png"); plt.close(fig)

# 8. audit: raw vs conditioned
au = asr["channel_audit"]; stats = list(au["per_statistic"])
labels = {"noise_floor_db": "Noise floor", "snr_proxy_db": "Dynamic range", "zcr_mean": "Zero-crossing rate",
          "duration": "Duration", "quiet_centroid_hz": "Quiet centroid", "quiet_tilt_db": "Quiet tilt",
          "band_share_hi": "Energy above band", "band_share_lo": "Energy below band",
          "trailing_quiet_s": "Trailing silence", "dc_abs": "DC offset"}
fig, ax = plt.subplots(figsize=(5.6, 3.2))
y = np.arange(len(stats)); h = 0.36
raw = [au["per_statistic_raw"][s]["auc_alone"] for s in stats]; cond = [au["per_statistic"][s]["auc_alone"] for s in stats]
ax.barh(y + h / 2, raw, h, color=GREY, label="raw file")
ax.barh(y - h / 2, cond, h, color=NAVY, label="after conditioning")
for yi, r, c in zip(y, raw, cond):
    ax.text(r + 0.01, yi + h / 2, f"{r:.2f}", va="center", fontsize=7, color=INK)
    ax.text(c + 0.01, yi - h / 2, f"{c:.2f}", va="center", fontsize=7, color=INK)
ax.axvline(0.75, color=ORANGE, lw=1, ls="--"); ax.text(0.755, len(stats) - 0.4, "stop at 0.75", color=ORANGE, fontsize=7)
ax.axvline(0.5, color=GREY, lw=0.6, ls=":")
ax.set_yticks(y, [labels[s] for s in stats]); ax.set_xlim(0.4, 1.0); ax.set_xlabel("AUC of the statistic alone")
ax.set_title("Confound audit of the 600-pair set"); ax.legend(frameon=False, loc="lower right")
fig.savefig(FIG / "chart_audit.png"); plt.close(fig)

# 9. robocall OOD
ro = load("robocall_ood.json")["models"]
fig, ax = plt.subplots(figsize=(5.2, 2.6))
metrics = [("recall_at_threshold", "Recall at 0.65"), ("cross_domain_auc", "Cross-domain AUC"), ("benign_false_alarm", "False alarms")]
x = np.arange(len(metrics)); w = 0.36
b1 = ax.bar(x - w / 2, [ro["rules"][k] for k, _ in metrics], w, color=GREY, label="Lexicon rules")
b2 = ax.bar(x + w / 2, [ro["tfidf"][k] for k, _ in metrics], w, color=NAVY, label="TF-IDF intent")
label_bars(ax, b1); label_bars(ax, b2)
ax.set_xticks(x, [n for _, n in metrics]); ax.set_ylim(0, 1.15)
ax.set_title("1,413 real robocalls the models never saw"); ax.legend(frameon=False, loc="upper right")
fig.savefig(FIG / "chart_robocall.png"); plt.close(fig)

# 10. time to detection
tt = load("ttd_results.json")
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axes[0]
ax.hist(tt["values"], bins=np.arange(5, 60, 5), color=NAVY, edgecolor="white")
ax.axvline(tt["median"], color=ORANGE, lw=1.2, ls="--"); ax.text(tt["median"] + 0.8, ax.get_ylim()[1] * 0.9, f"median {tt['median']:.1f} s", color=ORANGE, fontsize=8)
ax.set_xlabel("seconds of call heard before the alert"); ax.set_ylabel("calls"); ax.set_title(f"Time to detection, {tt['detected']} of {tt['total']} flagged")
ax = axes[1]
sc = sorted(tt["by_scenario"].items(), key=lambda kv: kv[1])
ax.barh([k.replace("_", " ") for k, _ in sc], [v for _, v in sc], color=NAVY, height=0.6)
for i, (_, v) in enumerate(sc):
    ax.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=7, color=INK)
ax.set_xlabel("median seconds"); ax.set_title("By scenario"); ax.set_xlim(0, 55)
fig.savefig(FIG / "chart_ttd.png"); plt.close(fig)

# 11. robustness
rb = load("robustness_results.json")
fig, ax = plt.subplots(figsize=(5.2, 2.6))
codecs = rb["codecs"]; xl = ["Clean", "G.711 mu", "G.711 A", "GSM"]
full = [rb["auc"][c] for c in codecs]; audio = [rb["auc_by_arm"]["audio_only"][c] for c in codecs]
ax.plot(xl, full, "-o", color=NAVY, lw=1.6, ms=5, label="Full arm")
ax.plot(xl, audio, "-o", color=ORANGE, lw=1.6, ms=5, label="Audio-only arm")
for i, (f, a) in enumerate(zip(full, audio)):
    ax.text(i, f + 0.02, f"{f:.3f}", ha="center", fontsize=7.5, color=INK); ax.text(i, a - 0.06, f"{a:.3f}", ha="center", fontsize=7.5, color=INK)
ax.set_ylim(0.4, 1.1); ax.set_ylabel("AUC"); ax.set_title("Channel robustness on the test split"); ax.legend(frameon=False, loc="center right")
fig.savefig(FIG / "chart_robustness.png"); plt.close(fig)

print("charts written to", FIG)
