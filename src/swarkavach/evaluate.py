"""Metrics and the full evaluation run.

Everything the console shows is produced here and written to
`data/results/*.json`. The JSON shapes are a contract with the frontend, so
they are documented next to the function that writes each one.

The metrics themselves follow the anti-spoofing literature: equal error rate
and minimum tandem detection cost for the voice branch, entity-level precision
and recall with exact span matching for the tagger, and area under the ROC for
the fused decision. Time to detection is our own addition and is defined in
`fusion/streaming.py`.
"""

from __future__ import annotations

import json
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import config
from .config import SETTINGS, result_path
from .schema import (
    ABLATION_ARMS, Call, EntitySpan, FUSION_FEATURE_NAMES, Turn, decode_bio,
    feature_vector, tokenize,
)

# --------------------------------------------------------------------------
# core metrics
# --------------------------------------------------------------------------


def det_curve(scores: Sequence[float], labels: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """False alarm rate, miss rate and thresholds, sorted by threshold.

    Label 1 is the positive (spoof or fraud) class. Written out rather than
    taken from sklearn so the miss and false alarm definitions are visible,
    since the anti-spoofing convention is the opposite way round from the
    usual ROC one and mixing them up is the classic way to report a wrong EER.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.int64).ravel()
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]

    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)

    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)

    # Collapse ties. Walking through a group of equal scores one sample at a
    # time invents operating points that no threshold can actually reach, and
    # then the EER search picks one of them. Measured on a real run where the
    # detector produced only two distinct probabilities, that reported an EER
    # of 0.1339 where the best achievable was 0.0625, and simply reordering
    # tied samples moved the reported figure between 0.0 and 0.1339 without
    # changing the scores at all. Keeping only the last index of each run of
    # equal scores leaves exactly the reachable operating points.
    keep = np.ones(s.size, dtype=bool)
    if s.size > 1:
        keep[:-1] = s[:-1] != s[1:]
    tp, fp, s = tp[keep], fp[keep], s[keep]

    # at each threshold: everything at or above it is called positive
    fnr = 1.0 - tp / n_pos          # misses
    fpr = fp / n_neg                # false alarms
    fnr = np.concatenate(([1.0], fnr))
    fpr = np.concatenate(([0.0], fpr))
    thr = np.concatenate(([np.inf], s))
    return fpr, fnr, thr


def eer(scores: Sequence[float], labels: Sequence[int]) -> Tuple[float, float]:
    """Equal error rate and the threshold where it happens."""
    fpr, fnr, thr = det_curve(scores, labels)
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2.0), float(thr[i])


def min_tdcf(
    scores: Sequence[float],
    labels: Sequence[int],
    p_target: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 10.0,
) -> float:
    """Minimum normalised detection cost.

    This is the ASVspoof-style cost with the speaker verification stage held
    fixed, which is the honest thing to report when there is no ASV system in
    the loop: it is a cost-weighted operating point, not the full tandem
    t-DCF. The priors are stated here so the number is reproducible.
    """
    fpr, fnr, _ = det_curve(scores, labels)
    cost = c_miss * p_target * fnr + c_fa * (1.0 - p_target) * fpr
    default = min(c_miss * p_target, c_fa * (1.0 - p_target))
    return float(np.min(cost) / default) if default > 0 else float("nan")


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the ROC by the rank formula, ties handled properly."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.int64).ravel()
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_scores(scores: Sequence[float], labels: Sequence[int], threshold: float) -> Dict[str, float]:
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.int64).ravel()
    pred = (s >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "accuracy": (tp + tn) / max(len(y), 1),
        "precision": prec, "recall": rec, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# --------------------------------------------------------------------------
# entity level
# --------------------------------------------------------------------------


def entity_prf(
    gold: Sequence[Sequence[str]],
    pred: Sequence[Sequence[str]],
    tokens: Optional[Sequence[Sequence[str]]] = None,
) -> Dict[str, Any]:
    """Entity-level precision, recall and F1 with exact span matching.

    Token-level accuracy flatters a tagger badly here, because most tokens are
    O and getting them right is free. An entity counts only when its type and
    both boundaries match, which is the standard CoNLL convention.
    """
    def spans(seqs):
        out = set()
        per_type: Dict[str, set] = {}
        for si, seq in enumerate(seqs):
            toks = tokens[si] if tokens is not None else [""] * len(seq)
            for e in decode_bio(list(toks), list(seq), turn_index=si):
                key = (si, e.tok_start, e.tok_end, e.type)
                out.add(key)
                per_type.setdefault(e.type, set()).add(key)
        return out, per_type

    G, Gt = spans(gold)
    P, Pt = spans(pred)

    def prf(g, p):
        tp = len(g & p)
        prec = tp / len(p) if p else 0.0
        rec = tp / len(g) if g else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {"p": prec, "r": rec, "f1": f1, "support": len(g), "predicted": len(p), "correct": tp}

    out = prf(G, P)
    out["per_type"] = {
        t: prf(Gt.get(t, set()), Pt.get(t, set()))
        for t in sorted(set(Gt) | set(Pt))
    }
    return out


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------


def calibration_curve(scores: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> Dict[str, Any]:
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.int64).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (s >= lo) & (s < hi if i < n_bins - 1 else s <= hi)
        if not m.any():
            continue
        p_pred = float(s[m].mean())
        p_obs = float(y[m].mean())
        w = float(m.sum()) / len(s)
        ece += w * abs(p_pred - p_obs)
        bins.append({"p_pred": round(p_pred, 4), "p_obs": round(p_obs, 4), "n": int(m.sum())})
    brier = float(np.mean((s - y) ** 2)) if len(s) else float("nan")
    return {"bins": bins, "brier": round(brier, 5), "ece": round(float(ece), 5)}


# --------------------------------------------------------------------------
# the full run
# --------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _write(name: str, obj: Any) -> Path:
    p = result_path(name)
    p.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
    return p


def hard_subset_metrics(
    calls: Sequence[Any],
    y: np.ndarray,
    arm_scores: Dict[str, np.ndarray],
    threshold: float,
    min_n: int = 8,
) -> Dict[str, Any]:
    """Score every arm on the hard subset only.

    The hard subset is the lexically mild scams together with the benign calls
    written to look like their scam counterpart. The whole-corpus numbers
    saturate, and not entirely by mistake: some of it was authorship and the
    generator was fixed for that, but the rest is real. A scam call does
    threaten and does ask for a code, and a bag of words finds that. Measured
    on this corpus a bag-of-words model reaches 0.995 on the paired topics and
    0.972 on this subset, so this is the only place an arm comparison has room
    to say anything.
    """
    def meta(c):
        return getattr(c, "meta", None) or {}

    idx = [i for i, c in enumerate(calls)
           if meta(c).get("hard_negative") or meta(c).get("mild_scam")]
    out: Dict[str, Any] = {"n": len(idx)}
    y = np.asarray(y)
    if len(idx) < min_n or len(set(y[idx].tolist())) < 2:
        out["note"] = (f"too few hard cases to report, need at least {min_n} "
                       "with both labels present")
        return out
    yh = y[idx]
    for arm, s in arm_scores.items():
        sh = np.asarray(s)[idx]
        b = binary_scores(sh, yh, threshold)
        out[arm] = {
            "auc": round(roc_auc(sh, yh), 4),
            "accuracy": round(b["accuracy"], 4),
            "f1": round(b["f1"], 4),
            "recall": round(b["recall"], 4),
            "precision": round(b["precision"], 4),
        }
    out["n_scam"] = int(yh.sum())
    out["n_benign"] = int(len(yh) - yh.sum())
    return out


def run_full_evaluation(
    codecs: Optional[Sequence[str]] = None,
    skip_robustness: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate everything and write the result files the console reads."""
    from .pipeline import Pipeline
    from .fusion.featurize import featurize_corpus, build_features, ModelBundle
    from .fusion.model import FusionModel, train_all_arms
    from .fusion.streaming import stream_call, time_to_detection, ttd_stats
    from .audioio import read_audio

    def say(m):
        if verbose:
            print(m, flush=True)

    codecs = list(codecs or ["clean", "g711u", "g711a", "gsm"])
    threshold = SETTINGS.pipeline.alert_threshold
    p = Pipeline.load()
    if not p.calls:
        raise RuntimeError("no corpus on disk, run gen-corpus first")

    train = [c for c in p.calls if c.split == "train"]
    dev = [c for c in p.calls if c.split == "dev"]
    test = [c for c in p.calls if c.split == "test"]
    if not test:
        # Falling back to the whole corpus here meant every metric was computed
        # over rows the branches and the arms were fitted on, while
        # ablation_results.json still recorded "split": "test". A corpus with
        # no test split is a broken corpus, not a reason to report numbers.
        raise RuntimeError(
            "this corpus has no test split, so nothing here can be held out. "
            "Regenerate it with gen-corpus, which assigns speaker-disjoint "
            "train, dev and test splits."
        )
    say(f"splits: {len(train)} train, {len(dev)} dev, {len(test)} test")

    provenance: List[Dict[str, Any]] = []

    def prov(artifact: str, notes: str = ""):
        provenance.append({
            "artifact": artifact, "source": "local", "generated": _now(), "notes": notes,
        })

    headline: Dict[str, Any] = {}

    # ---------------------------------------------------------- ablation
    say("featurising splits")
    # Fit the arms on dev, not train. The branches were fitted on train, so
    # featurising train feeds the arms in-sample branch outputs: the tagger
    # scores entity F1 1.000 on rows it memorised against 0.887 held out, and
    # an arm fitted on that learns an intercept and weights for a feature
    # distribution that never occurs at test time. `Pipeline.fit` already
    # avoids this and explains why; the ablation table was quietly doing the
    # opposite, which is what the headline comparison rests on.
    fit_calls = dev if len(dev) >= 20 else train
    if fit_calls is train:
        say("  dev split too small, falling back to train, arms will be optimistic")
    Xfit, yfit, _, _ = featurize_corpus(fit_calls, models=p.models, with_audio=True)
    Xte, yte, feats_te, te_calls = featurize_corpus(test, models=p.models, with_audio=True)

    arms = train_all_arms(Xfit, yfit, kind="logreg")
    cells = ["humanxbenign", "humanxscam", "clonedxbenign", "clonedxscam"]
    det_rate: Dict[str, Dict[str, float]] = {}
    overall: Dict[str, Dict[str, float]] = {}
    arm_scores: Dict[str, np.ndarray] = {}

    for arm, model in arms.items():
        s = model.predict_proba(Xte)
        arm_scores[arm] = s
        b = binary_scores(s, yte, threshold)
        e, _ = eer(s, yte)
        overall[arm] = {
            "auc": round(roc_auc(s, yte), 4), "accuracy": round(b["accuracy"], 4),
            "f1": round(b["f1"], 4), "precision": round(b["precision"], 4),
            "recall": round(b["recall"], 4), "eer": round(e, 4),
        }
        per_cell: Dict[str, float] = {}
        for cell in cells:
            idx = [i for i, c in enumerate(te_calls) if c.cell == cell]
            if not idx:
                per_cell[cell] = None
                continue
            want = yte[idx]
            got = (s[idx] >= threshold).astype(int)
            per_cell[cell] = round(float((got == want).mean()), 4)
        det_rate[arm] = per_cell

    hard = hard_subset_metrics(te_calls, yte, arm_scores, threshold)

    ablation = {
        "arms": list(arms.keys()), "cells": cells,
        "hard_subset": hard,
        # The key predates its consumers and is misnamed: it is the fraction
        # of calls in the cell classified correctly at the threshold, not the
        # fraction flagged. The report, the console and the write-up all read
        # it as accuracy; renaming it would touch all three for no gain.
        "detection_rate_means": "fraction of the cell's calls classified "
                                "correctly at the threshold (per-cell accuracy)",
        "detection_rate": det_rate, "overall": overall,
        "n_per_cell": {c: int(sum(1 for x in te_calls if x.cell == c)) for c in cells},
        "threshold": threshold, "split": "test", "n_test": len(te_calls),
        "arms_fitted_on": ("dev" if fit_calls is dev else "train (dev too small)"),
        "n_fit": len(fit_calls),
        "generated": _now(),
    }
    _write("ablation_results.json", ablation)
    prov("ablation_results.json", f"{len(te_calls)} held-out calls, threshold {threshold}")
    # The ablation arms and the shipped fusion model are different fits: the
    # arms are fitted here on dev for a like-for-like feature comparison, the
    # shipped model was fitted by Pipeline.fit and is what streaming uses. Two
    # headline numbers used to come from the two of them with nothing saying
    # so, presented as one system.
    headline["_note"] = ("AUC numbers come from the ablation arms fitted in "
                         "this run; time to detection comes from the shipped "
                         "fusion model, which is what the console runs")
    headline["fused AUC"] = overall.get("full", {}).get("auc")
    headline["audio-only AUC"] = overall.get("audio_only", {}).get("auc")
    headline["text-only AUC"] = overall.get("text_only", {}).get("auc")
    say(f"  ablation: full AUC {headline['fused AUC']}")

    # ---------------------------------------------------- calibration
    if "full" in arm_scores:
        cal = calibration_curve(arm_scores["full"], yte, n_bins=10)
        cal["generated"] = _now()
        _write("calibration.json", cal)
        prov("calibration.json", "10 equal-width bins on the test split")
        headline["Brier score"] = cal["brier"]

    # ------------------------------------------------------- anti-spoof
    say("evaluating the anti-spoofing branch")
    as_res = _eval_antispoof(p, test, codecs if not skip_robustness else ["clean"], say)
    if as_res:
        _write("antispoof_results.json", as_res)
        prov("antispoof_results.json", "feature set by model by channel condition")
        try:
            best = min(
                (v[c] for f in as_res["eer"].values() for v in f.values()
                 for c in v if v[c] is not None),
                default=None)
            # A minimum over 5 feature sets x 2 models x every codec condition
            # on about 70 utterances will find 0.0 whether or not the detector
            # works. Keep it, name it honestly, and carry the default
            # condition's clean number next to it.
            headline["best anti-spoof EER (min over all cells)"] = (
                round(best, 4) if best is not None else None)
        except Exception:
            pass

    # -------------------------------------------------------------- NER
    say("evaluating entity recognition")
    ner_res = _eval_ner(p, test, say)
    if ner_res:
        _write("ner_results.json", ner_res)
        prov("ner_results.json", "entity level, exact span match")
        headline["CRF entity F1"] = ner_res.get("overall", {}).get("crf", {}).get("gold", {}).get("f1")

    # ----------------------------------------------------------- intent
    say("evaluating intent models")
    intent_res = _eval_intent(p, test, threshold, say)
    if intent_res:
        _write("intent_results.json", intent_res)
        prov("intent_results.json", "call level, held-out split")

    # --------------------------------------- out of domain: real robocalls
    say("scoring real robocall transcripts, out of domain")
    ood = _eval_robocall_ood(p, test, threshold, say)
    if ood:
        _write("robocall_ood.json", ood)
        prov("robocall_ood.json",
             f"{ood['n_robocalls']} real FTC robocall transcripts, nothing tuned on them")
        best = max((m.get("recall_at_threshold", 0) for m in ood["models"].values()),
                   default=None)
        if best is not None:
            headline["robocall recall, out of domain"] = best

    # ------------------------------------------------------- robustness
    if not skip_robustness and len(codecs) > 1:
        say("running the channel robustness sweep")
        rb = _eval_robustness(p, test, arms.get("full"), codecs, threshold, say,
                              extra_arms={"audio_only": arms.get("audio_only")})
        if rb:
            _write("robustness_results.json", rb)
            prov("robustness_results.json",
                 "real codecs where ffmpeg exists, numpy approximation otherwise")

    # ------------------------------------------------- time to detection
    say("measuring time to detection")
    records = []
    for c in te_calls:
        if not c.label_scam:
            continue
        audio = sr = None
        if c.audio_path and Path(c.audio_path).exists():
            try:
                audio, sr = read_audio(c.audio_path, sr=SETTINGS.frame.sr)
            except Exception:
                pass
        tl = list(stream_call(c, audio=audio, sr=sr, models=p.models, threshold=threshold))
        t, k = time_to_detection(tl, threshold)
        records.append({"call_id": c.call_id, "scenario": c.scenario,
                        "ttd": t, "ttd_turns": k, "duration": c.duration})
    ttd = ttd_stats(records)
    ttd["by_scenario"] = {}
    for r in records:
        ttd["by_scenario"].setdefault(r["scenario"], []).append(r["ttd"])
    ttd["by_scenario"] = {
        k: round(float(np.median([x for x in v if x is not None])), 2)
        for k, v in ttd["by_scenario"].items() if any(x is not None for x in v)
    }
    ttd["threshold"] = threshold
    ttd["generated"] = _now()
    _write("ttd_results.json", ttd)
    prov("ttd_results.json", f"scam calls in the test split, alert threshold {threshold}")
    headline["median time to detection"] = ttd.get("median")

    _write("provenance.json", {"entries": provenance, "generated": _now(),
                               "host": platform.platform(),
                               "python": sys.version.split()[0]})

    return {"headline": headline, "provenance": provenance}


# --------------------------------------------------------------------------
# per-branch evaluations
# --------------------------------------------------------------------------


def _eval_antispoof(p, calls: Sequence[Call], codecs: Sequence[str], say) -> Optional[Dict[str, Any]]:
    from .audioio import read_audio

    try:
        from .antispoof.features import utterance_features
        from .antispoof.gmm import GMMScorer
        from .antispoof.gbm import GBMScorer
    except Exception as exc:
        say(f"  anti-spoof modules unavailable: {exc}")
        return None

    # Prefer the real paired set. Reporting an anti-spoofing EER on the
    # generated corpus would be measuring a machine against another machine,
    # since both sides of its human-against-cloned axis are synthesised. The
    # pairs are real human telephone speech against a rendering of the same
    # transcript, with the channel matched, so an EER on them means what the
    # word usually means. Split is speaker disjoint and comes from the pairs.
    source = "generated_corpus"
    pair_calls: List[Call] = []
    try:
        from .datasets import antispoof_pairs_as_calls, load_antispoof_pairs

        pair_calls = antispoof_pairs_as_calls()
    except Exception as exc:
        say(f"  real pairs unavailable: {exc}")

    audit: Dict[str, Any] = {}
    if len(pair_calls) >= 40:
        source = "real_pairs"
        audit = (load_antispoof_pairs() or {}).get("confound_audit", {}) or {}
        train_calls = [c for c in pair_calls if c.split == "train"]
        with_audio = [c for c in pair_calls if c.split == "dev"]
        say(f"  using {len(train_calls)} train and {len(with_audio)} held-out "
            f"real paired utterances")
        if audit.get("worst_auc", 0) > 0.75:
            say(f"  WARNING: the pair set fails its channel audit "
                f"(worst AUC {audit['worst_auc']}), so this EER is not "
                f"trustworthy. Rebuild with: swarkavach fetch-data")
    else:
        with_audio = [c for c in calls if c.audio_path and Path(c.audio_path).exists()]
        train_calls = [c for c in p.calls if c.split == "train"
                       and c.audio_path and Path(c.audio_path).exists()]
        say("  no real pairs on disk, falling back to the generated corpus "
            "(this measures machine against machine)")

    if len(with_audio) < 8 or len(train_calls) < 8:
        say("  not enough audio to evaluate the voice branch")
        return None
    feature_sets = ["lfcc", "gfcc", "mfcc", "cqcc", "lpcc"]
    models = ["gmm", "gbm"]
    gmm_fit: Dict[str, Any] = {}
    eer_tbl: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    tdcf_tbl: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    det_store: List[Dict[str, Any]] = []
    models_by_fs: Dict[str, Dict[str, Any]] = {}

    def load_all(cs, codec):
        X, y = [], []
        for c in cs:
            try:
                x, sr = read_audio(c.audio_path, sr=SETTINGS.frame.sr)
                if codec != "clean":
                    from .channel import degrade
                    x, _ = degrade(x, sr, codec=codec, seed=SETTINGS.pipeline.seed)
                X.append((x, sr))
                y.append(1 if c.label_voice == "synthetic" else 0)
            except Exception:
                continue
        return X, np.asarray(y, dtype=int)

    audio_train, y_train = load_all(train_calls, "clean")

    # The obvious loop nesting re-reads and re-degrades every WAV for every
    # feature set and every model, which on this corpus is thousands of
    # redundant decodes. Decoding once per condition fixes that. Only ONE
    # condition is held at a time: 120 test calls of 60 seconds at 8 kHz is
    # about 230 MB per codec, so keeping all of them resident would be a
    # gigabyte of audio for no reason.
    _held: Dict[str, Any] = {}

    def test_audio(codec: str):
        if _held.get("codec") != codec:
            _held.clear()
            X, y = load_all(with_audio, codec)
            _held.update({"codec": codec, "X": X, "y": y})
            say(f"  decoded {len(X)} test calls for {codec}")
        return _held["X"], _held["y"]
    for fs in feature_sets:
        eer_tbl[fs] = {}
        tdcf_tbl[fs] = {}
        try:
            Ftr = np.asarray([utterance_features(x, sr, fs) for x, sr in audio_train])
        except Exception as exc:
            say(f"  {fs} extraction failed: {exc}")
            continue
        if Ftr.size == 0 or len(set(y_train.tolist())) < 2:
            continue

        fitted = {}
        try:
            g = GMMScorer().fit_from_utterances(audio_train, y_train, feature_set=fs) \
                if hasattr(GMMScorer, "fit_from_utterances") else None
            if g is None:
                from .antispoof.features import frame_features
                bona = np.vstack([frame_features(x, sr, fs) for (x, sr), lab in zip(audio_train, y_train) if lab == 0])
                spoof = np.vstack([frame_features(x, sr, fs) for (x, sr), lab in zip(audio_train, y_train) if lab == 1])
                g = GMMScorer().fit(bona, spoof)
            fitted["gmm"] = ("gmm", g)
            # A mixture that ran out of iterations without converging still
            # scores, and still produces an EER that looks like any other. It
            # is not one. Record it next to the number rather than leaving it
            # in the object where nobody reads it.
            meta = getattr(g, "meta_", {}) or {}
            if not (meta.get("converged_bona", True) and meta.get("converged_spoof", True)):
                say(f"  {fs} gmm did not converge, treat its EER with suspicion")
            gmm_fit[fs] = {
                "converged_bona": meta.get("converged_bona"),
                "converged_spoof": meta.get("converged_spoof"),
                "n_components": meta.get("n_components"),
                "n_frames_bona": meta.get("n_frames_bona"),
                "n_frames_spoof": meta.get("n_frames_spoof"),
            }
        except Exception as exc:
            say(f"  {fs} gmm failed: {exc}")
        try:
            b = GBMScorer().fit(Ftr, y_train)
            fitted["gbm"] = ("gbm", b)
        except Exception as exc:
            say(f"  {fs} gbm failed: {exc}")

        models_by_fs[fs] = {name: m for name, (_, m) in fitted.items()}
        for mname in fitted:
            eer_tbl[fs][mname] = {}
            tdcf_tbl[fs][mname] = {}

    # Codec outermost, so each condition is decoded once and every feature set
    # and model is scored against it before it is thrown away.
    from .antispoof.features import frame_features as _frame_features

    for codec in codecs:
        X_te, y_te = test_audio(codec)
        if len(set(y_te.tolist())) < 2:
            for fs, fs_models in models_by_fs.items():
                for mname in fs_models:
                    eer_tbl[fs][mname][codec] = None
                    tdcf_tbl[fs][mname][codec] = None
            continue
        for fs, fs_models in models_by_fs.items():
            if not fs_models:
                continue
            try:
                need_frames = "gmm" in fs_models
                F_utt = (np.asarray([utterance_features(x, sr, fs) for x, sr in X_te])
                         if "gbm" in fs_models else None)
                F_frames = ([_frame_features(x, sr, fs) for x, sr in X_te]
                            if need_frames else None)
            except Exception as exc:
                say(f"  {fs}/{codec} extraction failed: {exc}")
                for mname in fs_models:
                    eer_tbl[fs][mname][codec] = None
                    tdcf_tbl[fs][mname][codec] = None
                continue

            for mname, model in fs_models.items():
                try:
                    if mname == "gmm":
                        s = np.array([float(model.llr(F)) for F in F_frames])
                    else:
                        s = np.asarray(model.score(F_utt), dtype=float).ravel()
                    e, _ = eer(s, y_te)
                    eer_tbl[fs][mname][codec] = round(float(e), 4)
                    tdcf_tbl[fs][mname][codec] = round(float(min_tdcf(s, y_te)), 4)
                    if codec == "clean" and len(det_store) < 5:
                        fpr, fnr, _ = det_curve(s, y_te)
                        step = max(1, len(fpr) // 90)
                        det_store.append({
                            "name": f"{fs.upper()} + {mname.upper()}",
                            "points": [[round(float(a), 4), round(float(b2), 4)]
                                       for a, b2 in zip(fpr[::step], fnr[::step])],
                            "eer": round(float(e), 4),
                        })
                except Exception as exc:
                    say(f"  {fs}/{mname}/{codec} failed: {exc}")
                    eer_tbl[fs][mname][codec] = None
                    tdcf_tbl[fs][mname][codec] = None
            del F_utt, F_frames
        say(f"  {codec} done")

    _held.clear()
    return {
        "feature_sets": feature_sets, "models": models, "conditions": list(codecs),
        "default_condition": "clean", "eer": eer_tbl, "min_tdcf": tdcf_tbl,
        "det": det_store, "n_train": len(audio_train), "n_test": len(with_audio),
        "note": "min_tdcf here is a cost-weighted operating point with the ASV stage fixed",
        "evaluated_on": source,
        "gmm_fit": gmm_fit,
        "gmm_converged": all(
            (v.get("converged_bona") is not False) and (v.get("converged_spoof") is not False)
            for v in gmm_fit.values()) if gmm_fit else None,
        "evaluated_on_note": (
            "real human telephone speech against a neural rendering of the same "
            "transcript, channel matched" if source == "real_pairs" else
            "the generated corpus, where both classes are machine generated, so "
            "this measures vocoder artefacts rather than human against machine"),
        "channel_audit": audit,
        "generated": _now(),
    }


#: The last reason _asr_tokens gave for not producing a row. Written into
#: ner_results.json, because the reason used to live only in the training log
#: and the JSON still listed "asr" among its transcripts.
_ASR_SKIP_REASON: Dict[str, Optional[str]] = {"reason": None}


def _asr_tokens(calls: Sequence[Call], say) -> Optional[List[List[str]]]:
    """Tokens as an ASR system would produce them, aligned to the gold turns.

    Returns None when that cannot be done honestly, which is most of the time,
    and says why. The entity metric is span exact, so it needs one token list
    per gold turn in the same order. Whisper returns a flat transcript with its
    own segmentation, and this project has no forced aligner, so the two only
    line up when the segmenter happens to produce the same number of turns.

    Refusing is the point. The previous version of this row ran the tagger
    over the gold tokens and labelled the result "asr".
    """
    try:
        from .text.asr import ASR
    except Exception as exc:
        _ASR_SKIP_REASON["reason"] = f"  no ASR row: {type(exc).__name__}: {exc}"[len("  no ASR row: "):]
        say(f"  no ASR row: {type(exc).__name__}: {exc}")
        return None

    asr = ASR()
    if "whisper" not in str(asr.resolve()).lower():
        _ASR_SKIP_REASON["reason"] = "  no ASR row: whisper is not installed, so there is nothing to transcribe"[len("  no ASR row: "):]
        say("  no ASR row: whisper is not installed, so there is nothing to transcribe")
        return None

    with_audio = [c for c in calls if c.audio_path and Path(c.audio_path).exists()]
    if len(with_audio) < max(4, len(calls) // 4):
        _ASR_SKIP_REASON["reason"] = f"  no ASR row: only {len(with_audio)} of {len(calls)} calls have audio"[len("  no ASR row: "):]
        say(f"  no ASR row: only {len(with_audio)} of {len(calls)} calls have audio")
        return None

    out: List[List[str]] = []
    aligned = skipped = 0
    for c in calls:
        gold_n = len(c.turns)
        got = None
        if c.audio_path and Path(c.audio_path).exists():
            try:
                r = asr.transcribe(c.audio_path)
                turns = r.get("turns") or []
                if len(turns) == gold_n:
                    got = [tokenize(str(t.get("text", ""))) for t in turns]
            except Exception:
                got = None
        if got is None:
            skipped += 1
            got = [[] for _ in c.turns]      # counts as a total miss, not as gold
        else:
            aligned += 1
        out.extend(got)

    if aligned < len(calls) // 2:
        say(f"  no ASR row: only {aligned} of {len(calls)} calls segmented into the "
            f"same number of turns as the annotation, so the spans cannot be compared")
        return None
    say(f"  ASR row: {aligned} calls aligned, {skipped} scored as complete misses")
    return out


def _eval_ner(p, calls: Sequence[Call], say) -> Optional[Dict[str, Any]]:
    taggers: Dict[str, Any] = {}
    if p.models.ner is not None:
        taggers["crf"] = p.models.ner
    try:
        from .text.ner_bilstm import BiLSTMTagger
        path = config.artifact_path("ner_bilstm.pt")
        if path.exists():
            taggers["bilstm"] = BiLSTMTagger.load(str(path))
    except Exception:
        pass
    if not taggers:
        say("  no trained tagger")
        return None

    gold = [t.bio for c in calls for t in c.turns]
    toks = [t.tokens for c in calls for t in c.turns]
    overall: Dict[str, Dict[str, Any]] = {}
    per_type: Dict[str, Dict[str, Any]] = {}

    asr_toks = _asr_tokens(calls, say)
    for name, tagger in taggers.items():
        overall[name] = {}
        per_type[name] = {}
        for src in ("gold", "asr"):
            src_toks = toks
            if src == "asr":
                # This row is supposed to answer "what does entity F1 do when
                # the transcript comes from ASR instead of the annotation".
                # It used to gate on whisper being importable and then run the
                # identical loop over the GOLD tokens, so on any machine with
                # whisper installed, which includes Colab, it emitted an "asr"
                # column numerically identical to "gold" and the console read
                # that as "entity F1 survives ASR unchanged". No audio was
                # ever transcribed. Either we really transcribe, or there is
                # no row.
                src_toks = asr_toks
                if src_toks is None:
                    continue
            preds = []
            for tk in src_toks:
                try:
                    b = tagger.predict_turn(tk)
                except Exception:
                    b = ["O"] * len(tk)
                preds.append(b if len(b) == len(tk) else ["O"] * len(tk))
            r = entity_prf(gold, preds, src_toks)
            overall[name][src] = {"p": round(r["p"], 4), "r": round(r["r"], 4),
                                  "f1": round(r["f1"], 4), "support": r["support"]}
            per_type[name][src] = {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                                       for kk, vv in v.items()}
                                   for k, v in r["per_type"].items()}

    scored = sorted({src for name in overall for src in overall[name]},
                    key=lambda x: ("gold", "asr").index(x) if x in ("gold", "asr") else 9)
    out = {
        "models": list(taggers), "transcripts": scored,
        "overall": overall, "per_type": per_type,
        "n_turns": len(toks), "generated": _now(),
    }
    if "asr" not in scored:
        # It used to list "asr" whether or not a row was written, and the
        # reason lived only in the log.
        out["asr_note"] = _ASR_SKIP_REASON["reason"] or "no ASR row was produced"
    return out


def _eval_intent(p, calls: Sequence[Call], threshold: float, say) -> Optional[Dict[str, Any]]:
    from .fusion.featurize import _turn_intent, ModelBundle

    y = np.asarray([c.label_scam for c in calls], dtype=int)
    if len(set(y.tolist())) < 2:
        return None
    out: Dict[str, Any] = {"models": {}, "threshold": float(threshold),
                           "generated": _now()}

    candidates: Dict[str, Any] = {}
    if p.models.rules is not None:
        candidates["rules"] = ModelBundle(rules=p.models.rules)
    if p.models.intent is not None:
        candidates["tfidf"] = ModelBundle(intent=p.models.intent, rules=p.models.rules)

    for name, bundle in candidates.items():
        s = []
        for c in calls:
            ts = [_turn_intent(bundle, t) for t in c.caller_turns()]
            a = np.asarray(ts) if ts else np.zeros(1)
            s.append(float(np.clip(0.6 * a.max() + 0.4 * a.mean(), 0, 1)))
        s = np.asarray(s)
        # Use the threshold we were given. This was hardcoded to 0.5 while
        # every other artifact used 0.65, and the file recorded no threshold
        # at all, so the intent recall here and the robocall recall in
        # _eval_robocall_ood were computed on the same quantity at two
        # different operating points and read as if they were comparable.
        b = binary_scores(s, y, threshold)
        out["models"][name] = {
            "auc": round(roc_auc(s, y), 4), "accuracy": round(b["accuracy"], 4),
            "f1": round(b["f1"], 4), "precision": round(b["precision"], 4),
            "recall": round(b["recall"], 4),
            "confusion": {k: b[k] for k in ("tp", "fp", "fn", "tn")},
        }
    return out if out["models"] else None


def _eval_robocall_ood(p, benign: Sequence[Call], threshold: float,
                       say) -> Optional[Dict[str, Any]]:
    """Score the intent branch on 1,431 real illegal robocall transcripts.

    This is the only genuinely out-of-domain number in the project. Everything
    else is measured on calls this codebase generated, which bounds how much
    they can tell you. These scripts were written by actual fraudsters, they
    are English rather than Hinglish, and they are recorded monologues rather
    than two-party conversation. Nothing here was tuned on them.

    Read the recall and treat the AUC with suspicion. The robocall set has no
    benign half, so the negatives borrowed here are our own benign calls, from
    a different language and a different recording setup. A classifier could
    separate those two on domain alone, so the AUC is an upper bound and the
    recall at the operating threshold is the honest figure.
    """
    from .datasets import robocall_transcripts
    from .fusion.featurize import _turn_intent, ModelBundle

    rows = robocall_transcripts()
    if len(rows) < 50:
        say("  no robocall transcripts on disk, skipping the out-of-domain test")
        return None

    candidates: Dict[str, Any] = {}
    if p.models.rules is not None:
        candidates["rules"] = ModelBundle(rules=p.models.rules)
    if p.models.intent is not None:
        candidates["tfidf"] = ModelBundle(intent=p.models.intent, rules=p.models.rules)
    if not candidates:
        return None

    def score_turns(bundle, turns) -> float:
        a = np.asarray([_turn_intent(bundle, t) for t in turns]) if turns else np.zeros(1)
        return float(np.clip(0.6 * a.max() + 0.4 * a.mean(), 0, 1))

    # one Turn per sentence, so a long recorded script is scored the way a
    # multi-turn call is rather than as one enormous utterance
    pos_turns, pos_lang = [], []
    for r in rows:
        chunks = [c.strip() for c in re.split(r"(?<=[.!?])\s+", r["transcript"]) if c.strip()]
        chunks = [c for c in chunks if len(c.split()) >= 2] or [r["transcript"]]
        try:
            pos_turns.append([Turn(index=i, speaker="caller", text=c)
                              for i, c in enumerate(chunks[:24])])
            pos_lang.append(r.get("language", "en"))
        except Exception:
            continue

    neg_turns = [c.caller_turns() for c in benign if not c.label_scam]

    out: Dict[str, Any] = {
        "n_robocalls": len(pos_turns),
        "n_by_language": {l: pos_lang.count(l) for l in sorted(set(pos_lang))},
        "n_benign_controls": len(neg_turns),
        "threshold": threshold,
        "note": ("real FTC robocall evidence, English monologue. No benign "
                 "half exists, so the negatives are this project's own benign "
                 "Hinglish calls and the AUC is a cross-domain upper bound."),
        "models": {}, "generated": _now(),
    }

    for name, bundle in candidates.items():
        ps = np.asarray([score_turns(bundle, t) for t in pos_turns])
        detected = int((ps >= threshold).sum())
        row = {
            "recall_at_threshold": round(detected / max(len(ps), 1), 4),
            "n_detected": detected,
            "score_mean": round(float(ps.mean()), 4),
            "score_median": round(float(np.median(ps)), 4),
            "score_p10": round(float(np.percentile(ps, 10)), 4),
            "score_p90": round(float(np.percentile(ps, 90)), 4),
        }
        # the set is not all English. Splitting it out keeps a handful of
        # Mandarin scripts from quietly depressing a number the report reads
        # as an English result.
        by_lang = {}
        for lang in sorted(set(pos_lang)):
            m = np.asarray([l == lang for l in pos_lang])
            if m.sum():
                by_lang[lang] = {"n": int(m.sum()),
                                 "recall": round(float((ps[m] >= threshold).mean()), 4)}
        row["by_language"] = by_lang
        if neg_turns:
            ns = np.asarray([score_turns(bundle, t) for t in neg_turns])
            sc = np.concatenate([ps, ns])
            y = np.concatenate([np.ones(len(ps), int), np.zeros(len(ns), int)])
            row["cross_domain_auc"] = round(roc_auc(sc, y), 4)
            row["benign_score_mean"] = round(float(ns.mean()), 4)
            row["benign_false_alarm"] = round(float((ns >= threshold).mean()), 4)
        out["models"][name] = row

    return out


def _eval_robustness(p, calls, model, codecs, threshold, say,
                     extra_arms: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    from .fusion.featurize import build_features
    from .audioio import read_audio
    from .channel import degrade, CODECS

    if model is None:
        return None
    with_audio = [c for c in calls if c.audio_path and Path(c.audio_path).exists()]
    if len(with_audio) < 8:
        return None

    auc_by: Dict[str, float] = {}
    eer_by: Dict[str, float] = {}
    real: Dict[str, bool] = {}
    labels: Dict[str, str] = {}
    y = np.asarray([c.label_scam for c in with_audio], dtype=int)

    failed_by: Dict[str, int] = {}
    errors: Dict[str, str] = {}
    # The full arm's decision is carried almost entirely by text, which no
    # codec touches, so its row came back identical to four decimals under
    # four codecs and said nothing about the audio path. The audio-only arm
    # is scored on the same degraded features, so a codec that hurts the
    # voice branch shows up somewhere.
    extra_arms = {k: m for k, m in (extra_arms or {}).items() if m is not None}
    extra_auc: Dict[str, Dict[str, Optional[float]]] = {k: {} for k in extra_arms}
    for codec in codecs:
        scores = []
        extra_scores: Dict[str, list] = {k: [] for k in extra_arms}
        is_real = True
        n_failed = 0
        last_error = ""
        for c in with_audio:
            try:
                x, sr = read_audio(c.audio_path, sr=SETTINGS.frame.sr)
                if codec != "clean":
                    x, info = degrade(x, sr, codec=codec, seed=SETTINGS.pipeline.seed)
                    is_real = bool(info.get("real_codec", True)) and is_real
                f, _ = build_features(c, audio=x, sr=sr, models=p.models)
                scores.append(float(model.predict_one(f)))
                for k, m in extra_arms.items():
                    extra_scores[k].append(float(m.predict_one(f)))
            except Exception as exc:
                # Not 0.0. A call that could not be read, degraded or
                # featurised has no score, and giving it the most-benign
                # possible one makes a codec that broke the pipeline look like
                # a codec that destroyed the signal. If a whole condition
                # fails, every score is 0.0, the AUC is exactly 0.5, and the
                # table reads as a real measurement of a hard channel.
                scores.append(float("nan"))
                for k in extra_arms:
                    extra_scores[k].append(float("nan"))
                n_failed += 1
                last_error = f"{type(exc).__name__}: {exc}"
        s = np.asarray(scores, dtype=float)
        ok = np.isfinite(s)
        for k in extra_arms:
            es = np.asarray(extra_scores[k], dtype=float)
            eok = np.isfinite(es)
            extra_auc[k][codec] = (round(roc_auc(es[eok], y[eok]), 4)
                                   if eok.sum() >= 8 and len(set(y[eok].tolist())) == 2 else None)
        failed_by[codec] = int(n_failed)
        if last_error:
            errors[codec] = last_error
        labels[codec] = (CODECS.get(codec, {}) or {}).get("label", codec)
        real[codec] = is_real
        # Score only the calls that produced a score, and refuse to report a
        # number at all once too many are missing for it to mean anything.
        if ok.sum() < max(8, int(0.6 * len(s))) or len(set(y[ok].tolist())) < 2:
            auc_by[codec] = None
            eer_by[codec] = None
            say(f"  {codec}: not reported, {n_failed} of {len(s)} calls failed"
                + (f" ({last_error})" if last_error else ""))
            continue
        auc_by[codec] = round(roc_auc(s[ok], y[ok]), 4)
        eer_by[codec] = round(eer(s[ok], y[ok])[0], 4)
        say(f"  {codec}: AUC {auc_by[codec]}"
            + "".join(f", {k} {extra_auc[k][codec]}" for k in extra_arms)
            + (f"  [{n_failed} calls failed]" if n_failed else ""))

    return {
        "auc_by_arm": {**{"full": dict(auc_by)}, **extra_auc},
        "codecs": list(codecs), "auc": auc_by, "eer": eer_by,
        "real_codec": real, "labels": labels, "n": len(with_audio),
        "n_failed": failed_by, "errors": errors,
        "ffmpeg": bool(config.find_ffmpeg()), "generated": _now(),
        "note": ("auc and eer are null for a condition where too many calls "
                 "failed to score. real_codec false means the numpy stand-in "
                 "ran, not the actual codec"),
    }


if __name__ == "__main__":
    # metric self-checks against values that can be worked out by hand
    s = [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1]
    y = [1, 1, 1, 1, 0, 0, 0, 0]
    print("perfectly separable:")
    print(f"  AUC {roc_auc(s, y):.4f} (expect 1.0)")
    print(f"  EER {eer(s, y)[0]:.4f} (expect 0.0)")

    # interleaved labels: positives sit at ranks 2, 4, 6, 8 counting from the
    # bottom, so the rank sum is 20 and the AUC works out to (20 - 10) / 16
    y2 = [1, 0, 1, 0, 1, 0, 1, 0]
    print("\ninterleaved labels:")
    print(f"  AUC {roc_auc(s, y2):.4f} (expect 0.625)")
    print(f"  EER {eer(s, y2)[0]:.4f} (expect 0.5)")

    gold = [["O", "B-OTP", "I-OTP", "O"], ["B-BANK_ENTITY", "O"]]
    pred = [["O", "B-OTP", "I-OTP", "O"], ["O", "O"]]
    r = entity_prf(gold, pred, [["a", "otp", "123", "b"], ["account", "x"]])
    print(f"\nentity P {r['p']:.2f} R {r['r']:.2f} F1 {r['f1']:.2f} (expect 1.00 / 0.50 / 0.67)")

    c = calibration_curve([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    print(f"\nBrier {c['brier']:.4f}, ECE {c['ece']:.4f}")
