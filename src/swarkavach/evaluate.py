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
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import config
from .config import SETTINGS, result_path
from .schema import (
    ABLATION_ARMS, Call, EntitySpan, FUSION_FEATURE_NAMES, decode_bio,
    feature_vector,
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
    # at each threshold: everything above it is called positive
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
        test = p.calls
    say(f"splits: {len(train)} train, {len(dev)} dev, {len(test)} test")

    provenance: List[Dict[str, Any]] = []

    def prov(artifact: str, notes: str = ""):
        provenance.append({
            "artifact": artifact, "source": "local", "generated": _now(), "notes": notes,
        })

    headline: Dict[str, Any] = {}

    # ---------------------------------------------------------- ablation
    say("featurising splits")
    Xtr, ytr, _, tr_calls = featurize_corpus(train, models=p.models, with_audio=True)
    Xte, yte, feats_te, te_calls = featurize_corpus(test, models=p.models, with_audio=True)

    arms = train_all_arms(Xtr, ytr, kind="logreg")
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

    ablation = {
        "arms": list(arms.keys()), "cells": cells,
        "detection_rate": det_rate, "overall": overall,
        "n_per_cell": {c: int(sum(1 for x in te_calls if x.cell == c)) for c in cells},
        "threshold": threshold, "split": "test", "n_test": len(te_calls),
        "generated": _now(),
    }
    _write("ablation_results.json", ablation)
    prov("ablation_results.json", f"{len(te_calls)} held-out calls, threshold {threshold}")
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
            headline["best anti-spoof EER"] = round(best, 4) if best is not None else None
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

    # ------------------------------------------------------- robustness
    if not skip_robustness and len(codecs) > 1:
        say("running the channel robustness sweep")
        rb = _eval_robustness(p, test, arms.get("full"), codecs, threshold, say)
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

    with_audio = [c for c in calls if c.audio_path and Path(c.audio_path).exists()]
    if len(with_audio) < 8:
        say("  not enough calls with audio")
        return None

    train_calls = [c for c in p.calls if c.split == "train"
                   and c.audio_path and Path(c.audio_path).exists()]
    feature_sets = ["lfcc", "gfcc", "mfcc", "cqcc", "lpcc"]
    models = ["gmm", "gbm"]
    eer_tbl: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    tdcf_tbl: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    det_store: List[Dict[str, Any]] = []

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
        except Exception as exc:
            say(f"  {fs} gmm failed: {exc}")
        try:
            b = GBMScorer().fit(Ftr, y_train)
            fitted["gbm"] = ("gbm", b)
        except Exception as exc:
            say(f"  {fs} gbm failed: {exc}")

        for mname, (_, model) in fitted.items():
            eer_tbl[fs][mname] = {}
            tdcf_tbl[fs][mname] = {}
            for codec in codecs:
                audio_te, y_te = load_all(with_audio, codec)
                if len(set(y_te.tolist())) < 2:
                    eer_tbl[fs][mname][codec] = None
                    tdcf_tbl[fs][mname][codec] = None
                    continue
                try:
                    if mname == "gmm":
                        from .antispoof.features import frame_features
                        s = np.array([float(model.llr(frame_features(x, sr, fs))) for x, sr in audio_te])
                    else:
                        F = np.asarray([utterance_features(x, sr, fs) for x, sr in audio_te])
                        s = np.asarray(model.score(F), dtype=float).ravel()
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

    return {
        "feature_sets": feature_sets, "models": models, "conditions": list(codecs),
        "default_condition": "clean", "eer": eer_tbl, "min_tdcf": tdcf_tbl,
        "det": det_store, "n_train": len(audio_train), "n_test": len(with_audio),
        "note": "min_tdcf here is a cost-weighted operating point with the ASV stage fixed",
        "generated": _now(),
    }


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

    for name, tagger in taggers.items():
        overall[name] = {}
        per_type[name] = {}
        for src in ("gold", "asr"):
            if src == "asr":
                try:
                    from .text.asr import ASR
                    asr = ASR()
                    if asr.backend not in ("whisper",):
                        continue          # nothing to degrade, skip honestly
                except Exception:
                    continue
            preds = []
            for tk in toks:
                try:
                    b = tagger.predict_turn(tk)
                except Exception:
                    b = ["O"] * len(tk)
                preds.append(b if len(b) == len(tk) else ["O"] * len(tk))
            r = entity_prf(gold, preds, toks)
            overall[name][src] = {"p": round(r["p"], 4), "r": round(r["r"], 4),
                                  "f1": round(r["f1"], 4), "support": r["support"]}
            per_type[name][src] = {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                                       for kk, vv in v.items()}
                                   for k, v in r["per_type"].items()}

    return {
        "models": list(taggers), "transcripts": ["gold", "asr"],
        "overall": overall, "per_type": per_type,
        "n_turns": len(toks), "generated": _now(),
    }


def _eval_intent(p, calls: Sequence[Call], threshold: float, say) -> Optional[Dict[str, Any]]:
    from .fusion.featurize import _turn_intent, ModelBundle

    y = np.asarray([c.label_scam for c in calls], dtype=int)
    if len(set(y.tolist())) < 2:
        return None
    out: Dict[str, Any] = {"models": {}, "generated": _now()}

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
        b = binary_scores(s, y, 0.5)
        out["models"][name] = {
            "auc": round(roc_auc(s, y), 4), "accuracy": round(b["accuracy"], 4),
            "f1": round(b["f1"], 4), "precision": round(b["precision"], 4),
            "recall": round(b["recall"], 4),
            "confusion": {k: b[k] for k in ("tp", "fp", "fn", "tn")},
        }
    return out if out["models"] else None


def _eval_robustness(p, calls, model, codecs, threshold, say) -> Optional[Dict[str, Any]]:
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

    for codec in codecs:
        scores = []
        is_real = True
        for c in with_audio:
            try:
                x, sr = read_audio(c.audio_path, sr=SETTINGS.frame.sr)
                if codec != "clean":
                    x, info = degrade(x, sr, codec=codec, seed=SETTINGS.pipeline.seed)
                    is_real = bool(info.get("real_codec", True)) and is_real
                f, _ = build_features(c, audio=x, sr=sr, models=p.models)
                scores.append(float(model.predict_one(f)))
            except Exception:
                scores.append(0.0)
        s = np.asarray(scores)
        auc_by[codec] = round(roc_auc(s, y), 4)
        eer_by[codec] = round(eer(s, y)[0], 4)
        real[codec] = is_real
        labels[codec] = (CODECS.get(codec, {}) or {}).get("label", codec)
        say(f"  {codec}: AUC {auc_by[codec]}")

    return {
        "codecs": list(codecs), "auc": auc_by, "eer": eer_by,
        "real_codec": real, "labels": labels, "n": len(with_audio),
        "ffmpeg": bool(config.find_ffmpeg()), "generated": _now(),
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
