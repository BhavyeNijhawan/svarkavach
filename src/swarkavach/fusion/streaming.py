"""Incremental scoring, turn by turn.

A finished-recording accuracy number is the wrong question for a system that
is supposed to interrupt a call in progress. What matters operationally is how
much of the call it had to hear before it was willing to raise an alert.

So the pipeline re-runs after every turn on the prefix heard so far, and the
first crossing of the alert threshold gives the time-to-detection. That is
both a capability (the live monitor is driven by this) and an evaluation
metric (the distribution over the test split is a reportable result).

Cost is quadratic in turns because each prefix is re-featurised. Calls here
run 8 to 20 turns, so the whole trajectory takes well under a second on CPU,
and keeping it simple beats an incremental cache that could silently drift out
of step with the batch path.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..config import SETTINGS
from ..schema import Call, Turn, risk_band, feature_vector
from .featurize import ModelBundle, build_features


def _prefix_call(call: Call, upto: int) -> Call:
    """A copy of the call containing only turns 0..upto inclusive."""
    return Call(
        call_id=call.call_id,
        turns=call.turns[: upto + 1],
        label_scam=call.label_scam,
        label_voice=call.label_voice,
        scenario=call.scenario,
        speaker_id=call.speaker_id,
        split=call.split,
        audio_path=call.audio_path,
        sample_rate=call.sample_rate,
        channel=call.channel,
        audio_source=call.audio_source,
        meta=dict(call.meta),
    )


def _fallback_risk(features: Dict[str, float]) -> float:
    """Risk when no fusion model is trained yet.

    A fixed, documented combination rather than a guess: the two branch
    probabilities plus the cross-modal term, weighted the way the trained
    model usually ends up weighting them. It exists so the console is usable
    before training, and it is always labelled as the untrained path.
    """
    a = float(features.get("as_score", 0.0))
    i = float(features.get("intent_score", 0.0))
    p = float(features.get("pim", 0.0))
    c = float(np.clip(features.get("coercion_slope", 0.0) * 2.0, 0.0, 1.0))
    ent = float(np.clip(
        (features.get("ent_otp", 0.0) + features.get("ent_personal", 0.0)
         + features.get("ent_authority", 0.0) + features.get("ent_threat", 0.0)) / 6.0,
        0.0, 1.0))
    score = 0.30 * i + 0.22 * a + 0.20 * p + 0.14 * ent + 0.14 * c
    return float(np.clip(score, 0.0, 1.0))


def score_features(models: ModelBundle, features: Dict[str, float]) -> Tuple[float, str]:
    """(risk, how it was produced) for one feature dict."""
    if models.fusion is not None and getattr(models.fusion, "trained", False):
        try:
            return float(models.fusion.predict_one(features)), "fusion"
        except Exception:
            pass
    return _fallback_risk(features), "untrained-heuristic"


def stream_call(
    call: Call,
    audio: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
    models: Optional[ModelBundle] = None,
    threshold: Optional[float] = None,
    every_turn: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Yield one incremental verdict per turn.

    Each event carries the turn itself so the console can render it, the
    evidence spans that landed inside that turn, and the running scores.
    """
    from .explain import build_evidence

    models = models or ModelBundle()
    threshold = SETTINGS.pipeline.alert_threshold if threshold is None else threshold

    seen_entities: set = set()
    alerted = False
    prev_end = 0.0
    # one scratch dict for the whole call, so per-turn prosody and the voice
    # branch are computed once rather than once per prefix
    cache: Dict[str, Any] = {}

    for k, turn in enumerate(call.turns):
        t_start = time.perf_counter()
        prefix = _prefix_call(call, k)

        clip = None
        if audio is not None and sr:
            n = int(round(turn.t_end * sr))
            clip = audio[: max(n, int(0.2 * sr))] if n > 0 else None

        features, detail = build_features(prefix, audio=clip, sr=sr, models=models,
                                          cache=cache)
        risk, how = score_features(models, features)

        # spans that belong to this turn
        spans: List[Dict[str, Any]] = []
        ev = build_evidence(prefix, features, [], detail, risk)
        for s in ev.spans:
            if s["turn_index"] == turn.index:
                spans.append(s)

        new_ents = []
        for e in detail.get("entities", []) or []:
            key = (e["turn_index"], e["tok_start"], e["type"])
            if key not in seen_entities:
                seen_entities.add(key)
                if e["turn_index"] == turn.index:
                    new_ents.append({"type": e["type"], "text": e.get("text", "")})

        alert_now = (not alerted) and risk >= threshold
        if alert_now:
            alerted = True

        yield {
            "turn_index": int(turn.index),
            "t_start": round(float(turn.t_start), 3),
            "t_end": round(float(turn.t_end), 3),
            "dt": round(max(0.0, float(turn.t_end) - prev_end), 3),
            "risk": round(float(risk), 5),
            "band": risk_band(risk),
            "intent": round(float(features.get("intent_score", 0.0)), 5),
            "authenticity": round(float(features.get("as_score", 0.0)), 5),
            "pim": round(float(features.get("pim", 0.0)), 5),
            "coercion": round(float(features.get("coercion_slope", 0.0)), 5),
            "turn": {
                "index": int(turn.index), "speaker": turn.speaker, "text": turn.text,
                "act": turn.act, "tokens": list(turn.tokens),
                "t_start": round(float(turn.t_start), 3),
                "t_end": round(float(turn.t_end), 3),
            },
            "spans": spans,
            "new_entities": new_ents,
            "reasons": [r for r in ev.reasons][:6],
            "alert": bool(alert_now),
            "flagged": bool(spans) and risk >= threshold * 0.75,
            "scored_by": how,
            "compute_ms": round((time.perf_counter() - t_start) * 1000.0, 2),
        }
        prev_end = float(turn.t_end)


def time_to_detection(
    timeline: Sequence[Dict[str, Any]],
    threshold: Optional[float] = None,
) -> Tuple[Optional[float], Optional[int]]:
    """First moment the running risk crossed the alert threshold."""
    threshold = SETTINGS.pipeline.alert_threshold if threshold is None else threshold
    for ev in timeline:
        if float(ev.get("risk", 0.0)) >= threshold:
            return float(ev.get("t_end", 0.0)), int(ev.get("turn_index", 0))
    return None, None


def ttd_stats(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise time-to-detection across a set of scored scam calls."""
    vals = [float(r["ttd"]) for r in records
            if r.get("ttd") is not None and np.isfinite(float(r["ttd"]))]
    total = len(records)
    out: Dict[str, Any] = {
        "values": [round(v, 3) for v in vals],
        "detected": len(vals),
        "total": int(total),
        "detection_rate": round(len(vals) / total, 4) if total else 0.0,
    }
    if vals:
        a = np.asarray(vals)
        out.update({
            "median": round(float(np.median(a)), 3),
            "mean": round(float(a.mean()), 3),
            "p90": round(float(np.percentile(a, 90)), 3),
            "min": round(float(a.min()), 3),
            "max": round(float(a.max()), 3),
        })
    by_turn = [int(r["ttd_turns"]) for r in records if r.get("ttd_turns") is not None]
    if by_turn:
        out["median_turns"] = float(np.median(by_turn))
    return out


if __name__ == "__main__":
    from ..schema import Turn as T

    call = Call(
        call_id="stream_demo", label_scam=1, label_voice="synthetic", scenario="digital_arrest",
        turns=[
            T(index=0, speaker="caller", text="Namaste, kya main Rohit ji se baat kar raha hoon", t_start=0.0, t_end=2.8),
            T(index=1, speaker="callee", text="Haan ji boliye", t_start=3.0, t_end=3.9),
            T(index=2, speaker="caller", text="Main cyber crime branch se inspector Verma bol raha hoon", t_start=4.2, t_end=7.6),
            T(index=3, speaker="callee", text="Kya hua sir", t_start=7.8, t_end=8.6),
            T(index=4, speaker="caller", text="Aapke account se illegal transaction hui hai, das minute mein account block ho jayega", t_start=9.0, t_end=13.8),
            T(index=5, speaker="caller", text="Turant OTP batao aur kisi ko mat batana, line pe raho", t_start=14.2, t_end=18.0),
        ],
    )
    tl = []
    print(f"{'turn':>4}  {'t_end':>6}  {'risk':>6}  {'intent':>6}  {'pim':>5}  band")
    for ev in stream_call(call):
        tl.append(ev)
        print(f"{ev['turn_index']:>4}  {ev['t_end']:>6.1f}  {ev['risk']:>6.3f}  "
              f"{ev['intent']:>6.3f}  {ev['pim']:>5.2f}  {ev['band']}"
              + ("   <-- ALERT" if ev["alert"] else ""))
    t, k = time_to_detection(tl)
    print(f"\ntime to detection: {t}s at turn {k}")
