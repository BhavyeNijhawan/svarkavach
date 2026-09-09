"""Prosody-Intent Mismatch.

The idea in one sentence: a cloned voice reading a threat script says
frightening things in a flat voice, and that gap is measurable.

Human speech carries affect. When a real person threatens you, tells you your
account is about to be frozen, or demands an OTP in the next ten seconds, the
pitch moves, the energy moves, and the speaking rate changes. Text-to-speech
reading the same words produces prosody that is smooth, regular and roughly
constant, because the model was trained to produce a neutral reading of
whatever text it was given.

So the system computes two per-turn quantities:

  lexical arousal   how much pressure the WORDS carry, from the weighted fraud
                    lexicon plus the entities present in the turn
  acoustic arousal  how much the VOICE moves, from pitch variation, energy
                    variation, pitch range and speaking-rate variation

and scores the gap. Neither branch can produce this on its own, which is why
it is a fusion feature rather than a feature of either branch. It is also the
feature that separates the two hard cells of the evaluation grid: a human
running a scam script has high lexical arousal AND high acoustic arousal, so
it stays low; a cloned voice reading a benign script has low lexical arousal,
so it also stays low.

The normalisation is deliberately absolute rather than per-call. If it were
normalised within the call, a uniformly flat synthetic call would look normal
because every turn matched every other turn, which is exactly the case the
feature exists to catch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import SETTINGS
from ..corpus.lexicon import PRESSURE_WEIGHTS, lexicon_hits
from ..schema import Call, Turn

# Anchors for the absolute normalisation. Each is (low, high): a value at or
# below `low` maps to 0, at or above `high` maps to 1. They come from what
# ordinary telephone speech does, measured on the corpus, and are held fixed
# so the feature means the same thing across calls and across channels.
ANCHORS: Dict[str, Tuple[float, float]] = {
    "f0_cv": (0.020, 0.160),      # pitch standard deviation over pitch mean
    "f0_span": (0.100, 0.750),    # pitch range over pitch mean
    "energy_std": (0.8, 6.5),     # short-time energy standard deviation, dB
    "rate_std": (0.05, 0.55),     # variation in syllable rate across the turn
}

#: How much each acoustic component contributes. Pitch dynamics carry most of
#: the affect in speech, so they get the larger share.
ACOUSTIC_WEIGHTS: Dict[str, float] = {
    "f0_cv": 0.34,
    "f0_span": 0.26,
    "energy_std": 0.24,
    "rate_std": 0.16,
}


def _anchor(value: float, key: str) -> float:
    lo, hi = ANCHORS[key]
    if not np.isfinite(value):
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


# --------------------------------------------------------------------------
# Lexical side
# --------------------------------------------------------------------------


def turn_lexical_arousal(turn: Turn, intent_score: Optional[float] = None) -> float:
    """How much pressure this turn's wording carries, in [0, 1].

    Built from the weighted lexicon so it stays inspectable: every point of
    arousal traces back to a specific matched phrase that the evidence panel
    can highlight. When an intent model score is available it is blended in,
    because the model catches phrasings the lexicon does not list.
    """
    hits = lexicon_hits(turn.text)
    score = 0.0
    for family, matches in hits.items():
        w = PRESSURE_WEIGHTS.get(family, 0.0)
        if not matches:
            continue
        # saturating sum: three threats are worse than one, but not three times
        # worse, and one long turn should not dominate the whole call
        s = sum(weight for _, weight in matches)
        score += w * (1.0 - np.exp(-0.9 * s))

    n_tok = max(len(turn.tokens), 1)
    density = score / (1.0 + np.log1p(n_tok / 12.0))
    lex = float(np.clip(density / 1.6, 0.0, 1.0))

    if intent_score is not None and np.isfinite(intent_score):
        lex = float(np.clip(0.6 * lex + 0.4 * intent_score, 0.0, 1.0))
    return lex


def lexical_arousal(
    call: Call,
    intent_scores: Optional[Sequence[float]] = None,
    caller_only: bool = True,
) -> Tuple[np.ndarray, List[int]]:
    """Per-turn lexical arousal plus the turn indices it was computed over."""
    turns = call.caller_turns() if caller_only else call.turns
    out = []
    for k, t in enumerate(turns):
        s = None
        if intent_scores is not None and k < len(intent_scores):
            s = float(intent_scores[k])
        out.append(turn_lexical_arousal(t, s))
    return np.asarray(out, dtype=np.float64), [t.index for t in turns]


# --------------------------------------------------------------------------
# Acoustic side
# --------------------------------------------------------------------------


def segment_arousal(x: np.ndarray, sr: int) -> Dict[str, float]:
    """Acoustic arousal of one audio segment, plus the parts it came from."""
    from ..dsp.prosody import prosody_summary

    if x is None or len(x) < int(0.12 * sr):
        return {"arousal": 0.0, "f0_cv": 0.0, "f0_span": 0.0,
                "energy_std": 0.0, "rate_std": 0.0, "voiced_ratio": 0.0}

    p = prosody_summary(np.asarray(x, dtype=np.float32), sr)
    f0m = float(p.get("f0_mean", 0.0)) or 0.0
    f0s = float(p.get("f0_std", 0.0)) or 0.0
    f0r = float(p.get("f0_range", 0.0)) or 0.0

    parts = {
        "f0_cv": _anchor(f0s / f0m if f0m > 1e-6 else 0.0, "f0_cv"),
        "f0_span": _anchor(f0r / f0m if f0m > 1e-6 else 0.0, "f0_span"),
        "energy_std": _anchor(float(p.get("energy_std", 0.0)), "energy_std"),
        "rate_std": _anchor(float(p.get("rate_std", 0.0)), "rate_std"),
    }
    arousal = sum(ACOUSTIC_WEIGHTS[k] * v for k, v in parts.items())

    # An unvoiced or near-silent segment has no prosody to measure. Reporting
    # 0 there would fake a mismatch, so the value is pulled toward the neutral
    # 0.5 in proportion to how little voicing there was.
    vr = float(p.get("voiced_ratio", 0.0))
    if vr < 0.25:
        conf = float(np.clip(vr / 0.25, 0.0, 1.0))
        arousal = conf * arousal + (1.0 - conf) * 0.5

    out = {"arousal": float(np.clip(arousal, 0.0, 1.0)), "voiced_ratio": vr}
    out.update(parts)
    return out


def acoustic_arousal(
    x: np.ndarray,
    sr: int,
    spans: Sequence[Tuple[float, float]],
    cache: Optional[Dict[Any, Dict[str, float]]] = None,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Acoustic arousal for each (start, end) span, in seconds.

    A turn's span never changes, so passing a `cache` dict makes the streaming
    path linear instead of quadratic. Without it, scoring an n-turn call turn
    by turn recomputes pitch tracking n(n+1)/2 times, which on a 40 second
    call is about ten seconds of pointless work.
    """
    vals: List[float] = []
    detail: List[Dict[str, float]] = []
    for (a, b) in spans:
        key = (round(float(a), 3), round(float(b), 3), int(sr))
        if cache is not None and key in cache:
            d = cache[key]
        else:
            i0 = max(0, int(round(a * sr)))
            i1 = min(len(x), int(round(b * sr)))
            d = segment_arousal(x[i0:i1] if i1 > i0 else np.zeros(0, dtype=np.float32), sr)
            if cache is not None:
                cache[key] = d
        vals.append(d["arousal"])
        detail.append(d)
    return np.asarray(vals, dtype=np.float64), detail


# --------------------------------------------------------------------------
# The mismatch
# --------------------------------------------------------------------------


def prosody_intent_mismatch(
    lex: np.ndarray,
    aco: np.ndarray,
    turn_indices: Optional[Sequence[int]] = None,
    times: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Combine the two arousal traces into one score in [0, 1].

    Two things are measured and blended:

      gap          how far the voice falls short of the words, weighted toward
                   the turns where the words carry the most pressure. A turn
                   that says nothing threatening contributes nothing, which is
                   what stops benign calls from scoring.
      incoherence  how uncorrelated the two traces are across the call. Real
                   speakers ramp their delivery with their content; a fixed
                   TTS voice does not, so the correlation collapses.
    """
    lex = np.asarray(lex, dtype=np.float64).ravel()
    aco = np.asarray(aco, dtype=np.float64).ravel()
    n = min(lex.size, aco.size)
    lex, aco = lex[:n], aco[:n]

    if n == 0:
        return {"pim": 0.0, "gap": 0.0, "incoherence": 0.0, "corr": 0.0,
                "per_turn": [], "n": 0}

    gaps = np.clip(lex - aco, 0.0, 1.0)
    w = lex.copy()
    wsum = float(w.sum())
    gap = float((w * gaps).sum() / wsum) if wsum > 1e-9 else 0.0

    corr = 0.0
    if n >= 3 and lex.std() > 1e-6 and aco.std() > 1e-6:
        corr = float(np.corrcoef(lex, aco)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0
    incoherence = float(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))

    # The correlation term only earns its weight once there is enough pressure
    # in the call for the correlation to mean anything. On a call with no
    # coercive language at all, a low correlation is noise, not evidence.
    pressure = float(np.clip(lex.max(initial=0.0), 0.0, 1.0))
    pim = 0.65 * gap + 0.35 * incoherence * pressure
    pim = float(np.clip(pim, 0.0, 1.0))

    per_turn = []
    for i in range(n):
        row: Dict[str, Any] = {
            "lex": round(float(lex[i]), 4),
            "aco": round(float(aco[i]), 4),
            "gap": round(float(gaps[i]), 4),
        }
        if turn_indices is not None and i < len(turn_indices):
            row["turn_index"] = int(turn_indices[i])
        if times is not None and i < len(times):
            row["t"] = round(float(times[i]), 3)
        per_turn.append(row)

    return {
        "pim": pim,
        "gap": round(gap, 4),
        "incoherence": round(incoherence, 4),
        "corr": round(corr, 4),
        "pressure": round(pressure, 4),
        "per_turn": per_turn,
        "n": int(n),
    }


def compute_pim(
    call: Call,
    audio: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
    intent_scores: Optional[Sequence[float]] = None,
    cache: Optional[Dict[Any, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Whole-call convenience wrapper used by the featuriser and the API."""
    lex, idx = lexical_arousal(call, intent_scores, caller_only=True)
    turns = call.caller_turns()
    spans = [(t.t_start, t.t_end) for t in turns]
    times = [t.t_end for t in turns]

    if audio is None or sr is None or not len(spans):
        # No audio available. The gap term is undefined, so the feature stays
        # at 0 rather than guessing, and the detail says why.
        out = {"pim": 0.0, "gap": 0.0, "incoherence": 0.0, "corr": 0.0,
               "per_turn": [{"lex": round(float(v), 4), "aco": 0.0, "gap": 0.0,
                             "turn_index": i, "t": t}
                            for v, i, t in zip(lex, idx, times)],
               "n": int(lex.size), "available": False}
        return out

    aco, adetail = acoustic_arousal(audio, sr, spans, cache=cache)
    out = prosody_intent_mismatch(lex, aco, idx, times)
    out["available"] = True
    for row, d in zip(out["per_turn"], adetail):
        row["voiced_ratio"] = round(float(d.get("voiced_ratio", 0.0)), 3)
    return out


if __name__ == "__main__":
    import json

    from ..schema import Turn as T

    scam = Call(
        call_id="demo_scam", label_scam=1, label_voice="synthetic",
        turns=[
            T(index=0, speaker="caller", text="Namaste, main cyber crime branch se bol raha hoon", t_start=0.0, t_end=3.0),
            T(index=1, speaker="caller", text="Aapke account se illegal transaction hua hai", t_start=3.5, t_end=6.5),
            T(index=2, speaker="caller", text="Das minute mein account block ho jayega, turant OTP batao", t_start=7.0, t_end=11.0),
            T(index=3, speaker="caller", text="Kisi ko mat batana, line pe raho warna warrant nikal jayega", t_start=11.5, t_end=15.5),
        ],
    )
    lex, idx = lexical_arousal(scam)
    print("lexical arousal per caller turn:", [round(v, 3) for v in lex])

    # a flat synthetic delivery against an animated human one
    flat = np.full(len(lex), 0.18)
    animated = np.array([0.30, 0.52, 0.78, 0.86])[: len(lex)]
    print("\nflat delivery (cloned):")
    print(json.dumps({k: v for k, v in prosody_intent_mismatch(lex, flat).items()
                      if k != "per_turn"}, indent=2))
    print("\nanimated delivery (human scammer):")
    print(json.dumps({k: v for k, v in prosody_intent_mismatch(lex, animated).items()
                      if k != "per_turn"}, indent=2))
