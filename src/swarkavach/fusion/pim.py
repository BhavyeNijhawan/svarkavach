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

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import SETTINGS
from ..corpus.lexicon import PRESSURE_WEIGHTS, lexicon_hits
from ..schema import Call, Turn

# Anchors for the absolute normalisation. Each is (low, high): a value at or
# below `low` maps to 0, at or above `high` maps to 1. They are held fixed so
# the feature means the same thing across calls and across channel conditions,
# which per-call normalisation would destroy.
#
# These defaults are measured, not guessed. The first version of this file used
# hand-set values and every one of them was far too low: real turns saturated
# the scale 81 percent of the time, so acoustic arousal came out at 1.0
# everywhere and the mismatch gap was always zero. `calibrate_anchors` below
# re-derives them from whatever corpus is on disk, and `Pipeline.fit` runs it
# on every retrain, so a change to the synthesiser cannot silently break this
# feature again.
#: Measured over 217 rendered caller turns, pooled across both voice classes,
#: at the 10th and 90th percentile. Under these the arousal scale separates as
#: it should (human mean 0.65, synthetic mean 0.37, saturation 1 percent), and
#: every component points the right way.
DEFAULT_ANCHORS: Dict[str, Tuple[float, float]] = {
    "f0_cv": (0.219, 0.549),        # pitch standard deviation over pitch mean
    "f0_span": (0.738, 1.447),      # pitch range over pitch mean
    "emphasis_var": (1.944, 6.591), # spread of syllable peak levels, dB
    "rate_std": (0.427, 1.101),     # variation in syllable rate across the turn
}

ANCHOR_KEYS: Tuple[str, ...] = tuple(DEFAULT_ANCHORS)

_ANCHORS: Dict[str, Tuple[float, float]] = dict(DEFAULT_ANCHORS)
_ANCHORS_LOADED = False

#: How much each acoustic component contributes. Pitch dynamics carry most of
#: the affect in speech, so they get the larger share.
#:
#: `emphasis_var` replaced a plain frame-level energy standard deviation, which
#: was measured pointing the WRONG way: synthetic turns scored higher than
#: human ones (separation -1.44). The reason is that frame-level energy
#: variance is dominated by the vowel to consonant to silence alternation,
#: which is a fact about articulation, not about how emphatic a speaker is
#: being. Measuring the spread of SYLLABLE PEAK levels instead captures what
#: emphasis actually means: hitting some syllables harder than others.
ACOUSTIC_WEIGHTS: Dict[str, float] = {
    "f0_cv": 0.34,
    "f0_span": 0.24,
    "emphasis_var": 0.26,
    "rate_std": 0.16,
}


def anchors() -> Dict[str, Tuple[float, float]]:
    """The anchors currently in force, loading the calibrated file once."""
    global _ANCHORS_LOADED
    if not _ANCHORS_LOADED:
        _ANCHORS_LOADED = True
        load_anchors()
    return _ANCHORS


def set_anchors(new: Dict[str, Sequence[float]]) -> None:
    """Replace the anchors in memory. Missing keys keep their current value."""
    global _ANCHORS_LOADED
    for k, v in (new or {}).items():
        if k in DEFAULT_ANCHORS and v is not None and len(v) == 2:
            lo, hi = float(v[0]), float(v[1])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                _ANCHORS[k] = (lo, hi)
    _ANCHORS_LOADED = True


def anchor_path() -> Path:
    from ..config import MODELS_DIR

    return Path(MODELS_DIR) / "pim_anchors.json"


def load_anchors(path: Optional[Path] = None) -> bool:
    """Load calibrated anchors if they exist. Returns whether any were found."""
    p = Path(path) if path else anchor_path()
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    set_anchors(d.get("anchors", d))
    return True


def _anchor(value: float, key: str) -> float:
    lo, hi = anchors()[key]
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


def syllable_emphasis_var(x: np.ndarray, sr: int) -> float:
    """Spread, in dB, of the peak level of each syllable in a segment.

    Emphasis is a relative thing: it means some syllables are hit harder than
    their neighbours. So the quantity to measure is the variation ACROSS
    syllable nuclei, not the variation across all frames. The frame-level
    version mostly measures how loud vowels are compared to consonants and
    pauses, which barely changes with how animated a speaker is.

    Peaks are picked from the smoothed energy envelope with the same
    prominence and spacing rules the speaking-rate estimator uses, so the two
    features agree about what counts as a syllable.
    """
    from scipy import signal as ssig

    from ..dsp.framing import frame_signal, short_time_energy

    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < int(0.2 * sr):
        return 0.0

    frame_len = max(8, int(round(0.025 * sr)))
    hop = max(4, int(round(0.010 * sr)))
    try:
        frames = frame_signal(x, frame_len, hop, window="hamming")
        energy = short_time_energy(frames)
    except Exception:
        return 0.0
    if energy.size < 5:
        return 0.0

    db = 10.0 * np.log10(np.maximum(energy, 1e-12))
    db = np.convolve(db, np.ones(5) / 5.0, mode="same")

    min_dist = max(int(round(0.080 * sr / hop)), 1)
    peaks, _ = ssig.find_peaks(db, prominence=3.0, distance=min_dist)
    # Ignore peaks buried in the noise floor: those are not syllables.
    if peaks.size:
        floor = np.percentile(db, 20) + 6.0
        peaks = peaks[db[peaks] > floor]
    if peaks.size < 2:
        return 0.0
    v = float(np.std(db[peaks]))
    return v if np.isfinite(v) else 0.0


def segment_components(x: np.ndarray, sr: int) -> Optional[Dict[str, float]]:
    """The four raw acoustic quantities for one segment, before anchoring.

    Split out from `segment_arousal` so that `calibrate_anchors` measures
    exactly what the feature will later consume. Returns None when the segment
    is too short to say anything about.
    """
    from ..dsp.prosody import prosody_summary

    if x is None or len(x) < int(0.12 * sr):
        return None

    p = prosody_summary(np.asarray(x, dtype=np.float32), sr)
    f0m = float(p.get("f0_mean", 0.0)) or 0.0
    f0s = float(p.get("f0_std", 0.0)) or 0.0
    f0r = float(p.get("f0_range", 0.0)) or 0.0
    return {
        "f0_cv": f0s / f0m if f0m > 1e-6 else 0.0,
        "f0_span": f0r / f0m if f0m > 1e-6 else 0.0,
        "emphasis_var": syllable_emphasis_var(x, sr),
        "rate_std": float(p.get("rate_std", 0.0)),
        "voiced_ratio": float(p.get("voiced_ratio", 0.0)),
    }


def segment_arousal(x: np.ndarray, sr: int) -> Dict[str, float]:
    """Acoustic arousal of one audio segment, plus the parts it came from."""
    raw = segment_components(x, sr)
    if raw is None:
        return {"arousal": 0.0, "f0_cv": 0.0, "f0_span": 0.0,
                "energy_std": 0.0, "rate_std": 0.0, "voiced_ratio": 0.0}

    p = {"voiced_ratio": raw["voiced_ratio"]}
    parts = {k: _anchor(raw[k], k) for k in ANCHOR_KEYS}
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


def calibrate_anchors(
    calls: Sequence[Call],
    max_calls: int = 160,
    low_pct: float = 10.0,
    high_pct: float = 90.0,
    save: bool = True,
) -> Dict[str, Any]:
    """Re-derive the anchors from corpus audio and optionally save them.

    Percentiles are taken over the pooled population, human and synthetic
    together, so the resulting scale spans the range both classes actually
    occupy. Calibrating on one class alone would push the other against a
    clip boundary and throw away the very difference the feature reads.

    Returns a report including the per-class medians, which is what tells you
    whether each component is pointing the right way. A component whose
    synthetic median sits ABOVE its human median is working against the
    feature, and that is worth knowing before trusting the number.
    """
    from pathlib import Path as _Path

    from ..audioio import read_audio

    raw: Dict[str, List[float]] = {k: [] for k in ANCHOR_KEYS}
    by_class: Dict[str, Dict[str, List[float]]] = {
        "human": {k: [] for k in ANCHOR_KEYS},
        "synthetic": {k: [] for k in ANCHOR_KEYS},
    }
    n_used = 0

    for call in list(calls)[:max_calls]:
        if not call.audio_path or not _Path(call.audio_path).exists():
            continue
        try:
            x, sr = read_audio(call.audio_path, sr=SETTINGS.frame.sr)
        except Exception:
            continue
        cls = "synthetic" if call.label_voice == "synthetic" else "human"
        used_this_call = False
        for t in call.caller_turns():
            i0, i1 = int(t.t_start * sr), int(t.t_end * sr)
            comp = segment_components(x[i0:i1], sr) if i1 > i0 else None
            if comp is None:
                continue
            for k in ANCHOR_KEYS:
                v = float(comp[k])
                if np.isfinite(v):
                    raw[k].append(v)
                    by_class[cls][k].append(v)
            used_this_call = True
        n_used += int(used_this_call)

    report: Dict[str, Any] = {
        "n_calls": n_used,
        "n_turns": len(raw[ANCHOR_KEYS[0]]),
        "percentiles": [low_pct, high_pct],
        "anchors": {},
        "medians": {},
        "direction_ok": {},
    }
    if report["n_turns"] < 30:
        report["status"] = "not enough audio, keeping the current anchors"
        return report

    new: Dict[str, Tuple[float, float]] = {}
    for k in ANCHOR_KEYS:
        arr = np.asarray(raw[k], dtype=np.float64)
        lo, hi = np.percentile(arr, [low_pct, high_pct])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
            new[k] = _ANCHORS[k]
        else:
            new[k] = (float(lo), float(hi))
        h = by_class["human"][k]
        s = by_class["synthetic"][k]
        mh = float(np.median(h)) if h else float("nan")
        ms = float(np.median(s)) if s else float("nan")
        report["medians"][k] = {"human": round(mh, 5), "synthetic": round(ms, 5)}
        # arousal is supposed to be HIGHER for human speech
        report["direction_ok"][k] = bool(np.isfinite(mh) and np.isfinite(ms) and mh > ms)
        report["anchors"][k] = [round(new[k][0], 5), round(new[k][1], 5)]

    set_anchors(new)
    report["status"] = "calibrated"
    if save:
        p = anchor_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["path"] = str(p)
    return report


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
