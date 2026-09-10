"""Turn a call (and optionally its audio) into the 25-dimensional fusion vector.

This is the seam between the two branches. Each branch fills its own group of
`schema.FUSION_FEATURES` and the cross-modal block is computed here, where
both sides are visible at once.

Everything is defensive on purpose. A missing model, missing audio or a branch
that raises must degrade to a usable vector rather than take down the call,
because the console has to keep working while models are still being trained.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import SETTINGS
from ..schema import (
    Call, Turn, EntitySpan, BranchScore, decode_bio, empty_feature_dict,
    ENTITY_TYPES,
)
from ..corpus.lexicon import lexicon_hits, PRESSURE_WEIGHTS
from .pim import compute_pim

# entity type -> fusion feature name
ENT_FEATURE = {
    "OTP": "ent_otp",
    "AUTHORITY_CLAIM": "ent_authority",
    "THREAT_DEADLINE": "ent_threat",
    "PAYMENT_HANDLE": "ent_payment",
    "PERSONAL_INFO_REQ": "ent_personal",
    "BANK_ENTITY": "ent_bank",
}


@dataclass
class ModelBundle:
    """Whatever models happen to be loaded. Every field may be None."""

    antispoof: Any = None
    ner: Any = None
    intent: Any = None
    rules: Any = None
    acthmm: Any = None
    fusion: Any = None
    embedder: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def backends(self) -> Dict[str, str]:
        def nm(o, default="none"):
            return type(o).__name__ if o is not None else default
        return {
            "antispoof": getattr(self.antispoof, "backend", None) or nm(self.antispoof),
            "ner": nm(self.ner, "rules"),
            "intent": nm(self.intent, "rules"),
            "fusion": nm(self.fusion, "untrained"),
        }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _safe(fn, default, *a, **kw):
    try:
        v = fn(*a, **kw)
        return default if v is None else v
    except Exception:
        return default


def _finite(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _turn_intent(models: ModelBundle, turn: Turn) -> float:
    """Score one turn for scam intent, falling back down the model ladder."""
    for m in (models.intent, models.rules):
        if m is None:
            continue
        for args in ((turn.tokens,), (turn.text,)):
            try:
                v = m.score_turn(*args)
                if v is not None and np.isfinite(float(v)):
                    return float(np.clip(float(v), 0.0, 1.0))
            except Exception:
                continue
    return _lexicon_turn_score(turn)


def _lexicon_turn_score(turn: Turn) -> float:
    """Last-resort intent score straight from the weighted lexicon."""
    hits = lexicon_hits(turn.text)
    s = 0.0
    for family, matches in hits.items():
        w = PRESSURE_WEIGHTS.get(family, 0.0)
        if matches:
            s += w * (1.0 - np.exp(-0.8 * sum(x for _, x in matches)))
    return float(np.clip(0.5 + 0.32 * s, 0.0, 1.0)) if s else 0.12


#: How much new audio has to arrive before the voice branch is re-scored
#: during streaming. Below this the previous score is reused. Voice
#: authenticity is a property of the speaker, not of the last half second, so
#: re-running the detector on every turn buys nothing and costs a lot.
VOICE_RESCORE_GAP_S = 4.0

#: Longest stretch of audio the voice branch looks at while streaming. Scoring
#: the whole prefix every time makes a call quadratic in its own length: a 60
#: second call ends up pushing about 12 minutes of audio through the front end.
#: A trailing window keeps each re-score constant-cost, and it is arguably the
#: better measurement anyway, since a spoofing artefact is a local property of
#: the signal and averaging it over a long call only dilutes it. The batch
#: path (cache=None) still scores the whole call in one pass.
VOICE_WINDOW_S = 20.0


def _score_voice(models: ModelBundle, audio: np.ndarray, sr: int,
                 cache: Optional[Dict[str, Any]]) -> BranchScore:
    """Anti-spoof score for the audio heard so far, with streaming throttled."""
    if cache is None:
        return models.antispoof.score(audio, sr)

    dur = len(audio) / float(sr or 1)
    last = cache.get("voice_at_s", -1e9)
    if cache.get("voice") is not None and (dur - last) < VOICE_RESCORE_GAP_S:
        return cache["voice"]

    n_win = int(VOICE_WINDOW_S * sr)
    clip = audio[-n_win:] if len(audio) > n_win else audio
    bs = models.antispoof.score(clip, sr)
    cache["voice"] = bs
    cache["voice_at_s"] = dur
    return bs


def tag_entities(
    models: ModelBundle,
    call: Call,
    use_gold: bool = False,
    notes: Optional[Dict[str, Any]] = None,
) -> List[EntitySpan]:
    """Predicted entity spans for a call, or the gold ones when explicitly asked.

    Falling back to `turn.bio` when the tagger is missing or throws was a
    silent evaluation leak: `turn.bio` is the GOLD annotation, so a held-out
    call scored with a broken tagger produced entity counts identical to the
    ground truth, and every entity feature plus the ablation table inherited
    that. An absent tagger now yields no entities at all, which is the honest
    answer, and the reason is recorded so it shows up instead of hiding.
    """
    spans: List[EntitySpan] = []
    n_failed = 0
    for t in call.turns:
        if use_gold:
            bio = t.bio
        elif models.ner is None:
            bio = ["O"] * len(t.tokens)
        else:
            bio = _safe(models.ner.predict_turn, None, t.tokens)
            if not bio or len(bio) != len(t.tokens):
                bio = ["O"] * len(t.tokens)
                n_failed += 1
        spans.extend(decode_bio(t.tokens, bio, turn_index=t.index))

    if notes is not None:
        if use_gold:
            notes["entity_source"] = "gold"
        elif models.ner is None:
            notes["entity_source"] = "none (no tagger loaded)"
        elif n_failed:
            notes["entity_source"] = f"predicted ({n_failed} turns failed)"
        else:
            notes["entity_source"] = "predicted"
    return spans


# --------------------------------------------------------------------------
# the featuriser
# --------------------------------------------------------------------------


def build_features(
    call: Call,
    audio: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
    models: Optional[ModelBundle] = None,
    use_gold_entities: bool = False,
    cache: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Return (feature dict with all 25 keys, detail dict for the evidence panel).

    `cache` is an optional per-call scratch dict. The streaming path passes one
    so that per-turn prosody and the voice branch are not recomputed from
    scratch on every prefix. Passing None (the batch path) simply recomputes,
    which is what you want when calls are independent.
    """
    models = models or ModelBundle()
    F = empty_feature_dict()
    detail: Dict[str, Any] = {}
    timing: Dict[str, float] = {}

    # ---------------------------------------------------------- voice group
    t0 = time.perf_counter()
    voice: Optional[BranchScore] = None
    if models.antispoof is not None and audio is not None and sr:
        try:
            voice = _score_voice(models, audio, sr, cache)
        except Exception as exc:
            detail["antispoof_error"] = f"{type(exc).__name__}: {exc}"
    if voice is not None:
        for k, v in (voice.features or {}).items():
            if k in F:
                F[k] = _finite(v)
        F["as_score"] = _finite(voice.score)
        detail["antispoof"] = voice.to_dict()
    else:
        # No audio path. Leave the voice group at zero and say so, rather than
        # inventing numbers that the fusion model would then trust.
        detail["antispoof"] = {
            "name": "antispoof", "score": 0.0, "backend": "unavailable",
            "raw": 0.0, "features": {}, "detail": {"reason": "no audio supplied"},
        }
    timing["antispoof_ms"] = (time.perf_counter() - t0) * 1000.0

    # --------------------------------------------------------- intent group
    t0 = time.perf_counter()
    caller = call.caller_turns()
    turn_scores = [_turn_intent(models, t) for t in caller]
    detail["turn_intent"] = [round(float(s), 4) for s in turn_scores]

    if turn_scores:
        arr = np.asarray(turn_scores, dtype=np.float64)
        # Max plus mean pooling: a scam needs only one extraction attempt to be
        # a scam, so the peak matters, but the mean stops a single unlucky turn
        # from carrying a benign call.
        F["intent_max_turn"] = float(arr.max())
        F["intent_score"] = float(np.clip(0.6 * arr.max() + 0.4 * arr.mean(), 0.0, 1.0))
    if models.intent is not None and hasattr(models.intent, "score_call"):
        try:
            cs = models.intent.score_call(call)
            if isinstance(cs, dict):
                # the pooler returns "intent_score"; matching only on "score"
                # meant this branch never fired, the dict fell through to
                # _finite(dict) which raised, and the exception was swallowed,
                # so the trained call-level model was silently unused
                key = next((k for k in ("intent_score", "score") if k in cs), None)
                if key is not None:
                    F["intent_score"] = _finite(cs[key], F["intent_score"])
                detail["intent_call"] = cs
            elif cs is not None:
                F["intent_score"] = _finite(cs, F["intent_score"])
        except Exception:
            pass

    spans = tag_entities(models, call, use_gold=use_gold_entities, notes=detail)
    detail["entities"] = [s.to_dict() for s in spans]
    n_tok = max(sum(len(t.tokens) for t in call.turns), 1)
    counts: Dict[str, int] = {}
    for s in spans:
        counts[s.type] = counts.get(s.type, 0) + 1
    for etype, fname in ENT_FEATURE.items():
        # per 100 tokens, so a long call does not automatically look worse
        F[fname] = float(100.0 * counts.get(etype, 0) / n_tok)
    detail["entity_counts"] = counts

    urgency = 0
    lex_spans: List[Dict[str, Any]] = []
    for t in call.turns:
        hits = lexicon_hits(t.text)
        for family in ("urgency", "threat", "isolation"):
            urgency += len(hits.get(family, []))
        for family, matches in hits.items():
            if family == "benign" or not matches:
                continue
            for term, weight in matches:
                lex_spans.append({"turn_index": t.index, "term": term,
                                  "type": family, "weight": float(weight)})
    F["urgency_density"] = float(100.0 * urgency / n_tok)
    detail["lexicon_spans"] = lex_spans
    timing["intent_ms"] = (time.perf_counter() - t0) * 1000.0

    # ---------------------------------------------------------- cross group
    t0 = time.perf_counter()

    # dialogue acts and the coercion trajectory
    acts: List[str] = []
    try:
        from ..text.coercion import act_classifier, coercion_features
        # same trap as the entity tagger: defaulting to t.act would hand the
        # gold dialogue act back on any failure
        acts = [_safe(act_classifier, "INFORM", t) for t in call.turns]
        cf = coercion_features(call, acts=acts)
        for k in ("coercion_slope", "coercion_peak", "callee_resist"):
            F[k] = _finite(cf.get(k), 0.0)
        if models.acthmm is not None:
            F["act_scam_llr"] = float(np.clip(_finite(_safe(models.acthmm.llr, 0.0, acts)), -10, 10))
        else:
            F["act_scam_llr"] = float(np.clip(_finite(cf.get("act_scam_llr"), 0.0), -10, 10))
        detail["acts"] = acts
        detail["coercion"] = cf
    except Exception as exc:
        detail["coercion_error"] = f"{type(exc).__name__}: {exc}"

    # code switching profile
    try:
        from ..text.langid import code_mixing_features, tag_languages
        toks: List[str] = []
        langs: List[str] = []
        bio: List[str] = []
        span_lookup = {(s.turn_index, i): s.type
                       for s in spans for i in range(s.tok_start, s.tok_end)}
        for t in call.turns:
            tl = t.lang if len(t.lang) == len(t.tokens) else _safe(tag_languages, [], t.tokens)
            if len(tl) != len(t.tokens):
                tl = ["univ"] * len(t.tokens)
            for i, tok in enumerate(t.tokens):
                toks.append(tok)
                langs.append(tl[i])
                et = span_lookup.get((t.index, i))
                bio.append(f"B-{et}" if et else "O")
        cm = code_mixing_features(toks, langs, bio)
        F["cmi"] = _finite(cm.get("cmi"))
        F["switch_entropy"] = _finite(cm.get("switch_entropy"))
        F["ent_lang_align"] = _finite(cm.get("ent_lang_align"))
        detail["code_mixing"] = cm
    except Exception as exc:
        detail["langid_error"] = f"{type(exc).__name__}: {exc}"

    # the cross-modal feature
    try:
        pim_cache = cache.setdefault("pim_segments", {}) if cache is not None else None
        pim = compute_pim(call, audio=audio, sr=sr, intent_scores=turn_scores,
                          cache=pim_cache)
        F["pim"] = _finite(pim.get("pim"))
        detail["pim"] = pim
    except Exception as exc:
        detail["pim_error"] = f"{type(exc).__name__}: {exc}"
        detail["pim"] = {"pim": 0.0, "per_turn": [], "available": False}

    timing["cross_ms"] = (time.perf_counter() - t0) * 1000.0
    detail["latency_ms"] = {k: round(v, 2) for k, v in timing.items()}

    for k in list(F):
        F[k] = _finite(F[k])
    return F, detail


def featurize_corpus(
    calls: Sequence[Call],
    models: Optional[ModelBundle] = None,
    with_audio: bool = True,
    progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]], List[Call]]:
    """Featurise a whole corpus. Returns (X, y, feature dicts, calls kept)."""
    from ..audioio import read_audio
    from pathlib import Path
    from ..schema import feature_vector

    X: List[List[float]] = []
    y: List[int] = []
    feats: List[Dict[str, float]] = []
    kept: List[Call] = []

    it = enumerate(calls)
    for i, c in it:
        audio = sr = None
        if with_audio and c.audio_path and Path(c.audio_path).exists():
            try:
                audio, sr = read_audio(c.audio_path, sr=SETTINGS.frame.sr)
            except Exception:
                audio = sr = None
        f, _ = build_features(c, audio=audio, sr=sr, models=models)
        X.append(feature_vector(f))
        y.append(int(c.label_scam))
        feats.append(f)
        kept.append(c)
        if progress and (i + 1) % 25 == 0:
            print(f"  featurised {i + 1}/{len(calls)}", flush=True)

    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64), feats, kept


if __name__ == "__main__":
    from ..schema import Turn as T

    call = Call(
        call_id="demo", label_scam=1, label_voice="synthetic", scenario="digital_arrest",
        turns=[
            T(index=0, speaker="caller", text="Namaste, main cyber crime branch se inspector bol raha hoon", t_start=0.0, t_end=3.4),
            T(index=1, speaker="callee", text="Ji kya baat hai", t_start=3.6, t_end=4.6),
            T(index=2, speaker="caller", text="Aapke account se illegal transaction hua hai, das minute mein block ho jayega", t_start=5.0, t_end=9.4),
            T(index=3, speaker="caller", text="Turant OTP batao aur kisi ko mat batana", t_start=9.8, t_end=12.8),
        ],
    )
    F, D = build_features(call)
    print("non-zero features:")
    for k, v in F.items():
        if abs(v) > 1e-9:
            print(f"  {k:20s} {v:8.4f}")
    print("\nentity counts:", D.get("entity_counts"))
    print("turn intent:", D.get("turn_intent"))
    print("pim:", round(D.get("pim", {}).get("pim", 0), 4))
    print("latency:", D.get("latency_ms"))
