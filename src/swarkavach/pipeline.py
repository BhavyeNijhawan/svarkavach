"""End to end: a call goes in, an explainable verdict comes out.

`Pipeline` owns the loaded corpus and whatever models exist on disk, and it is
the only object the Flask server and the CLI talk to. Every model is optional.
If nothing has been trained yet the pipeline still runs, using the rule
baselines and a documented heuristic fusion, and says so in `status()`. That
matters because a console that refuses to start until training finishes is a
console nobody uses while they are debugging training.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from . import config
from .audioio import read_audio, write_wav
from .config import SETTINGS, artifact_path
from .schema import (
    Call, Turn, Verdict, Evidence, feature_vector, risk_band, tokenize,
)
from .fusion.featurize import ModelBundle, build_features
from .fusion.model import FusionModel
from .fusion.explain import build_evidence
from .fusion.streaming import stream_call, time_to_detection, score_features


class Pipeline:
    """Loaded corpus plus loaded models, with the two entry points on top."""

    def __init__(self, calls: Optional[List[Call]] = None,
                 models: Optional[ModelBundle] = None):
        self.calls: List[Call] = calls or []
        self.models: ModelBundle = models or ModelBundle()
        self._by_id: Dict[str, Call] = {c.call_id: c for c in self.calls}
        self._extra: Dict[str, Call] = {}     # uploads and freshly synthesised calls
        self._load_notes: List[str] = []

    # ------------------------------------------------------------ loading

    @classmethod
    def load(cls, with_corpus: bool = True) -> "Pipeline":
        config.ensure_dirs()
        calls: List[Call] = []
        notes: List[str] = []

        if with_corpus:
            try:
                from .corpus.generator import load_corpus
                calls = load_corpus()
            except Exception as exc:
                notes.append(f"corpus not loaded: {type(exc).__name__}: {exc}")

        models = ModelBundle()

        # anti-spoofing
        try:
            from .antispoof.scorer import AntiSpoofScorer
            p = artifact_path("antispoof.joblib")
            models.antispoof = AntiSpoofScorer.load(str(p)) if p.exists() else AntiSpoofScorer()
        except Exception as exc:
            notes.append(f"antispoof unavailable: {type(exc).__name__}: {exc}")

        # entity tagger
        try:
            from .text.ner_crf import CRFTagger
            p = artifact_path("ner_crf.joblib")
            if p.exists():
                models.ner = CRFTagger.load(str(p))
        except Exception as exc:
            notes.append(f"ner unavailable: {type(exc).__name__}: {exc}")

        # intent
        try:
            from .text.intent_rules import RuleIntent
            models.rules = RuleIntent()
        except Exception as exc:
            notes.append(f"rule intent unavailable: {type(exc).__name__}: {exc}")
        try:
            from .text.intent_tfidf import TfidfIntent
            p = artifact_path("intent_tfidf.joblib")
            if p.exists():
                models.intent = TfidfIntent.load(str(p))
        except Exception as exc:
            notes.append(f"tfidf intent unavailable: {type(exc).__name__}: {exc}")

        # dialogue act Markov model
        try:
            from .text.coercion import ActHMM
            p = artifact_path("act_hmm.joblib")
            if p.exists():
                models.acthmm = ActHMM.load(str(p))
        except Exception as exc:
            notes.append(f"act model unavailable: {type(exc).__name__}: {exc}")

        # fusion
        try:
            p = artifact_path("fusion_full_logreg.joblib")
            if p.exists():
                models.fusion = FusionModel.load(str(p))
        except Exception as exc:
            notes.append(f"fusion unavailable: {type(exc).__name__}: {exc}")

        obj = cls(calls, models)
        obj._load_notes = notes
        return obj

    def status(self) -> Dict[str, Any]:
        b = self.models.backends()
        return {
            "antispoof_backend": b.get("antispoof"),
            "ner_backend": b.get("ner"),
            "intent_backend": b.get("intent"),
            "fusion_model": (
                f"{self.models.fusion.kind} / {self.models.fusion.arm}"
                if self.models.fusion is not None and self.models.fusion.trained else None
            ),
            "asr_backend": self._asr_backend(),
            "n_calls": len(self.calls),
            "notes": self._load_notes,
        }

    @staticmethod
    def _asr_backend() -> str:
        try:
            from .text.asr import ASR
            # ASR exposes resolve(), not a `backend` attribute. Reading the
            # wrong name raised AttributeError straight into the except, so
            # this always reported "gold" even with Whisper installed.
            return str(ASR().resolve())
        except Exception as exc:
            return f"unavailable ({type(exc).__name__})"

    # ------------------------------------------------------------ lookups

    def get_call(self, call_id: str) -> Optional[Call]:
        return self._by_id.get(call_id) or self._extra.get(call_id)

    def add_call(self, call: Call) -> Call:
        self._extra[call.call_id] = call
        return call

    # ----------------------------------------------------------- analysis

    def analyze(
        self,
        call: Call,
        audio: Optional[np.ndarray] = None,
        sr: Optional[int] = None,
        streaming: bool = True,
        use_gold_entities: bool = False,
    ) -> Verdict:
        """Score one call and assemble its full evidence panel."""
        t0 = time.perf_counter()
        features, detail = build_features(
            call, audio=audio, sr=sr, models=self.models,
            use_gold_entities=use_gold_entities,
        )
        risk, how = score_features(self.models, features)

        contributions: List[Dict[str, Any]] = []
        if self.models.fusion is not None and getattr(self.models.fusion, "trained", False):
            try:
                contributions = self.models.fusion.explain(features)
            except Exception:
                contributions = []
        if not contributions:
            contributions = self._heuristic_contributions(features)

        evidence = build_evidence(call, features, contributions, detail, risk)

        timeline: List[Dict[str, Any]] = []
        ttd = ttd_turns = None
        if streaming:
            try:
                timeline = list(self.stream(call, audio=audio, sr=sr))
                ttd, ttd_turns = time_to_detection(timeline)
            except Exception as exc:
                detail["stream_error"] = f"{type(exc).__name__}: {exc}"

        ab = detail.get("antispoof", {}) or {}
        pim = detail.get("pim", {}) or {}
        latency = dict(detail.get("latency_ms", {}))
        latency["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        return Verdict(
            call_id=call.call_id,
            risk=float(risk),
            band=risk_band(risk),
            authenticity=float(features.get("as_score", 0.0)),
            intent=float(features.get("intent_score", 0.0)),
            pim=float(features.get("pim", 0.0)),
            branches={
                "antispoof": ab,
                "intent": {
                    "name": "intent",
                    "score": float(features.get("intent_score", 0.0)),
                    "backend": self.models.backends().get("intent", "rules"),
                    "detail": {"turn_scores": detail.get("turn_intent", [])},
                },
                "pim": {"name": "pim", "score": float(features.get("pim", 0.0)),
                        "backend": "cross-modal", "detail": pim},
                "coercion": {"name": "coercion",
                             "score": float(features.get("coercion_slope", 0.0)),
                             "backend": "acts", "detail": detail.get("coercion", {})},
            },
            features={k: float(v) for k, v in features.items()},
            evidence=evidence,
            timeline=timeline,
            ttd=ttd,
            ttd_turns=ttd_turns,
            latency_ms=latency,
            transcript_source=str(call.meta.get("transcript_source", "gold")),
            label_scam=call.label_scam,
        )

    def stream(
        self, call: Call, audio: Optional[np.ndarray] = None, sr: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        return stream_call(call, audio=audio, sr=sr, models=self.models)

    @staticmethod
    def _heuristic_contributions(features: Dict[str, float]) -> List[Dict[str, Any]]:
        """Stand-in attributions for the untrained path.

        These are the fixed weights from the heuristic fusion, presented on the
        same scale as the Shapley values so the panel does not change shape
        depending on whether a model has been trained.
        """
        from .schema import FEATURE_GROUPS, FEATURE_LABELS

        w = {"intent_score": 0.30, "as_score": 0.22, "pim": 0.20,
             "coercion_slope": 0.14, "ent_otp": 0.06, "ent_personal": 0.05,
             "ent_authority": 0.04, "ent_threat": 0.04}
        out = []
        for k, weight in w.items():
            v = float(features.get(k, 0.0))
            out.append({
                "feature": k,
                "label": FEATURE_LABELS.get(k, k),
                "group": FEATURE_GROUPS.get(k, "intent"),
                "value": v,
                "contribution": round(weight * v * 4.0, 5),
            })
        out.sort(key=lambda d: abs(d["contribution"]), reverse=True)
        return out

    # ------------------------------------------------------- new material

    def synthesize_call(
        self,
        scenario: Optional[str] = None,
        scam: Optional[bool] = None,
        voice: Optional[str] = None,
        with_audio: bool = True,
        seed: Optional[int] = None,
    ) -> Call:
        """Generate a brand new call, so a live demo is never just a replay."""
        from .corpus.generator import generate_one

        call = generate_one(
            scenario=scenario, scam=scam, voice=voice,
            seed=seed if seed is not None else int(uuid.uuid4().int % (2 ** 31)),
        )
        if with_audio:
            try:
                from .corpus.tts import render_call_audio
                out = config.CORPUS_AUDIO_DIR / f"{call.call_id}.wav"
                render_call_audio(call, backend="sim", out_path=str(out))
                call.audio_path = str(out)
            except Exception:
                call.audio_path = None
        return self.add_call(call)

    def register_upload(
        self,
        x: np.ndarray,
        sr: int,
        name: str = "upload.wav",
        transcript: Optional[str] = None,
    ) -> Call:
        """Take uploaded audio and make it look like a corpus call.

        With no transcript the text branch has nothing to work with, so the
        turn list is a single placeholder and the verdict rests on the voice
        branch alone. The console states that rather than pretending
        otherwise.
        """
        config.ensure_dirs()
        call_id = f"upload_{uuid.uuid4().hex[:8]}"
        path = config.UPLOADS_DIR / f"{call_id}.wav"
        write_wav(str(path), x, sr)
        dur = float(len(x)) / float(sr or 1)

        turns: List[Turn] = []
        source = "manual"
        if transcript and transcript.strip():
            lines = [ln.strip() for ln in transcript.strip().splitlines() if ln.strip()]
            step = dur / max(len(lines), 1)
            for i, ln in enumerate(lines):
                speaker = "caller"
                text = ln
                if ":" in ln[:12]:
                    who, rest = ln.split(":", 1)
                    if who.strip().lower() in ("caller", "callee", "agent", "victim", "customer"):
                        speaker = "callee" if who.strip().lower() in ("callee", "victim", "customer") else "caller"
                        text = rest.strip()
                turns.append(Turn(index=i, speaker=speaker, text=text,
                                  t_start=round(i * step, 3), t_end=round((i + 1) * step, 3)))
        else:
            source = "none"
            turns.append(Turn(index=0, speaker="caller",
                              text="(no transcript supplied)",
                              t_start=0.0, t_end=round(dur, 3)))

        call = Call(
            call_id=call_id, turns=turns, label_scam=0, label_voice="human",
            scenario="uploaded", speaker_id="uploaded", split="none",
            audio_path=str(path), sample_rate=sr, channel="clean",
            audio_source="recorded",
            meta={"original_name": name, "transcript_source": source},
        )
        return self.add_call(call)

    # -------------------------------------------------------------- train

    def fit(self, calls: Optional[Sequence[Call]] = None, verbose: bool = True) -> Dict[str, Any]:
        """Train every trainable component from the corpus, in dependency order."""
        from .fusion.featurize import featurize_corpus
        from .fusion.model import train_all_arms

        calls = list(calls or self.calls)
        train = [c for c in calls if c.split == "train"] or calls
        # The anti-spoof scorer holds out a slice to fit its Platt
        # calibrator on, and prefers rows marked "dev" because the corpus
        # splits are speaker disjoint. Passing it only the train rows meant
        # no row was ever labelled dev, so it fell back to a random split of
        # the training pool and calibrated on speakers it had just fitted on.
        dev = [c for c in calls if c.split == "dev"]
        train_and_dev = train + dev
        report: Dict[str, Any] = {}

        def say(msg):
            if verbose:
                print(msg, flush=True)

        # 1. entity tagger
        try:
            from .text.ner_crf import CRFTagger
            say("training CRF entity tagger")
            ner = CRFTagger()
            ner.fit(train)
            ner.save(str(artifact_path("ner_crf.joblib")))
            self.models.ner = ner
            report["ner"] = "trained"
        except Exception as exc:
            report["ner"] = f"failed: {type(exc).__name__}: {exc}"
            say(f"  ner failed: {exc}")

        # 2. intent classifier
        try:
            from .text.intent_tfidf import TfidfIntent
            say("training TF-IDF intent classifier")
            it = TfidfIntent()
            it.fit(train)
            it.save(str(artifact_path("intent_tfidf.joblib")))
            self.models.intent = it
            report["intent"] = "trained"
        except Exception as exc:
            report["intent"] = f"failed: {type(exc).__name__}: {exc}"
            say(f"  intent failed: {exc}")

        # 3. dialogue act Markov model
        try:
            from .text.coercion import ActHMM
            say("fitting dialogue act sequence model")
            # Inference passes the rule classifier's predicted acts to llr(),
            # so fitting on gold acts trains on a different distribution than
            # it is ever scored on. On a generated corpus the gold act
            # sequence is a deterministic function of the scenario arc, which
            # makes that gap worse, not better.
            hmm = ActHMM(use_gold=False)
            hmm.fit(train)
            hmm.save(str(artifact_path("act_hmm.joblib")))
            self.models.acthmm = hmm
            report["act_hmm"] = "trained"
        except Exception as exc:
            report["act_hmm"] = f"failed: {type(exc).__name__}: {exc}"
            say(f"  act model failed: {exc}")

        # 4. prosody-intent anchors, before anything reads the audio.
        # The anchors set the scale acoustic arousal is measured on, so they
        # have to be right before the fusion features are built. Recalibrating
        # here means a change to the synthesiser or a new recording set cannot
        # silently push every turn against a clip boundary, which is exactly
        # what happened with the first hand-set values.
        try:
            from .fusion.pim import calibrate_anchors
            say("calibrating prosody-intent anchors")
            rep = calibrate_anchors(train)
            report["pim_anchors"] = {
                "status": rep.get("status"), "n_turns": rep.get("n_turns"),
                "anchors": rep.get("anchors"), "direction_ok": rep.get("direction_ok"),
            }
            bad = [k for k, ok in (rep.get("direction_ok") or {}).items() if not ok]
            if bad:
                say(f"  warning: these components read higher for synthetic "
                    f"speech than for human, so they fight the feature: {bad}")
        except Exception as exc:
            report["pim_anchors"] = f"failed: {type(exc).__name__}: {exc}"
            say(f"  anchor calibration failed: {exc}")

        # 5. anti-spoofing
        try:
            from .antispoof.scorer import AntiSpoofScorer
            say("training anti-spoofing branch")
            sc = self.models.antispoof or AntiSpoofScorer()
            sc.fit(train_and_dev)
            sc.save(str(artifact_path("antispoof.joblib")))
            self.models.antispoof = sc
            report["antispoof"] = getattr(sc, "backend", "trained")
        except Exception as exc:
            report["antispoof"] = f"failed: {type(exc).__name__}: {exc}"
            say(f"  antispoof failed: {exc}")

        # 6. fusion, which needs every branch above to be in place first.
        #
        # It is featurised on `dev`, NOT on `train`. The fusion model stacks on
        # top of the branches, so featurising the branches' own training rows
        # feeds it in-sample predictions: the CRF scores entity F1 1.000 there
        # against 0.887 on held-out data, and the resulting feature
        # distribution never occurs at test time. Worse, the shift is uneven
        # across branches, which is exactly what the audio-only against
        # text-only against full comparison is supposed to measure.
        fusion_calls = dev if len(dev) >= 40 else train
        if fusion_calls is train:
            say(f"  dev has only {len(dev)} calls, falling back to train for "
                f"fusion (the branch scores will be optimistic)")
        say(f"featurising {len(fusion_calls)} {'dev' if fusion_calls is dev else 'train'} "
            f"calls for fusion")
        X, y, feats, kept = featurize_corpus(fusion_calls, models=self.models,
                                             with_audio=True, progress=verbose)
        say(f"  {X.shape[0]} calls, {X.shape[1]} features")
        models = train_all_arms(X, y, kind="logreg")
        for arm, m in models.items():
            m.save(str(artifact_path(f"fusion_{arm}_logreg.joblib")))
        gbm = FusionModel("gbm", "full").fit(X, y)
        gbm.save(str(artifact_path("fusion_full_gbm.joblib")))
        self.models.fusion = models.get("full")
        report["fusion"] = {arm: m.train_meta for arm, m in models.items()}
        report["fusion_fitted_on"] = "dev" if fusion_calls is dev else "train"

        # a reference distribution for the radar chart in the console
        try:
            ref = {}
            for j, name in enumerate(__import__(
                    "swarkavach.schema", fromlist=["x"]).FUSION_FEATURE_NAMES):
                col = X[:, j]
                ref[name] = {"p05": float(np.percentile(col, 5)),
                             "p50": float(np.percentile(col, 50)),
                             "p95": float(np.percentile(col, 95))}
            (config.RESULTS_DIR / "feature_reference.json").write_text(
                json.dumps(ref, indent=2), encoding="utf-8")
        except Exception:
            pass

        return report


def analyze_file(
    audio_path: str,
    transcript: Optional[str] = None,
    pipeline: Optional[Pipeline] = None,
) -> Verdict:
    """Convenience entry point for scripts: one WAV in, one Verdict out."""
    p = pipeline or Pipeline.load()
    x, sr = read_audio(audio_path, sr=SETTINGS.frame.sr)
    sidecar = Path(audio_path).with_suffix(".json")
    if transcript is None and sidecar.exists():
        call = Call.load(str(sidecar))
        call.audio_path = audio_path
    else:
        call = p.register_upload(x, sr, name=Path(audio_path).name, transcript=transcript)
    return p.analyze(call, audio=x, sr=sr)


if __name__ == "__main__":
    p = Pipeline.load()
    print(json.dumps(p.status(), indent=2))
    if p.calls:
        c = p.calls[0]
        v = p.analyze(c)
        print(f"\n{c.call_id}  truth={'fraud' if c.label_scam else 'legit'}  "
              f"risk={v.risk:.3f} ({v.band})  ttd={v.ttd}")
        for r in v.evidence.reasons[:5]:
            print("  -", r["text"])
    else:
        print("\nNo corpus yet. Run: swarkavach gen-corpus")
