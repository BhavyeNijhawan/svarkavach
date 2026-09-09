"""The anti-spoof branch's public face: pick a back end, score a call, calibrate.

`AntiSpoofScorer` is what the rest of the pipeline imports. It hides which
detector is available and guarantees three things:

1. It always returns a `BranchScore`. A fresh checkout with no trained models
   still produces a number, through a documented heuristic over jitter,
   shimmer, spectral flatness variance and embedding consistency. A demo that
   crashes because nobody ran the training script is worse than a demo with a
   weak detector, and the heuristic is not that weak: on the simulated signals
   it separates human from synthetic cleanly.
2. The number is calibrated. Late fusion combines this branch with the text
   branch, and combining two uncalibrated scores means the fusion weights spend
   themselves fixing the scales instead of finding the interaction. So `fit`
   holds out a slice, fits Platt scaling on it, and everything downstream sees a
   probability that means what it says.
3. The eight `voice` group features from `schema.FUSION_FEATURES` come back with
   every score, filled in, finite.

Back end order when `backend="auto"`: rawnet, then gbm, then gmm, then heuristic.
Rawnet first because it sees the waveform and can catch artefacts a cepstral
front end smooths away; gbm before gmm because a discriminative model on pooled
statistics beats a pair of densities when the training corpus is small.

Run the self-test with:
    python -m swarkavach.antispoof.scorer
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..audioio import read_audio
from ..config import SETTINGS, TARGET_SR, artifact_path
from ..schema import BranchScore, FEATURE_GROUPS, FUSION_FEATURE_NAMES
from .embeddings import SpeakerEmbedder, cosine_consistency
from .features import antispoof_feature_dict, frame_features, utterance_features
from .gbm import GBMScorer
from .gmm import GMMScorer
from . import rawnet as rawnet_mod

try:
    import joblib
except Exception:  # pragma: no cover - environment dependent
    joblib = None

__all__ = ["AntiSpoofScorer", "PlattCalibrator", "VOICE_FEATURES", "BACKEND_ORDER"]

#: The eight fusion features this branch owns, taken from the schema rather than
#: retyped, so a change there shows up here as a test failure and not as a
#: silently missing key.
VOICE_FEATURES: Tuple[str, ...] = tuple(
    n for n in FUSION_FEATURE_NAMES if FEATURE_GROUPS[n] == "voice"
)

#: Preference order for backend="auto".
BACKEND_ORDER: Tuple[str, ...] = ("rawnet", "gbm", "gmm", "heuristic")

_EPS = 1e-12


def _sigmoid(z: float) -> float:
    z = float(np.clip(z, -40.0, 40.0))
    return 1.0 / (1.0 + math.exp(-z))


def _logit(p) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return math.log(p / (1.0 - p))


def _finite(v, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float(default)
    return f if math.isfinite(f) else float(default)


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


class PlattCalibrator:
    """One-dimensional logistic map from a detector score to a probability.

    Platt scaling: p = sigmoid(a * s + b), with a and b fitted by maximum
    likelihood on held-out scores. Two details that matter on a corpus of a few
    hundred calls:

    - The targets are Platt's corrected ones, (N+ + 1)/(N+ + 2) instead of 1 and
      1/(N- + 2) instead of 0. Fitting to hard 0/1 targets on a small held-out
      set drives the slope toward infinity and produces a calibrator that only
      ever says 0.001 or 0.999.
    - It is fitted on data the detector did not train on. Calibrating on the
      training scores would learn the overfit, not the detector.
    """

    def __init__(self) -> None:
        self.a: float = 1.0
        self.b: float = 0.0
        self.fitted: bool = False
        self.meta_: Dict[str, Any] = {}

    def fit(self, raw, y) -> "PlattCalibrator":
        s = np.asarray(raw, dtype=np.float64).ravel()
        y = np.asarray(y).ravel().astype(int)
        keep = np.isfinite(s)
        s, y = s[keep], y[keep]
        n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
        if n_pos < 1 or n_neg < 1 or s.size < 4 or np.ptp(s) < 1e-9:
            self.meta_ = {"status": "skipped", "n_pos": n_pos, "n_neg": n_neg}
            return self

        t_pos = (n_pos + 1.0) / (n_pos + 2.0)
        t_neg = 1.0 / (n_neg + 2.0)
        target = np.where(y == 1, t_pos, t_neg)

        try:
            from sklearn.linear_model import LogisticRegression

            # Soft targets through the standard duplicate-and-weight trick:
            # each score appears once as a positive with weight t and once as a
            # negative with weight 1 - t, which is exactly Platt's likelihood.
            X = np.concatenate([s, s])[:, None]
            yy = np.concatenate([np.ones_like(s), np.zeros_like(s)])
            w = np.concatenate([target, 1.0 - target])
            lr = LogisticRegression(C=1e4, solver="lbfgs", max_iter=1000)
            lr.fit(X, yy, sample_weight=w)
            self.a = float(lr.coef_[0, 0])
            self.b = float(lr.intercept_[0])
            self.fitted = True
            self.meta_ = {"status": "fitted", "n_pos": n_pos, "n_neg": n_neg,
                          "a": self.a, "b": self.b}
        except Exception as exc:  # pragma: no cover - sklearn should be present
            self.meta_ = {"status": f"failed: {exc}"}
        return self

    def transform(self, raw) -> float:
        if not self.fitted:
            return _sigmoid(float(raw))
        return _sigmoid(self.a * float(raw) + self.b)

    def to_dict(self) -> Dict[str, Any]:
        return {"a": self.a, "b": self.b, "fitted": self.fitted, "meta": self.meta_}

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "PlattCalibrator":
        obj = cls()
        if d:
            obj.a = float(d.get("a", 1.0))
            obj.b = float(d.get("b", 0.0))
            obj.fitted = bool(d.get("fitted", False))
            obj.meta_ = dict(d.get("meta", {}))
        return obj


# --------------------------------------------------------------------------
# The heuristic fallback
# --------------------------------------------------------------------------

#: Reference values and weights for the untrained fallback. Each reference is
#: the boundary, not the human value: a term is zero at the reference, negative
#: for a human-looking call and positive for a machine-looking one. The numbers
#: come from the simulated bona fide and spoofed signals in `features.py` and
#: from the ranges quoted in the prosody module (modal jitter 0.005 to 0.02,
#: modal shimmer 0.03 to 0.15). They are a starting point that `fit` replaces
#: with a trained model the first time anyone runs the training script.
HEURISTIC = {
    # (reference, kind, scale, weight). A ratio term reads log(reference /
    # value), so it is scale free: half the reference and twice the reference
    # are equal and opposite. A difference term reads (value - reference)/scale.
    #
    # The weights are ordered by how well each term ranked the simulated calls
    # on its own (AUC in brackets, from the hardest condition, where the two
    # classes differ only in micro-variation):
    "jitter":            {"ref": 0.0080, "kind": "ratio", "scale": 1.00, "w": 1.10},
    "shimmer":           {"ref": 0.0450, "kind": "ratio", "scale": 1.00, "w": 0.90},
    # Flatness variance gets the smallest weight because it is the only term
    # measured flipping sign: when a vocoder's over-regular noise floor is the
    # dominant artefact it drops as expected, but when the synthesis keeps the
    # pauses and fricatives and only smooths the excitation, quieter aspiration
    # widens the voiced-to-silent flatness gap and it rises instead.
    "spec_flatness_var": {"ref": 0.0022, "kind": "ratio", "scale": 1.00, "w": 0.35},
    "hnr":               {"ref": 13.0,   "kind": "diff",  "scale": 6.00, "w": 0.55},
    # Consistency needs segments of a second or more to mean anything, so on a
    # short clip this term is mostly noise. It is weighted for that.
    "spk_consistency":   {"ref": 0.0,    "kind": "diff",  "scale": 0.10, "w": 0.45},
}

#: Terms are clipped to this many units before weighting, so one wild
#: measurement (a call with no voiced frames, say) cannot swing the verdict on
#: its own.
_TERM_CLIP = 2.0


def _heuristic_terms(feats: Dict[str, float]) -> Dict[str, float]:
    """Per-feature contribution to the fallback score, positive meaning spoofed."""
    out: Dict[str, float] = {}
    for name, spec in HEURISTIC.items():
        value = _finite(feats.get(name, 0.0))
        if spec["kind"] == "ratio":
            # A measurement of exactly zero means "could not measure", which is
            # itself weak evidence of over-regularity, so it floors at the clip
            # rather than running off to infinity.
            ratio = spec["ref"] / max(value, 1e-6)
            term = math.log(max(ratio, 1e-6))
        else:
            term = (value - spec["ref"]) / spec["scale"]
        out[name] = float(np.clip(term, -_TERM_CLIP, _TERM_CLIP)) * spec["w"]
    return out


def heuristic_score(feats: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """Fallback raw score (a log odds) and its per-feature breakdown.

    Positive means the call looks synthetic. The four contract features plus the
    embedding consistency each push in the direction the literature says they
    should: a vocoder is too steady (low jitter, low shimmer), too clean (high
    HNR), too regular in its noise floor (low flatness variance) and too stable
    across segments (high consistency).
    """
    terms = _heuristic_terms(feats)
    return float(sum(terms.values())), terms


# --------------------------------------------------------------------------
# The scorer
# --------------------------------------------------------------------------


class AntiSpoofScorer:
    """Chooses a back end, scores a call, and reports the voice feature group."""

    def __init__(
        self,
        backend: str = "auto",
        feature_set: str = "lfcc",
        n_segments: int = 4,
        seed: Optional[int] = None,
    ) -> None:
        self.backend = str(backend).lower()
        self.feature_set = str(feature_set).lower()
        self.n_segments = int(n_segments)
        self.seed = int(SETTINGS.pipeline.seed if seed is None else seed)

        self.gmm: Optional[GMMScorer] = None
        self.gbm: Optional[GBMScorer] = None
        self.rawnet = None
        self.embedder = SpeakerEmbedder(seed=self.seed)
        self.calibrator = PlattCalibrator()
        self.meta_: Dict[str, Any] = {}

    # -- back end selection -----------------------------------------------

    def available(self) -> List[str]:
        """Which back ends could run right now, in preference order."""
        out = []
        if self.rawnet is not None:
            out.append("rawnet")
        if self.gbm is not None and self.gbm.is_fitted:
            out.append("gbm")
        if self.gmm is not None and self.gmm.is_fitted:
            out.append("gmm")
        out.append("heuristic")
        return out

    def active_backend(self) -> str:
        """The back end `score` will actually use."""
        if self.backend in ("auto", "", None):
            for name in BACKEND_ORDER:
                if name in self.available():
                    return name
            return "heuristic"
        if self.backend in self.available():
            return self.backend
        # An explicit request for something untrained falls back rather than
        # raising: the pipeline must keep running.
        return "heuristic"

    def load_artifacts(self, feature_set: Optional[str] = None) -> "AntiSpoofScorer":
        """Pick up whatever has been trained and left in the models directory."""
        fs = feature_set or self.feature_set
        try:
            p = GMMScorer.default_path(fs)
            if p.exists():
                self.gmm = GMMScorer.load(p)
        except Exception:
            self.gmm = None
        try:
            p = GBMScorer.default_path(fs)
            if p.exists():
                self.gbm = GBMScorer.load(p)
        except Exception:
            self.gbm = None
        self.rawnet = rawnet_mod.load_rawnet()
        emb = SpeakerEmbedder.load(artifact_path("antispoof_embedder.pt"))
        if emb is not None:
            self.embedder = emb
        return self

    # -- scoring ----------------------------------------------------------

    def _raw_for(self, backend: str, x, sr: int,
                 feats: Dict[str, float]) -> Tuple[float, float]:
        """(raw score, calibration input) for one back end.

        The two differ for the probability valued back ends: the raw number the
        caller sees is a probability, but Platt scaling wants something that is
        roughly linear in the log odds, so the calibrator is fed the logit.
        """
        if backend == "rawnet" and self.rawnet is not None:
            p = float(self.rawnet.score(x, sr))
            return p, _logit(p)
        if backend == "gbm" and self.gbm is not None and self.gbm.is_fitted:
            p = float(self.gbm.score(utterance_features(x, sr, self.feature_set)))
            return p, _logit(p)
        if backend == "gmm" and self.gmm is not None and self.gmm.is_fitted:
            llr = float(feats.get("as_llr", 0.0))
            return llr, llr
        raw, _ = heuristic_score(feats)
        return raw, raw

    def score(self, x, sr: int = TARGET_SR) -> BranchScore:
        """Score one call. Returns a BranchScore with the voice feature group."""
        t0 = time.time()
        backend = self.active_backend()

        gmm = self.gmm if (self.gmm is not None and self.gmm.is_fitted) else None
        feats = antispoof_feature_dict(x, sr, gmm=gmm, feature_set=self.feature_set)

        try:
            embs = self.embedder.embed_segments(x, sr, self.n_segments)
            consistency = cosine_consistency(embs)
        except Exception:
            embs, consistency = np.zeros((1, 1)), 1.0
        feats["spk_consistency"] = _finite(consistency, 1.0)

        raw, calib_in = self._raw_for(backend, x, sr, feats)
        prob = self.calibrator.transform(calib_in)
        prob = float(np.clip(_finite(prob, 0.5), 0.0, 1.0))

        # Distance from the detector's own decision boundary, rescaled to
        # [0, 1]. A confident call sits near 1, a coin flip near 0.
        margin = float(np.clip(2.0 * abs(prob - 0.5), 0.0, 1.0))

        _, terms = heuristic_score(feats)
        features = {
            "as_score": prob,
            "as_llr": _finite(feats.get("as_llr", 0.0)),
            "as_margin": margin,
            "spk_consistency": _finite(feats.get("spk_consistency", 1.0), 1.0),
            "jitter": _finite(feats.get("jitter", 0.0)),
            "shimmer": _finite(feats.get("shimmer", 0.0)),
            "hnr": _finite(feats.get("hnr", 0.0)),
            "spec_flatness_var": _finite(feats.get("spec_flatness_var", 0.0)),
        }
        detail = {
            "diagnostics": {k: v for k, v in feats.items() if k.startswith("dx_")},
            "heuristic_terms": terms,
            "calibrated": bool(self.calibrator.fitted),
            "available": self.available(),
            "n_segments": int(np.atleast_2d(embs).shape[0]),
            "feature_set": self.feature_set,
            "latency_ms": round(1000.0 * (time.time() - t0), 2),
        }
        return BranchScore(name="antispoof", score=prob, backend=backend,
                           raw=_finite(raw), features=features, detail=detail)

    def score_batch(self, signals: Sequence, sr: int = TARGET_SR) -> np.ndarray:
        return np.array([self.score(x, sr).score for x in signals], dtype=np.float64)

    # -- training ---------------------------------------------------------

    def fit(
        self,
        calls: Sequence,
        sr: int = TARGET_SR,
        with_rawnet: bool = False,
        calib_frac: float = 0.25,
        rawnet_epochs: int = 10,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Train from a corpus of Call objects that carry audio.

        Calls with no readable audio are skipped rather than fatal, so a corpus
        that was generated without the audio pass still gets you a text branch.
        Returns a summary dict; look at `["error"]` if nothing could be trained.
        """
        signals, labels, splits = [], [], []
        for call in calls:
            path = getattr(call, "audio_path", None)
            if not path or not Path(str(path)).exists():
                continue
            try:
                sig, _ = read_audio(str(path), sr)
            except Exception:
                continue
            if np.asarray(sig).size < sr // 2:
                continue
            signals.append(np.asarray(sig, dtype=np.float32))
            labels.append(1 if getattr(call, "label_voice", "human") == "synthetic" else 0)
            splits.append(getattr(call, "split", "train"))
        if not signals:
            return {"error": "no calls with readable audio"}
        return self.fit_signals(signals, labels, sr=sr, splits=splits,
                                with_rawnet=with_rawnet, calib_frac=calib_frac,
                                rawnet_epochs=rawnet_epochs, verbose=verbose)

    def fit_signals(
        self,
        signals: Sequence,
        labels: Sequence[int],
        sr: int = TARGET_SR,
        splits: Optional[Sequence[str]] = None,
        with_rawnet: bool = False,
        calib_frac: float = 0.25,
        rawnet_epochs: int = 10,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """The body of `fit`, on waveforms rather than Call objects.

        Kept public because the tests and the Colab notebooks both want to train
        on arrays without writing WAV files first.
        """
        y = np.asarray(labels).ravel().astype(int)
        n = len(signals)
        if y.size != n:
            return {"error": "labels and signals have different lengths"}
        if np.unique(y).size < 2:
            return {"error": "training needs both human and synthetic calls"}

        train_idx, calib_idx = self._split(y, splits, calib_frac)
        summary: Dict[str, Any] = {
            "n_total": int(n), "n_train": int(train_idx.size),
            "n_calib": int(calib_idx.size), "feature_set": self.feature_set,
            "seed": self.seed, "backends": {},
        }
        if verbose:
            print(f"  fitting on {train_idx.size} calls, calibrating on {calib_idx.size}")

        # -- GMM over frames ----------------------------------------------
        try:
            bona = [frame_features(signals[i], sr, self.feature_set)
                    for i in train_idx if y[i] == 0]
            spoof = [frame_features(signals[i], sr, self.feature_set)
                     for i in train_idx if y[i] == 1]
            gmm = GMMScorer(feature_set=self.feature_set, seed=self.seed)
            gmm.fit(bona, spoof)
            self.gmm = gmm
            summary["backends"]["gmm"] = gmm.meta_
        except Exception as exc:
            summary["backends"]["gmm"] = {"error": str(exc)}

        # -- GBM over pooled vectors --------------------------------------
        try:
            X = np.vstack([utterance_features(signals[i], sr, self.feature_set)
                           for i in train_idx])
            gbm = GBMScorer(feature_set=self.feature_set, seed=self.seed)
            gbm.fit(X, y[train_idx])
            self.gbm = gbm
            summary["backends"]["gbm"] = gbm.meta_
        except Exception as exc:
            summary["backends"]["gbm"] = {"error": str(exc)}

        # -- RawNet, only when asked ---------------------------------------
        if with_rawnet:
            try:
                net, hist = rawnet_mod.train_rawnet(
                    [signals[i] for i in train_idx], y[train_idx], sr=sr,
                    epochs=rawnet_epochs, seed=self.seed, verbose=verbose)
                if net is not None:
                    self.rawnet = net
                summary["backends"]["rawnet"] = {
                    k: v for k, v in hist.items() if k != "path"}
            except Exception as exc:
                summary["backends"]["rawnet"] = {"error": str(exc)}

        # -- calibration on the held-out slice -----------------------------
        backend = self.active_backend()
        summary["active_backend"] = backend
        self.calibrator = PlattCalibrator()
        if calib_idx.size >= 4:
            candidates = self.available()
            raws: Dict[str, List[float]] = {name: [] for name in candidates}
            for i in calib_idx:
                feats = antispoof_feature_dict(
                    signals[i], sr,
                    gmm=self.gmm if (self.gmm and self.gmm.is_fitted) else None,
                    feature_set=self.feature_set)
                feats["spk_consistency"] = self.embedder.consistency(
                    signals[i], sr, self.n_segments)
                for name in candidates:
                    raws[name].append(self._raw_for(name, signals[i], sr, feats)[1])
            self.calibrator.fit(np.asarray(raws[backend]), y[calib_idx])
            summary["calibration"] = dict(self.calibrator.meta_)
            summary["calibration"]["dev_auc"] = _auc(np.asarray(raws[backend]),
                                                     y[calib_idx])
            # Every back end scored on the same held-out slice. The auto order is
            # fixed by contract, but the report needs to know what the others did.
            summary["dev_auc_by_backend"] = {
                name: _auc(np.asarray(vals), y[calib_idx]) for name, vals in raws.items()
            }
        else:
            summary["calibration"] = {"status": "skipped: held-out slice too small"}

        self.meta_ = summary
        return summary

    def _split(self, y: np.ndarray, splits: Optional[Sequence[str]],
               calib_frac: float) -> Tuple[np.ndarray, np.ndarray]:
        """Train and calibration indices, stratified, deterministic.

        If the corpus already carries a dev split, that is the calibration slice:
        the corpus splits are speaker disjoint and a calibration set that shares
        speakers with training is optimistic.
        """
        idx = np.arange(y.size)
        if splits is not None:
            splits = np.asarray([str(s) for s in splits])
            dev = idx[splits == "dev"]
            if dev.size >= 4 and np.unique(y[dev]).size == 2:
                return idx[splits != "dev"], dev

        rng = np.random.default_rng(self.seed)
        calib: List[int] = []
        for cls in (0, 1):
            cls_idx = idx[y == cls]
            rng.shuffle(cls_idx)
            take = int(round(calib_frac * cls_idx.size))
            calib.extend(cls_idx[:take].tolist())
        calib_arr = np.array(sorted(calib), dtype=int)
        train = np.array([i for i in idx if i not in set(calib)], dtype=int)
        if train.size < 4 or np.unique(y[train]).size < 2:
            return idx, calib_arr
        return train, calib_arr

    # -- persistence ------------------------------------------------------

    def save(self, path: Union[str, Path] = None) -> str:
        """Write the whole scorer: joblib for the sklearn side, torch alongside.

        The torch models go to `<path>.torch` rather than into the joblib blob,
        so a checkpoint stays loadable by torch on its own and joblib never has
        to pickle a live nn.Module.
        """
        if joblib is None:
            raise RuntimeError("joblib is required to save an AntiSpoofScorer")
        path = Path(path or artifact_path("antispoof_scorer.joblib"))
        path.parent.mkdir(parents=True, exist_ok=True)

        blob = {
            "kind": "AntiSpoofScorer",
            "backend": self.backend,
            "feature_set": self.feature_set,
            "n_segments": self.n_segments,
            "seed": self.seed,
            "gmm": None if self.gmm is None else {
                "feature_set": self.gmm.feature_set,
                "gmm_bona": self.gmm.gmm_bona,
                "gmm_spoof": self.gmm.gmm_spoof,
                "dim": self.gmm.dim_,
                "n_components": self.gmm.n_components_,
                "seed": self.gmm.seed,
                "meta": self.gmm.meta_,
            },
            "gbm": None if self.gbm is None else {
                "feature_set": self.gbm.feature_set,
                "model": self.gbm.model,
                "dim": self.gbm.dim_,
                "seed": self.gbm.seed,
                "meta": self.gbm.meta_,
            },
            "calibrator": self.calibrator.to_dict(),
            "embedder": {
                "feature_set": self.embedder.feature_set,
                "dim": self.embedder.dim,
                "width": self.embedder.width,
                "seed": self.embedder.seed,
                "n_in": self.embedder.n_in,
                "trained": self.embedder.trained,
                "center": (None if self.embedder.center is None
                           else np.asarray(self.embedder.center)),
            },
            "meta": self.meta_,
            "has_torch": False,
        }

        torch_path = Path(str(path) + ".torch")
        if rawnet_mod.TORCH_AVAILABLE:
            import torch

            state = {}
            if self.rawnet is not None:
                state["rawnet"] = {"config": self.rawnet.config(),
                                   "state_dict": self.rawnet.state_dict()}
            if self.embedder.net is not None:
                state["embedder"] = {"n_in": self.embedder.n_in,
                                     "state_dict": self.embedder.net.state_dict()}
            if state:
                torch.save(state, torch_path)
                blob["has_torch"] = True
        if not blob["has_torch"]:
            torch_path.unlink(missing_ok=True)

        joblib.dump(blob, path)
        return str(path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AntiSpoofScorer":
        if joblib is None:
            raise RuntimeError("joblib is required to load an AntiSpoofScorer")
        path = Path(path)
        blob = joblib.load(path)

        obj = cls(backend=blob.get("backend", "auto"),
                  feature_set=blob.get("feature_set", "lfcc"),
                  n_segments=int(blob.get("n_segments", 4)),
                  seed=blob.get("seed"))

        g = blob.get("gmm")
        if g:
            gmm = GMMScorer(feature_set=g.get("feature_set", "lfcc"), seed=g.get("seed"))
            gmm.gmm_bona = g.get("gmm_bona")
            gmm.gmm_spoof = g.get("gmm_spoof")
            gmm.dim_ = int(g.get("dim", 0))
            gmm.n_components_ = int(g.get("n_components", 0))
            gmm.meta_ = dict(g.get("meta", {}))
            obj.gmm = gmm

        b = blob.get("gbm")
        if b:
            gbm = GBMScorer(feature_set=b.get("feature_set", "lfcc"), seed=b.get("seed"))
            gbm.model = b.get("model")
            gbm.dim_ = int(b.get("dim", 0))
            gbm.meta_ = dict(b.get("meta", {}))
            if gbm.model is not None:
                gbm.classes_ = gbm.model.classes_
            obj.gbm = gbm

        obj.calibrator = PlattCalibrator.from_dict(blob.get("calibrator"))
        obj.meta_ = dict(blob.get("meta", {}))

        e = blob.get("embedder") or {}
        emb = SpeakerEmbedder(feature_set=e.get("feature_set", "mfcc"),
                              dim=int(e.get("dim", 192)),
                              width=int(e.get("width", 96)),
                              seed=e.get("seed"))
        if e.get("n_in"):
            emb._build(int(e["n_in"]))
        if e.get("center") is not None:
            emb.center = np.asarray(e["center"], dtype=np.float64)
        emb.trained = bool(e.get("trained", False))
        obj.embedder = emb

        torch_path = Path(str(path) + ".torch")
        if blob.get("has_torch") and rawnet_mod.TORCH_AVAILABLE and torch_path.exists():
            try:
                import torch

                state = torch.load(torch_path, map_location="cpu", weights_only=False)
                if "rawnet" in state:
                    cfg = dict(state["rawnet"]["config"])
                    net = rawnet_mod.RawNetLite(**cfg)
                    net.load_state_dict(state["rawnet"]["state_dict"])
                    net.eval()
                    obj.rawnet = net
                if "embedder" in state and obj.embedder.net is not None:
                    obj.embedder.net.load_state_dict(state["embedder"]["state_dict"])
                    obj.embedder.net.eval()
            except Exception:
                pass  # a stale torch sidecar must not stop the sklearn side loading
        return obj


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank based AUC, ties handled, no sklearn import needed."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
    # Average the ranks inside each tie group, otherwise ties bias the answer.
    uniq, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


if __name__ == "__main__":  # pragma: no cover - self-test
    from .features import demo_bonafide, demo_spoof

    t0 = time.time()
    sr = TARGET_SR
    print("antispoof.scorer self-test")

    scorer = AntiSpoofScorer()
    print(f"  fresh scorer: available={scorer.available()} "
          f"active={scorer.active_backend()}")

    bona = [demo_bonafide(sr, 3.0, seed=i, f0_base=105.0 + 9.0 * (i % 7)) for i in range(10)]
    spoof = [demo_spoof(sr, 3.0, seed=300 + i, f0_base=105.0 + 9.0 * (i % 7)) for i in range(10)]
    y = np.array([0] * 10 + [1] * 10)

    heur = np.array([scorer.score(x, sr).score for x in bona + spoof])
    print(f"  heuristic: bona fide {heur[y == 0].mean():.3f}, "
          f"spoofed {heur[y == 1].mean():.3f}, AUC {_auc(heur, y):.3f}")

    bs = scorer.score(bona[0], sr)
    print(f"  BranchScore: name={bs.name} backend={bs.backend} score={bs.score:.3f} "
          f"raw={bs.raw:+.3f} latency={bs.detail['latency_ms']:.0f} ms")
    print(f"  features: {sorted(bs.features)}")
    print(f"  covers the voice group: {sorted(bs.features) == sorted(VOICE_FEATURES)}")
    print("  heuristic terms: "
          + ", ".join(f"{k}={v:+.2f}" for k, v in bs.detail["heuristic_terms"].items()))

    fit_bona = [demo_bonafide(sr, 3.0, seed=50 + i, f0_base=100.0 + 11.0 * (i % 6))
                for i in range(14)]
    fit_spoof = [demo_spoof(sr, 3.0, seed=800 + i, f0_base=100.0 + 11.0 * (i % 6))
                 for i in range(14)]
    trained = AntiSpoofScorer()
    summary = trained.fit_signals(fit_bona + fit_spoof, [0] * 14 + [1] * 14,
                                 sr=sr, verbose=True)
    print(f"  trained backends: {list(summary['backends'])}, "
          f"active={summary['active_backend']}")
    print(f"  calibration: {summary['calibration']}")
    print("  held-out AUC per back end: "
          + ", ".join(f"{k}={v:.3f}"
                      for k, v in summary.get("dev_auc_by_backend", {}).items()))

    p = np.array([trained.score(x, sr).score for x in bona + spoof])
    print(f"  trained on held-out demo calls: bona fide {p[y == 0].mean():.3f}, "
          f"spoofed {p[y == 1].mean():.3f}, AUC {_auc(p, y):.3f}")

    path = artifact_path("selftest_scorer.joblib")
    trained.save(path)
    again = AntiSpoofScorer.load(path)
    p2 = np.array([again.score(x, sr).score for x in bona + spoof])
    print(f"  save/load round trip: max |delta| {np.max(np.abs(p - p2)):.2e}, "
          f"backend {again.active_backend()}")
    Path(path).unlink(missing_ok=True)
    Path(str(path) + ".torch").unlink(missing_ok=True)

    long_call = np.tile(demo_bonafide(sr, 3.0, seed=1), 10)
    t1 = time.time()
    trained.score(long_call, sr)
    print(f"  30 s call scored in {time.time() - t1:.2f} s")
    print(f"  done in {time.time() - t0:.1f} s")
