"""The LFCC-GMM baseline: two Gaussian mixtures and a log-likelihood ratio.

This is the ASVspoof 2019 and 2021 baseline system, and it is here for the same
reason the organisers shipped it: it is generative, it needs no negative mining,
it trains in seconds on CPU, and on unseen attacks it often loses to a neural
system by much less than people expect. It is also the only back end in this
branch whose score has a defined meaning outside the training set, which is why
its output goes into the fusion vector as `as_llr` and not just as a probability.

How it works: fit p_bona over the frames of bona fide calls and p_spoof over the
frames of spoofed calls, both diagonal covariance, then score a new utterance by

    llr = (1/T) * sum_t [ log p_spoof(x_t) - log p_bona(x_t) ]

Two decisions worth stating.

The average, not the sum. A sum grows with the number of frames, so a 40 second
call and a 5 second call would need different thresholds. Averaging makes the
score a per frame quantity and the threshold a property of the detector rather
than of the call length.

The sign. The ASVspoof convention is bona fide minus spoof, so that a higher
score means more genuine. This project's fusion contract says every feature with
direction +1 is larger when the call is more suspicious, and `as_llr` is one of
them, so the sign is flipped here: positive llr means the frames look spoofed.

Run the self-test with:
    python -m swarkavach.antispoof.gmm
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from ..config import SETTINGS, TARGET_SR, artifact_path
from .features import frame_features

try:  # joblib ships with scikit-learn, but do not assume it
    import joblib
except Exception:  # pragma: no cover - environment dependent
    joblib = None

from sklearn.mixture import GaussianMixture

__all__ = ["GMMScorer", "LLR_CLIP"]

#: The fusion contract clips log ratios to this range, so the scorer does too.
LLR_CLIP = 10.0

#: Frames per mixture component. The ASVspoof baseline uses 512 components on
#: tens of hours of audio, roughly two thousand frames per component. Asking for
#: 512 components when a small corpus supplies 400 frames gives singular
#: components and a likelihood that is pure overfit, so the count is derived
#: from the data instead of fixed.
FRAMES_PER_COMPONENT = 120

#: Never more than this many components, whatever the data size. Beyond about 64
#: the gain on telephone-band cepstra is small and the fit time is not.
MAX_COMPONENTS = 64
MIN_COMPONENTS = 2

#: Cap on the frames used for one fit. Above this the estimate has converged and
#: the extra frames only cost time.
MAX_FIT_FRAMES = 200_000


def _as_frame_matrix(X) -> np.ndarray:
    """Accept a frame matrix, a list of frame matrices, or a list of vectors."""
    if isinstance(X, np.ndarray) and X.ndim == 2:
        out = X
    elif isinstance(X, np.ndarray) and X.ndim == 1:
        out = X[None, :]
    else:
        parts = []
        for item in X:
            arr = np.asarray(item, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.size:
                parts.append(arr)
        if not parts:
            return np.zeros((0, 1), dtype=np.float64)
        width = min(p.shape[1] for p in parts)
        out = np.concatenate([p[:, :width] for p in parts], axis=0)
    return np.nan_to_num(np.asarray(out, dtype=np.float64),
                         nan=0.0, posinf=0.0, neginf=0.0)


class GMMScorer:
    """Two diagonal covariance GMMs and the frame averaged LLR between them."""

    def __init__(
        self,
        feature_set: str = "lfcc",
        n_components: Optional[int] = None,
        max_components: int = MAX_COMPONENTS,
        frames_per_component: int = FRAMES_PER_COMPONENT,
        reg_covar: Optional[float] = None,
        max_iter: int = 120,
        seed: Optional[int] = None,
    ) -> None:
        self.feature_set = str(feature_set).lower()
        self.n_components = n_components
        self.max_components = int(max_components)
        self.frames_per_component = int(max(frames_per_component, 1))
        self.reg_covar = reg_covar
        self.max_iter = int(max_iter)
        self.seed = int(SETTINGS.pipeline.seed if seed is None else seed)

        self.gmm_bona: Optional[GaussianMixture] = None
        self.gmm_spoof: Optional[GaussianMixture] = None
        self.dim_: int = 0
        self.n_components_: int = 0
        self.meta_: Dict[str, Any] = {}

    # -- state ------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self.gmm_bona is not None and self.gmm_spoof is not None

    def __repr__(self) -> str:  # pragma: no cover - display only
        state = (f"fitted k={self.n_components_} d={self.dim_}"
                 if self.is_fitted else "unfitted")
        return f"GMMScorer({self.feature_set}, {state})"

    # -- fitting ----------------------------------------------------------

    def _choose_k(self, n_frames: int) -> int:
        if self.n_components:
            return int(np.clip(int(self.n_components), MIN_COMPONENTS,
                               max(self.max_components, MIN_COMPONENTS)))
        k = n_frames // self.frames_per_component
        return int(np.clip(k, MIN_COMPONENTS, self.max_components))

    def _fit_one(self, X: np.ndarray, k: int, reg: float) -> GaussianMixture:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            reg_covar=reg,
            max_iter=self.max_iter,
            n_init=1,
            init_params="kmeans",
            random_state=self.seed,
        )
        gmm.fit(X)
        return gmm

    def _subsample(self, X: np.ndarray) -> np.ndarray:
        if X.shape[0] <= MAX_FIT_FRAMES:
            return X
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(X.shape[0], MAX_FIT_FRAMES, replace=False)
        return X[np.sort(idx)]

    def fit(self, X_bona, X_spoof) -> "GMMScorer":
        """Fit both mixtures. Each argument is a frame matrix or a list of them.

        Both models get the same component count so the ratio is not biased by
        one side simply having more capacity to fit its own data.
        """
        bona = self._subsample(_as_frame_matrix(X_bona))
        spoof = self._subsample(_as_frame_matrix(X_spoof))
        if bona.shape[0] < MIN_COMPONENTS or spoof.shape[0] < MIN_COMPONENTS:
            raise ValueError(
                f"need at least {MIN_COMPONENTS} frames per class, got "
                f"{bona.shape[0]} bona fide and {spoof.shape[0]} spoofed"
            )
        width = min(bona.shape[1], spoof.shape[1])
        bona, spoof = bona[:, :width], spoof[:, :width]

        k = min(self._choose_k(min(bona.shape[0], spoof.shape[0])),
                bona.shape[0], spoof.shape[0])
        k = max(k, MIN_COMPONENTS)

        # A ridge on the diagonal, scaled to the data. CMVN leaves the features
        # near unit variance, so this is about a thousandth of the spread: big
        # enough that a component holding a handful of near identical frames
        # cannot collapse to a spike, small enough not to blur the model.
        if self.reg_covar is None:
            spread = float(np.mean(np.var(np.vstack([bona, spoof]), axis=0)))
            reg = max(1e-6, 1e-3 * (spread if np.isfinite(spread) else 1.0))
        else:
            reg = float(self.reg_covar)

        self.gmm_bona = self._fit_one(bona, k, reg)
        self.gmm_spoof = self._fit_one(spoof, k, reg)
        self.dim_ = int(width)
        self.n_components_ = int(k)
        self.meta_ = {
            "feature_set": self.feature_set,
            "n_components": int(k),
            "dim": int(width),
            "reg_covar": float(reg),
            "n_frames_bona": int(bona.shape[0]),
            "n_frames_spoof": int(spoof.shape[0]),
            "converged_bona": bool(self.gmm_bona.converged_),
            "converged_spoof": bool(self.gmm_spoof.converged_),
            "seed": int(self.seed),
        }
        return self

    def fit_audio(self, bona_signals: Iterable, spoof_signals: Iterable,
                  sr: int = TARGET_SR) -> "GMMScorer":
        """Convenience wrapper: run the front end over two lists of waveforms."""
        bona = [frame_features(x, sr, self.feature_set) for x in bona_signals]
        spoof = [frame_features(x, sr, self.feature_set) for x in spoof_signals]
        return self.fit(bona, spoof)

    # -- scoring ----------------------------------------------------------

    def llr_frames(self, X) -> np.ndarray:
        """Per frame log-likelihood ratio, positive meaning spoofed."""
        if not self.is_fitted:
            raise RuntimeError("GMMScorer.llr called before fit")
        F = _as_frame_matrix(X)
        if F.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        if F.shape[1] != self.dim_:
            if F.shape[1] > self.dim_:
                F = F[:, : self.dim_]
            else:
                F = np.pad(F, ((0, 0), (0, self.dim_ - F.shape[1])))
        ll_bona = self.gmm_bona.score_samples(F)
        ll_spoof = self.gmm_spoof.score_samples(F)
        return np.nan_to_num(ll_spoof - ll_bona, nan=0.0, posinf=0.0, neginf=0.0)

    def llr(self, X) -> float:
        """Frame averaged log-likelihood ratio, clipped to the fusion range."""
        per_frame = self.llr_frames(X)
        if per_frame.size == 0:
            return 0.0
        return float(np.clip(np.mean(per_frame), -LLR_CLIP, LLR_CLIP))

    def llr_audio(self, x, sr: int = TARGET_SR) -> float:
        """Score a waveform end to end, running our own front end on it."""
        return self.llr(frame_features(x, sr, self.feature_set))

    def score(self, X) -> float:
        """P(spoof) from the LLR through a logistic, for a uniform back end API.

        This is not calibrated. The scorer that owns this object fits a Platt
        model on held-out data and uses that instead; this exists so the GMM can
        stand alone in a quick experiment.
        """
        return float(1.0 / (1.0 + np.exp(-self.llr(X))))

    # -- persistence ------------------------------------------------------

    def save(self, path: Union[str, Path]) -> str:
        if joblib is None:
            raise RuntimeError("joblib is required to save a GMMScorer")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "kind": "GMMScorer",
                "feature_set": self.feature_set,
                "gmm_bona": self.gmm_bona,
                "gmm_spoof": self.gmm_spoof,
                "dim": self.dim_,
                "n_components": self.n_components_,
                "seed": self.seed,
                "meta": self.meta_,
            },
            path,
        )
        return str(path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GMMScorer":
        if joblib is None:
            raise RuntimeError("joblib is required to load a GMMScorer")
        blob = joblib.load(Path(path))
        obj = cls(feature_set=blob.get("feature_set", "lfcc"),
                  seed=blob.get("seed"))
        obj.gmm_bona = blob.get("gmm_bona")
        obj.gmm_spoof = blob.get("gmm_spoof")
        obj.dim_ = int(blob.get("dim", 0))
        obj.n_components_ = int(blob.get("n_components", 0))
        obj.meta_ = dict(blob.get("meta", {}))
        return obj

    @classmethod
    def default_path(cls, feature_set: str = "lfcc") -> Path:
        return artifact_path(f"antispoof_gmm_{feature_set}.joblib")


if __name__ == "__main__":  # pragma: no cover - self-test
    import time

    from .features import demo_bonafide, demo_spoof

    t0 = time.time()
    sr = TARGET_SR
    print("antispoof.gmm self-test")

    n_train, n_test = 12, 8
    bona_tr = [demo_bonafide(sr, 2.0, seed=i, f0_base=110.0 + 6.0 * (i % 5))
               for i in range(n_train)]
    spoof_tr = [demo_spoof(sr, 2.0, seed=500 + i, f0_base=110.0 + 6.0 * (i % 5))
                for i in range(n_train)]
    bona_te = [demo_bonafide(sr, 2.0, seed=1000 + i, f0_base=125.0 + 5.0 * (i % 4))
               for i in range(n_test)]
    spoof_te = [demo_spoof(sr, 2.0, seed=2000 + i, f0_base=125.0 + 5.0 * (i % 4))
                for i in range(n_test)]

    scorer = GMMScorer(feature_set="lfcc")
    scorer.fit_audio(bona_tr, spoof_tr, sr)
    print(f"  {scorer}")
    print(f"  meta: {scorer.meta_}")

    from .features import frame_features as _ff

    s_bona = [scorer.llr_audio(x, sr) for x in bona_te]
    s_spoof = [scorer.llr_audio(x, sr) for x in spoof_te]
    u_bona = [float(np.mean(scorer.llr_frames(_ff(x, sr, "lfcc")))) for x in bona_te]
    u_spoof = [float(np.mean(scorer.llr_frames(_ff(x, sr, "lfcc")))) for x in spoof_te]
    print(f"  held-out llr  bona fide {np.mean(s_bona):+7.3f} "
          f"+- {np.std(s_bona):.3f}  (unclipped {np.mean(u_bona):+8.1f})")
    print(f"  held-out llr  spoofed   {np.mean(s_spoof):+7.3f} "
          f"+- {np.std(s_spoof):.3f}  (unclipped {np.mean(u_spoof):+8.1f})")
    print("  both sides sit on the clip, which is what a cleanly separated pair "
          "looks like: the ranking survives, the magnitude does not")

    scores = np.array(s_bona + s_spoof)
    labels = np.array([0] * len(s_bona) + [1] * len(s_spoof))
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    print(f"  AUC on the held-out pair: {auc:.3f}")

    p = Path(artifact_path("selftest_gmm.joblib"))
    scorer.save(p)
    again = GMMScorer.load(p)
    delta = abs(again.llr_audio(bona_te[0], sr) - scorer.llr_audio(bona_te[0], sr))
    print(f"  save/load round trip: llr difference {delta:.2e}")
    p.unlink(missing_ok=True)
    print(f"  done in {time.time() - t0:.2f} s")
