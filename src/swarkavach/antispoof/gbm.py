"""Gradient boosted trees over pooled utterance vectors.

The GMM models each class's frame density and then asks which fits better. This
back end asks the discriminative question directly: given the whole utterance
summarised as one vector, how separable are the two classes. On a small corpus
that usually wins, because it can key on a single pooled statistic (the
kurtosis of one cepstral coefficient, say) that a density model spreads thin
across sixty dimensions.

Trees also suit this feature vector. The pooled statistics are on wildly
different scales (a log energy mean near 10, a kurtosis near 0, a delta std near
0.1) and boosted trees do not care, so no scaler has to be fitted, saved and
kept in sync with the model.

`HistGradientBoostingClassifier` is the default because it bins the features
first, which makes it fast on 480 columns and gives free regularisation on a
corpus of a few hundred calls. The hyper-parameters shrink automatically when
the training set is small, otherwise sklearn's defaults (20 samples per leaf,
early stopping on a 10 percent split) refuse to grow any tree at all.

Run the self-test with:
    python -m swarkavach.antispoof.gbm
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from ..config import SETTINGS, TARGET_SR, artifact_path
from .features import pooled_feature_names, utterance_features

try:
    import joblib
except Exception:  # pragma: no cover - environment dependent
    joblib = None

from sklearn.ensemble import HistGradientBoostingClassifier

__all__ = ["GBMScorer"]

#: Below this many training calls the model runs with hand-set small-data
#: parameters instead of sklearn's defaults.
_SMALL_DATA = 400


def _as_matrix(X) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


class GBMScorer:
    """Boosted trees on `utterance_features` vectors, scoring P(spoof)."""

    def __init__(
        self,
        feature_set: str = "lfcc",
        max_iter: int = 250,
        learning_rate: float = 0.08,
        max_leaf_nodes: Optional[int] = None,
        l2_regularization: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.feature_set = str(feature_set).lower()
        self.max_iter = int(max_iter)
        self.learning_rate = float(learning_rate)
        self.max_leaf_nodes = max_leaf_nodes
        self.l2_regularization = float(l2_regularization)
        self.seed = int(SETTINGS.pipeline.seed if seed is None else seed)

        self.model: Optional[HistGradientBoostingClassifier] = None
        self.dim_: int = 0
        self.classes_: Optional[np.ndarray] = None
        self.meta_: Dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        return self.model is not None

    def __repr__(self) -> str:  # pragma: no cover - display only
        state = f"fitted d={self.dim_}" if self.is_fitted else "unfitted"
        return f"GBMScorer({self.feature_set}, {state})"

    # -- fitting ----------------------------------------------------------

    def fit(self, X, y) -> "GBMScorer":
        """Fit on pooled vectors X (n_calls, d) and binary labels y (1 = spoof)."""
        X = _as_matrix(X)
        y = np.asarray(y).ravel().astype(int)
        if X.shape[0] != y.size:
            raise ValueError(f"{X.shape[0]} feature rows for {y.size} labels")
        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError(
                f"GBMScorer needs both classes, got only {classes.tolist()}"
            )
        n = X.shape[0]
        small = n < _SMALL_DATA
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=self.learning_rate,
            max_iter=self.max_iter if not small else min(self.max_iter, 150),
            max_leaf_nodes=self.max_leaf_nodes or (8 if small else 31),
            # sklearn's default of 20 samples per leaf leaves a corpus of 40
            # calls unable to split at all.
            min_samples_leaf=max(2, min(20, n // 8)),
            l2_regularization=self.l2_regularization,
            max_bins=min(255, max(8, n - 1)),
            early_stopping=not small,
            random_state=self.seed,
        )
        model.fit(X, y)
        self.model = model
        self.dim_ = int(X.shape[1])
        self.classes_ = model.classes_
        self.meta_ = {
            "feature_set": self.feature_set,
            "n_train": int(n),
            "dim": int(X.shape[1]),
            "n_spoof": int((y == 1).sum()),
            "n_bona": int((y == 0).sum()),
            "n_iter": int(getattr(model, "n_iter_", 0)),
            "small_data": bool(small),
            "seed": int(self.seed),
        }
        return self

    def fit_audio(self, signals: Iterable, labels: Sequence[int],
                  sr: int = TARGET_SR) -> "GBMScorer":
        """Convenience wrapper that runs the pooling front end for you."""
        X = np.vstack([utterance_features(x, sr, self.feature_set) for x in signals])
        return self.fit(X, labels)

    # -- scoring ----------------------------------------------------------

    def score(self, X) -> Union[float, np.ndarray]:
        """P(spoof). A single vector gives a float, a matrix gives an array."""
        if not self.is_fitted:
            raise RuntimeError("GBMScorer.score called before fit")
        single = np.asarray(X, dtype=np.float64).ndim == 1
        M = _as_matrix(X)
        if M.shape[1] != self.dim_:
            if M.shape[1] > self.dim_:
                M = M[:, : self.dim_]
            else:
                M = np.pad(M, ((0, 0), (0, self.dim_ - M.shape[1])))
        proba = self.model.predict_proba(M)
        spoof_col = int(np.argmax(self.model.classes_ == 1)) if 1 in self.model.classes_ else -1
        p = proba[:, spoof_col]
        p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
        return float(p[0]) if single else p

    def score_audio(self, x, sr: int = TARGET_SR) -> float:
        return float(self.score(utterance_features(x, sr, self.feature_set)))

    def margin(self, X) -> Union[float, np.ndarray]:
        """Log odds of the probability, which is the raw boosted score.

        The fusion vector carries `as_margin` as a confidence, and a log odds is
        the natural unbounded version of it for a tree ensemble.
        """
        p = np.clip(np.atleast_1d(np.asarray(self.score(X), dtype=np.float64)),
                    1e-6, 1 - 1e-6)
        out = np.log(p / (1.0 - p))
        return float(out[0]) if out.size == 1 else out

    def feature_names(self) -> List[str]:
        return pooled_feature_names(self.feature_set)

    # -- persistence ------------------------------------------------------

    def save(self, path: Union[str, Path]) -> str:
        if joblib is None:
            raise RuntimeError("joblib is required to save a GBMScorer")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "kind": "GBMScorer",
                "feature_set": self.feature_set,
                "model": self.model,
                "dim": self.dim_,
                "seed": self.seed,
                "meta": self.meta_,
            },
            path,
        )
        return str(path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GBMScorer":
        if joblib is None:
            raise RuntimeError("joblib is required to load a GBMScorer")
        from ..compat import install_pickle_aliases
        install_pickle_aliases()
        blob = joblib.load(Path(path))
        obj = cls(feature_set=blob.get("feature_set", "lfcc"), seed=blob.get("seed"))
        obj.model = blob.get("model")
        obj.dim_ = int(blob.get("dim", 0))
        obj.meta_ = dict(blob.get("meta", {}))
        if obj.model is not None:
            obj.classes_ = obj.model.classes_
        return obj

    @classmethod
    def default_path(cls, feature_set: str = "lfcc") -> Path:
        return artifact_path(f"antispoof_gbm_{feature_set}.joblib")


if __name__ == "__main__":  # pragma: no cover - self-test
    import time

    from .features import demo_bonafide, demo_spoof

    t0 = time.time()
    sr = TARGET_SR
    print("antispoof.gbm self-test")

    def make(n, start, fn):
        return [fn(sr, 2.0, seed=start + i, f0_base=105.0 + 7.0 * (i % 6))
                for i in range(n)]

    train = make(14, 0, demo_bonafide) + make(14, 300, demo_spoof)
    y_train = [0] * 14 + [1] * 14
    test = make(8, 1000, demo_bonafide) + make(8, 1300, demo_spoof)
    y_test = np.array([0] * 8 + [1] * 8)

    scorer = GBMScorer()
    t1 = time.time()
    scorer.fit_audio(train, y_train, sr)
    print(f"  {scorer}, fit in {time.time() - t1:.2f} s")
    print(f"  meta: {scorer.meta_}")

    p = np.array([scorer.score_audio(x, sr) for x in test])
    print(f"  held-out P(spoof)  bona fide mean {p[y_test == 0].mean():.3f}, "
          f"spoofed mean {p[y_test == 1].mean():.3f}")
    acc = float(((p >= 0.5).astype(int) == y_test).mean())
    print(f"  accuracy at 0.5: {acc:.3f}")
    print(f"  margin on the first test call: {scorer.margin(utterance_features(test[0], sr)):+.3f}")

    path = Path(artifact_path("selftest_gbm.joblib"))
    scorer.save(path)
    again = GBMScorer.load(path)
    print(f"  save/load round trip: max |delta| "
          f"{max(abs(again.score_audio(x, sr) - s) for x, s in zip(test, p)):.2e}")
    path.unlink(missing_ok=True)
    print(f"  done in {time.time() - t0:.2f} s")
