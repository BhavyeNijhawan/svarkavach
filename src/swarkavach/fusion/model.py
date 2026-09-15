"""Late fusion over the 25 features, with calibration.

Two estimators are supported. Logistic regression is the default because its
Shapley values have a closed form, so the evidence panel shows exact
attributions rather than a sampled approximation. Gradient boosting is
available for the results table, where the question is how much accuracy a
non-linear model buys.

Calibration is not decoration here. The whole argument of the project is that
two branch scores can be combined, and combining uncalibrated scores gives a
fusion that tracks whichever branch happens to be more confident rather than
whichever is more correct. Plan B section 5 says to check calibration first
when fusion underperforms, so it is built in from the start.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import SETTINGS, artifact_path
from ..schema import (
    ABLATION_ARMS, FEATURE_GROUPS, FEATURE_LABELS, FUSION_FEATURE_NAMES,
    feature_vector,
)


class FusionModel:
    """Calibrated late-fusion classifier over a named subset of the features."""

    def __init__(self, kind: str = "logreg", arm: str = "full"):
        if arm not in ABLATION_ARMS:
            raise ValueError(f"unknown ablation arm {arm!r}")
        self.kind = kind
        self.arm = arm
        self.names: Tuple[str, ...] = ABLATION_ARMS[arm]
        self.clf = None
        self.calibrator = None
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.trained = False
        self.train_meta: Dict[str, Any] = {}

    # -- plumbing ---------------------------------------------------------

    def _idx(self) -> List[int]:
        return [FUSION_FEATURE_NAMES.index(n) for n in self.names]

    def _select(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] == len(self.names):
            return X
        return X[:, self._idx()]

    def _standardise(self, Z: np.ndarray) -> np.ndarray:
        return (Z - self.mean_) / self.scale_

    # -- training ---------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[Sequence[str]] = None) -> "FusionModel":
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import StratifiedKFold

        Z = self._select(X)
        y = np.asarray(y, dtype=np.int64).ravel()
        seed = SETTINGS.pipeline.seed

        self.mean_ = Z.mean(axis=0)
        sd = Z.std(axis=0)
        # A feature that never varies in training carries no information, and
        # dividing by its zero standard deviation would produce NaN, so it is
        # pinned to scale 1 and ends up contributing nothing.
        self.scale_ = np.where(sd < 1e-8, 1.0, sd)
        Zs = self._standardise(Z)

        n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
        n_min = min(n_pos, n_neg)

        if self.kind == "gbm":
            base = HistGradientBoostingClassifier(
                max_iter=220, learning_rate=0.07, max_depth=4,
                min_samples_leaf=max(5, len(y) // 40), l2_regularization=1.0,
                random_state=seed,
            )
        else:
            base = LogisticRegression(
                C=0.7, max_iter=4000, solver="lbfgs",
                class_weight="balanced", random_state=seed,
            )

        # Calibrate with cross-validation when there is enough of the minority
        # class to split; below that, fit plainly and record why.
        if n_min >= 10:
            folds = int(min(5, n_min))
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            method = "isotonic" if n_min >= 60 else "sigmoid"
            self.clf = CalibratedClassifierCV(base, method=method, cv=cv)
            self.clf.fit(Zs, y)
            self.calibrator = method
        else:
            base.fit(Zs, y)
            self.clf = base
            self.calibrator = None

        self.trained = True
        self.train_meta = {
            "n": int(len(y)), "n_pos": n_pos, "n_neg": n_neg,
            "kind": self.kind, "arm": self.arm, "calibrator": self.calibrator,
            "n_features": len(self.names), "seed": seed,
        }
        return self

    # -- inference --------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("fusion model is not trained")
        Zs = self._standardise(self._select(X))
        return self.clf.predict_proba(Zs)[:, 1]

    def predict_one(self, features: Dict[str, float]) -> float:
        v = np.asarray(feature_vector(features), dtype=np.float64).reshape(1, -1)
        return float(self.predict_proba(v)[0])

    # -- explanation ------------------------------------------------------

    def linear_terms(self) -> Optional[np.ndarray]:
        """Coefficients in standardised space, averaged over calibration folds.

        Returns None for a non-linear estimator, which is the signal to fall
        back on the sampling explainer.
        """
        if self.kind != "logreg" or not self.trained:
            return None
        if hasattr(self.clf, "coef_"):
            return np.asarray(self.clf.coef_, dtype=np.float64).ravel()
        cc = getattr(self.clf, "calibrated_classifiers_", None)
        if not cc:
            return None
        coefs = []
        for c in cc:
            est = getattr(c, "estimator", None) or getattr(c, "base_estimator", None)
            if est is not None and hasattr(est, "coef_"):
                coefs.append(np.asarray(est.coef_, dtype=np.float64).ravel())
        return np.mean(coefs, axis=0) if coefs else None

    def explain(self, features: Dict[str, float]) -> List[Dict[str, Any]]:
        """Per-feature contribution to the log-odds, largest magnitude first."""
        from .explain import linear_shap, kernel_shap

        x = np.asarray(feature_vector(features), dtype=np.float64).reshape(1, -1)
        w = self.linear_terms()
        if w is not None:
            zs = self._standardise(self._select(x)).ravel()
            return linear_shap(w, zs, self.names)
        return kernel_shap(
            lambda M: self.predict_proba(M),
            self._select(x).ravel(),
            background=self.mean_,
            names=self.names,
        )

    # -- persistence ------------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        import joblib

        path = str(path or artifact_path(f"fusion_{self.arm}_{self.kind}.joblib"))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "kind": self.kind, "arm": self.arm, "names": list(self.names),
            "clf": self.clf, "calibrator": self.calibrator,
            "mean": self.mean_, "scale": self.scale_,
            "trained": self.trained, "train_meta": self.train_meta,
        }, path)
        return path

    @classmethod
    def load(cls, path: str) -> "FusionModel":
        import joblib

        from ..compat import install_pickle_aliases
        install_pickle_aliases()
        d = joblib.load(path)
        m = cls(kind=d["kind"], arm=d["arm"])
        m.names = tuple(d["names"])
        m.clf = d["clf"]
        m.calibrator = d.get("calibrator")
        m.mean_ = d["mean"]
        m.scale_ = d["scale"]
        m.trained = bool(d.get("trained", True))
        m.train_meta = d.get("train_meta", {})
        return m


def train_all_arms(
    X: np.ndarray, y: np.ndarray, kind: str = "logreg"
) -> Dict[str, FusionModel]:
    """One model per ablation arm, so the comparison is like for like."""
    out: Dict[str, FusionModel] = {}
    for arm in ABLATION_ARMS:
        try:
            out[arm] = FusionModel(kind=kind, arm=arm).fit(X, y)
        except Exception as exc:
            print(f"  arm {arm} failed to train: {type(exc).__name__}: {exc}")
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    d = len(FUSION_FEATURE_NAMES)
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, d)) * 0.4
    # make three features actually informative
    for j, name in enumerate(FUSION_FEATURE_NAMES):
        if name in ("as_score", "intent_score", "pim"):
            X[:, j] += y * 1.3

    m = FusionModel("logreg", "full").fit(X, y)
    p = m.predict_proba(X)
    acc = float(((p > 0.5).astype(int) == y).mean())
    print(f"train accuracy {acc:.3f}  calibrator={m.calibrator}")

    feats = {n_: float(v) for n_, v in zip(FUSION_FEATURE_NAMES, X[0])}
    print(f"\nP(fraud) for row 0: {m.predict_one(feats):.4f}  (truth {y[0]})")
    print("top contributions:")
    for c in m.explain(feats)[:5]:
        print(f"  {c['label']:34s} {c['contribution']:+.3f}")

    import tempfile, os
    p1 = m.predict_one(feats)
    tmp = os.path.join(tempfile.gettempdir(), "fusion_selftest.joblib")
    m.save(tmp)
    p2 = FusionModel.load(tmp).predict_one(feats)
    print(f"\nsave/load round trip: {p1:.6f} vs {p2:.6f}  ok={abs(p1-p2) < 1e-9}")
