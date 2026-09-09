"""Linear-chain conditional random field, written from scratch.

Why a CRF and not a per-token classifier: BIO tagging has hard sequential
structure. `I-OTP` is only legal after `B-OTP` or `I-OTP`, and the evidence for
"block ho jayega" being one THREAT_DEADLINE span lives in the whole three-token
sequence, not in any single token. A logistic regression over token features
scores each position independently and has to be patched up afterwards; a CRF
puts the transition scores inside the model and normalises over whole label
sequences, so decoding returns the best globally consistent sequence.

Why hand-written: `sklearn-crfsuite` is not installable in this offline setup,
and the derivation is the assessable part. The model is the standard one,

    p(y | x) = exp( sum_t s(y_t, x, t) + sum_t a(y_{t-1}, y_t)
                    + start(y_0) + stop(y_{T-1}) ) / Z(x)

with a sparse linear state score s and a full transition matrix a. Training
minimises the L2-regularised negative conditional log-likelihood with
L-BFGS-B. The gradient is the difference between empirical feature counts and
the counts expected under the model, and the expectations come from
forward-backward marginals:

    d logL / d w_k = sum_t f_k(y_t*, x, t)
                     - sum_t sum_l p(y_t = l | x) f_k(l, x, t)

Three implementation points that matter:

* Everything is in log space through `scipy.special.logsumexp`. Underflow in
  the forward recursion is the usual way a hand-written CRF quietly breaks once
  sequences pass thirty or so tokens.
* Forward-backward runs on all sequences at once, padded to a common length,
  so each time step is a handful of array operations rather than one per
  sequence. Empirical counts and the sparse state-score product are computed
  once and reused across every L-BFGS evaluation. Without this a few hundred
  training sequences take minutes instead of seconds.
* Illegal transitions can be masked to -inf. They then carry zero probability
  in the marginals and drop out of the gradient on their own, so a constrained
  CRF needs no special case: Z sums over valid label sequences only and Viterbi
  cannot emit an invalid one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import logsumexp

__all__ = ["LinearChainCRF", "CRFData", "numerical_gradient"]

#: One token's features: either a collection of active binary feature names, or
#: a name to value mapping when a feature is real-valued.
TokenFeatures = Union[Sequence[str], Dict[str, float]]

#: Cap on the padded (batch, time, label, label) working array, in elements.
#: Roughly 64 MB of float64, which keeps forward-backward inside cache-friendly
#: chunks on a laptop.
_CHUNK_ELEMS = 8_000_000


@dataclass
class CRFData:
    """A dataset encoded once so training never re-indexes.

    Holds the sparse token-by-feature matrix, the padded layout used by the
    batched forward-backward, and the empirical counts, which do not depend on
    the weights and so are computed a single time.
    """

    X: csr_matrix                 # (n_tokens, n_features)
    lengths: np.ndarray           # (n_seq,)
    idx_pad: np.ndarray           # (n_seq, T_max) row index into X
    valid: np.ndarray             # (n_seq, T_max) bool
    n_seq: int
    y: Optional[np.ndarray] = None            # (n_tokens,) gold label index
    emp_W: Optional[np.ndarray] = None        # (n_features, n_labels)
    emp_trans: Optional[np.ndarray] = None    # (n_labels, n_labels)
    emp_start: Optional[np.ndarray] = None    # (n_labels,)
    emp_stop: Optional[np.ndarray] = None     # (n_labels,)

    @property
    def n_tokens(self) -> int:
        return int(self.X.shape[0])


def _iter_feats(item: TokenFeatures) -> Iterable[Tuple[str, float]]:
    if isinstance(item, dict):
        for k, v in item.items():
            yield str(k), float(v)
    else:
        for k in item:
            yield str(k), 1.0


class LinearChainCRF:
    """First-order linear-chain CRF with sparse binary or real-valued features.

    Parameters
    ----------
    l2:
        Coefficient of the 0.5 * l2 * ||w||^2 penalty, applied to state,
        transition, start and stop weights alike.
    max_iter:
        L-BFGS-B iteration cap.
    min_freq:
        Features firing on fewer than this many tokens are dropped. On a small
        corpus most features fire once and can only memorise, so raising this
        shrinks the model without costing accuracy.
    labels:
        Fixed label order. Pass it when the output space is known in advance
        (BIO_LABELS), so a training split that happens to miss a rare tag still
        yields a model covering the full space.
    constraints:
        Callable taking the label list and returning
        (allowed_trans, allowed_start, allowed_stop) boolean arrays. Anything
        false scores -inf and can never appear in a decode.
    """

    def __init__(
        self,
        l2: float = 1.0,
        max_iter: int = 200,
        min_freq: int = 1,
        labels: Optional[Sequence[str]] = None,
        constraints: Optional[Callable[[List[str]], Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
        tol: float = 1e-7,
        verbose: bool = False,
    ) -> None:
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.min_freq = int(min_freq)
        self.tol = float(tol)
        self.verbose = bool(verbose)
        self._constraint_fn = constraints

        self.labels_: List[str] = list(labels) if labels else []
        self.label_index_: Dict[str, int] = {l: i for i, l in enumerate(self.labels_)}
        self.features_: List[str] = []
        self.feature_index_: Dict[str, int] = {}
        self.w_: Optional[np.ndarray] = None
        self.mask_trans_: Optional[np.ndarray] = None   # additive, 0.0 or -inf
        self.mask_start_: Optional[np.ndarray] = None
        self.mask_stop_: Optional[np.ndarray] = None
        self.history_: Dict[str, Any] = {}

    # -- sizes -------------------------------------------------------------

    @property
    def n_labels(self) -> int:
        return len(self.labels_)

    @property
    def n_features(self) -> int:
        return len(self.features_)

    @property
    def n_params(self) -> int:
        L, F = self.n_labels, self.n_features
        return F * L + L * L + 2 * L

    def _unpack(self, w: np.ndarray):
        L, F = self.n_labels, self.n_features
        i = F * L
        W = w[:i].reshape(F, L)
        trans = w[i:i + L * L].reshape(L, L)
        start = w[i + L * L:i + L * L + L]
        stop = w[i + L * L + L:]
        return W, trans, start, stop

    def _effective(self, trans, start, stop):
        """Raw parameters plus the -inf constraint mask, used by Z and Viterbi."""
        if self.mask_trans_ is None:
            return trans, start, stop
        return trans + self.mask_trans_, start + self.mask_start_, stop + self.mask_stop_

    # -- indexing ----------------------------------------------------------

    def _build_index(
        self,
        X: Sequence[Sequence[TokenFeatures]],
        y: Optional[Sequence[Sequence[str]]],
    ) -> None:
        if y is not None:
            for seq in y:
                for tag in seq:
                    if tag not in self.label_index_:
                        self.label_index_[tag] = len(self.labels_)
                        self.labels_.append(tag)

        counts: Dict[str, int] = {}
        for seq in X:
            for item in seq:
                for name, _ in _iter_feats(item):
                    counts[name] = counts.get(name, 0) + 1
        self.features_ = sorted(n for n, c in counts.items() if c >= self.min_freq)
        self.feature_index_ = {n: i for i, n in enumerate(self.features_)}

        if self._constraint_fn is not None:
            a_trans, a_start, a_stop = self._constraint_fn(self.labels_)
            self.mask_trans_ = np.where(np.asarray(a_trans, dtype=bool), 0.0, -np.inf)
            self.mask_start_ = np.where(np.asarray(a_start, dtype=bool), 0.0, -np.inf)
            self.mask_stop_ = np.where(np.asarray(a_stop, dtype=bool), 0.0, -np.inf)

    def prepare(
        self,
        X: Sequence[Sequence[TokenFeatures]],
        y: Optional[Sequence[Sequence[str]]] = None,
        build_index: bool = False,
    ) -> CRFData:
        """Encode sequences into a sparse matrix, a padded layout and counts."""
        if build_index:
            self._build_index(X, y)
        L = self.n_labels
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        lengths = np.asarray([len(s) for s in X], dtype=np.int64)
        if lengths.size and lengths.min() < 1:
            raise ValueError("empty sequences are not allowed, drop them before fitting")
        t_max = int(lengths.max()) if lengths.size else 0
        idx_pad = np.zeros((len(X), t_max), dtype=np.int64)
        valid = np.zeros((len(X), t_max), dtype=bool)

        row = 0
        labs: List[int] = []
        for si, seq in enumerate(X):
            for ti, item in enumerate(seq):
                for name, v in _iter_feats(item):
                    j = self.feature_index_.get(name)
                    if j is None:
                        continue        # unseen at training time, ignored
                    rows.append(row)
                    cols.append(j)
                    vals.append(v)
                idx_pad[si, ti] = row
                valid[si, ti] = True
                row += 1
            if y is not None:
                for tag in y[si]:
                    labs.append(self.label_index_[tag])

        Xs = csr_matrix(
            (np.asarray(vals, dtype=np.float64),
             (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
            shape=(row, max(1, self.n_features)),
        )
        data = CRFData(X=Xs, lengths=lengths, idx_pad=idx_pad, valid=valid, n_seq=len(X))
        if y is None:
            return data

        y_arr = np.asarray(labs, dtype=np.int64)
        data.y = y_arr
        onehot = csr_matrix(
            (np.ones(y_arr.size), (np.arange(y_arr.size), y_arr)),
            shape=(y_arr.size, max(1, L)),
        )
        data.emp_W = np.asarray((Xs.T @ onehot).todense())
        emp_trans = np.zeros((L, L))
        emp_start = np.zeros(L)
        emp_stop = np.zeros(L)
        off = 0
        for si in range(len(X)):
            n = int(lengths[si])
            ys = y_arr[off:off + n]
            off += n
            emp_start[ys[0]] += 1.0
            emp_stop[ys[-1]] += 1.0
            if n > 1:
                flat = ys[:-1] * L + ys[1:]
                emp_trans += np.bincount(flat, minlength=L * L).reshape(L, L)
        data.emp_trans = emp_trans
        data.emp_start = emp_start
        data.emp_stop = emp_stop
        self._check_constraints(data)
        return data

    def _check_constraints(self, data: CRFData) -> None:
        """Fail loudly if a gold path uses a transition the mask forbids.

        Without this the gold score would contain a -inf and the whole
        objective would turn into a NaN somewhere deep inside L-BFGS.
        """
        if self.mask_trans_ is None or data.emp_trans is None:
            return
        bad = (data.emp_trans > 0) & ~np.isfinite(self.mask_trans_)
        if bad.any():
            i, j = np.argwhere(bad)[0]
            raise ValueError(
                f"gold data uses forbidden transition "
                f"{self.labels_[i]} -> {self.labels_[j]}"
            )
        if (data.emp_start > 0).any() and not np.isfinite(
            self.mask_start_[data.emp_start > 0]
        ).all():
            raise ValueError("gold data starts on a forbidden label")

    # -- inference ---------------------------------------------------------

    @staticmethod
    def _forward_backward(
        E: np.ndarray, valid: np.ndarray, lengths: np.ndarray,
        trans: np.ndarray, start: np.ndarray, stop: np.ndarray,
        want_pair: bool = True,
    ):
        """Batched log-space forward-backward over padded sequences.

        E is (B, T, L) with padding positions unconstrained (they are frozen
        out by `valid`). Returns log Z per sequence, unary marginals masked to
        the real positions, and the summed pairwise marginals.
        """
        B, T, L = E.shape
        alpha = np.empty((B, T, L))
        alpha[:, 0] = start[None, :] + E[:, 0]
        for t in range(1, T):
            cand = E[:, t] + logsumexp(alpha[:, t - 1][:, :, None] + trans[None], axis=1)
            # Finished sequences hold their alpha, so alpha[:, T-1] is always
            # the value at each sequence's own final token.
            alpha[:, t] = np.where(valid[:, t][:, None], cand, alpha[:, t - 1])
        logZ = logsumexp(alpha[:, T - 1] + stop[None, :], axis=1)      # (B,)

        beta = np.empty((B, T, L))
        beta[:, T - 1] = stop[None, :]
        for t in range(T - 2, -1, -1):
            cand = logsumexp(trans[None] + (E[:, t + 1] + beta[:, t + 1])[:, None, :], axis=2)
            # If position t+1 is padding then t is this sequence's last token
            # and its beta is the stop score.
            beta[:, t] = np.where(valid[:, t + 1][:, None], cand, stop[None, :])

        unary = np.exp(alpha + beta - logZ[:, None, None]) * valid[:, :, None]
        pair = np.zeros((L, L))
        if want_pair and T > 1:
            M = (alpha[:, :-1][:, :, :, None] + trans[None, None]
                 + (E[:, 1:] + beta[:, 1:])[:, :, None, :]
                 - logZ[:, None, None, None])
            pair = (np.exp(M) * valid[:, 1:][:, :, None, None]).sum(axis=(0, 1))
        return logZ, unary, pair

    @staticmethod
    def _viterbi(E: np.ndarray, trans: np.ndarray, start: np.ndarray, stop: np.ndarray):
        """Single-sequence Viterbi. Decoding is outside the training loop."""
        T, L = E.shape
        delta = start + E[0]
        back = np.zeros((T, L), dtype=np.int64)
        for t in range(1, T):
            scores = delta[:, None] + trans           # (prev, next)
            back[t] = np.argmax(scores, axis=0)
            delta = E[t] + scores[back[t], np.arange(L)]
        final = delta + stop
        best = int(np.argmax(final))
        path = [best]
        for t in range(T - 1, 0, -1):
            best = int(back[t, best])
            path.append(best)
        path.reverse()
        return path, float(final[int(np.argmax(final))])

    # -- objective ---------------------------------------------------------

    def loss_and_grad(self, w: np.ndarray, data: CRFData) -> Tuple[float, np.ndarray]:
        """Regularised negative log-likelihood and its analytic gradient.

        Public because the gradient check in the test suite calls it directly.
        That check is the only real proof that the derivation above made it
        into the code intact, so the entry point is part of the interface.
        """
        w = np.asarray(w, dtype=np.float64)
        L = self.n_labels
        W, trans_raw, start_raw, stop_raw = self._unpack(w)
        trans, start, stop = self._effective(trans_raw, start_raw, stop_raw)

        E_all = np.asarray(data.X @ W)                      # (n_tokens, L)
        P_all = np.zeros_like(E_all)
        exp_trans = np.zeros((L, L))
        exp_start = np.zeros(L)
        exp_stop = np.zeros(L)
        logZ_total = 0.0

        t_max = data.idx_pad.shape[1]
        chunk = max(1, int(_CHUNK_ELEMS / max(1, t_max * L * L)))
        for a in range(0, data.n_seq, chunk):
            b = min(data.n_seq, a + chunk)
            idx = data.idx_pad[a:b]
            valid = data.valid[a:b]
            E = E_all[idx] * valid[:, :, None]
            logZ, unary, pair = self._forward_backward(
                E, valid, data.lengths[a:b], trans, start, stop
            )
            logZ_total += float(logZ.sum())
            P_all[idx[valid]] = unary[valid]
            exp_trans += pair
            exp_start += unary[:, 0].sum(axis=0)
            exp_stop += unary[np.arange(b - a), data.lengths[a:b] - 1].sum(axis=0)

        # The gold score is linear in the parameters, so it is just the inner
        # product with the empirical counts. Masked transitions never occur in
        # a gold path (checked in `prepare`), so the raw parameters are the
        # right ones to use here.
        gold = (
            float((W * data.emp_W).sum())
            + float((trans_raw * data.emp_trans).sum())
            + float(start_raw @ data.emp_start)
            + float(stop_raw @ data.emp_stop)
        )
        nll = logZ_total - gold

        g_W = np.asarray(data.X.T @ P_all) - data.emp_W
        g_trans = exp_trans - data.emp_trans
        g_start = exp_start - data.emp_start
        g_stop = exp_stop - data.emp_stop

        grad = np.concatenate([g_W.ravel(), g_trans.ravel(), g_start, g_stop])
        loss = nll + 0.5 * self.l2 * float(w @ w)
        grad = grad + self.l2 * w
        return float(loss), grad

    # -- training ----------------------------------------------------------

    def fit(self, X: Sequence[Sequence[TokenFeatures]], y: Sequence[Sequence[str]]) -> "LinearChainCRF":
        if len(X) != len(y):
            raise ValueError(f"{len(X)} feature sequences but {len(y)} label sequences")
        for i, (xs, ys) in enumerate(zip(X, y)):
            if len(xs) != len(ys):
                raise ValueError(f"sequence {i}: {len(xs)} tokens but {len(ys)} labels")
        data = self.prepare(X, y, build_index=True)
        w0 = np.zeros(self.n_params, dtype=np.float64)
        n_calls = {"n": 0}

        def fun(w):
            n_calls["n"] += 1
            return self.loss_and_grad(w, data)

        res = minimize(
            fun, w0, jac=True, method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol, "gtol": self.tol},
        )
        self.w_ = res.x
        self.history_ = {
            "loss": float(res.fun),
            "n_iter": int(res.nit),
            "n_fun_calls": int(n_calls["n"]),
            "success": bool(res.success),
            "message": str(res.message),
            "n_features": self.n_features,
            "n_labels": self.n_labels,
            "n_params": self.n_params,
            "n_sequences": len(X),
            "n_tokens": int(data.n_tokens),
        }
        if self.verbose:
            print(json.dumps(self.history_, indent=2))
        return self

    # -- prediction --------------------------------------------------------

    def _emissions(self, seq: Sequence[TokenFeatures]) -> np.ndarray:
        W, _, _, _ = self._unpack(self.w_)
        E = np.zeros((len(seq), self.n_labels))
        for t, item in enumerate(seq):
            for name, v in _iter_feats(item):
                j = self.feature_index_.get(name)
                if j is not None:
                    E[t] += v * W[j]
        return E

    def predict_one(self, seq: Sequence[TokenFeatures]) -> List[str]:
        if self.w_ is None:
            raise RuntimeError("CRF is not fitted")
        if len(seq) == 0:
            return []
        _, tr, st, sp = self._unpack(self.w_)
        trans, start, stop = self._effective(tr, st, sp)
        path, _ = self._viterbi(self._emissions(seq), trans, start, stop)
        return [self.labels_[i] for i in path]

    def predict(self, X: Sequence[Sequence[TokenFeatures]]) -> List[List[str]]:
        return [self.predict_one(seq) for seq in X]

    def predict_marginals(self, seq: Sequence[TokenFeatures]) -> List[Dict[str, float]]:
        """Per-token posterior over labels, used for span confidence scores."""
        if self.w_ is None:
            raise RuntimeError("CRF is not fitted")
        if len(seq) == 0:
            return []
        _, tr, st, sp = self._unpack(self.w_)
        trans, start, stop = self._effective(tr, st, sp)
        E = self._emissions(seq)[None, :, :]
        valid = np.ones(E.shape[:2], dtype=bool)
        _, unary, _ = self._forward_backward(
            E, valid, np.asarray([E.shape[1]]), trans, start, stop, want_pair=False
        )
        return [{l: float(p) for l, p in zip(self.labels_, row)} for row in unary[0]]

    def log_marginal(self, seq: Sequence[TokenFeatures]) -> float:
        """log Z for one sequence. Handy for debugging the batched path."""
        _, tr, st, sp = self._unpack(self.w_)
        trans, start, stop = self._effective(tr, st, sp)
        E = self._emissions(seq)[None, :, :]
        valid = np.ones(E.shape[:2], dtype=bool)
        logZ, _, _ = self._forward_backward(
            E, valid, np.asarray([E.shape[1]]), trans, start, stop, want_pair=False
        )
        return float(logZ[0])

    def sequence_score(self, seq: Sequence[TokenFeatures], tags: Sequence[str]) -> float:
        """log p(tags | seq), the quantity Viterbi maximises."""
        _, tr, st, sp = self._unpack(self.w_)
        trans, start, stop = self._effective(tr, st, sp)
        E = self._emissions(seq)
        idx = np.array([self.label_index_[t] for t in tags], dtype=np.int64)
        s = float(E[np.arange(len(idx)), idx].sum() + start[idx[0]] + stop[idx[-1]])
        if len(idx) > 1:
            s += float(trans[idx[:-1], idx[1:]].sum())
        return s - self.log_marginal(seq)

    def score(self, X: Sequence[Sequence[TokenFeatures]], y: Sequence[Sequence[str]]) -> float:
        """Token-level accuracy. Entity-level scoring lives in `ner_crf`."""
        correct = total = 0
        for seq, gold in zip(X, y):
            for p, g in zip(self.predict_one(seq), gold):
                correct += int(p == g)
                total += 1
        return correct / max(1, total)

    def top_weights(self, k: int = 15) -> List[Tuple[str, str, float]]:
        """Largest state weights, as (feature, label, weight). For the report."""
        if self.w_ is None:
            return []
        W, _, _, _ = self._unpack(self.w_)
        flat = np.argsort(-np.abs(W), axis=None)[:k]
        out = []
        for f in flat:
            i, j = divmod(int(f), self.n_labels)
            out.append((self.features_[i], self.labels_[j], float(W[i, j])))
        return out

    # -- persistence -------------------------------------------------------

    def save(self, path) -> None:
        import joblib

        joblib.dump(
            {
                "l2": self.l2, "max_iter": self.max_iter, "min_freq": self.min_freq,
                "tol": self.tol, "labels": self.labels_, "features": self.features_,
                "w": self.w_, "mask_trans": self.mask_trans_,
                "mask_start": self.mask_start_, "mask_stop": self.mask_stop_,
                "history": self.history_,
            },
            path,
        )

    @classmethod
    def load(cls, path) -> "LinearChainCRF":
        import joblib

        d = joblib.load(path)
        m = cls(l2=d["l2"], max_iter=d["max_iter"], min_freq=d["min_freq"], tol=d["tol"])
        m.labels_ = list(d["labels"])
        m.label_index_ = {l: i for i, l in enumerate(m.labels_)}
        m.features_ = list(d["features"])
        m.feature_index_ = {n: i for i, n in enumerate(m.features_)}
        m.w_ = d["w"]
        m.mask_trans_ = d["mask_trans"]
        m.mask_start_ = d["mask_start"]
        m.mask_stop_ = d["mask_stop"]
        m.history_ = d.get("history", {})
        return m


def numerical_gradient(crf: LinearChainCRF, w: np.ndarray, data: CRFData, eps: float = 1e-5) -> np.ndarray:
    """Central finite-difference gradient of `crf.loss_and_grad`.

    Two objective evaluations per parameter, so keep the problem tiny. This
    exists for the correctness test and nothing else.
    """
    w = np.asarray(w, dtype=np.float64).copy()
    g = np.zeros_like(w)
    for k in range(w.size):
        orig = w[k]
        w[k] = orig + eps
        f_plus, _ = crf.loss_and_grad(w, data)
        w[k] = orig - eps
        f_minus, _ = crf.loss_and_grad(w, data)
        w[k] = orig
        g[k] = (f_plus - f_minus) / (2 * eps)
    return g


if __name__ == "__main__":
    import time

    rng = np.random.default_rng(20230100)

    # A toy tagging problem with real sequential structure: the label of a
    # token depends on the token AND on whether the previous token opened a
    # span, which is exactly what an independent per-token classifier cannot
    # represent.
    vocab = ["the", "otp", "is", "445566", "sir", "bank", "account", "please"]

    def make_seq(n):
        toks = list(rng.choice(vocab, size=n))
        tags, prev = [], "O"
        for t in toks:
            if t in ("otp", "bank"):
                tag = "B-E"
            elif t in ("account", "445566") and prev in ("B-E", "I-E"):
                tag = "I-E"
            else:
                tag = "O"
            tags.append(tag)
            prev = tag
        return [[f"w={t}", f"len={len(t)}", "bias"] for t in toks], tags

    X, Y = zip(*[make_seq(int(rng.integers(4, 24))) for _ in range(400)])
    X, Y = list(X), list(Y)

    crf = LinearChainCRF(l2=0.5, max_iter=200)
    t0 = time.time()
    crf.fit(X[:320], Y[:320])
    print(json.dumps(crf.history_, indent=2))
    print(f"fit in {time.time() - t0:.2f}s")
    print("train token accuracy   :", round(crf.score(X[:320], Y[:320]), 4))
    print("held-out token accuracy:", round(crf.score(X[320:], Y[320:]), 4))
    print("example:", list(zip([f[0][2:] for f in X[320]], crf.predict_one(X[320]))))

    # Batched forward-backward must agree with a single-sequence run.
    data_all = crf.prepare(X[:8], Y[:8])
    W, tr, st, sp = crf._unpack(crf.w_)
    trans, start, stop = crf._effective(tr, st, sp)
    E_all = np.asarray(data_all.X @ W)
    E = E_all[data_all.idx_pad] * data_all.valid[:, :, None]
    logZ_batch, _, _ = crf._forward_backward(
        E, data_all.valid, data_all.lengths, trans, start, stop
    )
    logZ_single = np.array([crf.log_marginal(seq) for seq in X[:8]])
    print("batched vs single logZ max diff:",
          f"{float(np.max(np.abs(logZ_batch - logZ_single))):.3e}")
    assert np.allclose(logZ_batch, logZ_single, atol=1e-8)

    # Gradient check on a deliberately small problem.
    small = LinearChainCRF(l2=0.7, min_freq=1)
    small._build_index(X[:6], Y[:6])
    data = small.prepare(X[:6], Y[:6])
    w = rng.normal(scale=0.4, size=small.n_params)
    _, ana = small.loss_and_grad(w, data)
    num = numerical_gradient(small, w, data)
    err = float(np.max(np.abs(ana - num)))
    print(f"gradient check on {small.n_params} params: max abs diff = {err:.3e}")
    assert err < 1e-6, err

    m = crf.predict_marginals(X[0])
    assert all(abs(sum(row.values()) - 1.0) < 1e-9 for row in m)
    print("marginals normalised: ok")
