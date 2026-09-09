"""Dynamic coefficients and per-utterance normalisation.

Static cepstra describe one 25 ms slice. What separates a real voice from a
vocoded one is often how the slices move: synthesis systems generate frames
that are individually plausible but transition too smoothly, so the delta and
delta-delta streams carry a good part of the anti-spoofing signal.

CMVN then removes the channel. A convolutional channel (a handset, a codec's
fixed response) is additive in the log spectral domain and therefore additive in
the cepstral domain, so subtracting the utterance mean subtracts most of it.
That is cheap insurance for a system that has to survive G.711, GSM and AMR.

Run the self-test with:
    python -m swarkavach.dsp.deltas
"""

from __future__ import annotations

import numpy as np

__all__ = ["delta", "add_deltas", "cmvn", "stack_context"]

_EPS = 1e-8


def delta(feat: np.ndarray, width: int = 2) -> np.ndarray:
    """Regression based first difference, same shape as the input.

        d_t = sum_{n=1..W} n (f_{t+n} - f_{t-n}) / (2 sum_{n=1..W} n^2)

    This is the HTK formula, a least squares slope over a 2W+1 frame window,
    not a raw difference: a raw difference amplifies frame to frame noise.
    Edges are handled by replicating the first and last frame, which is what
    keeps the output the same length as the input.
    """
    feat = np.atleast_2d(np.asarray(feat, dtype=np.float64))
    n_frames = feat.shape[0]
    width = max(int(width), 1)
    if n_frames == 1:
        return np.zeros_like(feat)

    padded = np.pad(feat, ((width, width), (0, 0)), mode="edge")
    denom = 2.0 * sum(n * n for n in range(1, width + 1))
    out = np.zeros_like(feat)
    for n in range(1, width + 1):
        ahead = padded[width + n: width + n + n_frames]
        behind = padded[width - n: width - n + n_frames]
        out += n * (ahead - behind)
    return out / denom


def add_deltas(feat: np.ndarray, width: int = 2, order: int = 2) -> np.ndarray:
    """Append derivatives: order=2 gives (n_frames, n_ceps * 3)."""
    feat = np.atleast_2d(np.asarray(feat, dtype=np.float64))
    parts = [feat]
    cur = feat
    for _ in range(max(int(order), 0)):
        cur = delta(cur, width)
        parts.append(cur)
    return np.concatenate(parts, axis=1)


def cmvn(feat: np.ndarray, var_norm: bool = True, eps: float = _EPS) -> np.ndarray:
    """Cepstral mean and variance normalisation over the utterance.

    Constant columns (a silent recording, a dead channel) are left at zero
    instead of being divided by an epsilon and blown up into noise.
    """
    feat = np.atleast_2d(np.asarray(feat, dtype=np.float64))
    if feat.shape[0] == 0:
        return feat
    centred = feat - feat.mean(axis=0, keepdims=True)
    if not var_norm:
        return centred
    std = centred.std(axis=0, keepdims=True)
    alive = std > eps
    out = np.zeros_like(centred)
    np.divide(centred, std, out=out, where=alive)
    return out


def stack_context(feat: np.ndarray, left: int = 0, right: int = 0) -> np.ndarray:
    """Splice neighbouring frames into each frame's vector, edges replicated.

    Not part of the contract, but the GMM and GBM scorers want a wider view than
    one frame and this is cheaper than training a sequence model.
    """
    feat = np.atleast_2d(np.asarray(feat, dtype=np.float64))
    if left <= 0 and right <= 0:
        return feat
    n = feat.shape[0]
    padded = np.pad(feat, ((left, right), (0, 0)), mode="edge")
    cols = [padded[i: i + n] for i in range(left + right + 1)]
    return np.concatenate(cols, axis=1)


if __name__ == "__main__":  # pragma: no cover - self-test
    import time

    t0 = time.time()
    rng = np.random.default_rng(0)
    feat = rng.normal(loc=3.0, scale=2.0, size=(200, 20))

    print("deltas self-test")
    d1 = delta(feat)
    d2 = delta(d1)
    full = add_deltas(feat, width=2, order=2)
    print(f"  static {feat.shape} -> delta {d1.shape} -> +deltas {full.shape}")
    print(f"  delta std {d1.std():.4f}, delta-delta std {d2.std():.4f} "
          "(each derivative should be smaller)")

    # A straight ramp has a known slope: delta must recover it exactly away
    # from the edges.
    ramp = np.arange(50, dtype=np.float64)[:, None] * np.array([[1.0, -2.0]])
    dr = delta(ramp, width=2)
    print(f"  ramp of slope [1, -2] -> delta rows 5 and 44: "
          f"{np.round(dr[5], 6)} {np.round(dr[44], 6)}")

    norm = cmvn(feat)
    print(f"  cmvn mean {np.abs(norm.mean(axis=0)).max():.2e}, "
          f"std {np.abs(norm.std(axis=0) - 1).max():.2e}")

    const = np.ones((30, 4))
    print(f"  cmvn on a constant matrix stays finite: "
          f"{bool(np.isfinite(cmvn(const)).all())}, max abs "
          f"{np.abs(cmvn(const)).max():.1f}")
    print(f"  stack_context(2, 2): {stack_context(feat, 2, 2).shape}")
    print(f"  done in {time.time() - t0:.2f} s")
