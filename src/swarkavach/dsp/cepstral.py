"""Cepstral front ends: MFCC, GFCC, LFCC, CQCC and LPCC, all written out here.

The DCT, the filterbank projection, the autocorrelation, the Levinson-Durbin
recursion and the LPC to cepstrum recursion are implemented in this file. That
is the point of the exercise: the only thing borrowed from scipy is the FFT.

Shared pipeline for the four filterbank features:

    signal -> pre-emphasis -> frame -> window -> |FFT|^2
           -> filterbank -> compression -> DCT-II -> lifter

They differ only in the filterbank and the compression:

    mfcc   mel bank,      log compression
    lfcc   linear bank,   log compression      (ASVspoof baseline front end)
    gfcc   gammatone bank, cube root           (loudness law, the usual GFCC choice)
    cqcc   geometric bank, log compression     (approximates Todisco's CQCC)

LPCC takes the other route entirely: model the frame as an all-pole filter and
convert those poles to a cepstrum. It carries different information from the
filterbank features (vocal tract resonances rather than a smoothed spectrum),
which is why the anti-spoof branch scores it as a separate system.

Run the self-test with:
    python -m swarkavach.dsp.cepstral
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, Optional

import numpy as np
from scipy import fft as sfft

from ..config import SETTINGS, CepstralConfig, FrameConfig
from .filterbanks import get_filterbank
from .framing import frame_signal, preemphasis
from .spectral import power_spectrum

__all__ = [
    "dct2",
    "dct_matrix",
    "lifter",
    "filterbank_energies",
    "mfcc",
    "gfcc",
    "lfcc",
    "cqcc",
    "lpcc",
    "autocorrelation",
    "levinson_durbin",
    "lpc_to_cepstrum",
    "extract",
    "FEATURE_EXTRACTORS",
]

_EPS = 1e-12
_TINY = 1e-10


def _frame_cfg(cfg: Optional[FrameConfig]) -> FrameConfig:
    return SETTINGS.frame if cfg is None else cfg


def _cep_cfg(cep: Optional[CepstralConfig]) -> CepstralConfig:
    return SETTINGS.cepstral if cep is None else cep


def _n_fft(cfg: FrameConfig) -> int:
    return int(max(cfg.n_fft, cfg.frame_len))


# --------------------------------------------------------------------------
# DCT-II
# --------------------------------------------------------------------------


@lru_cache(maxsize=32)
def dct_matrix(n_in: int, n_out: int) -> np.ndarray:
    """Orthonormal DCT-II matrix, shape (n_out, n_in).

    D[k, n] = s(k) cos(pi k (2n + 1) / (2 N)),  s(0) = sqrt(1/N), s(k>0) = sqrt(2/N)

    Same normalisation as scipy.fft.dct(..., type=2, norm="ortho"), which the
    test checks. Cached because a 20 by 40 matrix gets rebuilt otherwise on
    every single utterance.
    """
    n = np.arange(n_in, dtype=np.float64)
    k = np.arange(n_out, dtype=np.float64)
    d = np.cos(np.pi * k[:, None] * (2.0 * n[None, :] + 1.0) / (2.0 * n_in))
    scale = np.full(n_out, np.sqrt(2.0 / n_in))
    if n_out > 0:
        scale[0] = np.sqrt(1.0 / n_in)
    d *= scale[:, None]
    d.flags.writeable = False
    return d


def dct2(x: np.ndarray, n_out: int) -> np.ndarray:
    """DCT-II along the last axis, orthonormal, keeping the first n_out coefficients.

    The DCT is here to decorrelate the filterbank log energies (adjacent filters
    overlap, so their outputs are heavily correlated) and to truncate: the low
    order coefficients hold the spectral envelope, the high order ones hold the
    pitch harmonics we do not want in an envelope feature.
    """
    arr = np.asarray(x, dtype=np.float64)
    one_d = arr.ndim == 1
    if one_d:
        arr = arr[None, :]
    n_in = arr.shape[-1]
    keep = int(min(n_out, n_in))
    out = arr @ dct_matrix(n_in, keep).T
    if n_out > keep:  # asked for more coefficients than the transform has
        out = np.pad(out, ((0, 0), (0, n_out - keep)))
    return out[0] if one_d else out


def lifter(cep: np.ndarray, n_lifter: int = 22) -> np.ndarray:
    """Sinusoidal liftering, w[n] = 1 + (L/2) sin(pi n / L).

    Higher cepstral coefficients have much smaller dynamic range than the low
    ones. Without liftering a diagonal covariance GMM effectively ignores them.
    L=22 is the HTK default.
    """
    if not n_lifter or n_lifter <= 0:
        return cep
    n = np.arange(cep.shape[-1], dtype=np.float64)
    w = 1.0 + (n_lifter / 2.0) * np.sin(np.pi * n / n_lifter)
    return cep * w


# --------------------------------------------------------------------------
# Filterbank features
# --------------------------------------------------------------------------


def filterbank_energies(
    x: np.ndarray,
    cfg: Optional[FrameConfig] = None,
    cep: Optional[CepstralConfig] = None,
    kind: str = "mel",
) -> np.ndarray:
    """Project the power spectrum onto a filterbank, shape (n_frames, n_filters)."""
    cfg, cep = _frame_cfg(cfg), _cep_cfg(cep)
    n_fft = _n_fft(cfg)
    spec = power_spectrum(x, cfg)
    fb = get_filterbank(kind, cfg.sr, n_fft, cep.n_filters, cep.fmin, cep.fmax)
    return spec @ fb.T


def _cepstra(
    x: np.ndarray,
    cfg: FrameConfig,
    cep: CepstralConfig,
    kind: str,
    compression: str = "log",
) -> np.ndarray:
    spec = power_spectrum(x, cfg)
    fb = get_filterbank(kind, cfg.sr, _n_fft(cfg), cep.n_filters, cep.fmin, cep.fmax)
    energies = spec @ fb.T

    if compression == "cuberoot":
        # Stevens' loudness law. GFCC uses it instead of log because the cube
        # root keeps low energy channels from swinging wildly under noise.
        comp = np.cbrt(np.maximum(energies, 0.0))
    else:
        comp = np.log(np.maximum(energies, _EPS))

    out = dct2(comp, cep.n_ceps)
    out = lifter(out, cep.lifter)

    if cep.use_energy and out.shape[1] > 0:
        # By Parseval the summed power spectrum is the frame energy up to a
        # constant factor, so no second pass over the waveform is needed.
        out[:, 0] = np.log(np.maximum(spec.sum(axis=1), _EPS))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def mfcc(x, cfg: Optional[FrameConfig] = None, cep: Optional[CepstralConfig] = None) -> np.ndarray:
    """Mel frequency cepstral coefficients, shape (n_frames, cep.n_ceps)."""
    return _cepstra(x, _frame_cfg(cfg), _cep_cfg(cep), "mel", "log")


def lfcc(x, cfg: Optional[FrameConfig] = None, cep: Optional[CepstralConfig] = None) -> np.ndarray:
    """Linear frequency cepstral coefficients.

    The ASVspoof 2019 and 2021 organisers shipped LFCC-GMM as a baseline because
    linear spacing keeps full resolution in the top of the band, where vocoder
    and waveform-model artefacts live. Mel spacing throws that resolution away
    to match human hearing, which is the wrong objective for spoof detection.
    """
    return _cepstra(x, _frame_cfg(cfg), _cep_cfg(cep), "linear", "log")


def gfcc(x, cfg: Optional[FrameConfig] = None, cep: Optional[CepstralConfig] = None) -> np.ndarray:
    """Gammatone frequency cepstral coefficients (ERB spaced, cube root compressed)."""
    return _cepstra(x, _frame_cfg(cfg), _cep_cfg(cep), "gammatone", "cuberoot")


def cqcc(x, cfg: Optional[FrameConfig] = None, cep: Optional[CepstralConfig] = None) -> np.ndarray:
    """Constant-Q cepstral coefficients, approximated on the FFT grid.

    Real CQCC (Todisco, Delgado and Evans 2016) runs a constant-Q transform and
    then resamples the log-frequency axis uniformly before the DCT. Here the
    geometric filterbank already places channels uniformly in log frequency, so
    taking the DCT straight across those channels does the same job. The
    difference from true CQCC is the time resolution: a real CQT uses a longer
    window at low frequencies, this uses one window length for everything.
    """
    return _cepstra(x, _frame_cfg(cfg), _cep_cfg(cep), "cqt", "log")


# --------------------------------------------------------------------------
# Linear prediction
# --------------------------------------------------------------------------


def autocorrelation(frames: np.ndarray, order: int) -> np.ndarray:
    """Biased short-time autocorrelation, lags 0..order, shape (n_frames, order+1).

    Computed through the FFT: r = IFFT(|FFT(frame)|^2). Same answer as the
    direct double loop, much faster, and the zero padding to at least twice the
    frame length is what keeps it linear rather than circular.
    """
    frames = np.atleast_2d(np.asarray(frames, dtype=np.float64))
    n = frames.shape[1]
    nfft = int(2 ** np.ceil(np.log2(max(2 * n, 2))))
    spec = sfft.rfft(frames, n=nfft, axis=1)
    r = sfft.irfft(spec.real ** 2 + spec.imag ** 2, n=nfft, axis=1)
    return r[:, : order + 1]


def levinson_durbin(r: np.ndarray, order: int):
    """Solve the Yule-Walker system for every frame at once.

    r is (n_frames, order+1). Returns (a, err) where a is (n_frames, order+1)
    with a[:, 0] = 1 and the prediction is x[n] = -sum_i a_i x[n-i], and err is
    the residual energy per frame.

    Two guards keep this from producing NaN on a frame that carries no signal:
    a small ridge on r[0] (the standard white noise correction, which also fixes
    ill-conditioning on a strongly periodic frame), and clipping the reflection
    coefficients to just inside the unit circle so the synthesis filter stays
    stable and the residual energy stays positive.
    """
    r = np.atleast_2d(np.asarray(r, dtype=np.float64)).copy()
    n_frames = r.shape[0]
    order = int(min(order, r.shape[1] - 1))

    r[:, 0] = r[:, 0] * 1.0001 + _TINY

    a = np.zeros((n_frames, order + 1), dtype=np.float64)
    a[:, 0] = 1.0
    err = np.maximum(r[:, 0].copy(), _TINY)

    for i in range(1, order + 1):
        if i > 1:
            acc = r[:, i] + np.sum(a[:, 1:i] * r[:, i - 1:0:-1], axis=1)
        else:
            acc = r[:, 1]
        k = np.clip(-acc / err, -0.999999, 0.999999)
        if i > 1:
            prev = a[:, 1:i].copy()
            a[:, 1:i] = prev + k[:, None] * prev[:, ::-1]
        a[:, i] = k
        err = np.maximum(err * (1.0 - k * k), _TINY)

    return a, err


def lpc_to_cepstrum(a: np.ndarray, err: np.ndarray, n_ceps: int) -> np.ndarray:
    """Recursion from LPC coefficients to the real cepstrum of the all-pole model.

    With A(z) = 1 + sum_i a_i z^-i and gain G:

        c_0 = log(G^2)  (taken as log of the residual energy)
        c_m = -a_m - sum_{k=1}^{min(m-1, p)} (1 - k/m) a_k c_{m-k}

    Coefficients past the model order keep coming from the recursion with
    a_m = 0, which is how you get 20 cepstral coefficients out of a 16th order
    model.
    """
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    err = np.asarray(err, dtype=np.float64).ravel()
    n_frames = a.shape[0]
    order = a.shape[1] - 1
    n_ceps = int(n_ceps)

    c = np.zeros((n_frames, max(n_ceps, 1)), dtype=np.float64)
    c[:, 0] = np.log(np.maximum(err, _TINY))

    for m in range(1, n_ceps):
        k_max = min(m - 1, order)
        if k_max >= 1:
            k = np.arange(1, k_max + 1)
            w = 1.0 - k / float(m)
            acc = np.sum(a[:, k] * c[:, m - k] * w[None, :], axis=1)
        else:
            acc = 0.0
        a_m = a[:, m] if m <= order else 0.0
        c[:, m] = -a_m - acc
    return c[:, :n_ceps]


def lpcc(
    x,
    cfg: Optional[FrameConfig] = None,
    order: int = 16,
    n_ceps: int = 20,
    cep: Optional[CepstralConfig] = None,
) -> np.ndarray:
    """Linear prediction cepstral coefficients, shape (n_frames, n_ceps).

    Order 16 at 8 kHz follows the usual rule of thumb of about two poles per
    kilohertz plus a few for the glottal source and radiation, so there are
    enough poles for four formants inside the telephone band.

    Frames with no energy (leading silence, a dropped packet) produce a singular
    autocorrelation matrix. Rather than let that propagate NaN through the whole
    feature matrix, those frames get a constant floor vector, and the caller can
    still spot them because c0 sits at the energy floor.
    """
    cfg = _frame_cfg(cfg)
    if cep is not None:
        n_ceps = cep.n_ceps
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        x = np.zeros(1, dtype=np.float64)
    if cfg.preemphasis:
        x = preemphasis(x, cfg.preemphasis)

    frames = frame_signal(x, cfg.frame_len, cfg.hop_len, cfg.window)
    energy = np.sum(frames * frames, axis=1)

    r = autocorrelation(frames, order)
    a, err = levinson_durbin(r, order)
    c = lpc_to_cepstrum(a, err, n_ceps)

    silent = energy <= _TINY
    if np.any(silent):
        c[silent, :] = 0.0
        c[silent, 0] = np.log(_TINY)
    return np.nan_to_num(np.clip(c, -1e6, 1e6), nan=0.0, posinf=0.0, neginf=0.0)


def _lpcc_entry(x, cfg: Optional[FrameConfig] = None, cep: Optional[CepstralConfig] = None):
    """Adapter so LPCC has the same (x, cfg, cep) signature as the rest."""
    cep = _cep_cfg(cep)
    return lpcc(x, cfg, order=16, n_ceps=cep.n_ceps)


#: Name to callable, every entry taking (x, cfg=None, cep=None).
FEATURE_EXTRACTORS: Dict[str, Callable] = {
    "mfcc": mfcc,
    "gfcc": gfcc,
    "lfcc": lfcc,
    "cqcc": cqcc,
    "lpcc": _lpcc_entry,
}


def extract(
    x,
    name: str,
    cfg: Optional[FrameConfig] = None,
    cep: Optional[CepstralConfig] = None,
) -> np.ndarray:
    """Look up a front end by name and run it."""
    key = str(name).lower()
    if key not in FEATURE_EXTRACTORS:
        raise ValueError(f"unknown feature set {name!r}, expected {sorted(FEATURE_EXTRACTORS)}")
    return FEATURE_EXTRACTORS[key](x, cfg, cep)


if __name__ == "__main__":  # pragma: no cover - self-test
    import time

    from scipy import signal as ssig

    cfg, cep = SETTINGS.frame, SETTINGS.cepstral
    sr = cfg.sr
    rng = np.random.default_rng(SETTINGS.pipeline.seed)
    n = sr * 30  # the 30 second call the timing budget is quoted for

    # A crude voiced signal: 130 Hz pulse train through three formants. The
    # formants are two pole resonators run with lfilter, which is linear in the
    # signal length. Convolving with an explicit impulse response would be
    # quadratic and would dominate the timing this self-test is meant to report.
    f0 = 130.0
    src = np.zeros(n)
    src[:: int(sr / f0)] = 1.0
    x = np.zeros(n)
    for fc, bw, g in ((600.0, 80.0, 1.0), (1200.0, 100.0, 0.6), (2600.0, 160.0, 0.3)):
        r = np.exp(-np.pi * bw / sr)
        w = 2 * np.pi * fc / sr
        x += g * ssig.lfilter([1.0], [1.0, -2 * r * np.cos(w), r * r], src)
    x = 0.5 * x / max(np.max(np.abs(x)), 1e-9)
    x += 0.005 * rng.standard_normal(n)

    print("cepstral self-test")
    print(f"  input: {n / sr:.0f} s at {sr} Hz")

    d = dct2(np.eye(8)[0], 8)
    print(f"  dct2 of a unit impulse (first 4): {np.round(d[:4], 4)}")

    total = 0.0
    for name in ("mfcc", "gfcc", "lfcc", "cqcc", "lpcc"):
        t0 = time.time()
        feat = extract(x, name, cfg, cep)
        dt = time.time() - t0
        total += dt
        print(f"  {name:5s} shape={feat.shape} finite={bool(np.isfinite(feat).all())} "
              f"c0_mean={feat[:, 0].mean():9.3f} c1_mean={feat[:, 1].mean():8.3f} "
              f"std={feat.std():7.3f}  {dt:.2f} s")
    print(f"  all five front ends: {total:.2f} s (budget 3.0 s)")

    sil = np.zeros(sr)
    print(f"  lpcc on pure silence: finite={bool(np.isfinite(lpcc(sil, cfg)).all())} "
          f"c0={lpcc(sil, cfg)[0, 0]:.2f}")
    print(f"  mfcc on pure silence: finite={bool(np.isfinite(mfcc(sil, cfg, cep)).all())}")

    a, err = levinson_durbin(autocorrelation(
        frame_signal(x[:2000], cfg.frame_len, cfg.hop_len), 16), 16)
    roots_ok = all(np.max(np.abs(np.roots(ai))) < 1.0 for ai in a[:5])
    print(f"  levinson_durbin: err[0]={err[0]:.4e}, first 5 filters stable={roots_ok}")
