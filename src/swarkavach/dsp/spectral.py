"""STFT, power spectra and the five spectral shape descriptors.

The descriptors (centroid, bandwidth, rolloff, flatness, flux) are cheap and
they carry real anti-spoofing signal: a vocoder that reconstructs a waveform
from a mel spectrogram tends to produce a flatter, less variable high band than
a real telephone recording, so the variance of spectral flatness across a call
is one of the 25 fusion features (`spec_flatness_var`).

Each descriptor accepts either a 1-D signal or an already computed power
spectrum of shape (n_frames, n_fft//2+1). Passing the spectrum avoids five
redundant FFTs when you want all five.

Run the self-test with:
    python -m swarkavach.dsp.spectral
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import fft as sfft

from ..config import SETTINGS, FrameConfig
from .framing import frame_signal, preemphasis

__all__ = [
    "stft",
    "power_spectrum",
    "magnitude_spectrum",
    "spectrogram_db",
    "fft_frequencies",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "spectral_flatness",
    "spectral_flux",
    "spectral_descriptors",
]

_EPS = 1e-12


def _cfg(cfg: Optional[FrameConfig]) -> FrameConfig:
    return SETTINGS.frame if cfg is None else cfg


def _n_fft(cfg: FrameConfig) -> int:
    # A transform shorter than the frame would silently throw samples away.
    return int(max(cfg.n_fft, cfg.frame_len))


def fft_frequencies(cfg: Optional[FrameConfig] = None) -> np.ndarray:
    """Centre frequency in Hz of every rfft bin, shape (n_fft//2+1,)."""
    cfg = _cfg(cfg)
    return sfft.rfftfreq(_n_fft(cfg), d=1.0 / float(cfg.sr))


def stft(
    x: np.ndarray,
    cfg: Optional[FrameConfig] = None,
    apply_preemph: bool = True,
) -> np.ndarray:
    """Short-time Fourier transform, complex, shape (n_frames, n_fft//2+1).

    Pre-emphasis is part of the analysis chain, not of the caller's job, so it
    is applied here by default using cfg.preemphasis. Pass apply_preemph=False
    when you want the raw spectrum (pitch analysis wants the low harmonics left
    alone, for example).
    """
    cfg = _cfg(cfg)
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        x = np.zeros(1, dtype=np.float64)
    if apply_preemph and cfg.preemphasis:
        x = preemphasis(x, cfg.preemphasis)
    if cfg.center:
        pad = cfg.frame_len // 2
        mode = "reflect" if x.size > pad else "constant"
        x = np.pad(x, (pad, pad), mode=mode)
    frames = frame_signal(x, cfg.frame_len, cfg.hop_len, cfg.window)
    return sfft.rfft(frames, n=_n_fft(cfg), axis=1)


def magnitude_spectrum(x: np.ndarray, cfg: Optional[FrameConfig] = None, **kw) -> np.ndarray:
    """|STFT|, shape (n_frames, n_fft//2+1)."""
    return np.abs(stft(x, cfg, **kw))


def power_spectrum(x: np.ndarray, cfg: Optional[FrameConfig] = None, **kw) -> np.ndarray:
    """|STFT|^2. This is the input to every filterbank in this package."""
    spec = stft(x, cfg, **kw)
    return (spec.real ** 2 + spec.imag ** 2)


def spectrogram_db(
    x: np.ndarray,
    cfg: Optional[FrameConfig] = None,
    top_db: float = 80.0,
) -> np.ndarray:
    """Power spectrogram in dB relative to its own maximum, floored at -top_db.

    0 dB is the loudest bin in the utterance, so the dashboard gets a picture
    with a fixed dynamic range whatever the recording level was.
    """
    s = _as_power(x, cfg)
    db = 10.0 * np.log10(np.maximum(s, _EPS))
    peak = float(db.max()) if db.size else 0.0
    db = db - peak
    return np.maximum(db, -abs(float(top_db)))


def _as_power(x: np.ndarray, cfg: Optional[FrameConfig]) -> np.ndarray:
    """Accept a waveform or an already computed power spectrum."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 2:
        return arr
    return power_spectrum(arr, cfg)


def spectral_centroid(x: np.ndarray, cfg: Optional[FrameConfig] = None) -> np.ndarray:
    """Energy-weighted mean frequency per frame, in Hz. The "brightness" of the
    frame. Silent frames report 0.0 rather than 0/0."""
    cfg = _cfg(cfg)
    s = _as_power(x, cfg)
    f = fft_frequencies(cfg)
    total = s.sum(axis=1)
    cent = (s @ f) / np.maximum(total, _EPS)
    return np.where(total > _EPS, cent, 0.0)


def spectral_bandwidth(
    x: np.ndarray,
    cfg: Optional[FrameConfig] = None,
    p: float = 2.0,
) -> np.ndarray:
    """Energy-weighted spread around the centroid, in Hz (p=2 gives a std)."""
    cfg = _cfg(cfg)
    s = _as_power(x, cfg)
    f = fft_frequencies(cfg)
    total = np.maximum(s.sum(axis=1), _EPS)
    cent = (s @ f) / total
    dev = np.abs(f[None, :] - cent[:, None]) ** p
    bw = ((s * dev).sum(axis=1) / total) ** (1.0 / p)
    return np.where(s.sum(axis=1) > _EPS, bw, 0.0)


def spectral_rolloff(
    x: np.ndarray,
    cfg: Optional[FrameConfig] = None,
    roll_percent: float = 0.85,
) -> np.ndarray:
    """Frequency below which `roll_percent` of the frame energy sits, in Hz.

    On telephone audio this mostly tracks the codec's low-pass corner, which is
    exactly why it separates a clean synthetic waveform from one that has been
    through G.711 or GSM.
    """
    cfg = _cfg(cfg)
    s = _as_power(x, cfg)
    f = fft_frequencies(cfg)
    total = s.sum(axis=1)
    cum = np.cumsum(s, axis=1)
    target = roll_percent * np.maximum(total, _EPS)
    idx = np.argmax(cum >= target[:, None], axis=1)
    out = f[idx]
    return np.where(total > _EPS, out, 0.0)


def spectral_flatness(x: np.ndarray, cfg: Optional[FrameConfig] = None) -> np.ndarray:
    """Geometric mean over arithmetic mean of the power spectrum, in [0, 1].

    1.0 is white noise, near 0.0 is a clean harmonic stack. Its variance across
    a call is `spec_flatness_var` in the fusion vector.
    """
    cfg = _cfg(cfg)
    s = np.maximum(_as_power(x, cfg), _EPS)
    geo = np.exp(np.mean(np.log(s), axis=1))
    ari = np.mean(s, axis=1)
    return np.clip(geo / np.maximum(ari, _EPS), 0.0, 1.0)


def spectral_flux(x: np.ndarray, cfg: Optional[FrameConfig] = None) -> np.ndarray:
    """How much the (level normalised) magnitude spectrum changed since the
    previous frame. Rectified, so only rises count: that is the onset detector
    behaviour that makes it useful for syllable rate. First frame is 0."""
    cfg = _cfg(cfg)
    s = np.sqrt(np.maximum(_as_power(x, cfg), 0.0))
    norm = np.maximum(np.linalg.norm(s, axis=1, keepdims=True), _EPS)
    s = s / norm
    d = np.diff(s, axis=0)
    flux = np.sqrt(np.sum(np.maximum(d, 0.0) ** 2, axis=1))
    return np.concatenate([[0.0], flux])


def spectral_descriptors(
    x: np.ndarray,
    cfg: Optional[FrameConfig] = None,
) -> dict:
    """All five descriptors from one FFT pass. Keys: centroid, bandwidth,
    rolloff, flatness, flux, each shape (n_frames,)."""
    cfg = _cfg(cfg)
    s = _as_power(x, cfg)
    return {
        "centroid": spectral_centroid(s, cfg),
        "bandwidth": spectral_bandwidth(s, cfg),
        "rolloff": spectral_rolloff(s, cfg),
        "flatness": spectral_flatness(s, cfg),
        "flux": spectral_flux(s, cfg),
    }


if __name__ == "__main__":  # pragma: no cover - self-test
    import time

    t0 = time.time()
    cfg = SETTINGS.frame
    sr = cfg.sr
    n = sr * 3
    t = np.arange(n) / sr
    rng = np.random.default_rng(0)

    tone = 0.5 * np.sin(2 * np.pi * 500.0 * t)
    noise = 0.5 * rng.standard_normal(n)
    sweep = 0.5 * np.sin(2 * np.pi * (200.0 + 1000.0 * t / 3.0) * t)

    print("spectral self-test")
    S = stft(tone, cfg)
    P = power_spectrum(tone, cfg)
    print(f"  stft shape {S.shape} dtype {S.dtype}")
    print(f"  power shape {P.shape}, n_fft={max(cfg.n_fft, cfg.frame_len)}, "
          f"bin width {sr / max(cfg.n_fft, cfg.frame_len):.2f} Hz")

    peak_bin = int(np.argmax(P.mean(axis=0)))
    print(f"  500 Hz tone peaks at bin {peak_bin} = "
          f"{fft_frequencies(cfg)[peak_bin]:.1f} Hz")

    db = spectrogram_db(tone, cfg)
    print(f"  spectrogram_db range [{db.min():.1f}, {db.max():.1f}] dB")

    for name, sig in (("tone", tone), ("noise", noise), ("sweep", sweep)):
        d = spectral_descriptors(sig, cfg)
        print(f"  {name:6s} centroid={d['centroid'].mean():7.1f} Hz  "
              f"bw={d['bandwidth'].mean():7.1f} Hz  "
              f"rolloff={d['rolloff'].mean():7.1f} Hz  "
              f"flatness={d['flatness'].mean():.4f}  "
              f"flux={d['flux'].mean():.4f}")

    # One frame checked against a hand rolled DFT.
    cfg_raw = FrameConfig(sr=sr, preemphasis=0.0)
    frames = frame_signal(tone, cfg_raw.frame_len, cfg_raw.hop_len, cfg_raw.window)
    ref = np.fft.rfft(frames[5], n=max(cfg_raw.n_fft, cfg_raw.frame_len))
    got = stft(tone, cfg_raw)[5]
    print(f"  stft vs numpy.fft on frame 5: max abs diff "
          f"{np.max(np.abs(ref - got)):.2e}")
    print(f"  done in {time.time() - t0:.2f} s")
