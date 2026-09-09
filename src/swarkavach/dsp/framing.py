"""Short-time analysis: pre-emphasis, framing, windows, energy and ZCR.

Every other module in `dsp` starts here. Spectra, filterbanks, cepstra, pitch
and VAD all need the signal cut into overlapping frames, and they all need to
agree on how many frames there are and where each one begins, so that decision
lives in exactly one place: `frame_signal`.

Convention used throughout the package: analysis is done in float64 even though
audio arrives as float32. The extra precision costs nothing at these sizes and
it matters for the Levinson-Durbin recursion in `cepstral.lpcc`, where a 16th
order autocorrelation matrix on a quiet frame is close to singular.

Run the self-test with:
    python -m swarkavach.dsp.framing
"""

from __future__ import annotations

from typing import Union

import numpy as np

__all__ = [
    "preemphasis",
    "get_window",
    "frame_signal",
    "frame_count",
    "frame_times",
    "short_time_energy",
    "log_energy",
    "rms_db",
    "zero_crossing_rate",
    "WINDOWS",
]

#: Window names this module accepts. Aliases are folded onto the four the
#: contract asks for.
WINDOWS = ("hamming", "hann", "rect", "blackman")

_WINDOW_ALIASES = {
    "hamming": "hamming",
    "hann": "hann",
    "hanning": "hann",
    "rect": "rect",
    "rectangular": "rect",
    "boxcar": "rect",
    "none": "rect",
    "blackman": "blackman",
}

_EPS = 1e-12


def preemphasis(x: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """First order high-pass: y[n] = x[n] - coef * x[n-1].

    Speech has roughly a -6 dB per octave spectral tilt from the glottal source
    and lip radiation. Flattening it before the FFT stops the low frequencies
    from dominating the filterbank energies, and it improves the conditioning of
    the LPC autocorrelation matrix. 0.97 is the usual value.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0 or coef is None or coef <= 0.0:
        return x.copy()
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - coef * x[:-1]
    return y


def get_window(name: str, n: int) -> np.ndarray:
    """Return an n-point analysis window.

    Symmetric (not periodic) definitions, which is the textbook choice for
    analysis. The periodic variant only matters when you plan to overlap-add
    back to a waveform, and nothing here does.
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    key = _WINDOW_ALIASES.get(str(name).strip().lower())
    if key is None:
        raise ValueError(f"unknown window {name!r}, expected one of {WINDOWS}")
    if n == 1:
        return np.ones(1, dtype=np.float64)
    if key == "hamming":
        return np.hamming(n).astype(np.float64)
    if key == "hann":
        return np.hanning(n).astype(np.float64)
    if key == "blackman":
        return np.blackman(n).astype(np.float64)
    return np.ones(n, dtype=np.float64)


def frame_count(n_samples: int, frame_len: int, hop_len: int) -> int:
    """How many whole frames fit in a signal of this length.

    A signal shorter than one frame still gets one frame (zero padded), because
    returning an empty feature matrix pushes the special case into every caller.
    """
    if frame_len <= 0 or hop_len <= 0:
        raise ValueError("frame_len and hop_len must be positive")
    if n_samples < frame_len:
        return 1
    return 1 + (n_samples - frame_len) // hop_len


def frame_signal(
    x: np.ndarray,
    frame_len: int,
    hop_len: int,
    window: Union[str, np.ndarray] = "hamming",
) -> np.ndarray:
    """Cut a signal into overlapping windowed frames, shape (n_frames, frame_len).

    The tail that does not fill a whole frame is dropped, which is standard and
    keeps frame timing exact: frame i starts at sample i * hop_len.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if frame_len <= 0 or hop_len <= 0:
        raise ValueError("frame_len and hop_len must be positive")

    n_frames = frame_count(x.size, frame_len, hop_len)
    need = (n_frames - 1) * hop_len + frame_len
    if x.size < need:
        x = np.pad(x, (0, need - x.size))

    view = np.lib.stride_tricks.sliding_window_view(x, frame_len)[::hop_len]
    frames = np.array(view[:n_frames], dtype=np.float64, copy=True)

    if isinstance(window, str):
        w = get_window(window, frame_len)
    else:
        w = np.asarray(window, dtype=np.float64).ravel()
        if w.size != frame_len:
            raise ValueError(f"window length {w.size} != frame_len {frame_len}")
    frames *= w
    return frames


def frame_times(n_frames: int, hop_len: int, sr: int, frame_len: int = 0) -> np.ndarray:
    """Centre time in seconds of each frame. Handy for plotting and for turning
    a per-frame VAD mask back into second-valued segments."""
    idx = np.arange(int(n_frames), dtype=np.float64)
    return (idx * hop_len + frame_len / 2.0) / float(sr)


def short_time_energy(frames: np.ndarray) -> np.ndarray:
    """Sum of squares per frame, the classic E_n = sum_m x(m)^2.

    Sum rather than mean, to match the textbook definition. Anything that wants
    a length-independent number should use `rms_db`.
    """
    frames = np.atleast_2d(np.asarray(frames, dtype=np.float64))
    return np.sum(frames * frames, axis=1)


def log_energy(frames: np.ndarray, floor: float = _EPS) -> np.ndarray:
    """Natural log of the frame energy, floored so silence gives a finite number.

    Natural log, not dB, because this is what replaces c0 in the MFCC vector and
    the rest of that vector is a log-magnitude spectrum in the same units.
    """
    return np.log(np.maximum(short_time_energy(frames), floor))


def rms_db(frames: np.ndarray, floor: float = _EPS) -> np.ndarray:
    """Per-frame RMS level in dB. Used by VAD and by the prosody energy stats,
    where a threshold like "6 dB above the noise floor" has to mean something."""
    frames = np.atleast_2d(np.asarray(frames, dtype=np.float64))
    power = np.mean(frames * frames, axis=1)
    return 10.0 * np.log10(np.maximum(power, floor))


def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    """Fraction of adjacent sample pairs that change sign, in [0, 1].

    A sine at f0 crosses zero 2*f0 times a second, so the rate here is about
    2*f0/sr. High ZCR with low energy is the signature of an unvoiced fricative,
    which is why VAD looks at both.
    """
    frames = np.atleast_2d(np.asarray(frames, dtype=np.float64))
    if frames.shape[1] < 2:
        return np.zeros(frames.shape[0], dtype=np.float64)
    # Treat exact zero as non-negative so a silent frame reports no crossings
    # instead of one per sample.
    nonneg = frames >= 0.0
    crossings = np.count_nonzero(nonneg[:, 1:] != nonneg[:, :-1], axis=1)
    return crossings.astype(np.float64) / float(frames.shape[1] - 1)


if __name__ == "__main__":  # pragma: no cover - self-test
    import time

    t0 = time.time()
    sr = 8000
    n = sr * 3
    t = np.arange(n) / sr
    tone = 0.6 * np.sin(2 * np.pi * 220.0 * t)
    square = np.sign(np.sin(2 * np.pi * 100.0 * t))

    print("framing self-test")
    print(f"  signal: {n} samples at {sr} Hz ({n / sr:.1f} s)")

    frames = frame_signal(tone, 200, 80, "hamming")
    print(f"  frames (25 ms / 10 ms): {frames.shape}, expected n_frames="
          f"{frame_count(n, 200, 80)}")

    for name in WINDOWS:
        w = get_window(name, 200)
        print(f"  window {name:9s} sum={w.sum():8.2f} max={w.max():.3f} "
              f"edge={w[0]:.4f}")

    e = short_time_energy(frames)
    le = log_energy(frames)
    print(f"  energy: mean={e.mean():.4f} min={e.min():.4f} max={e.max():.4f}")
    print(f"  log energy range: [{le.min():.2f}, {le.max():.2f}]")

    sq_frames = frame_signal(square, 200, 80, "rect")
    z = zero_crossing_rate(sq_frames)
    expected = 2.0 * 100.0 / sr
    print(f"  ZCR on a 100 Hz square wave: {z.mean():.5f} "
          f"(expected about {expected:.5f})")

    z_noise = zero_crossing_rate(frame_signal(
        np.random.default_rng(0).standard_normal(n), 200, 80, "rect"))
    print(f"  ZCR on white noise: {z_noise.mean():.3f} (expected about 0.5)")

    rec = frame_signal(tone, 100, 100, "rect").ravel()
    err = float(np.max(np.abs(rec - tone[: rec.size])))
    print(f"  rect framing at hop=frame reconstructs, max error {err:.2e}")

    short = frame_signal(np.ones(37), 200, 80, "hamming")
    print(f"  signal shorter than a frame -> {short.shape} (padded, never empty)")
    print(f"  done in {time.time() - t0:.2f} s")
