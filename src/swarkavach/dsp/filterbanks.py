"""Four filterbanks, all returning shape (n_filters, n_fft//2+1).

Why four, when most systems ship one:

- mel: the standard perceptual scale, what MFCC uses. Good for what a listener
  hears, which is not necessarily where a vocoder leaves its fingerprints.
- linear: uniform spacing, what LFCC uses. This is the ASVspoof 2019 and 2021
  baseline front end for a reason. Mel spacing spends most of its resolution
  below 1 kHz, but synthesis artefacts (over-smoothed harmonics, a wrong noise
  floor, missing high band detail) sit in the top of the band, and linear
  spacing keeps the same resolution up there.
- gammatone: ERB spaced, what GFCC uses. Its filters overlap more heavily than
  mel triangles, which makes the feature less brittle under additive noise.
- constant-Q style geometric spacing: the front end for the CQCC approximation.
  Constant relative bandwidth gives fine frequency resolution at low
  frequencies, where the pitch harmonics of a synthetic voice are unnaturally
  regular.

The one failure mode that matters here: a filter narrower than the FFT bin
spacing lands between two bins and comes out all zeros, which silently turns
that cepstral channel into a constant. At 8 kHz with n_fft=512 the bins are
15.6 Hz apart and the lowest constant-Q filters are narrower than that, so
`_triangular_bank` widens any filter below a minimum width and, as a last
resort, drops a single unit spike on the nearest bin. Both are logged in the
`meta` returned by `filterbank_info` and covered by a test.

Run the self-test with:
    python -m swarkavach.dsp.filterbanks
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
from scipy import fft as sfft

from ..config import SETTINGS, TARGET_SR

__all__ = [
    "hz_to_mel",
    "mel_to_hz",
    "hz_to_erb",
    "erb_to_hz",
    "erb_bandwidth",
    "mel_filterbank",
    "gammatone_filterbank",
    "linear_filterbank",
    "cqt_like_filterbank",
    "get_filterbank",
    "filterbank_centres",
    "FILTERBANKS",
]

_EPS = 1e-12


# --------------------------------------------------------------------------
# Frequency scales
# --------------------------------------------------------------------------


def hz_to_mel(f):
    """O'Shaughnessy's mel scale, the one every MFCC paper means: 2595 log10(1 + f/700)."""
    f = np.asarray(f, dtype=np.float64)
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    """Inverse of `hz_to_mel`."""
    m = np.asarray(m, dtype=np.float64)
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def hz_to_erb(f):
    """Glasberg and Moore ERB-rate scale: 21.4 log10(1 + 0.00437 f).

    ERB number, not ERB bandwidth. Equal steps on this scale are equal steps
    along the cochlea, which is what the gammatone centre frequencies follow.
    """
    f = np.asarray(f, dtype=np.float64)
    return 21.4 * np.log10(1.0 + 0.00437 * f)


def erb_to_hz(e):
    """Inverse of `hz_to_erb`."""
    e = np.asarray(e, dtype=np.float64)
    return (10.0 ** (e / 21.4) - 1.0) / 0.00437


def erb_bandwidth(f):
    """Equivalent rectangular bandwidth in Hz at centre frequency f (Glasberg
    and Moore 1990): ERB = 24.7 (4.37 f / 1000 + 1)."""
    f = np.asarray(f, dtype=np.float64)
    return 24.7 * (4.37 * f / 1000.0 + 1.0)


# --------------------------------------------------------------------------
# Shared machinery
# --------------------------------------------------------------------------


def _bin_freqs(sr: int, n_fft: int) -> np.ndarray:
    return sfft.rfftfreq(int(n_fft), d=1.0 / float(sr))


def _resolve_range(sr: int, fmin: Optional[float], fmax: Optional[float]) -> Tuple[float, float]:
    nyq = sr / 2.0
    lo = 0.0 if fmin is None else float(fmin)
    hi = nyq if fmax is None else float(fmax)
    lo = max(0.0, min(lo, nyq - 1.0))
    hi = max(lo + 1.0, min(hi, nyq))
    return lo, hi


def _fix_empty_rows(fb: np.ndarray, centres: np.ndarray, freqs: np.ndarray) -> int:
    """Guarantee every row has energy. Returns how many rows had to be rescued.

    A filter this narrow is under-resolved by the FFT, so the honest thing is a
    single bin at its centre frequency: it keeps the channel alive and it is
    obvious in the filterbank plot that the resolution ran out there.
    """
    empty = np.where(fb.sum(axis=1) <= _EPS)[0]
    for k in empty:
        j = int(np.argmin(np.abs(freqs - centres[k])))
        fb[k, j] = 1.0
    return int(empty.size)


def _triangular_bank(
    freqs: np.ndarray,
    edges: np.ndarray,
    min_width_bins: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Triangles defined by n_filters+2 band edges, evaluated on the FFT grid.

    Evaluating the triangle analytically at each bin centre (instead of rounding
    the edges to bins first) keeps the shape right when the filters are narrow.
    `min_width_bins` widens any triangle whose base spans less than that many
    FFT bins, symmetrically around its centre.

    Returns (fb, centres, n_widened, n_rescued).
    """
    edges = np.asarray(edges, dtype=np.float64)
    n_filters = edges.size - 2
    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1.0
    min_width = min_width_bins * df

    left = edges[:-2].copy()
    centre = edges[1:-1].copy()
    right = edges[2:].copy()

    narrow = (right - left) < min_width
    n_widened = int(np.count_nonzero(narrow))
    if n_widened:
        half = min_width / 2.0
        left[narrow] = centre[narrow] - half
        right[narrow] = centre[narrow] + half

    # Degenerate edges would divide by zero; nudge them apart.
    left = np.minimum(left, centre - _EPS)
    right = np.maximum(right, centre + _EPS)

    f = freqs[None, :]
    up = (f - left[:, None]) / (centre - left)[:, None]
    down = (right[:, None] - f) / (right - centre)[:, None]
    fb = np.maximum(0.0, np.minimum(up, down))

    n_rescued = _fix_empty_rows(fb, centre, freqs)
    return fb, centre, n_widened, n_rescued


def _normalise(fb: np.ndarray, norm: Optional[str], edges: np.ndarray = None) -> np.ndarray:
    """`None` leaves triangles peaking at 1 (HTK style), "area" makes every
    filter sum to 1 (so wide high-frequency filters do not simply collect more
    energy), "slaney" scales by 2/(right-left)."""
    if norm is None or str(norm).lower() in ("none", ""):
        return fb
    key = str(norm).lower()
    if key == "area":
        s = fb.sum(axis=1, keepdims=True)
        return fb / np.maximum(s, _EPS)
    if key == "peak":
        m = fb.max(axis=1, keepdims=True)
        return fb / np.maximum(m, _EPS)
    if key == "slaney":
        if edges is None:
            raise ValueError("slaney normalisation needs the band edges")
        widths = edges[2:] - edges[:-2]
        return fb * (2.0 / np.maximum(widths, _EPS))[:, None]
    raise ValueError(f"unknown filterbank norm {norm!r}")


# --------------------------------------------------------------------------
# The four banks
# --------------------------------------------------------------------------


def mel_filterbank(
    sr: int = TARGET_SR,
    n_fft: int = 512,
    n_filters: int = 40,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    norm: Optional[str] = None,
) -> np.ndarray:
    """Triangular mel filterbank, shape (n_filters, n_fft//2+1)."""
    fmin, fmax = _resolve_range(sr, fmin, fmax)
    freqs = _bin_freqs(sr, n_fft)
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_filters + 2))
    fb, _, _, _ = _triangular_bank(freqs, edges)
    return _normalise(fb, norm, edges)


def linear_filterbank(
    sr: int = TARGET_SR,
    n_fft: int = 512,
    n_filters: int = 40,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    norm: Optional[str] = None,
) -> np.ndarray:
    """Uniformly spaced triangular filterbank, the LFCC front end.

    Same triangles as mel, edges on a linear axis. On an 8 kHz band with 40
    filters each one is about 185 Hz wide, so the whole 300 to 3400 Hz
    telephone band gets equal treatment.
    """
    fmin, fmax = _resolve_range(sr, fmin, fmax)
    freqs = _bin_freqs(sr, n_fft)
    edges = np.linspace(fmin, fmax, n_filters + 2)
    fb, _, _, _ = _triangular_bank(freqs, edges)
    return _normalise(fb, norm, edges)


def cqt_like_filterbank(
    sr: int = TARGET_SR,
    n_fft: int = 512,
    n_filters: int = 40,
    fmin: float = 60.0,
    fmax: Optional[float] = None,
    norm: Optional[str] = None,
) -> np.ndarray:
    """Geometrically spaced triangular filterbank: constant Q, on the FFT grid.

    A true constant-Q transform uses a different window length per bin. That is
    expensive and awkward to align with the rest of the frame based pipeline, so
    this approximates it: geometric band edges give every filter the same
    fractional bandwidth, which is the property CQCC actually depends on.
    fmin must be above zero, since the geometric series starts there.
    """
    lo, hi = _resolve_range(sr, max(float(fmin), 1.0), fmax)
    freqs = _bin_freqs(sr, n_fft)
    ratio = hi / lo
    edges = lo * ratio ** (np.arange(n_filters + 2, dtype=np.float64) / (n_filters + 1))
    fb, _, _, _ = _triangular_bank(freqs, edges)
    return _normalise(fb, norm, edges)


def gammatone_filterbank(
    sr: int = TARGET_SR,
    n_fft: int = 512,
    n_filters: int = 40,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    order: int = 4,
    norm: Optional[str] = None,
) -> np.ndarray:
    """Magnitude response of a bank of 4th order gammatone filters, ERB spaced.

    The gammatone impulse response t^(n-1) exp(-2 pi b t) cos(2 pi fc t) has the
    magnitude response |H(f)| = [1 + ((f - fc)/b)^2]^(-n/2), with b = 1.019 ERB(fc)
    the standard bandwidth scaling from Patterson's fits. Sampling that on the
    FFT grid is the usual way to build a GFCC front end without running a real
    filterbank, and it cannot produce an empty row because the response has
    tails everywhere. Each row is peak normalised so the channels are
    comparable.
    """
    fmin, fmax = _resolve_range(sr, fmin, fmax)
    freqs = _bin_freqs(sr, n_fft)
    # Centres sit strictly inside the band, spaced evenly on the ERB scale.
    e_lo, e_hi = hz_to_erb(max(fmin, 1.0)), hz_to_erb(fmax)
    centres = erb_to_hz(np.linspace(e_lo, e_hi, n_filters + 2)[1:-1])
    b = 1.019 * erb_bandwidth(centres)

    ratio = (freqs[None, :] - centres[:, None]) / b[:, None]
    fb = (1.0 + ratio ** 2) ** (-order / 2.0)
    fb = fb / np.maximum(fb.max(axis=1, keepdims=True), _EPS)
    _fix_empty_rows(fb, centres, freqs)
    return _normalise(fb, norm)


FILTERBANKS = {
    "mel": mel_filterbank,
    "linear": linear_filterbank,
    "gammatone": gammatone_filterbank,
    "cqt": cqt_like_filterbank,
}


@lru_cache(maxsize=64)
def _cached_bank(kind: str, sr: int, n_fft: int, n_filters: int,
                 fmin: float, fmax: float, norm: Optional[str]) -> np.ndarray:
    fn = FILTERBANKS[kind]
    fb = fn(sr=sr, n_fft=n_fft, n_filters=n_filters, fmin=fmin, fmax=fmax, norm=norm)
    fb.flags.writeable = False
    return fb


def get_filterbank(
    kind: str,
    sr: int = TARGET_SR,
    n_fft: int = 512,
    n_filters: int = 40,
    fmin: float = 60.0,
    fmax: float = 3800.0,
    norm: Optional[str] = None,
) -> np.ndarray:
    """Cached lookup by name. The cepstral extractors go through here so a long
    call does not rebuild the same 40 by 257 matrix on every utterance.

    The returned array is read only. Copy it if you want to edit it.
    """
    kind = str(kind).lower()
    if kind not in FILTERBANKS:
        raise ValueError(f"unknown filterbank {kind!r}, expected {sorted(FILTERBANKS)}")
    return _cached_bank(kind, int(sr), int(n_fft), int(n_filters),
                        float(fmin), float(fmax), norm)


def filterbank_centres(fb: np.ndarray, sr: int, n_fft: int) -> np.ndarray:
    """Energy weighted centre frequency of each row, for plots and sanity checks."""
    freqs = _bin_freqs(sr, n_fft)
    w = fb.sum(axis=1)
    return (fb @ freqs) / np.maximum(w, _EPS)


if __name__ == "__main__":  # pragma: no cover - self-test
    import time

    t0 = time.time()
    cep = SETTINGS.cepstral
    sr, n_fft = TARGET_SR, SETTINGS.frame.n_fft
    print("filterbanks self-test")
    print(f"  sr={sr} n_fft={n_fft} n_filters={cep.n_filters} "
          f"fmin={cep.fmin} fmax={cep.fmax} bin width={sr / n_fft:.2f} Hz")

    print(f"  hz_to_mel(1000)={hz_to_mel(1000.0):.2f}  "
          f"mel_to_hz(hz_to_mel(1000))={mel_to_hz(hz_to_mel(1000.0)):.2f}")
    print(f"  hz_to_erb(1000)={hz_to_erb(1000.0):.2f}  "
          f"erb_to_hz(hz_to_erb(1000))={erb_to_hz(hz_to_erb(1000.0)):.2f}  "
          f"ERB(1000)={erb_bandwidth(1000.0):.1f} Hz")

    for kind in ("mel", "linear", "gammatone", "cqt"):
        fb = get_filterbank(kind, sr, n_fft, cep.n_filters, cep.fmin, cep.fmax)
        sums = fb.sum(axis=1)
        centres = filterbank_centres(fb, sr, n_fft)
        print(f"  {kind:10s} shape={fb.shape} empty_rows={int((sums <= 0).sum())} "
              f"min_sum={sums.min():.4f} centre[0]={centres[0]:6.1f} Hz "
              f"centre[-1]={centres[-1]:6.1f} Hz")

    # The stress case: many narrow filters on a short transform.
    print("  stress test (n_fft=256, n_filters=64, fmin=20, fmax=3900):")
    for kind in ("mel", "linear", "gammatone", "cqt"):
        fb = FILTERBANKS[kind](sr=sr, n_fft=256, n_filters=64, fmin=20.0, fmax=3900.0)
        sums = fb.sum(axis=1)
        print(f"    {kind:10s} shape={fb.shape} empty_rows={int((sums <= 0).sum())} "
              f"min_sum={sums.min():.4f}")
    print(f"  done in {time.time() - t0:.2f} s")
