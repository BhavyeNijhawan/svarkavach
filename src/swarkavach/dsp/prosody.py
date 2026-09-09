"""Pitch, voice quality and rhythm.

This module feeds two things. The anti-spoof branch takes jitter, shimmer and
HNR directly, because a vocoder driven by a smooth predicted F0 contour
produces cycle to cycle variation that is far too regular: a real larynx is
never that steady. The fusion layer takes the whole `prosody_summary` dict as
the ACOUSTIC AROUSAL side of the prosody intent mismatch feature, where the
interesting case is words that are shouting (deadline, arrest, block your
account) delivered with a flat contour and no rate change.

Because PIM depends on it, `prosody_summary` has one hard rule: it returns all
thirteen keys as finite floats, always. Silence, white noise, a single sample,
a constant, a clipped signal: all of them produce a dict, never a NaN and never
an exception. Anything that cannot be estimated is 0.0.

Pitch tracking is YIN style (de Cheveigne and Kawahara 2002):

- the cumulative mean normalised difference function, which is what removes
  YIN's bias toward long lags and therefore the downward octave errors,
- the "first local minimum below an absolute threshold" rule rather than the
  global minimum, which is the guard against picking a multiple of the period,
- parabolic interpolation around that minimum, so the F0 resolution is not
  limited to sr/tau (at 8 kHz and 200 Hz that would be a 5 Hz quantisation,
  enough to fake a jitter measurement all by itself),
- a normalised cross correlation at the chosen lag for the voicing decision.

Search range 60 to 400 Hz. That covers adult male floor to raised female
speech, which is what an Indian telephone fraud call contains, and it excludes
the child range where 8 kHz telephone audio has too few harmonics left to be
reliable anyway.

Run the self-test with:
    python -m swarkavach.dsp.prosody
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import fft as sfft
from scipy import signal as ssig

from ..config import SETTINGS, TARGET_SR, ProsodyConfig
from .framing import frame_count, frame_signal, rms_db
from .vad import frame_grid, vad_mask

__all__ = [
    "f0_track",
    "jitter",
    "shimmer",
    "hnr",
    "speech_rate",
    "prosody_summary",
    "PROSODY_KEYS",
]

#: The exact key set `prosody_summary` promises. The fusion layer indexes this.
PROSODY_KEYS: Tuple[str, ...] = (
    "f0_mean", "f0_std", "f0_range", "f0_slope", "voiced_ratio",
    "jitter", "shimmer", "hnr",
    "energy_std", "energy_range",
    "rate", "rate_std", "pause_ratio",
)

_EPS = 1e-12

#: YIN's absolute threshold on the cumulative mean normalised difference. The
#: paper uses 0.1; 0.15 is a little more permissive, which telephone audio
#: needs because the codec strips the higher harmonics that make the dip deep.
_YIN_THRESHOLD = 0.15

#: Upper bound on aperiodicity at the accepted lag. A second gate next to the
#: correlation threshold, to stop band-limited noise from occasionally looking
#: periodic at some long lag.
_MAX_APERIODICITY = 0.60

#: Below this RMS a frame is silence at 16 bit resolution (1 LSB is about 3e-5),
#: so no pitch is claimed there whatever the correlation says.
_SILENCE_RMS = 1e-5

#: Frames more than this far below the loudest frame are not pitch tracked.
_LEVEL_FLOOR_DB = 50.0


def _cfg(cfg: Optional[ProsodyConfig]) -> ProsodyConfig:
    return SETTINGS.prosody if cfg is None else cfg


def _finite(value, default: float = 0.0) -> float:
    """Every number leaving this module goes through here."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float(default)
    return f if math.isfinite(f) else float(default)


def _lowpass(x: np.ndarray, sr: int, cutoff: float = 1000.0) -> np.ndarray:
    """Gentle low-pass before pitch analysis.

    Formant energy and fricative noise above 1 kHz add structure to the
    autocorrelation that has nothing to do with the period. Praat and RAPT both
    low-pass first for the same reason. Signals too short to filter are passed
    through untouched.
    """
    nyq = sr / 2.0
    wn = min(cutoff, 0.45 * sr) / nyq
    if x.size < 64 or not (0.0 < wn < 1.0):
        return x
    try:
        sos = ssig.butter(4, wn, btype="low", output="sos")
        return ssig.sosfiltfilt(sos, x)
    except ValueError:  # signal shorter than the filter's padding needs
        return x


def _pitch_frames(x: np.ndarray, sr: int, cfg: ProsodyConfig):
    """Frame the signal for pitch analysis and return the lag geometry.

    YIN needs the integration window to hold at least two periods of the lowest
    F0 searched. At 60 Hz and 8 kHz that is 267 samples, so the analysis window
    here is about 50 ms even though the config asks for 40. The frame count and
    the hop still follow the config, so these frames line up with everything
    else in the package.
    """
    hop = max(int(round(sr * cfg.hop_ms / 1000.0)), 1)
    base_len = max(int(round(sr * cfg.frame_ms / 1000.0)), 2)

    tau_min = max(int(math.floor(sr / max(cfg.f0_max, 1.0))), 2)
    tau_max = max(int(math.ceil(sr / max(cfg.f0_min, 1.0))), tau_min + 2)

    win = max(base_len, 3 * tau_max + 1)
    n_frames = frame_count(x.size, base_len, hop)
    need = (n_frames - 1) * hop + win
    if x.size < need:
        x = np.pad(x, (0, need - x.size))

    view = np.lib.stride_tricks.sliding_window_view(x, win)[::hop]
    frames = np.array(view[:n_frames], dtype=np.float64, copy=True)
    integ = win - tau_max - 1  # samples summed in the difference function
    return frames, hop, win, integ, tau_min, tau_max


def _correlations(frames: np.ndarray, integ: int, tau_max: int):
    """Difference function, its cumulative mean normalisation, and the NCCF.

    All three come out of one FFT pass per frame, for lags 0..tau_max+1.
    """
    n_frames, win = frames.shape
    head = frames[:, :integ]

    nfft = int(2 ** math.ceil(math.log2(max(win + integ, 4))))
    fh = sfft.rfft(head, n=nfft, axis=1)
    ff = sfft.rfft(frames, n=nfft, axis=1)
    cc = sfft.irfft(np.conj(fh) * ff, n=nfft, axis=1)[:, : tau_max + 2]

    sq = np.pad(frames * frames, ((0, 0), (1, 0)))
    cs = np.cumsum(sq, axis=1)
    lags = np.arange(tau_max + 2)
    e_tau = cs[:, lags + integ] - cs[:, lags]      # energy of the shifted window
    e0 = e_tau[:, :1]                              # energy of the fixed window

    d = np.maximum(e0 + e_tau - 2.0 * cc, 0.0)

    # Cumulative mean normalisation: d'(tau) = d(tau) / mean(d(1..tau)).
    run = np.cumsum(d[:, 1:], axis=1)
    denom = run / np.arange(1, d.shape[1], dtype=np.float64)[None, :]
    dp = np.ones_like(d)
    np.divide(d[:, 1:], denom, out=dp[:, 1:], where=denom > _EPS)
    dp[:, 1:][denom <= _EPS] = 1.0

    nccf = np.zeros_like(d)
    scale = np.sqrt(np.maximum(e0 * e_tau, 0.0))
    np.divide(cc, scale, out=nccf, where=scale > _EPS)
    return d, np.clip(dp, 0.0, 4.0), np.clip(nccf, -1.0, 1.0), e0.ravel()


def _pitch_analysis(x: np.ndarray, sr: int, cfg: ProsodyConfig) -> Dict[str, np.ndarray]:
    """Everything the pitch stage produces, in one pass."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 2:
        z = np.zeros(1)
        return {"f0": z, "voiced": np.zeros(1, dtype=bool), "r": z,
                "tau": z, "hop": 1, "rms": z}

    xf = _lowpass(x, sr)
    frames, hop, win, integ, tau_min, tau_max = _pitch_frames(xf, sr, cfg)
    _, dp, nccf, e0 = _correlations(frames, integ, tau_max)

    # Local minima of the normalised difference function.
    local_min = np.zeros_like(dp, dtype=bool)
    local_min[:, 1:-1] = (dp[:, 1:-1] < dp[:, :-2]) & (dp[:, 1:-1] <= dp[:, 2:])

    band = slice(tau_min, tau_max + 1)
    cand = local_min[:, band] & (dp[:, band] < _YIN_THRESHOLD)
    has_cand = cand.any(axis=1)
    # argmax on a boolean row gives the FIRST True, which is exactly YIN's
    # "first local minimum below the threshold" rule and therefore the octave
    # guard: the global minimum is often at twice the true period.
    idx = np.where(has_cand, np.argmax(cand, axis=1), np.argmin(dp[:, band], axis=1))
    tau = idx + tau_min

    rows = np.arange(frames.shape[0])
    y0 = dp[rows, tau - 1]
    y1 = dp[rows, tau]
    y2 = dp[rows, tau + 1]
    denom = y0 - 2.0 * y1 + y2
    shift = np.zeros_like(y1)
    np.divide(0.5 * (y0 - y2), denom, out=shift, where=np.abs(denom) > _EPS)
    tau_ref = tau + np.clip(shift, -1.0, 1.0)

    f0 = sr / np.maximum(tau_ref, 1.0)
    f0 = np.clip(f0, cfg.f0_min, cfg.f0_max)

    rms = np.sqrt(np.maximum(e0 / max(integ, 1), 0.0))
    level = 20.0 * np.log10(np.maximum(rms, 1e-12))
    loud_enough = (rms > _SILENCE_RMS) & (level > level.max() - _LEVEL_FLOOR_DB)

    r_best = nccf[rows, tau]
    voiced = (r_best > cfg.voicing_threshold) & (y1 < _MAX_APERIODICITY) & loud_enough

    # Three point median on the F0 contour, applied only where the frame and
    # both neighbours are voiced. Cleans up the isolated halving or doubling
    # that survives the threshold rule.
    if f0.size >= 3:
        trio = np.stack([f0[:-2], f0[1:-1], f0[2:]])
        keep = voiced[:-2] & voiced[1:-1] & voiced[2:]
        f0[1:-1] = np.where(keep, np.median(trio, axis=0), f0[1:-1])

    f0 = np.where(voiced, f0, 0.0)
    return {"f0": f0, "voiced": voiced, "r": r_best, "tau": tau.astype(np.float64),
            "hop": hop, "rms": rms}


def f0_track(
    x: np.ndarray,
    sr: int = TARGET_SR,
    cfg: Optional[ProsodyConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per frame F0 in Hz and a voiced flag. Unvoiced frames report 0.0 Hz."""
    cfg = _cfg(cfg)
    out = _pitch_analysis(x, sr, cfg)
    return out["f0"], out["voiced"]


def _voiced_runs(voiced: np.ndarray, min_len: int = 2):
    """Contiguous voiced stretches as (start, stop) index pairs."""
    v = np.asarray(voiced, dtype=bool)
    if v.size == 0:
        return []
    edges = np.flatnonzero(np.diff(v.astype(np.int8)))
    starts = np.concatenate([[0], edges + 1])
    stops = np.concatenate([edges + 1, [v.size]])
    return [(int(a), int(b)) for a, b in zip(starts, stops)
            if v[a] and b - a >= min_len]


def jitter(f0: np.ndarray, voiced: np.ndarray) -> float:
    """Local jitter: mean absolute period difference over mean period.

    Returned as a fraction, so 0.012 means 1.2 percent. Healthy modal voice
    sits near 0.005 to 0.02; a vocoder that resynthesises from a smoothed F0
    contour comes out an order of magnitude below that, which is the whole
    point of measuring it here.

    Only consecutive voiced frames are compared. Differencing across an
    unvoiced gap would measure the pause, not the larynx.
    """
    f0 = np.asarray(f0, dtype=np.float64).ravel()
    voiced = np.asarray(voiced, dtype=bool).ravel()
    if f0.size != voiced.size or f0.size < 3:
        return 0.0

    diffs, periods = [], []
    for a, b in _voiced_runs(voiced, min_len=3):
        f = f0[a:b]
        f = f[f > 0]
        if f.size < 3:
            continue
        t = 1.0 / f
        diffs.append(np.abs(np.diff(t)))
        periods.append(t)
    if not diffs:
        return 0.0
    all_diff = np.concatenate(diffs)
    all_per = np.concatenate(periods)
    mean_per = float(np.mean(all_per))
    if mean_per <= _EPS or all_diff.size == 0:
        return 0.0
    return _finite(np.clip(float(np.mean(all_diff)) / mean_per, 0.0, 2.0))


def _cycle_amplitudes(x, sr, f0, voiced, hop):
    """Peak amplitude of each glottal cycle, grouped by voiced run."""
    n = x.size
    groups = []
    for a, b in _voiced_runs(voiced, min_len=2):
        amps = []
        pos = a * hop
        stop = min(b * hop, n)
        while pos < stop:
            fi = min(pos // hop, f0.size - 1)
            f = f0[fi]
            if f <= 0:
                break
            period = int(round(sr / f))
            if period < 2 or pos + period > n:
                break
            amps.append(float(np.max(np.abs(x[pos: pos + period]))))
            pos += period
        if len(amps) >= 3:
            groups.append(np.asarray(amps, dtype=np.float64))
    return groups


def shimmer(
    x: np.ndarray,
    sr: int = TARGET_SR,
    f0: Optional[np.ndarray] = None,
    voiced: Optional[np.ndarray] = None,
    cfg: Optional[ProsodyConfig] = None,
) -> float:
    """Local shimmer: mean absolute amplitude difference between neighbouring
    glottal cycles, divided by the mean amplitude. Fraction, not percent.

    Genuinely cycle by cycle, not frame by frame: the cycle boundaries are
    walked using the local period from the F0 track. f0 and voiced must come
    from `f0_track` with the same config, so the hop matches.
    """
    cfg = _cfg(cfg)
    x = np.asarray(x, dtype=np.float64).ravel()
    if f0 is None or voiced is None:
        f0, voiced = f0_track(x, sr, cfg)
    f0 = np.asarray(f0, dtype=np.float64).ravel()
    voiced = np.asarray(voiced, dtype=bool).ravel()
    if x.size < 4 or f0.size == 0 or not voiced.any():
        return 0.0

    hop = max(int(round(sr * cfg.hop_ms / 1000.0)), 1)
    groups = _cycle_amplitudes(x, sr, f0, voiced, hop)
    if not groups:
        return 0.0
    diffs = np.concatenate([np.abs(np.diff(g)) for g in groups])
    amps = np.concatenate(groups)
    mean_amp = float(np.mean(amps))
    if mean_amp <= _EPS or diffs.size == 0:
        return 0.0
    return _finite(np.clip(float(np.mean(diffs)) / mean_amp, 0.0, 2.0))


def hnr(
    x: np.ndarray,
    sr: int = TARGET_SR,
    f0: Optional[np.ndarray] = None,
    voiced: Optional[np.ndarray] = None,
    cfg: Optional[ProsodyConfig] = None,
) -> float:
    """Harmonic to noise ratio in dB, averaged over voiced frames.

    Boersma's estimator: with r the normalised autocorrelation at the pitch
    period, the periodic part accounts for a fraction r of the power and the
    noise for 1 - r, so HNR = 10 log10(r / (1 - r)). Using the cross correlation
    of two shifted windows rather than the autocorrelation of one windowed
    frame avoids having to divide out the window's own autocorrelation.

    Clipped to [-10, 40] dB. Above 40 the estimate is dominated by numerical
    noise, and a perfectly periodic synthetic signal would otherwise return
    infinity.
    """
    cfg = _cfg(cfg)
    x = np.asarray(x, dtype=np.float64).ravel()
    if f0 is None or voiced is None:
        f0, voiced = f0_track(x, sr, cfg)
    f0 = np.asarray(f0, dtype=np.float64).ravel()
    voiced = np.asarray(voiced, dtype=bool).ravel()
    if x.size < 4 or not voiced.any():
        return 0.0

    frames, hop, win, integ, tau_min, tau_max = _pitch_frames(_lowpass(x, sr), sr, cfg)
    _, _, nccf, _ = _correlations(frames, integ, tau_max)

    n = min(frames.shape[0], f0.size, voiced.size)
    if n == 0:
        return 0.0
    idx = np.flatnonzero(voiced[:n] & (f0[:n] > 0))
    if idx.size == 0:
        return 0.0
    tau = np.clip(np.round(sr / f0[idx]).astype(int), tau_min, tau_max)
    r = np.clip(nccf[idx, tau], 1e-6, 1.0 - 1e-6)
    per_frame = 10.0 * np.log10(r / (1.0 - r))
    return _finite(np.clip(float(np.mean(per_frame)), -10.0, 40.0))


def _energy_envelope(x: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    """Per frame level in dB on the VAD frame grid, plus the hop in samples."""
    frame_len, hop = frame_grid(sr)
    frames = frame_signal(x, frame_len, hop, "hamming")
    return rms_db(frames), hop


def speech_rate(x: np.ndarray, sr: int = TARGET_SR) -> float:
    """Syllable-like rate in events per second of speech.

    Counts peaks in the smoothed energy envelope, which tracks the syllable
    nucleus well enough for a relative measure. Divided by speech duration, not
    total duration, so it is an articulation rate: pausing is reported
    separately as pause_ratio, and mixing the two would hide exactly the
    contrast the PIM feature looks for.

    A peak needs 3 dB of prominence and 80 ms of separation, which caps the
    reading near 12 syllables per second, above any real speaker.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < sr // 10:
        return 0.0
    level, hop = _energy_envelope(x, sr)
    if level.size < 5:
        return 0.0

    kernel = np.ones(5) / 5.0
    smooth = np.convolve(level, kernel, mode="same")
    mask = vad_mask(x, sr)
    n = min(mask.size, smooth.size)
    speech_s = int(mask[:n].sum()) * hop / float(sr)

    min_dist = max(int(round(0.080 * sr / hop)), 1)
    peaks, _ = ssig.find_peaks(smooth[:n], prominence=3.0, distance=min_dist)
    if speech_s >= 0.2:
        if peaks.size:
            peaks = peaks[mask[:n][peaks]]
    else:
        # No VAD speech to divide by (a sustained vowel has no envelope for the
        # VAD to latch onto). Use the whole signal rather than returning 0.0 and
        # pretending the question was unanswerable: with no energy peaks the
        # answer comes out near zero anyway.
        speech_s = x.size / float(sr)
        if speech_s < 0.2:
            return 0.0
    return _finite(np.clip(peaks.size / speech_s, 0.0, 12.0))


def prosody_summary(
    x: np.ndarray,
    sr: int = TARGET_SR,
    cfg: Optional[ProsodyConfig] = None,
) -> Dict[str, float]:
    """The thirteen prosody numbers, every one guaranteed finite.

    Keys: f0_mean f0_std f0_range f0_slope voiced_ratio jitter shimmer hnr
    energy_std energy_range rate rate_std pause_ratio.

    Units: F0 values in Hz, f0_slope in Hz per second, jitter and shimmer as
    fractions, hnr in dB, energy_std and energy_range in dB, rate in events per
    second of speech, the two ratios in [0, 1].
    """
    cfg = _cfg(cfg)
    out = {k: 0.0 for k in PROSODY_KEYS}
    x = np.asarray(x, dtype=np.float64).ravel()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.size < 8 or not np.any(np.abs(x) > 0):
        out["pause_ratio"] = 1.0
        return out

    try:
        pitch = _pitch_analysis(x, sr, cfg)
    except Exception:  # a broken signal must not take the pipeline down
        pitch = {"f0": np.zeros(1), "voiced": np.zeros(1, dtype=bool)}
    f0, voiced = pitch["f0"], pitch["voiced"]

    out["voiced_ratio"] = _finite(np.mean(voiced) if voiced.size else 0.0)

    fv = f0[voiced] if voiced.any() else np.zeros(0)
    if fv.size >= 2:
        out["f0_mean"] = _finite(np.mean(fv))
        out["f0_std"] = _finite(np.std(fv))
        out["f0_range"] = _finite(np.percentile(fv, 95) - np.percentile(fv, 5))
    elif fv.size == 1:
        out["f0_mean"] = _finite(fv[0])

    if fv.size >= 3:
        # Declination slope over the utterance, in Hz per second.
        hop_s = cfg.hop_ms / 1000.0
        t = np.flatnonzero(voiced).astype(np.float64) * hop_s
        t = t - t.mean()
        var = float(np.dot(t, t))
        if var > _EPS:
            slope = float(np.dot(t, fv - fv.mean()) / var)
            out["f0_slope"] = _finite(np.clip(slope, -500.0, 500.0))

    out["jitter"] = _finite(jitter(f0, voiced))
    out["shimmer"] = _finite(shimmer(x, sr, f0, voiced, cfg))
    out["hnr"] = _finite(hnr(x, sr, f0, voiced, cfg))

    try:
        level, _ = _energy_envelope(x, sr)
        mask = vad_mask(x, sr)
        n = min(level.size, mask.size)
        sel = level[:n][mask[:n]] if mask[:n].any() else level[:n]
        # Keep only frames within 60 dB of the loudest one. Padding a segment
        # can pull in a few frames of digital silence at -120 dB, and those
        # would otherwise set energy_range on their own.
        if sel.size:
            sel = sel[sel > sel.max() - 60.0]
        if sel.size >= 2:
            out["energy_std"] = _finite(np.std(sel))
            out["energy_range"] = _finite(np.percentile(sel, 95) - np.percentile(sel, 5))

        # A frame that carries pitch is not a pause, whatever the energy VAD
        # thinks. The two masks sit on slightly different frame grids (25 ms
        # against 40 ms with the same hop), so compare over the shorter one.
        m = min(mask.size, voiced.size)
        if m:
            active = mask[:m] | voiced[:m]
            out["pause_ratio"] = _finite(1.0 - float(active.mean()), 1.0)
        else:
            out["pause_ratio"] = 1.0
    except Exception:
        out["pause_ratio"] = 1.0

    out["rate"] = _finite(speech_rate(x, sr))

    # Rate variability across one second windows. Speaking at a steady clip and
    # speaking in bursts are different arousal signatures.
    win = int(sr)
    if x.size >= 2 * win:
        rates = [speech_rate(x[i: i + win], sr) for i in range(0, x.size - win + 1, win)]
        rates = [r for r in rates if r > 0]
        if len(rates) >= 2:
            out["rate_std"] = _finite(np.std(rates))

    return {k: _finite(out[k]) for k in PROSODY_KEYS}


if __name__ == "__main__":  # pragma: no cover - self-test
    import time

    t0 = time.time()
    sr = TARGET_SR
    cfg = SETTINGS.prosody
    rng = np.random.default_rng(SETTINGS.pipeline.seed)

    def vowel(dur_s: float, f0_hz: float, jitter_pct: float = 0.01,
              shimmer_pct: float = 0.05, breath: float = 0.02) -> np.ndarray:
        """Source-filter vowel: jittered pulse train through three formants.

        The formants are two pole resonators driven with lfilter, so this stays
        linear in the duration. Jitter and shimmer are injected deliberately, so
        the measured values below can be compared against what went in.
        """
        n = int(dur_s * sr)
        out = np.zeros(n)
        pos, r = 0.0, np.random.default_rng(3)
        while pos < n:
            out[int(pos)] = 1.0 + shimmer_pct * r.standard_normal()
            period = sr / (f0_hz * (1.0 + jitter_pct * r.standard_normal()))
            pos += max(period, 2.0)
        y = np.zeros(n)
        for fc, bw, g in ((700.0, 90.0, 1.0), (1220.0, 110.0, 0.5), (2600.0, 170.0, 0.25)):
            rp = np.exp(-np.pi * bw / sr)
            w = 2 * np.pi * fc / sr
            y += g * ssig.lfilter([1.0], [1.0, -2 * rp * np.cos(w), rp * rp], out)
        y = 0.5 * y / max(np.max(np.abs(y)), 1e-9)
        return y + breath * r.standard_normal(n) * np.abs(y)

    cases = {
        "silence": np.zeros(3 * sr),
        "white noise": 0.3 * rng.standard_normal(3 * sr),
        "pure sine 200 Hz": 0.5 * np.sin(2 * np.pi * 200.0 * np.arange(3 * sr) / sr),
        "synthetic vowel": vowel(3.0, 150.0),
        "vowel + pauses": np.concatenate([
            np.zeros(sr // 2), vowel(0.9, 165.0), np.zeros(sr // 2),
            vowel(1.1, 140.0), np.zeros(sr // 2)]),
    }

    print("prosody self-test")
    for name, sig in cases.items():
        f0, voiced = f0_track(sig, sr, cfg)
        s = prosody_summary(sig, sr, cfg)
        ok = all(math.isfinite(v) for v in s.values())
        med = float(np.median(f0[voiced])) if voiced.any() else 0.0
        print(f"  {name:18s} voiced={s['voiced_ratio']:.2f} "
              f"f0_med={med:6.1f} f0_mean={s['f0_mean']:6.1f} "
              f"jit={s['jitter']:.4f} shim={s['shimmer']:.4f} "
              f"hnr={s['hnr']:6.2f} rate={s['rate']:4.1f} "
              f"pause={s['pause_ratio']:.2f} finite={ok}")

    full = prosody_summary(cases["vowel + pauses"], sr)
    print(f"  keys returned: {len(full)} of {len(PROSODY_KEYS)} expected")
    print("  full dict for 'vowel + pauses':")
    for k in PROSODY_KEYS:
        print(f"    {k:14s} {full[k]:9.4f}")
    t1 = time.time()
    prosody_summary(np.concatenate([cases['synthetic vowel']] * 10), sr)
    print(f"  30 s call: prosody_summary in {time.time() - t1:.2f} s")
    print(f"  done in {time.time() - t0:.2f} s")
