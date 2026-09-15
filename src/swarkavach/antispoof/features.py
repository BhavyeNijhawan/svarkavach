"""Front end for the anti-spoofing branch: frame features, pooled vectors and
the small dictionary of interpretable voice-quality numbers.

Three levels of description, each feeding a different model:

1. `frame_features` gives one vector per 25 ms frame (cepstra plus deltas plus
   CMVN). That is what the GMM back end consumes, because a GMM is a density
   over frames and needs many samples per utterance.
2. `utterance_features` pools those frames into one fixed length vector so a
   discriminative classifier (the GBM) can be trained on whole calls.
3. `antispoof_feature_dict` returns the handful of numbers a human can read and
   argue with, and the four of them that go into the 25 feature fusion vector.

Why LFCC is the default rather than MFCC: mel spacing spends most of its
resolution below 1 kHz because that is where hearing is sharpest, but vocoder
and waveform-model artefacts (over-smoothed harmonics, a wrong noise floor,
missing high band detail) sit at the top of the band. Linear spacing keeps the
resolution up there, which is why the ASVspoof 2019 and 2021 organisers shipped
LFCC-GMM as the baseline and not MFCC-GMM.

Why skewness and kurtosis are pooled alongside mean and variance: a synthetic
voice produces cepstral coefficients whose distribution across a call is more
Gaussian than a real speaker's. Real speech has heavy tails, from creak, glottal
irregularity, breath and level jumps; a vocoder driven by a smooth predicted
contour does not. Mean and variance cannot see that difference, the third and
fourth moments can.

Run the self-test with:
    python -m swarkavach.antispoof.features
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..audioio import rms_normalize
from ..config import SETTINGS, TARGET_SR, CepstralConfig, FrameConfig
from ..dsp.cepstral import FEATURE_EXTRACTORS
from ..dsp.deltas import add_deltas, cmvn
from ..dsp.prosody import prosody_summary
from ..dsp.spectral import (
    fft_frequencies,
    power_spectrum,
    spectral_centroid,
    spectral_flatness,
    spectral_flux,
)
from ..dsp.vad import vad_mask
from ..dsp.framing import frame_signal, zero_crossing_rate

__all__ = [
    "FEATURE_SETS",
    "POOLING",
    "frame_features",
    "utterance_features",
    "utterance_feature_dim",
    "pooled_feature_names",
    "antispoof_feature_dict",
    "ANTISPOOF_KEYS",
    "demo_bonafide",
    "demo_spoof",
]

#: Front ends this branch can run. All five come from dsp.cepstral.
FEATURE_SETS: Tuple[str, ...] = ("lfcc", "gfcc", "mfcc", "cqcc", "lpcc")

#: The pooling functions applied per coefficient, in output order. The first
#: four run on the frame values, the last four on the first differences.
POOLING: Tuple[str, ...] = (
    "mean", "std", "skew", "kurtosis",
    "d_mean", "d_std", "d_skew", "d_kurtosis",
)

#: Keys `antispoof_feature_dict` always returns, before the dx_ diagnostics.
ANTISPOOF_KEYS: Tuple[str, ...] = (
    "as_llr", "jitter", "shimmer", "hnr", "spec_flatness_var",
)

_EPS = 1e-12

#: A frame matrix with fewer speech frames than this is pooled over every
#: frame instead. Below roughly 100 ms of speech the VAD decision is less
#: reliable than just using the whole signal.
_MIN_SPEECH_FRAMES = 8


# --------------------------------------------------------------------------
# Config helpers
# --------------------------------------------------------------------------


def _configs(sr: int) -> Tuple[FrameConfig, CepstralConfig]:
    """Frame and cepstral configs adapted to this sample rate.

    SETTINGS is written for 8 kHz telephone audio, where the 3800 Hz filterbank
    ceiling is just under Nyquist. If a caller hands us wideband audio the
    ceiling scales with it, otherwise half the band would go unanalysed.
    """
    sr = int(sr)
    cfg = replace(SETTINGS.frame, sr=sr)
    cep = SETTINGS.cepstral
    fmax = min(cep.fmax * sr / float(TARGET_SR), 0.49 * sr)
    if abs(fmax - cep.fmax) > 1.0:
        cep = replace(cep, fmax=float(fmax))
    return cfg, cep


def _clean(x) -> np.ndarray:
    """Signal in, finite float64 out. Never empty, never NaN."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0:
        return np.zeros(1, dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def trim_silence(x: np.ndarray, sr: int, rel_db: float = 40.0,
                 pad_s: float = 0.1) -> np.ndarray:
    """Cut leading and trailing quiet.

    A frame is quiet if it sits more than `rel_db` below the loudest frame.
    `pad_s` of context is kept before the first loud frame, so onsets are
    not clipped; nothing is kept after the last one, because a trailing pad
    of any fixed length is itself a difference between a clip that had a
    silent tail and one that did not. Signals with nothing above the
    threshold come back untouched.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = int(0.02 * sr)
    hop = max(n // 2, 1)
    if x.size < 2 * n:
        return x
    n_fr = 1 + (x.size - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(n_fr)[:, None]
    e = 10.0 * np.log10(np.mean(x[idx] ** 2, axis=1) + 1e-12)
    loud = np.flatnonzero(e > e.max() - rel_db)
    if loud.size == 0:
        return x
    a = max(int(loud[0]) * hop - int(pad_s * sr), 0)
    b = min(int(loud[-1]) * hop + n, x.size)
    return x[a:b] if b - a >= n else x


#: The band every anti-spoofing cepstrum is computed from. Measured on the
#: real Hindi telephone recordings: their MP3 encoder's low-pass starts
#: just under 3000 Hz (-28 dB by 3200), not at the 3400 Hz of the textbook
#: telephone band, and below 300 Hz they carry 11 dB less than a neural
#: render does. The top edge sits at 2800 so the filter's own transition
#: band lies where both sides are still flat; at 3000 the last hundred
#: hertz of transition was reading the recordings' roll-off against the
#: renders' content and separated them at AUC 0.82 on its own.
#: SETTINGS.cepstral.fmin / fmax are set to the same edges.
BAND_LOW_HZ = 300.0
BAND_HIGH_HZ = 2800.0


def _brick_wall(x: np.ndarray, sr: int, low: float, high: float,
                width_hz: float = 100.0, stop_db: float = 60.0) -> np.ndarray:
    """A steep zero-phase FIR band-pass.

    A Butterworth of any sane order leaves a transition band a few hundred
    hertz wide, and on this data the spoofs carry 15 dB more than the real
    recordings right there. Kaiser-windowed FIR with a 100 Hz transition,
    run forwards and backwards, so the stop band is really stopped.
    """
    from scipy import signal as ssig

    x = np.asarray(x, dtype=np.float64).ravel()
    nyq = sr / 2.0
    if x.size < 64 or not (0.0 < low < high < nyq):
        return x
    beta = 0.1102 * (stop_db - 8.7) if stop_db > 50 else 0.5842 * (stop_db - 21) ** 0.4 + 0.07886 * (stop_db - 21)
    numtaps = int(np.ceil((stop_db - 8.0) / (2.285 * 2.0 * np.pi * width_hz / sr)))
    numtaps += 1 - numtaps % 2                     # odd, so the delay is integral
    numtaps = max(numtaps, 65)
    if x.size <= 3 * numtaps:
        return x
    taps = ssig.firwin(numtaps, [low, min(high, 0.98 * nyq)], pass_zero=False,
                       window=("kaiser", beta), fs=sr)
    return ssig.filtfilt(taps, [1.0], x)


def condition(x, sr: int = TARGET_SR) -> np.ndarray:
    """What every anti-spoofing cepstrum is computed from.

    Four things, in this order: remove the DC offset, band-pass to the
    band the real recordings actually occupy (BAND_LOW_HZ to BAND_HIGH_HZ,
    brick wall), trim leading and trailing silence, add dither.

    This exists because the real Hindi recordings are 8 kHz telephone audio
    with nothing above 3.4 kHz, and the spoofs were 24 kHz renders
    resampled through a short filter that leaves energy up to 4 kHz. The
    share of energy above 3.4 kHz alone separated the two classes at AUC
    1.000, the spoofs carried a second of trailing digital silence to the
    recordings' fifth, and the vocoded half had a DC offset. All three are
    facts about the pipeline, not the voice, and the level and colour
    matching in the pair build could not see any of them.

    Applied here, in the one function every extractor goes through, so
    training, evaluation and the live detector see the same signal. A phone
    call is band-limited anyway; this only makes the detector honest about
    it.
    """
    x = _clean(x)
    x = x - float(np.mean(x))
    x = _brick_wall(x, sr, BAND_LOW_HZ, BAND_HIGH_HZ)
    x = trim_silence(x, sr)
    # Dither at -70 dB relative to the signal. An 8 kHz MP3 decodes to
    # exactly zero above its encoder's low-pass, a resampled render does
    # not, and with a log floor of 1e-12 the ratio of nothing to almost
    # nothing was still a perfect classifier after the band-pass. Both
    # sides now share the same noise floor in the stop bands. Seeded from
    # the length so the same input always conditions the same way.
    rms = float(np.sqrt(np.mean(x ** 2))) or 1e-6
    rng = np.random.default_rng(x.size)
    return x + rng.standard_normal(x.size) * (rms * 10 ** (-70 / 20))


def _speech_index(x: np.ndarray, sr: int, n_frames: int) -> np.ndarray:
    """Indices of the frames the VAD calls speech, or every frame if too few.

    Several published ASVspoof results turned out to be driven partly by
    silence, because the bona fide and spoofed sets carried different amounts of
    it. Pooling over speech frames only removes that shortcut.
    """
    try:
        mask = vad_mask(x, sr)
    except Exception:
        return np.arange(n_frames)
    n = min(mask.size, n_frames)
    if n == 0:
        return np.arange(n_frames)
    idx = np.flatnonzero(mask[:n])
    if idx.size < _MIN_SPEECH_FRAMES:
        return np.arange(n_frames)
    return idx


# --------------------------------------------------------------------------
# Frame level
# --------------------------------------------------------------------------


def frame_features(
    x,
    sr: int = TARGET_SR,
    feature_set: str = "lfcc",
    *,
    deltas: Optional[bool] = None,
    apply_cmvn: Optional[bool] = None,
    voiced_only: bool = True,
) -> np.ndarray:
    """Per frame feature matrix, shape (n_frames, n_ceps * 3) by default.

    Dispatches to the dsp cepstral extractors, appends first and second
    derivatives and applies CMVN, all following SETTINGS.cepstral. Pass
    `deltas` or `apply_cmvn` explicitly to override the config, which is what
    `utterance_features` does and why the knobs exist.

    CMVN is on by default because a convolutional channel (a handset response,
    a codec's fixed shaping) is additive in the cepstral domain, so subtracting
    the utterance mean subtracts most of it. Without that, a GMM trained on one
    channel scores a different channel as spoofed.
    """
    key = str(feature_set).lower()
    if key not in FEATURE_EXTRACTORS:
        raise ValueError(
            f"unknown feature set {feature_set!r}, expected {sorted(FEATURE_SETS)}"
        )
    cep_cfg = SETTINGS.cepstral
    use_deltas = cep_cfg.deltas if deltas is None else bool(deltas)
    use_cmvn = cep_cfg.cmvn if apply_cmvn is None else bool(apply_cmvn)

    x = condition(x, sr)
    # Level normalisation after conditioning: a detector that keys on
    # recording gain would score the same voice differently on two handsets,
    # and the gain has to be measured on the band and span the detector reads.
    x = np.asarray(rms_normalize(x.astype(np.float32)), dtype=np.float64)

    cfg, cep = _configs(sr)
    feat = np.atleast_2d(FEATURE_EXTRACTORS[key](x, cfg, cep))
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    if voiced_only and feat.shape[0] > _MIN_SPEECH_FRAMES:
        feat = feat[_speech_index(x, sr, feat.shape[0])]
    if feat.shape[0] == 0:
        feat = np.zeros((1, cep.n_ceps), dtype=np.float64)

    if use_deltas:
        feat = add_deltas(feat, width=cep_cfg.delta_width, order=2)
    if use_cmvn:
        feat = cmvn(feat)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------
# Utterance level pooling
# --------------------------------------------------------------------------


def _moments(a: np.ndarray) -> np.ndarray:
    """Mean, std, skewness and excess kurtosis down axis 0, shape (4, d).

    Written out rather than taken from scipy.stats so the zero variance case
    (a constant column, which silence produces) returns 0.0 instead of NaN.
    """
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    if a.shape[0] == 0:
        return np.zeros((4, a.shape[1]), dtype=np.float64)
    mean = a.mean(axis=0)
    centred = a - mean
    m2 = np.mean(centred ** 2, axis=0)
    alive = m2 > 1e-10
    std = np.sqrt(np.maximum(m2, 0.0))
    m3 = np.mean(centred ** 3, axis=0)
    m4 = np.mean(centred ** 4, axis=0)
    skew = np.zeros_like(m2)
    kurt = np.zeros_like(m2)
    np.divide(m3, np.power(np.maximum(m2, _EPS), 1.5), out=skew, where=alive)
    np.divide(m4, np.maximum(m2, _EPS) ** 2, out=kurt, where=alive)
    kurt = np.where(alive, kurt - 3.0, 0.0)   # excess kurtosis: 0 for a Gaussian
    return np.nan_to_num(
        np.stack([mean, std, skew, kurt]), nan=0.0, posinf=0.0, neginf=0.0
    )


def utterance_features(x, sr: int = TARGET_SR, feature_set: str = "lfcc") -> np.ndarray:
    """One fixed length vector per call, shape (n_ceps * 3 * 8,) by default.

    Layout: for each coefficient, mean, std, skew and excess kurtosis of the
    frame values, then the same four statistics of the frame to frame first
    differences. The difference block is where over-smoothing shows up: a
    synthesis system produces frames that are individually plausible but move
    between each other too gently, so the differences are small and unnaturally
    symmetric.

    CMVN is deliberately switched off for this path. CMVN forces every column
    to zero mean and unit variance over the utterance, so pooling the mean and
    the std afterwards would return a vector of zeros and ones and throw away
    half the pooled statistics. Skewness and kurtosis are invariant to that
    affine map, so they are the same either way.
    """
    feat = frame_features(x, sr, feature_set, apply_cmvn=False)
    stats = _moments(feat)
    if feat.shape[0] >= 2:
        d_stats = _moments(np.diff(feat, axis=0))
    else:
        d_stats = np.zeros_like(stats)
    out = np.concatenate([stats.ravel(), d_stats.ravel()])
    # Cepstra are unbounded in principle. Clipping keeps one broken frame from
    # producing a feature that dominates every tree split in the GBM.
    out = np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1e4, 1e4)
    return out.astype(np.float64)


def utterance_feature_dim(feature_set: str = "lfcc") -> int:
    """Length of the vector `utterance_features` returns, without running it."""
    cep = SETTINGS.cepstral
    per_frame = cep.n_ceps * (3 if cep.deltas else 1)
    return per_frame * len(POOLING)


def pooled_feature_names(feature_set: str = "lfcc") -> List[str]:
    """Human readable name per pooled dimension, for the report and the GBM
    importance plot."""
    cep = SETTINGS.cepstral
    blocks = ["c", "d", "dd"] if cep.deltas else ["c"]
    names: List[str] = []
    for pool in POOLING:
        for block in blocks:
            for i in range(cep.n_ceps):
                names.append(f"{feature_set}_{block}{i}_{pool}")
    return names


# --------------------------------------------------------------------------
# Interpretable dictionary
# --------------------------------------------------------------------------


def _finite(value, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float(default)
    return f if math.isfinite(f) else float(default)


def antispoof_feature_dict(
    x,
    sr: int = TARGET_SR,
    gmm=None,
    feature_set: str = "lfcc",
) -> Dict[str, float]:
    """The interpretable acoustic numbers, every value a finite float.

    Always present: as_llr, jitter, shimmer, hnr, spec_flatness_var. Everything
    else is prefixed dx_ and is diagnostic: the heuristic back end reads some of
    it, the dashboard shows some of it, and none of it is part of the fusion
    contract.

    `as_llr` stays 0.0 unless a fitted GMMScorer is passed in, so this function
    works before anything has been trained.

    What each of the four fusion numbers is for:

    jitter, shimmer: cycle to cycle irregularity of period and amplitude. A real
    larynx is never perfectly steady. A vocoder resynthesising from a smoothed
    F0 contour is, so both come out low.

    hnr: harmonic to noise ratio. Synthesis with too little aspiration noise
    scores unnaturally high.

    spec_flatness_var: how much the per frame spectral flatness moves across the
    call. Real speech alternates voiced (peaky, low flatness) with fricatives
    and pauses (flat), and the noise floor wobbles. A vocoder regularises the
    noise floor, so the flatness sits in a narrow band and its variance drops.
    """
    x = _clean(x)
    out: Dict[str, float] = {k: 0.0 for k in ANTISPOOF_KEYS}

    # -- voice quality ----------------------------------------------------
    try:
        pros = prosody_summary(x, sr)
    except Exception:
        pros = {}
    out["jitter"] = _finite(pros.get("jitter", 0.0))
    out["shimmer"] = _finite(pros.get("shimmer", 0.0))
    out["hnr"] = _finite(pros.get("hnr", 0.0))
    out["dx_f0_mean"] = _finite(pros.get("f0_mean", 0.0))
    out["dx_f0_std"] = _finite(pros.get("f0_std", 0.0))
    out["dx_voiced_ratio"] = _finite(pros.get("voiced_ratio", 0.0))
    out["dx_pause_ratio"] = _finite(pros.get("pause_ratio", 0.0))
    out["dx_energy_std"] = _finite(pros.get("energy_std", 0.0))

    # -- spectral shape ---------------------------------------------------
    cfg, _ = _configs(sr)
    try:
        spec = power_spectrum(x, cfg)
        flat = spectral_flatness(spec, cfg)
        idx = _speech_index(x, sr, flat.size)
        sel = flat[idx] if idx.size >= 2 else flat
        out["spec_flatness_var"] = _finite(np.var(sel))
        out["dx_flatness_mean"] = _finite(np.mean(sel))
        # Flatness spans several orders of magnitude, so its plain variance is
        # a very small number. The dB version is the one the heuristic reads,
        # because a variance of 4 dB^2 and one of 0.4 dB^2 are easy to threshold.
        flat_db = 10.0 * np.log10(np.maximum(sel, 1e-10))
        out["dx_flatness_db_var"] = _finite(np.var(flat_db))
        out["dx_centroid_mean"] = _finite(np.mean(spectral_centroid(spec, cfg)))
        flux = spectral_flux(spec, cfg)
        out["dx_flux_mean"] = _finite(np.mean(flux))
        out["dx_flux_std"] = _finite(np.std(flux))
        # Share of the energy above 2.5 kHz. Mel-spectrogram vocoders leave the
        # top of the telephone band thin, which this catches directly.
        freqs = fft_frequencies(cfg)
        hi = spec[:, freqs >= 2500.0].sum()
        total = spec.sum()
        out["dx_hf_ratio"] = _finite(hi / total if total > _EPS else 0.0)
    except Exception:
        for k in ("dx_flatness_mean", "dx_flatness_db_var", "dx_centroid_mean",
                  "dx_flux_mean", "dx_flux_std", "dx_hf_ratio"):
            out.setdefault(k, 0.0)

    # -- cepstral dynamics and tail shape ---------------------------------
    try:
        feat = frame_features(x, sr, feature_set, apply_cmvn=False)
        stats = _moments(feat)
        out["dx_kurtosis_mean"] = _finite(np.mean(stats[3]))
        out["dx_skew_absmean"] = _finite(np.mean(np.abs(stats[2])))
        if feat.shape[0] >= 3:
            d = np.diff(feat, axis=0)
            # Frame to frame movement relative to the spread of the frames
            # themselves. Low means over-smoothed: the frames barely move
            # compared with how far apart they are.
            spread = np.maximum(feat.std(axis=0), 1e-6)
            out["dx_delta_ratio"] = _finite(np.mean(d.std(axis=0) / spread))
        else:
            out["dx_delta_ratio"] = 0.0
    except Exception:
        out.setdefault("dx_kurtosis_mean", 0.0)
        out.setdefault("dx_skew_absmean", 0.0)
        out.setdefault("dx_delta_ratio", 0.0)

    # -- zero crossing spread ---------------------------------------------
    try:
        frames = frame_signal(x, cfg.frame_len, cfg.hop_len, cfg.window)
        out["dx_zcr_std"] = _finite(np.std(zero_crossing_rate(frames)))
    except Exception:
        out["dx_zcr_std"] = 0.0

    # -- optional GMM log-likelihood ratio --------------------------------
    if gmm is not None:
        try:
            out["as_llr"] = _finite(gmm.llr(frame_features(x, sr, feature_set)))
        except Exception:
            out["as_llr"] = 0.0

    return {k: _finite(v) for k, v in out.items()}


# --------------------------------------------------------------------------
# Signals for the self-tests
# --------------------------------------------------------------------------
#
# Every module in this package needs a bona fide-like and a spoof-like signal to
# exercise itself, and the corpus generator is written by someone else and may
# not exist yet. These two functions are the fallback: a source-filter
# synthesiser where the only differences are the ones that anti-spoofing is
# supposed to detect. They are not a substitute for real ASVspoof data, they are
# there so nothing in this package is untestable offline.


def _formant_filter(exc: np.ndarray, sr: int, tracks: Sequence[np.ndarray],
                    bws: Sequence[float], gains: Sequence[float],
                    block: int = 160) -> np.ndarray:
    """Run the excitation through time-varying two-pole resonators.

    Coefficients are updated once per block and the filter state is carried
    across blocks, which is how a formant can glide without a click at every
    update.
    """
    from scipy import signal as ssig

    n = exc.size
    n_blocks = int(np.ceil(n / block))
    out = np.zeros(n, dtype=np.float64)
    states = [np.zeros(2) for _ in tracks]
    for bi in range(n_blocks):
        a0, a1 = bi * block, min((bi + 1) * block, n)
        chunk = exc[a0:a1]
        acc = np.zeros(a1 - a0, dtype=np.float64)
        for fi, track in enumerate(tracks):
            fc = float(track[min(bi, track.size - 1)])
            bw = float(bws[fi])
            r = np.exp(-np.pi * bw / sr)
            theta = 2.0 * np.pi * fc / sr
            b = np.array([1.0 - r])
            a = np.array([1.0, -2.0 * r * np.cos(theta), r * r])
            y, states[fi] = ssig.lfilter(b, a, chunk, zi=states[fi])
            acc += gains[fi] * y
        out[a0:a1] = acc
    return out


def _glottal_source(n: int, sr: int, f0_track: np.ndarray, jitter: float,
                    shimmer: float, rng: np.random.Generator,
                    block: int = 160) -> np.ndarray:
    """Impulse train with per period jitter and shimmer."""
    src = np.zeros(n, dtype=np.float64)
    pos = 0.0
    while pos < n - 1:
        i = int(pos)
        f0 = float(f0_track[min(i // block, f0_track.size - 1)])
        src[i] = 1.0 + shimmer * rng.standard_normal()
        period = sr / max(f0 * (1.0 + jitter * rng.standard_normal()), 20.0)
        pos += max(period, 2.0)
    return src


#: Micro-variation levels the demo synthesiser interpolates between: what a
#: larynx does, and what a vocoder driven by a smoothed contour does.
_MICRO_HUMAN = (0.018, 0.090, 0.030)      # jitter, shimmer, breath noise
_MICRO_MACHINE = (0.0004, 0.004, 0.002)


def _micro_levels(micro: float):
    """Geometric interpolation between the machine and human levels.

    Geometric rather than linear because all three quantities are ratios
    spanning more than an order of magnitude, so halfway should mean the
    geometric mean, not the arithmetic one.
    """
    m = float(np.clip(micro, 0.0, 1.0))
    return tuple(lo * (hi / lo) ** m for lo, hi in zip(_MICRO_MACHINE, _MICRO_HUMAN))


def _synth(sr: int, dur_s: float, seed: int, synthetic: bool,
           f0_base: float, micro: float) -> np.ndarray:
    """Shared body of demo_bonafide and demo_spoof."""
    rng = np.random.default_rng(seed)
    block = max(int(round(sr * 0.02)), 8)
    n = int(dur_s * sr)
    n_blocks = int(np.ceil(n / block)) + 1
    t_blk = np.arange(n_blocks) * block / float(sr)
    jit, shim, breath = _micro_levels(micro)

    if synthetic:
        # A predicted F0 contour: smooth, low order, no micro-variation. This is
        # what a text-to-speech duration and pitch predictor emits.
        f0 = f0_base * (1.0 + 0.05 * np.sin(2 * np.pi * 0.4 * t_blk))
        tracks = [
            620.0 + 60.0 * np.sin(2 * np.pi * 0.7 * t_blk),
            1250.0 + 120.0 * np.sin(2 * np.pi * 0.5 * t_blk + 1.0),
            2550.0 + 90.0 * np.sin(2 * np.pi * 0.3 * t_blk + 2.0),
        ]
        env = 0.9 + 0.1 * np.sin(2 * np.pi * 1.1 * t_blk)
    else:
        # A real contour: declination plus a random walk plus micro-tremor.
        walk = np.cumsum(rng.standard_normal(n_blocks)) * 1.6
        f0 = (f0_base - 6.0 * t_blk + walk
              + 3.0 * np.sin(2 * np.pi * 5.0 * t_blk))
        f0 = np.clip(f0, f0_base * 0.7, f0_base * 1.4)
        tracks = []
        for centre, spread in ((640.0, 190.0), (1300.0, 320.0), (2600.0, 260.0)):
            w = np.cumsum(rng.standard_normal(n_blocks)) * spread / 12.0
            tracks.append(np.clip(centre + w, centre * 0.6, centre * 1.5))
        env = np.clip(0.8 + 0.35 * np.cumsum(rng.standard_normal(n_blocks)) / 14.0,
                      0.25, 1.4)

    src = _glottal_source(n, sr, f0, jit, shim, rng)
    y = _formant_filter(src, sr, tracks,
                        bws=(90.0, 110.0, 170.0), gains=(1.0, 0.55, 0.28),
                        block=block)

    amp = np.repeat(env, block)[:n]
    y = y * amp
    peak = float(np.max(np.abs(y))) or 1.0
    y = 0.5 * y / peak

    if synthetic:
        # Vocoder style: a thin, over-regular noise floor and a soft top end,
        # because the mel spectrogram the waveform was rebuilt from does not
        # carry the fine detail of the upper band.
        y = y + breath * rng.standard_normal(n)
        from scipy import signal as ssig
        sos = ssig.butter(2, min(2800.0, 0.45 * sr) / (sr / 2.0),
                          btype="low", output="sos")
        y = 0.75 * y + 0.25 * ssig.sosfilt(sos, y)
    else:
        # Breath noise that follows the envelope, plus fricative bursts and two
        # pauses, which is what makes the flatness of a real call move around.
        y = y + breath * rng.standard_normal(n) * (0.3 + np.abs(y) / 0.5)
        for _ in range(max(int(dur_s), 1)):
            a = rng.integers(0, max(n - int(0.09 * sr), 1))
            w = int(0.06 * sr)
            y[a:a + w] += 0.12 * rng.standard_normal(min(w, n - a))
        for _ in range(2):
            a = rng.integers(0, max(n - int(0.25 * sr), 1))
            w = int(0.18 * sr)
            y[a:a + w] *= 0.02

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(y))) or 1.0
    return (0.6 * y / peak).astype(np.float32)


def demo_bonafide(sr: int = TARGET_SR, dur_s: float = 3.0, seed: int = 0,
                  f0_base: float = 130.0, micro: float = 1.0) -> np.ndarray:
    """A human-like signal: jittered periods, shimmering amplitudes, wandering
    formants, breath noise that tracks the envelope, fricatives and pauses.

    `micro` scales the cycle to cycle variation from machine-steady (0.0) to
    fully human (1.0). Lower it to simulate an unusually steady speaker, which
    is the bona fide case a spoof detector is most likely to get wrong.
    """
    return _synth(sr, dur_s, seed, synthetic=False, f0_base=f0_base, micro=micro)


def demo_spoof(sr: int = TARGET_SR, dur_s: float = 3.0, seed: int = 0,
               f0_base: float = 130.0, micro: float = 0.0) -> np.ndarray:
    """A vocoder-like signal: the same source-filter model with the micro
    variation removed, a smooth predicted F0 contour, a flat noise floor and a
    soft top band.

    Raise `micro` toward 1.0 to simulate a synthesis system that models jitter
    and shimmer instead of ignoring them. Recent neural vocoders do, which is
    why a detector that leans only on voice quality ages badly, and why this
    branch also carries cepstral and waveform back ends.
    """
    return _synth(sr, dur_s, seed, synthetic=True, f0_base=f0_base, micro=micro)


if __name__ == "__main__":  # pragma: no cover - self-test
    import time

    t0 = time.time()
    sr = TARGET_SR
    print("antispoof.features self-test")

    bona = demo_bonafide(sr, 3.0, seed=1)
    spoof = demo_spoof(sr, 3.0, seed=1)
    sil = np.zeros(2 * sr, dtype=np.float32)
    noise = (0.2 * np.random.default_rng(0).standard_normal(2 * sr)).astype(np.float32)
    tone = (0.4 * np.sin(2 * np.pi * 220.0 * np.arange(2 * sr) / sr)).astype(np.float32)

    for name in FEATURE_SETS:
        f = frame_features(bona, sr, name)
        u = utterance_features(bona, sr, name)
        print(f"  {name:5s} frames={f.shape} pooled={u.shape} "
              f"finite={bool(np.isfinite(u).all())} "
              f"expected_dim={utterance_feature_dim(name)}")

    print(f"  pooled names: {len(pooled_feature_names('lfcc'))} "
          f"e.g. {pooled_feature_names('lfcc')[0]}, {pooled_feature_names('lfcc')[-1]}")

    for name, sig in (("silence", sil), ("noise", noise), ("tone", tone),
                      ("bonafide", bona), ("spoof", spoof)):
        u = utterance_features(sig, sr)
        d = antispoof_feature_dict(sig, sr)
        print(f"  {name:9s} pooled finite={bool(np.isfinite(u).all())} "
              f"jit={d['jitter']:.4f} shim={d['shimmer']:.4f} hnr={d['hnr']:6.2f} "
              f"sfv={d['spec_flatness_var']:.3e} sfv_db={d['dx_flatness_db_var']:6.2f} "
              f"dratio={d['dx_delta_ratio']:.3f} kurt={d['dx_kurtosis_mean']:6.2f}")

    print(f"  dict keys: {sorted(antispoof_feature_dict(bona, sr))}")
    t1 = time.time()
    utterance_features(np.tile(bona, 10), sr)
    print(f"  30 s call pooled in {time.time() - t1:.2f} s")
    print(f"  done in {time.time() - t0:.2f} s")
