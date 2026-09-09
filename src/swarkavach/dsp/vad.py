"""Voice activity detection: energy plus zero crossing rate, with hangover.

Deliberately simple and deliberately unsupervised. A trained VAD would be more
accurate, but it would also be one more model to explain, and this system only
needs the mask for three jobs: measuring pause ratio for the prosody features,
cutting turn boundaries out of a call, and keeping silence out of the
anti-spoofing statistics.

That last one matters more than it sounds. Several published ASVspoof results
turned out to be partly driven by silence: the bona fide and spoofed sets had
different amounts of leading silence, and the model learned that instead of the
artefacts. Anything that pools frame statistics over a call should pool over
speech frames only.

The decision has three parts:

1. An adaptive energy threshold, chosen from the spread between a low
   percentile of the frame levels (config default: the 35th, the presumed noise
   floor) and a high one (the 95th, the presumed speech level). Three regimes:
   a spread under 3 dB with a noise-like ZCR means stationary noise and nothing
   is speech; a spread under 12 dB means the recording has no real pauses, so
   the low percentile is itself speech and the threshold has to come off the
   peak instead; otherwise the threshold sits 6 dB above the floor, or 35 dB
   below the peak, whichever is higher. Getting the middle case wrong is the
   classic bug: a threshold anchored to the 35th percentile can never mark more
   than 65 percent of a continuously spoken call as speech, and a synthesised
   turn with no pauses in it is exactly that case.
2. A zero crossing escape hatch, so unvoiced fricatives (/s/, /sh/) that are
   10 to 15 dB below a vowel still count as speech.
3. Hangover smoothing: fill short gaps, drop short bursts, pad the edges. Speech
   does not switch on and off every 10 ms, and the pad recovers the low energy
   onset and offset that the threshold clips.

Known limit, stated plainly: with no contrast in the recording this is a
relative measure with nothing to be relative to. A steady tone comes out as
speech, because energy and ZCR give no reason to call it anything else. The
assumption is that the input is a call, meaning it contains speech and gaps.

Run the self-test with:
    python -m swarkavach.dsp.vad
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..config import SETTINGS, TARGET_SR, VADConfig
from .framing import frame_signal, rms_db, zero_crossing_rate

__all__ = ["vad_mask", "speech_segments", "trim_silence", "speech_ratio", "frame_grid"]

#: A speech frame has to be at least this far above the estimated noise floor.
#: 6 dB is a factor of four in power, comfortably outside the frame to frame
#: wobble of stationary line noise and still low enough to catch a soft vowel.
_MIN_MARGIN_DB = 6.0

#: Nothing this far below the loud end of the recording is speech. This is the
#: only criterion left when the recording has no silence in it to measure.
_DROP_DB = 35.0

#: Floor to peak spread below which the percentile floor is not a noise
#: estimate at all, it is just a quieter speech frame. A call recorded with no
#: pauses lands here, and then the level relative to the peak is all we have.
_MIN_DYNAMIC_DB = 12.0

#: Below this spread the signal has no envelope modulation at all. That alone
#: does not make it noise: a sustained vowel measures about 0.7 dB of spread
#: and broadband noise about 1.0 dB, so energy cannot separate them. The second
#: cue does. Noise crosses zero on about half of all sample pairs, voiced
#: speech on under a fifth, so a flat signal whose median ZCR is above
#: _STATIONARY_ZCR is noise and gets no speech frames.
_STATIONARY_DB = 3.0
_STATIONARY_ZCR = 0.30

#: A recording whose loud end is below this is digital silence (about 6 LSB at
#: 16 bits) and gets no speech frames at all. Every other decision in here is
#: relative, so without this one, silence would be normalised into "speech".
_DIGITAL_SILENCE_DB = -75.0

#: ZCR above this, with energy above a reduced threshold, is treated as an
#: unvoiced fricative. At 8 kHz a strong /s/ crosses zero on roughly a third of
#: sample pairs, while voiced speech stays well under 0.2.
_FRICATIVE_ZCR = 0.28
_FRICATIVE_RELIEF_DB = 6.0


def _cfg(cfg: Optional[VADConfig]) -> VADConfig:
    return SETTINGS.vad if cfg is None else cfg


def frame_grid(sr: int, cfg: Optional[VADConfig] = None) -> Tuple[int, int]:
    """(frame_len, hop_len) in samples for this VAD config."""
    cfg = _cfg(cfg)
    return (int(round(sr * cfg.frame_ms / 1000.0)),
            int(round(sr * cfg.hop_ms / 1000.0)))


def _smooth(mask: np.ndarray, hop_len: int, sr: int, cfg: VADConfig) -> np.ndarray:
    """Fill short silences, drop short speech bursts, then pad each run."""
    if mask.size == 0:
        return mask
    frames_per_s = float(sr) / float(hop_len)

    def n_frames(ms: float) -> int:
        return max(int(round(ms * frames_per_s / 1000.0)), 1)

    out = mask.copy()
    min_sil = n_frames(cfg.min_silence_ms)
    min_sp = n_frames(cfg.min_speech_ms)
    pad = int(round(cfg.pad_ms * frames_per_s / 1000.0))

    for target, min_run in ((False, min_sil), (True, min_sp)):
        for start, stop in _runs(out, target):
            if stop - start < min_run:
                out[start:stop] = not target

    if pad > 0:
        padded = out.copy()
        for start, stop in _runs(out, True):
            padded[max(0, start - pad): min(out.size, stop + pad)] = True
        out = padded
    return out


def _runs(mask: np.ndarray, value: bool) -> List[Tuple[int, int]]:
    """Half open [start, stop) index pairs of every run equal to `value`."""
    if mask.size == 0:
        return []
    m = mask.astype(bool) == bool(value)
    edges = np.flatnonzero(np.diff(m.astype(np.int8)))
    starts = np.concatenate([[0], edges + 1])
    stops = np.concatenate([edges + 1, [m.size]])
    return [(int(a), int(b)) for a, b in zip(starts, stops) if m[a]]


def vad_mask(
    x: np.ndarray,
    sr: int = TARGET_SR,
    cfg: Optional[VADConfig] = None,
) -> np.ndarray:
    """Per frame speech decision, shape (n_frames,), dtype bool.

    Frames are cut with the VAD config's own frame and hop, which is 25 ms and
    10 ms by default, matching the feature frames.
    """
    cfg = _cfg(cfg)
    x = np.asarray(x, dtype=np.float64).ravel()
    frame_len, hop_len = frame_grid(sr, cfg)
    if x.size == 0:
        return np.zeros(0, dtype=bool)

    frames = frame_signal(x, frame_len, hop_len, "hamming")
    level = rms_db(frames)
    zcr = zero_crossing_rate(frames)

    if not np.any(np.isfinite(level)):
        return np.zeros(frames.shape[0], dtype=bool)

    peak = float(np.percentile(level, 95.0))
    if peak < _DIGITAL_SILENCE_DB:
        return np.zeros(frames.shape[0], dtype=bool)

    floor = float(np.percentile(level, cfg.energy_percentile))
    if peak - floor < _STATIONARY_DB and float(np.median(zcr)) > _STATIONARY_ZCR:
        return np.zeros(frames.shape[0], dtype=bool)

    if peak - floor < _MIN_DYNAMIC_DB:
        # No usable silence in the recording, so the percentile "floor" is a
        # speech frame. Fall back to a pure relative-to-peak decision, which
        # marks a continuously spoken call as speech throughout instead of
        # rejecting its quieter two thirds.
        thresh = peak - _DROP_DB
    else:
        thresh = max(floor + _MIN_MARGIN_DB, peak - _DROP_DB)

    speech = level > thresh
    fricative = (zcr > _FRICATIVE_ZCR) & (level > thresh - _FRICATIVE_RELIEF_DB)
    mask = speech | fricative
    return _smooth(mask, hop_len, sr, cfg)


def speech_segments(
    x: np.ndarray,
    sr: int = TARGET_SR,
    cfg: Optional[VADConfig] = None,
) -> List[Tuple[float, float]]:
    """Speech runs as (start_seconds, end_seconds), clipped to the signal."""
    cfg = _cfg(cfg)
    mask = vad_mask(x, sr, cfg)
    frame_len, hop_len = frame_grid(sr, cfg)
    dur = len(np.asarray(x).ravel()) / float(sr)
    out = []
    for start, stop in _runs(mask, True):
        t0 = start * hop_len / float(sr)
        t1 = min(((stop - 1) * hop_len + frame_len) / float(sr), dur)
        if t1 > t0:
            out.append((round(float(t0), 3), round(float(t1), 3)))
    return out


def speech_ratio(
    x: np.ndarray,
    sr: int = TARGET_SR,
    cfg: Optional[VADConfig] = None,
) -> float:
    """Fraction of frames marked speech, in [0, 1]. 0.0 for an empty signal."""
    mask = vad_mask(x, sr, cfg)
    return float(mask.mean()) if mask.size else 0.0


def trim_silence(
    x: np.ndarray,
    sr: int = TARGET_SR,
    cfg: Optional[VADConfig] = None,
    keep_internal: bool = True,
) -> np.ndarray:
    """Cut leading and trailing silence.

    keep_internal=True (default) keeps the pauses between words, so timings
    inside the returned signal still line up with the original. Pass False to
    concatenate the speech segments instead, which is what the anti-spoof
    pooling wants when silence would otherwise dominate the statistics.

    A signal with no detected speech comes back unchanged. Returning an empty
    array here would make every caller handle a special case.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    segs = speech_segments(x, sr, cfg)
    if not segs or x.size == 0:
        return x
    if keep_internal:
        a = int(round(segs[0][0] * sr))
        b = int(round(segs[-1][1] * sr))
        return x[max(a, 0): min(b, x.size)]
    parts = [x[int(round(s * sr)): int(round(e * sr))] for s, e in segs]
    parts = [p for p in parts if p.size]
    return np.concatenate(parts) if parts else x


if __name__ == "__main__":  # pragma: no cover - self-test
    import time

    t0 = time.time()
    sr = TARGET_SR
    cfg = SETTINGS.vad
    rng = np.random.default_rng(7)

    def burst(dur_s: float, f0: float = 140.0) -> np.ndarray:
        t = np.arange(int(dur_s * sr)) / sr
        sig = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3, 4))
        env = np.clip(np.sin(np.pi * t / max(dur_s, 1e-9)), 0, 1)
        return 0.4 * sig * env

    quiet = lambda d: 0.0008 * rng.standard_normal(int(d * sr))
    x = np.concatenate([quiet(0.5), burst(0.8), quiet(0.6), burst(1.1), quiet(0.4)])
    x += 0.0008 * rng.standard_normal(x.size)

    print("vad self-test")
    print(f"  signal {x.size / sr:.2f} s, true speech at 0.50-1.30 and 1.90-3.00 s")
    mask = vad_mask(x, sr, cfg)
    segs = speech_segments(x, sr, cfg)
    print(f"  mask {mask.shape}, speech frames {int(mask.sum())} "
          f"({100 * mask.mean():.1f} %)")
    for s, e in segs:
        print(f"    segment {s:.2f} to {e:.2f} s  ({e - s:.2f} s)")

    trimmed = trim_silence(x, sr, cfg)
    packed = trim_silence(x, sr, cfg, keep_internal=False)
    print(f"  trim_silence: {x.size / sr:.2f} s -> {trimmed.size / sr:.2f} s "
          f"(packed {packed.size / sr:.2f} s)")

    sil = 1e-5 * rng.standard_normal(2 * sr)
    print(f"  pure noise floor: speech ratio {speech_ratio(sil, sr, cfg):.3f} "
          "(want near 0)")
    print(f"  pure zeros: speech ratio {speech_ratio(np.zeros(sr), sr, cfg):.3f}")
    print(f"  loud stationary noise: speech ratio "
          f"{speech_ratio(0.2 * rng.standard_normal(2 * sr), sr, cfg):.3f} "
          "(no envelope, so not speech)")
    loud = np.concatenate([burst(1.0), burst(1.0)])
    print(f"  continuous speech, no pauses: speech ratio "
          f"{speech_ratio(loud, sr, cfg):.3f} (want near 1)")
    print(f"  done in {time.time() - t0:.2f} s")
