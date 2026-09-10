"""Telephone channel simulation: codecs, band limiting and additive noise.

Why this module exists at all. An anti-spoofing model trained on clean 16 kHz
studio audio collapses when a real phone line touches the signal, and that was
the headline finding of ASVspoof 2021: the same systems that scored a 1 percent
EER on clean data went past 20 percent once codecs were in the loop. A fraud
detector for Indian telephony is only ever going to see codec output, so the
training data has to contain it and the evaluation has to report per condition
numbers. This module is what produces those conditions.

What is exact and what is approximate, stated up front so the report can be
honest about it:

- `g711u` and `g711a` are the real ITU-T G.711 mu-law and A-law companding
  curves, the segment tables and all, implemented in numpy. Encode to 8 bits,
  decode back. Nothing about them is an approximation.
- `narrowband` and `packet_loss` are exactly what they claim to be.
- `gsm` (GSM 06.10 full rate) and `amrnb` are real codecs and this module
  cannot implement them from scratch, so it shells out to ffmpeg. With no
  ffmpeg on the system there is a documented numpy stand-in (band limit,
  mu-law quantisation, spectral smearing) and every result carries
  `real_codec: False`. A stand-in that is quietly reported as the real thing
  would be worse than no condition at all.

Run the self-test with:
    python -m swarkavach.channel
"""

from __future__ import annotations

import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import fft as sfft
from scipy import ndimage
from scipy import signal as ssig

from .audioio import read_audio, resample, write_wav
from .config import TARGET_SR, find_ffmpeg

__all__ = [
    "CODECS",
    "apply_codec",
    "add_noise",
    "telephone_band",
    "degrade",
    "codec_available",
    "linear_to_ulaw",
    "ulaw_to_linear",
    "linear_to_alaw",
    "alaw_to_linear",
    "mu_law_roundtrip",
    "a_law_roundtrip",
    "packet_loss",
    "make_noise",
]

_EPS = 1e-12

#: Registry the evaluation loop iterates over. "real" says whether the
#: implementation reproduces the actual standard; "needs_ffmpeg" says whether
#: that depends on an external binary being present.
CODECS: Dict[str, Dict] = {
    "clean": {"label": "Clean (no codec)", "real": True, "needs_ffmpeg": False},
    "g711u": {"label": "G.711 mu-law (PCMU, 64 kbit/s)", "real": True, "needs_ffmpeg": False},
    "g711a": {"label": "G.711 A-law (PCMA, 64 kbit/s)", "real": True, "needs_ffmpeg": False},
    "gsm": {"label": "GSM 06.10 full rate (13 kbit/s)", "real": True, "needs_ffmpeg": True},
    "amrnb": {"label": "AMR-NB (12.2 kbit/s)", "real": True, "needs_ffmpeg": True},
    "narrowband": {"label": "Telephone band 300-3400 Hz", "real": True, "needs_ffmpeg": False},
    "packet_loss": {"label": "Packet loss, no concealment", "real": True, "needs_ffmpeg": False},
}


@lru_cache(maxsize=8)
def _ffmpeg_can_encode(kind: str) -> bool:
    """Can this ffmpeg actually encode `kind`, as opposed to merely existing.

    Asking whether the binary is on PATH is not the same question. Colab ships
    an ffmpeg with the native GSM 06.10 encoder but no AMR-NB one, because the
    AMR encoder lives in libopencore_amrnb and most distribution builds leave
    it out over licensing. So `find_ffmpeg() is not None` reported AMR-NB as
    real while every encode quietly fell back to the numpy stand-in, and the
    results would have carried a condition labelled as the true codec when it
    was not.

    Rather than parse `ffmpeg -encoders` and guess which encoder name the
    container will select, this runs a fifth of a second of noise through the
    exact path `apply_codec` uses. Whatever that path can do is what gets
    reported. Once per codec per process.
    """
    if find_ffmpeg() is None:
        return False
    probe = np.sin(2.0 * np.pi * 440.0 * np.arange(1600) / 8000.0) * 0.2
    try:
        return _ffmpeg_codec(probe, TARGET_SR, kind) is not None
    except Exception:
        return False


def codec_available(codec: str) -> bool:
    """True when this codec can run its real implementation right now."""
    info = CODECS.get(str(codec).lower())
    if info is None:
        return False
    if not info["real"]:
        return False
    if not info["needs_ffmpeg"]:
        return True
    return _ffmpeg_can_encode(str(codec).lower())


# --------------------------------------------------------------------------
# G.711, the exact companding curves
# --------------------------------------------------------------------------

# Segment end points from the ITU-T G.711 reference implementation. mu-law
# works in a 14 bit domain (the input is shifted right by 2), A-law in a 13 bit
# one (shifted right by 3). Both compress a 12 to 14 bit linear sample into 8
# bits by splitting the range into 8 exponential segments with 16 uniform steps
# each, which is what buys roughly 38 dB of SNR across a 30 dB dynamic range on
# a 64 kbit/s link.
_ULAW_SEG_END = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF], dtype=np.int32)
_ALAW_SEG_END = np.array([0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF], dtype=np.int32)
_ULAW_BIAS = 0x84      # 132
_ULAW_CLIP = 8159      # largest value the 14 bit domain encodes


def _to_pcm16(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return np.clip(np.rint(x * 32768.0), -32768, 32767).astype(np.int32)


def _from_pcm16(p: np.ndarray) -> np.ndarray:
    return np.asarray(p, dtype=np.float64) / 32768.0


def linear_to_ulaw(pcm: np.ndarray) -> np.ndarray:
    """16 bit linear PCM to 8 bit mu-law, G.711 exactly. Returns uint8."""
    p = np.asarray(pcm, dtype=np.int32) >> 2          # 14 bit dynamic range
    mask = np.where(p < 0, 0x7F, 0xFF).astype(np.int32)
    v = np.minimum(np.abs(p), _ULAW_CLIP) + (_ULAW_BIAS >> 2)
    seg = np.searchsorted(_ULAW_SEG_END, v, side="left").astype(np.int32)
    seg_c = np.minimum(seg, 7)
    coded = ((seg_c << 4) | ((v >> (seg_c + 1)) & 0xF)) ^ mask
    return np.where(seg >= 8, 0x7F ^ mask, coded).astype(np.uint8)


def ulaw_to_linear(code: np.ndarray) -> np.ndarray:
    """8 bit mu-law back to 16 bit linear PCM, G.711 exactly."""
    u = (~np.asarray(code, dtype=np.uint8)).astype(np.int32)
    t = ((u & 0x0F) << 3) + _ULAW_BIAS
    t = t << ((u & 0x70) >> 4)
    return np.where(u & 0x80, _ULAW_BIAS - t, t - _ULAW_BIAS).astype(np.int32)


def linear_to_alaw(pcm: np.ndarray) -> np.ndarray:
    """16 bit linear PCM to 8 bit A-law, G.711 exactly. Returns uint8.

    A-law is the European variant (mu-law is North America and Japan). Its
    first two segments are linear rather than exponential, which gives slightly
    better SNR on quiet passages and slightly worse on loud ones.
    """
    p = np.asarray(pcm, dtype=np.int32) >> 3          # 13 bit dynamic range
    mask = np.where(p >= 0, 0xD5, 0x55).astype(np.int32)
    v = np.where(p >= 0, p, -p - 1)
    seg = np.searchsorted(_ALAW_SEG_END, v, side="left").astype(np.int32)
    seg_c = np.minimum(seg, 7)
    shift = np.where(seg_c < 2, 1, seg_c)
    coded = ((seg_c << 4) | ((v >> shift) & 0xF)) ^ mask
    return np.where(seg >= 8, 0x7F ^ mask, coded).astype(np.uint8)


def alaw_to_linear(code: np.ndarray) -> np.ndarray:
    """8 bit A-law back to 16 bit linear PCM, G.711 exactly."""
    a = (np.asarray(code, dtype=np.uint8) ^ 0x55).astype(np.int32)
    t = (a & 0x0F) << 4
    seg = (a & 0x70) >> 4
    t = np.where(
        seg == 0, t + 8,
        np.where(seg == 1, t + 0x108, (t + 0x108) << np.maximum(seg - 1, 0)),
    )
    return np.where(a & 0x80, t, -t).astype(np.int32)


def mu_law_roundtrip(x: np.ndarray) -> np.ndarray:
    """Float in, float out, through an 8 bit mu-law channel."""
    return _from_pcm16(ulaw_to_linear(linear_to_ulaw(_to_pcm16(x))))


def a_law_roundtrip(x: np.ndarray) -> np.ndarray:
    """Float in, float out, through an 8 bit A-law channel."""
    return _from_pcm16(alaw_to_linear(linear_to_alaw(_to_pcm16(x))))


# --------------------------------------------------------------------------
# Band limiting and packet loss
# --------------------------------------------------------------------------


def telephone_band(
    x: np.ndarray,
    sr: int = TARGET_SR,
    low: float = 300.0,
    high: float = 3400.0,
    order: int = 4,
) -> np.ndarray:
    """Band-pass to the 300 to 3400 Hz passband every phone network imposes.

    Zero phase (filtfilt), so the group delay does not smear the onsets that
    the streaming detector times its verdicts against. Signals too short to
    filter come back untouched.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    nyq = sr / 2.0
    lo = max(float(low), 1.0) / nyq
    hi = min(float(high), 0.99 * nyq) / nyq
    if x.size < 64 or not (0.0 < lo < hi < 1.0):
        return x
    try:
        sos = ssig.butter(order, [lo, hi], btype="band", output="sos")
        return ssig.sosfiltfilt(sos, x)
    except ValueError:
        return x


def packet_loss(
    x: np.ndarray,
    sr: int = TARGET_SR,
    loss_rate: float = 0.05,
    burst_ms: float = 20.0,
    seed: int = 0,
) -> np.ndarray:
    """Drop whole packets, with no concealment.

    20 ms is one RTP packet at 8 kHz. Real endpoints run some form of
    concealment (repeat the last pitch period, fade out), so zero filling is
    the worst case rather than the typical one. Short raised cosine ramps at
    each edge keep the gaps from adding a click, which would be an artefact of
    the simulation rather than of the network.
    """
    x = np.asarray(x, dtype=np.float64).ravel().copy()
    n_burst = max(int(round(burst_ms * sr / 1000.0)), 1)
    n_packets = int(np.ceil(x.size / n_burst))
    if n_packets == 0:
        return x
    rng = np.random.default_rng(seed)
    drop = rng.random(n_packets) < float(loss_rate)
    ramp = max(int(0.002 * sr), 1)
    fade = 0.5 * (1.0 - np.cos(np.pi * np.arange(ramp) / ramp))
    for i in np.flatnonzero(drop):
        a, b = i * n_burst, min((i + 1) * n_burst, x.size)
        if b - a < 2 * ramp:
            x[a:b] = 0.0
            continue
        x[a: a + ramp] *= fade[::-1]
        x[a + ramp: b - ramp] = 0.0
        x[b - ramp: b] *= fade
    return x


# --------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------


def _white(n: int, rng: np.random.Generator, sr: int) -> np.ndarray:
    return rng.standard_normal(n)


def _hum(n: int, rng: np.random.Generator, sr: int) -> np.ndarray:
    """Mains hum: 50 Hz and its harmonics with 1/k amplitudes.

    50 Hz because this is Indian telephony (the Americas would be 60). The
    harmonics matter more than the fundamental on a phone line, since the
    300 Hz high-pass removes the 50 and 100 Hz components almost entirely and
    what survives is the buzz at 300 Hz and above.
    """
    t = np.arange(n, dtype=np.float64) / sr
    out = np.zeros(n, dtype=np.float64)
    for k in range(1, 9):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        out += np.sin(2.0 * np.pi * 50.0 * k * t + phase) / k
    out += 0.05 * rng.standard_normal(n)   # a little hiss, no line is pure tone
    return out


def _babble(n: int, rng: np.random.Generator, sr: int, n_talkers: int = 6) -> np.ndarray:
    """Several band-limited, syllable-modulated noise streams summed together.

    Babble is the hard noise case because it is speech shaped: a detector
    cannot simply high-pass it away. Each stream gets its own sub-band inside
    the telephone passband and its own 2 to 7 Hz modulation, which is the
    syllable rate range, so the sum has the envelope statistics of a room full
    of people without needing any recorded speech.
    """
    t = np.arange(n, dtype=np.float64) / sr
    nyq = sr / 2.0
    out = np.zeros(n, dtype=np.float64)
    for _ in range(n_talkers):
        lo = rng.uniform(200.0, 900.0)
        hi = lo * rng.uniform(2.2, 4.0)
        hi = min(hi, 0.95 * nyq)
        stream = rng.standard_normal(n)
        if hi > lo + 50.0 and n > 64:
            try:
                sos = ssig.butter(2, [lo / nyq, hi / nyq], btype="band", output="sos")
                stream = ssig.sosfilt(sos, stream)
            except ValueError:
                pass
        f_mod = rng.uniform(2.0, 7.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        env = 0.35 + 0.65 * (0.5 * (1.0 + np.sin(2.0 * np.pi * f_mod * t + phase))) ** 1.5
        out += stream * env * rng.uniform(0.6, 1.4)
    return out


_NOISE_KINDS = {"white": _white, "hum": _hum, "babble": _babble}


def make_noise(
    n: int,
    kind: str = "babble",
    sr: int = TARGET_SR,
    seed: int = 0,
) -> np.ndarray:
    """Unit variance noise of the requested kind, reproducible from `seed`."""
    key = str(kind).lower()
    if key not in _NOISE_KINDS:
        raise ValueError(f"unknown noise kind {kind!r}, expected {sorted(_NOISE_KINDS)}")
    rng = np.random.default_rng(seed)
    noise = _NOISE_KINDS[key](int(n), rng, int(sr))
    rms = float(np.sqrt(np.mean(noise * noise)))
    return noise / rms if rms > _EPS else noise


def add_noise(
    x: np.ndarray,
    snr_db: float,
    kind: str = "babble",
    sr: int = TARGET_SR,
    seed: int = 0,
) -> np.ndarray:
    """Add noise at an exact global SNR.

    The scale is solved for, not searched: with signal power P and unit power
    noise, scaling the noise by sqrt(P) * 10^(-snr/20) puts the noise power at
    P * 10^(-snr/10), so 10 log10(P / noise power) is the requested value to
    within floating point. Measured over the whole signal, silence included, so
    a call with long pauses gets a lower speech-to-noise ratio than the number
    suggests. That is deliberate: it is the definition the ASVspoof noise
    conditions use, and it stays reproducible without a VAD in the loop.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return x
    p_sig = float(np.mean(x * x))
    if p_sig <= _EPS:
        return x
    noise = make_noise(x.size, kind, sr, seed)
    p_noise = float(np.mean(noise * noise))
    if p_noise <= _EPS:
        return x
    scale = np.sqrt(p_sig / p_noise) * (10.0 ** (-float(snr_db) / 20.0))
    return x + scale * noise


def measured_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """SNR in dB between a signal and its noisy version. Used by the tests and
    by the provenance report, so the condition labels can be verified."""
    clean = np.asarray(clean, dtype=np.float64).ravel()
    noisy = np.asarray(noisy, dtype=np.float64).ravel()
    n = min(clean.size, noisy.size)
    noise = noisy[:n] - clean[:n]
    p_sig = float(np.mean(clean[:n] ** 2))
    p_noise = float(np.mean(noise ** 2))
    if p_noise <= _EPS or p_sig <= _EPS:
        return float("inf")
    return float(10.0 * np.log10(p_sig / p_noise))


# --------------------------------------------------------------------------
# Codec simulation
# --------------------------------------------------------------------------


def _wola_smear(x: np.ndarray, width: int = 3, n_fft: int = 256) -> np.ndarray:
    """Smooth the magnitude spectrum across frequency, keep the phase.

    This is the stand-in for what a low bit rate codec does to the fine spectral
    structure: an LPC based coder transmits a smooth envelope plus a coarsely
    quantised residual, so the deep valleys between harmonics get filled in.
    Weighted overlap-add with an explicit window sum in the denominator, so the
    reconstruction is exact when width=1 and nothing is smoothed.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    hop = n_fft // 4
    if x.size < n_fft or width <= 1:
        return x
    w = np.hanning(n_fft)
    n_frames = 1 + (x.size - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * w

    spec = sfft.rfft(frames, axis=1)
    mag = np.abs(spec)
    # Moving average across frequency bins, edges held rather than zero padded
    # so the band edges do not get attenuated as an artefact of the smoothing.
    smooth = ndimage.uniform_filter1d(mag, size=width, axis=1, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        phase = np.where(mag > _EPS, spec / np.maximum(mag, _EPS), 1.0)
    out_frames = sfft.irfft(smooth * phase, n=n_fft, axis=1) * w

    out = np.zeros(x.size, dtype=np.float64)
    wsum = np.zeros(x.size, dtype=np.float64)
    ww = w * w
    for i in range(n_frames):
        a = i * hop
        out[a: a + n_fft] += out_frames[i]
        wsum[a: a + n_fft] += ww
    tail = n_frames * hop + n_fft
    out[tail:] = x[tail:]
    wsum[tail:] = 1.0
    return np.where(wsum > 1e-6, out / np.maximum(wsum, _EPS), x)


def _codec_approx(x: np.ndarray, sr: int, kind: str, seed: int = 0) -> np.ndarray:
    """Documented numpy stand-in for GSM and AMR-NB, used when ffmpeg is absent.

    Three effects, in the order a real coder applies them:

    1. Band limit to 300 to 3400 Hz. Both codecs are narrowband by definition.
    2. Smear the magnitude spectrum across neighbouring bins, which imitates the
       coarse quantisation of the LPC residual filling in the inter-harmonic
       valleys. AMR-NB at 12.2 kbit/s gets a wider kernel than GSM full rate at
       13 kbit/s, which is roughly the right ordering of how much structure each
       one throws away.
    3. Quantise through mu-law and add band-limited noise to reach the coder's
       rough SNR (about 30 dB for GSM 06.10, about 24 dB for AMR-NB at 12.2k).

    This is NOT GSM and it is NOT AMR. It puts a comparable amount of the right
    kind of damage on the signal so the pipeline can be exercised end to end,
    and every caller gets real_codec False so the report can say so.
    """
    width, snr = (3, 30.0) if kind == "gsm" else (5, 24.0)
    y = telephone_band(x, sr)
    y = _wola_smear(y, width=width)
    y = mu_law_roundtrip(np.clip(y, -1.0, 1.0))
    y = add_noise(y, snr, kind="white", sr=sr, seed=seed)
    return telephone_band(y, sr)


def _ffmpeg_codec(x: np.ndarray, sr: int, kind: str) -> Optional[Tuple[np.ndarray, int]]:
    """Run the real codec through ffmpeg. Returns None if anything goes wrong.

    GSM and AMR-NB are both 8 kHz only, so the signal is resampled on the way
    in and comes back at 8 kHz.
    """
    exe = find_ffmpeg()
    if exe is None:
        return None
    x = np.asarray(x, dtype=np.float64).ravel()
    if sr != TARGET_SR:
        x = resample(x.astype(np.float32), sr, TARGET_SR)
    ext, extra = ("gsm", []) if kind == "gsm" else ("amr", ["-b:a", "12.2k"])

    try:
        with tempfile.TemporaryDirectory(prefix="swarkavach_codec_") as tmp:
            tmp = Path(tmp)
            src, enc, dec = tmp / "in.wav", tmp / f"coded.{ext}", tmp / "out.wav"
            write_wav(src, x, TARGET_SR, peak=None)
            base = [exe, "-y", "-hide_banner", "-loglevel", "error"]
            steps = [
                base + ["-i", str(src), "-ar", "8000", "-ac", "1"] + extra + [str(enc)],
                base + ["-i", str(enc), "-ar", "8000", "-ac", "1", str(dec)],
            ]
            for cmd in steps:
                r = subprocess.run(cmd, capture_output=True, timeout=120)
                if r.returncode != 0 or not Path(cmd[-1]).exists():
                    return None
            y, out_sr = read_audio(dec, sr=TARGET_SR)
        return np.asarray(y, dtype=np.float64), int(out_sr)
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError):
        return None


def apply_codec(
    x: np.ndarray,
    sr: int = TARGET_SR,
    codec: str = "g711u",
    return_info: bool = False,
    seed: int = 0,
    **kw,
):
    """Push a signal through one channel condition.

    Returns (y, sr). With return_info=True returns (y, sr, info), where info
    carries `real_codec`: True when the output came from the actual standard
    (exact G.711 in numpy, or ffmpeg for GSM and AMR-NB) and False when it came
    from the documented stand-in. `degrade` always reports this flag.

    Extra keyword arguments go to the specific condition: low and high for
    narrowband, loss_rate and burst_ms for packet_loss.
    """
    name = str(codec).lower()
    if name not in CODECS:
        raise ValueError(f"unknown codec {codec!r}, expected {sorted(CODECS)}")
    x = np.asarray(x, dtype=np.float64).ravel()
    info = {"codec": name, "label": CODECS[name]["label"], "real_codec": True,
            "sr_in": int(sr), "sr_out": int(sr), "backend": "numpy"}

    if x.size == 0:
        return (x, sr, info) if return_info else (x, sr)

    if name == "clean":
        y, out_sr = x, sr
    elif name == "g711u":
        y, out_sr = mu_law_roundtrip(np.clip(x, -1.0, 1.0)), sr
    elif name == "g711a":
        y, out_sr = a_law_roundtrip(np.clip(x, -1.0, 1.0)), sr
    elif name == "narrowband":
        y = telephone_band(x, sr, kw.get("low", 300.0), kw.get("high", 3400.0))
        out_sr = sr
    elif name == "packet_loss":
        y = packet_loss(x, sr, kw.get("loss_rate", 0.05), kw.get("burst_ms", 20.0), seed)
        out_sr = sr
    else:  # gsm, amrnb
        real = _ffmpeg_codec(x, sr, name)
        if real is not None:
            y, out_sr = real
            info["backend"] = "ffmpeg"
        else:
            y, out_sr = _codec_approx(x, sr, name, seed), sr
            info["real_codec"] = False
            info["backend"] = "numpy-approx"
            why = ("ffmpeg not found" if find_ffmpeg() is None
                   else f"this ffmpeg build has no {name} encoder")
            info["note"] = (f"{why}, used the documented numpy approximation "
                            "(band limit, spectral smearing, mu-law "
                            "quantisation)")

    y = np.nan_to_num(np.asarray(y, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    info["sr_out"] = int(out_sr)
    return (y, int(out_sr), info) if return_info else (y, int(out_sr))


def degrade(
    x: np.ndarray,
    sr: int = TARGET_SR,
    codec: str = "g711u",
    snr_db: Optional[float] = None,
    seed: int = 0,
    noise_kind: str = "babble",
    **kw,
) -> Tuple[np.ndarray, Dict]:
    """One channel condition end to end: noise first, then the codec.

    That order is the physical one. Line and room noise are picked up before
    the encoder sees the signal, so the codec has to encode the noise too, and
    a low bit rate coder handles noisy input noticeably worse than clean input.
    Adding noise after the codec would produce a signal no phone network can.

    Returns (y, info). info records the codec, whether it was the real thing,
    the requested SNR, the SNR actually measured on the pre-codec signal, and
    the seed, so a condition can be reproduced exactly from the results file.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    info: Dict = {
        "codec": str(codec).lower(),
        "snr_db": None if snr_db is None else float(snr_db),
        "noise_kind": None,
        "seed": int(seed),
        "sr_in": int(sr),
    }

    noisy = x
    if snr_db is not None and x.size:
        noisy = add_noise(x, float(snr_db), noise_kind, sr, seed)
        info["noise_kind"] = str(noise_kind).lower()
        info["snr_measured_db"] = round(measured_snr(x, noisy), 3)

    y, out_sr, codec_info = apply_codec(noisy, sr, codec, return_info=True, seed=seed, **kw)
    info.update(codec_info)
    info["sr"] = int(out_sr)
    info["duration_s"] = round(y.size / float(out_sr), 3) if out_sr else 0.0
    return y, info


if __name__ == "__main__":  # pragma: no cover - self-test
    import time

    t0 = time.time()
    sr = TARGET_SR
    n = sr * 3
    t = np.arange(n) / sr
    rng = np.random.default_rng(11)

    # Speech-like test signal: a 140 Hz pulse train through three formants.
    src = np.zeros(n)
    src[:: int(sr / 140.0)] = 1.0
    x = np.zeros(n)
    for fc, bw, g in ((700.0, 90.0, 1.0), (1200.0, 110.0, 0.5), (2600.0, 170.0, 0.25)):
        r = np.exp(-np.pi * bw / sr)
        w = 2 * np.pi * fc / sr
        x += g * ssig.lfilter([1.0], [1.0, -2 * r * np.cos(w), r * r], src)
    x = 0.6 * x / np.max(np.abs(x))

    print("channel self-test")
    print(f"  ffmpeg: {find_ffmpeg() or 'not found, GSM and AMR fall back to numpy'}")

    for law, trip in (("mu-law", mu_law_roundtrip), ("A-law", a_law_roundtrip)):
        y = trip(x)
        err = y - x
        snr = 10 * np.log10(np.mean(x ** 2) / max(np.mean(err ** 2), 1e-20))
        print(f"  G.711 {law}: SNR {snr:5.2f} dB, max abs error {np.max(np.abs(err)):.4f}, "
              f"unique codes {len(np.unique(linear_to_ulaw(_to_pcm16(x)))) if law == 'mu-law' else len(np.unique(linear_to_alaw(_to_pcm16(x))))}")

    # Known values from the G.711 tables: silence and full scale.
    print(f"  ulaw(0)=0x{int(linear_to_ulaw(np.array([0]))[0]):02X} "
          f"alaw(0)=0x{int(linear_to_alaw(np.array([0]))[0]):02X} "
          f"ulaw(32767)=0x{int(linear_to_ulaw(np.array([32767]))[0]):02X} "
          f"ulaw(-32768)=0x{int(linear_to_ulaw(np.array([-32768]))[0]):02X}")

    print("  add_noise, requested against measured:")
    for kind in ("white", "hum", "babble"):
        for want in (20.0, 10.0, 0.0):
            y = add_noise(x, want, kind, sr, seed=7)
            got = measured_snr(x, y)
            print(f"    {kind:7s} want {want:5.1f} dB, got {got:6.2f} dB, "
                  f"error {abs(got - want):.3f}")

    a = add_noise(x, 10.0, "babble", sr, seed=3)
    b = add_noise(x, 10.0, "babble", sr, seed=3)
    c = add_noise(x, 10.0, "babble", sr, seed=4)
    print(f"  determinism: same seed identical {np.array_equal(a, b)}, "
          f"different seed differs {not np.array_equal(a, c)}")

    print("  codecs:")
    for name in CODECS:
        y, info = degrade(x, sr, codec=name, snr_db=None, seed=1)
        d = y[: x.size] - x[: y.size]
        snr = 10 * np.log10(np.mean(x ** 2) / max(np.mean(d ** 2), 1e-20))
        print(f"    {name:12s} real={str(info['real_codec']):5s} "
              f"backend={info['backend']:12s} out {y.size / sr:.2f} s "
              f"SNR vs clean {snr:6.2f} dB")

    y, info = degrade(x, sr, codec="g711u", snr_db=15.0, noise_kind="babble", seed=5)
    print(f"  degrade(g711u, 15 dB babble): {info['label']}, "
          f"measured pre-codec SNR {info['snr_measured_db']} dB")

    band = telephone_band(x, sr)
    spec = np.abs(np.fft.rfft(band * np.hanning(band.size)))
    freqs = np.fft.rfftfreq(band.size, 1 / sr)
    keep = spec[(freqs > 400) & (freqs < 3300)].mean()
    cut = spec[(freqs < 150)].mean()
    print(f"  telephone_band: passband mean {keep:.1f}, below 150 Hz {cut:.4f} "
          f"({20 * np.log10(cut / max(keep, 1e-12)):.1f} dB down)")
    print(f"  done in {time.time() - t0:.2f} s")
