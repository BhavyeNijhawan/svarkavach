"""A source-filter synthesiser for paired human-sounding and cloned-sounding speech.

Read this before judging the output: this is NOT an attempt at intelligible
speech. Nobody will understand a word of it. What it is trying to do is
produce a signal whose acoustic statistics separate natural speech from
vocoder output the same way real recordings separate from real TTS, so the
anti-spoofing branch has genuine structure to learn from with zero downloads
and zero network access. The dialogue text drives the segment sequence, which
is what makes two renderings of the same call comparable, and the per-speaker
voice dictionary drives the vocal tract, which is what makes two speakers
separable.

The model:

* Source. A glottal flow pulse train, one pulse per pitch period, built from a
  two-slope Rosenberg style pulse and then differentiated so the closure
  instant produces the sharp excitation that a real glottis does. Plus an
  aperiodic noise stream for breath and frication.
* Filter. Three to five formant resonators in cascade, each a two-pole IIR,
  with centre frequencies that move across a per-token vowel target sequence.
  Coefficients are updated every 5 ms frame and the filter state is carried
  across frames, so the trajectory is continuous.
* Radiation. Already folded in: the source is the flow derivative.

What differs between the two paths, and why:

| property            | human (synthetic=False) | cloned (synthetic=True) |
|---------------------|-------------------------|-------------------------|
| cycle jitter        | 0.5 to 1.5 percent      | about 0.02 percent      |
| amplitude shimmer   | 4 to 9 percent          | about 0.3 percent       |
| formant trajectory  | lightly smoothed        | heavily low-passed      |
| high band noise     | breath present          | reduced and duller      |
| segment timing      | random per segment      | quantised to a grid     |
| spectral tilt       | speaker dependent       | extra low-pass tilt     |
| harmonic phase      | free running            | constant phase buzz     |

Those are exactly the artefacts published anti-spoofing features respond to:
low jitter and shimmer, over-smoothed spectral envelopes, missing aperiodic
energy and unnaturally regular phase. The human path is not a recording, so
this corpus cannot stand in for ASVspoof; it is the offline fallback that
keeps the whole pipeline runnable on a laptop, and the report says so.

Everything is deterministic given (text, voice, synthetic, seed).
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..audioio import rms_normalize
from ..config import TARGET_SR
from ..schema import tokenize

try:  # scipy is a hard dependency of the project, but stay defensive
    from scipy.signal import lfilter as _lfilter
except Exception:  # pragma: no cover - exercised only without scipy
    _lfilter = None

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

HOP_S = 0.005                     # frame hop for the trajectory tracks
TOKENS_PER_SEC = 2.6              # matches generator.SPEAK_RATE_TOKENS_PER_S
MIN_TURN_S = 0.45

#: Neutral formant targets in Hz for a reference vocal tract. Scaled per
#: speaker by voice["vt_scale"], which is what a shorter or longer vocal tract
#: does to every formant at once.
VOWEL_TARGETS: Dict[str, Tuple[float, float, float]] = {
    "a": (730.0, 1090.0, 2440.0),
    "aa": (850.0, 1220.0, 2810.0),
    "e": (530.0, 1840.0, 2480.0),
    "i": (270.0, 2290.0, 3010.0),
    "o": (570.0, 840.0, 2410.0),
    "u": (300.0, 870.0, 2240.0),
    "@": (500.0, 1500.0, 2500.0),
}

#: Segment classes. V voiced vowel, N nasal or liquid, F fricative,
#: S stop (closure then burst), P pause.
_BASE_DURATION = {"V": 0.110, "N": 0.060, "F": 0.080, "S": 0.070, "P": 0.220}

_VOWEL_LETTERS = "aeiou"
_FRICATIVES = set("sfhzv")
_STOPS = set("ptkbdgcjqx")
_SONORANTS = set("mnlrwy")

_DIGIT_SYLLABLES = {
    "0": "ju", "1": "wa", "2": "tu", "3": "ti", "4": "fo",
    "5": "fa", "6": "si", "7": "se", "8": "e", "9": "na",
}


def _rng_from(*parts: object) -> np.random.Generator:
    """A numpy generator seeded stably from arbitrary parts.

    Python's hash() is salted per process, so string parts go through blake2b
    instead. Same inputs, same audio, every run and every machine.
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


# --------------------------------------------------------------------------
# Voices
# --------------------------------------------------------------------------


def make_voice(speaker_id: str, seed: int = 0) -> Dict[str, float]:
    """Stable vocal tract and source parameters for one speaker.

    The same speaker_id always gives the same voice, and two different
    speaker_ids give voices that differ in pitch, formant scaling, bandwidth,
    tilt and speaking rate, which is enough for a speaker embedding to tell
    them apart.
    """
    rng = _rng_from("voice", speaker_id, seed)
    # Bimodal F0: a lower and a higher voice population, as in a mixed pool of
    # speakers, rather than one wide unimodal spread.
    if rng.random() < 0.5:
        f0 = float(rng.uniform(92.0, 135.0))
        vt = float(rng.uniform(0.90, 1.02))
    else:
        f0 = float(rng.uniform(165.0, 235.0))
        vt = float(rng.uniform(1.04, 1.18))
    return {
        "speaker_id": speaker_id,
        "f0": f0,
        "f0_range": float(rng.uniform(0.05, 0.13)),     # relative accent depth
        "vt_scale": vt,                                  # formant scaling
        "bw_scale": float(rng.uniform(0.85, 1.30)),      # formant bandwidths
        "tilt": float(rng.uniform(0.0, 0.22)),           # source spectral tilt
        "breath": float(rng.uniform(0.020, 0.075)),      # aperiodic level
        "oq": float(rng.uniform(0.48, 0.72)),            # glottal open quotient
        "rate": float(rng.uniform(0.88, 1.14)),          # speaking rate factor
        "jitter": float(rng.uniform(0.005, 0.015)),      # cycle to cycle, 0.5 to 1.5 percent
        "shimmer": float(rng.uniform(0.040, 0.090)),
        "vowel_warp": float(rng.uniform(-0.06, 0.06)),   # personal vowel space skew
    }


#: What the cloned path overrides. Pitch and vocal tract survive, because a
#: voice clone keeps the speaker identity; the micro-variation does not.
SYNTHETIC_OVERRIDES: Dict[str, float] = {
    "jitter": 0.0002,
    "shimmer": 0.003,
    "breath": 0.006,
}

#: Extra artefacts only the cloned path gets.
SYNTHETIC_SMOOTH_FRAMES = 11      # formant trajectory low-pass width
HUMAN_SMOOTH_FRAMES = 3
SYNTHETIC_TILT_EXTRA = 0.30       # duller high band
SYNTHETIC_TIMING_GRID_S = 0.020   # segment durations snap to this
SYNTHETIC_BUZZ_GAIN = 0.030       # constant phase harmonic stack


# --------------------------------------------------------------------------
# Text to segments
# --------------------------------------------------------------------------


def _letters_to_segments(word: str) -> List[Tuple[str, str]]:
    """Split a romanised token into (class, vowel_key) segments."""
    out: List[Tuple[str, str]] = []
    i = 0
    w = word.lower()
    n = len(w)
    while i < n:
        ch = w[i]
        if ch in _VOWEL_LETTERS:
            j = i
            while j + 1 < n and w[j + 1] == ch:
                j += 1
            doubled = j > i
            key = {"a": "aa" if doubled else "a", "e": "e", "i": "i",
                   "o": "o", "u": "u"}[ch]
            out.append(("V", key))
            i = j + 1
            continue
        if ch in _FRICATIVES:
            # sh, ch and th behave like single fricatives
            if i + 1 < n and w[i + 1] == "h":
                i += 1
            out.append(("F", "@"))
        elif ch in _STOPS:
            if i + 1 < n and w[i + 1] == "h":
                i += 1
            out.append(("S", "@"))
        elif ch in _SONORANTS:
            out.append(("N", "@"))
        else:
            out.append(("N", "@"))
        i += 1
    if not out:
        out.append(("V", "@"))
    return out


def token_segments(token: str) -> List[Tuple[str, str]]:
    """Segment sequence for one token, including digit strings.

    Digits are spelled out one syllable each, which is what a speaker reading
    an OTP aloud actually does and which makes those turns audibly longer.
    """
    if token.isdigit() or (token[:1].isdigit() and any(c.isdigit() for c in token)):
        out: List[Tuple[str, str]] = []
        for ch in token:
            if ch in _DIGIT_SYLLABLES:
                out.extend(_letters_to_segments(_DIGIT_SYLLABLES[ch]))
        return out or [("V", "@")]
    if not token[:1].isalpha():
        return [("P", "@")]
    return _letters_to_segments(token)


def plan_segments(
    text: str,
    voice: Dict[str, float],
    synthetic: bool,
    rng: np.random.Generator,
) -> List[Tuple[str, str, float]]:
    """(class, vowel key, duration in seconds) for the whole turn.

    Durations are rescaled so the turn lands near the same speaking rate the
    corpus timing model assumes, then either jittered (human) or snapped to a
    fixed grid (cloned). Regular segment timing is one of the artefacts that
    gives vocoded speech away, so it is deliberate here.
    """
    tokens = tokenize(text)
    if not tokens:
        tokens = ["hmm"]
    plan: List[Tuple[str, str, float]] = []
    for i, token in enumerate(tokens):
        for kind, key in token_segments(token):
            plan.append((kind, key, _BASE_DURATION[kind]))
        if i != len(tokens) - 1:
            plan.append(("P", "@", 0.055))

    target = max(MIN_TURN_S, len(tokens) / TOKENS_PER_SEC * float(voice["rate"]))
    total = sum(d for _, _, d in plan)
    scale = target / total if total > 0 else 1.0

    out: List[Tuple[str, str, float]] = []
    for kind, key, dur in plan:
        d = dur * scale
        if synthetic:
            grid = SYNTHETIC_TIMING_GRID_S
            d = max(grid, round(d / grid) * grid)
        else:
            d *= float(1.0 + rng.normal(0.0, 0.08))
            d = max(0.020, d)
        out.append((kind, key, d))
    return out


# --------------------------------------------------------------------------
# Frame level tracks
# --------------------------------------------------------------------------


def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    """Centred moving average, used as the formant trajectory low-pass."""
    if width <= 1 or x.size == 0:
        return x
    kernel = np.ones(int(width), dtype=np.float64) / float(width)
    pad = int(width) // 2
    padded = np.concatenate([np.full(pad, x[0]), x, np.full(pad, x[-1])])
    return np.convolve(padded, kernel, mode="same")[pad:pad + x.size]


def _formant_count(sr: int) -> int:
    """Three to five resonators, as many as fit under the Nyquist limit."""
    if sr >= 16000:
        return 5
    if sr >= 11025:
        return 4
    return 4 if sr >= 8000 else 3


def build_tracks(
    plan: Sequence[Tuple[str, str, float]],
    sr: int,
    voice: Dict[str, float],
    synthetic: bool,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Per-frame formants, bandwidths, F0, amplitude, voicing and noise gain."""
    hop = max(1, int(round(HOP_S * sr)))
    n_frames = max(1, int(round(sum(d for _, _, d in plan) / HOP_S)))
    n_formants = _formant_count(sr)

    freqs = np.zeros((n_frames, n_formants))
    bws = np.zeros((n_frames, n_formants))
    amp = np.zeros(n_frames)
    voiced = np.zeros(n_frames, dtype=bool)
    noise = np.zeros(n_frames)

    vt = float(voice["vt_scale"])
    warp = float(voice["vowel_warp"])
    bw_scale = float(voice["bw_scale"])
    nyq = 0.5 * sr

    # Upper formants sit at fixed places; only F1 to F3 move with the vowel.
    fixed_upper = [3400.0, 4500.0]
    base_bw = [70.0, 95.0, 135.0, 200.0, 260.0]

    frame = 0
    for kind, key, dur in plan:
        n = max(1, int(round(dur / HOP_S)))
        f1, f2, f3 = VOWEL_TARGETS.get(key, VOWEL_TARGETS["@"])
        f1 *= vt * (1.0 + warp)
        f2 *= vt * (1.0 - warp)
        f3 *= vt
        if kind == "N":
            # Nasals and liquids: low F1, damped, F2 pulled toward the middle.
            f1 *= 0.55
            f2 = 0.85 * f2 + 0.15 * 1500.0
        for i in range(min(n, n_frames - frame)):
            f = frame + i
            row = [f1, f2, f3] + fixed_upper
            freqs[f] = [min(v * vt if j >= 3 else v, nyq * 0.92)
                        for j, v in enumerate(row[:n_formants])]
            widen = 1.0 if kind == "V" else (2.0 if kind == "N" else 1.4)
            bws[f] = [base_bw[j] * bw_scale * widen for j in range(n_formants)]
            if kind == "V":
                amp[f], voiced[f], noise[f] = 1.00, True, float(voice["breath"])
            elif kind == "N":
                amp[f], voiced[f], noise[f] = 0.55, True, float(voice["breath"]) * 0.7
            elif kind == "F":
                amp[f], voiced[f] = 0.0, False
                noise[f] = 0.45 if not synthetic else 0.32
            elif kind == "S":
                # closure then burst
                burst = i > 0.62 * n
                amp[f], voiced[f] = 0.0, False
                noise[f] = (0.55 if not synthetic else 0.40) if burst else 0.005
            else:
                amp[f], voiced[f], noise[f] = 0.0, False, 0.0
        frame += n
        if frame >= n_frames:
            break

    # Coarticulation. The cloned path gets a much wider window, which is the
    # over-smoothed spectral envelope that vocoders are known for.
    width = SYNTHETIC_SMOOTH_FRAMES if synthetic else HUMAN_SMOOTH_FRAMES
    for j in range(n_formants):
        freqs[:, j] = _smooth(freqs[:, j], width)
        bws[:, j] = _smooth(bws[:, j], max(3, width // 2))
    # The amplitude envelope is smoothed the same way on both paths. The
    # over-smoothing that matters for spoofing detection is spectral, and
    # flattening the loudness contour as well would only add an amplitude
    # artefact that no real vocoder has.
    amp = _smooth(amp, 3)
    if synthetic:
        # A vocoder does carry a smoothed aperiodicity parameter, so the
        # contrast between a vowel and a fricative is much weaker than in real
        # speech. That is what flattens spectral-flatness variance over time,
        # one of the features the anti-spoof branch reads.
        noise = _smooth(noise, 9)
        noise = 0.55 * noise + 0.45 * float(np.mean(noise))

    # F0 contour: declination over the utterance, accents on voiced runs, a
    # slow drift for the human path only.
    t = np.linspace(0.0, 1.0, n_frames)
    f0 = np.full(n_frames, float(voice["f0"]))
    f0 *= 1.0 + 0.06 * (1.0 - t) - 0.05 * t          # declination
    accent = np.sin(2.0 * np.pi * t * 2.5) * float(voice["f0_range"]) * 0.5
    f0 *= 1.0 + accent
    if not synthetic:
        drift = _smooth(rng.normal(0.0, 1.0, n_frames), max(3, n_frames // 8))
        scale = np.std(drift) or 1.0
        f0 *= 1.0 + 0.035 * drift / scale
        f0 *= 1.0 + rng.normal(0.0, 0.004, n_frames)
    else:
        f0 *= 1.0 + 0.0015 * np.sin(2.0 * np.pi * t * 7.0)

    return {
        "freqs": freqs,
        "bws": bws,
        "f0": np.clip(f0, 60.0, 400.0),
        "amp": amp,
        "voiced": voiced,
        "noise": noise,
        "hop": np.array([hop]),
    }


# --------------------------------------------------------------------------
# Source
# --------------------------------------------------------------------------


def glottal_pulse(period: int, oq: float, skew: float = 0.66) -> np.ndarray:
    """One glottal flow derivative pulse, two-slope Rosenberg style.

    The flow rises over the open phase, falls sharply at closure, then stays
    shut. Differentiating gives the negative spike at the closure instant that
    actually excites the vocal tract, and a source spectrum that rolls off the
    way a real glottis does.
    """
    period = max(8, int(period))
    open_len = max(4, int(period * oq))
    rise = max(2, int(open_len * skew))
    fall = max(2, open_len - rise)

    flow = np.zeros(period, dtype=np.float64)
    tr = np.arange(rise)
    flow[:rise] = 0.5 * (1.0 - np.cos(np.pi * tr / rise))
    tf = np.arange(fall)
    flow[rise:rise + fall] = np.cos(np.pi * tf / (2.0 * fall))

    deriv = np.diff(flow, prepend=0.0)
    peak = np.max(np.abs(deriv))
    return deriv / peak if peak > 0 else deriv


class _PulseCache:
    """Pulses are reused constantly; building them once pays for itself."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[int, int], np.ndarray] = {}

    def get(self, period: float, oq: float) -> np.ndarray:
        key = (int(round(period)), int(round(oq * 100)))
        pulse = self._cache.get(key)
        if pulse is None:
            pulse = glottal_pulse(key[0], key[1] / 100.0)
            self._cache[key] = pulse
        return pulse


def build_excitation(
    tracks: Dict[str, np.ndarray],
    sr: int,
    voice: Dict[str, float],
    synthetic: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Glottal pulse train plus aperiodic noise, at sample rate."""
    hop = int(tracks["hop"][0])
    n_frames = tracks["f0"].size
    n = n_frames * hop
    exc = np.zeros(n + 4 * hop, dtype=np.float64)

    f0 = tracks["f0"]
    amp = tracks["amp"]
    voiced = tracks["voiced"]
    jitter = float(voice["jitter"])
    shimmer = float(voice["shimmer"])
    oq = float(voice["oq"])
    cache = _PulseCache()

    pos = 0
    while pos < n:
        fr = min(pos // hop, n_frames - 1)
        if not voiced[fr]:
            pos = (fr + 1) * hop
            continue
        period = sr / float(f0[fr])
        if jitter > 0:
            period *= 1.0 + float(rng.normal(0.0, jitter))
        period = max(8.0, period)
        if synthetic:
            # Constant open quotient and integer period placement: unnaturally
            # regular phase, which is exactly what a vocoder produces.
            pulse = cache.get(period, oq)
        else:
            pulse = cache.get(period, oq * (1.0 + float(rng.normal(0.0, 0.05))))
        gain = float(amp[fr]) * (1.0 + float(rng.normal(0.0, shimmer)))
        exc[pos:pos + pulse.size] += pulse * gain
        pos += int(round(period))

    exc = exc[:n]

    # Aperiodic stream. The human path keeps a high-band breath component; the
    # cloned path gets less of it and what is left is duller.
    noise_env = np.repeat(tracks["noise"], hop)[:n]
    white = rng.normal(0.0, 1.0, n)
    if synthetic:
        white = _one_pole(white, 0.55)          # low-passed, less high band
    else:
        white = white - _one_pole(white, 0.72)  # high-passed breath
    exc = exc + noise_env * white * 0.9

    if synthetic:
        # A faint harmonic stack locked to the pitch contour with a fixed phase
        # relationship between the harmonics. Real vocoder output has this kind
        # of over-regular harmonic structure: it lifts HNR and flattens the
        # spectral-flatness variance. The stack follows F0 rather than sitting
        # at a fixed frequency, otherwise it beats against the pulse train and
        # produces amplitude modulation that no vocoder actually has.
        f0_dense = np.repeat(f0, hop)[:n]
        phase = 2.0 * np.pi * np.cumsum(f0_dense) / float(sr)
        base = float(np.median(f0))
        buzz = np.zeros(n)
        k = 1
        while k * base < 0.45 * sr and k <= 20:
            buzz += np.cos(k * phase) / k
            k += 1
        speech = np.repeat(np.maximum(tracks["amp"], tracks["noise"]), hop)[:n]
        exc = exc + SYNTHETIC_BUZZ_GAIN * buzz * speech
    return exc


def _one_pole(x: np.ndarray, coef: float) -> np.ndarray:
    """y[n] = (1 - c) x[n] + c y[n-1], the tilt and smoothing workhorse."""
    if _lfilter is not None:
        return _lfilter([1.0 - coef], [1.0, -coef], x)
    out = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc = (1.0 - coef) * v + coef * acc
        out[i] = acc
    return out


# --------------------------------------------------------------------------
# Filter
# --------------------------------------------------------------------------


def _resonator(freq: float, bw: float, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Two-pole resonator with unity gain at DC."""
    r = math.exp(-math.pi * bw / sr)
    theta = 2.0 * math.pi * min(freq, 0.48 * sr) / sr
    a1 = -2.0 * r * math.cos(theta)
    a2 = r * r
    b0 = 1.0 + a1 + a2
    return np.array([b0]), np.array([1.0, a1, a2])


def apply_vocal_tract(
    exc: np.ndarray,
    tracks: Dict[str, np.ndarray],
    sr: int,
) -> np.ndarray:
    """Run the excitation through the moving formant cascade.

    Coefficients are frozen inside one 5 ms frame and the filter state is
    carried across frames, which is the standard way to run a time-varying
    formant synthesiser without clicks at the boundaries.
    """
    hop = int(tracks["hop"][0])
    freqs = tracks["freqs"]
    bws = tracks["bws"]
    n_frames, n_formants = freqs.shape
    n = min(exc.size, n_frames * hop)
    out = np.zeros(n, dtype=np.float64)

    if _lfilter is None:  # pragma: no cover - scipy is present in this project
        return exc[:n]

    states = [np.zeros(2) for _ in range(n_formants)]
    for fr in range(n_frames):
        start = fr * hop
        end = min(start + hop, n)
        if start >= n:
            break
        block = exc[start:end]
        for j in range(n_formants):
            b, a = _resonator(float(freqs[fr, j]), float(bws[fr, j]), sr)
            block, states[j] = _lfilter(b, a, block, zi=states[j])
        out[start:end] = block
    return out


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def synthesize_turn(
    text: str,
    sr: int = TARGET_SR,
    voice: Optional[Dict[str, float]] = None,
    synthetic: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """Render one turn of speech-like audio.

    Deterministic for a given (text, voice, synthetic, seed). Returns float32
    samples in [-1, 1], loudness normalised so downstream detector scores stay
    comparable across turns.
    """
    voice = dict(voice or make_voice("spk_default", seed))
    if synthetic:
        voice.update(SYNTHETIC_OVERRIDES)
    rng = _rng_from("turn", text, voice.get("speaker_id", "?"), int(synthetic), seed)

    plan = plan_segments(text, voice, synthetic, rng)
    tracks = build_tracks(plan, sr, voice, synthetic, rng)
    exc = build_excitation(tracks, sr, voice, synthetic, rng)
    y = apply_vocal_tract(exc, tracks, sr)

    tilt = float(voice["tilt"]) + (SYNTHETIC_TILT_EXTRA if synthetic else 0.0)
    if tilt > 0:
        y = _one_pole(y, min(0.85, tilt))
    y = y - _one_pole(y, 0.995)              # DC block
    if not synthetic:
        # A touch of wideband room noise. Real recordings are never this clean.
        y = y + rng.normal(0.0, 0.0015, y.size)

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = rms_normalize(y.astype(np.float32), target_dbfs=-20.0)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def voice_summary(voice: Dict[str, float]) -> Dict[str, float]:
    """The subset of the voice worth recording in a manifest."""
    keys = ("f0", "vt_scale", "bw_scale", "tilt", "breath", "rate", "jitter", "shimmer")
    return {k: round(float(voice[k]), 5) for k in keys if k in voice}


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------


def _measure(x: np.ndarray, sr: int) -> Dict[str, float]:
    """Crude jitter, shimmer, HNR and flatness variance, for the self-test.

    The real versions live in dsp/prosody.py and antispoof/features.py. This
    is only here so the module can prove, without importing them, that the two
    paths really do separate.
    """
    frame = int(0.030 * sr)
    hop = int(0.010 * sr)
    n = 1 + max(0, (x.size - frame) // hop)
    window = np.hanning(frame)
    flat: List[float] = []
    # (frame index, f0, frame energy, periodicity) for accepted voiced frames.
    voiced: List[Tuple[int, float, float, float]] = []

    for i in range(n):
        seg = x[i * hop:i * hop + frame]
        if seg.size < frame:
            break
        spec = np.abs(np.fft.rfft(seg * window)) ** 2 + 1e-12
        flat.append(float(np.exp(np.mean(np.log(spec))) / np.mean(spec)))

        energy = float(np.sqrt(np.mean(seg ** 2)))
        if energy < 1e-3:
            continue
        ac = np.correlate(seg, seg, mode="full")[frame - 1:]
        lo, hi = int(sr / 300), int(sr / 70)
        if hi >= ac.size or ac[0] <= 0:
            continue
        lag = int(np.argmax(ac[lo:hi])) + lo
        r = float(ac[lag] / ac[0])
        if r <= 0.35 or not (0 < lag < ac.size - 1):
            continue
        # Parabolic peak interpolation. Without it the integer lag quantises
        # F0 at about 2 percent, which swamps the jitter being measured.
        y0, y1, y2 = ac[lag - 1], ac[lag], ac[lag + 1]
        denom = y0 - 2.0 * y1 + y2
        shift = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        voiced.append((i, sr / (lag + float(np.clip(shift, -1.0, 1.0))), energy, r))

    # Only adjacent frames are comparable. Differencing across a gap measures
    # the segment boundary, not the cycle to cycle variation.
    d_f0: List[float] = []
    for (i0, f0_0, _, _), (i1, f0_1, _, _) in zip(voiced, voiced[1:]):
        if i1 == i0 + 1:
            d_f0.append(abs(f0_1 - f0_0) / (0.5 * (f0_1 + f0_0) + 1e-9))

    # Shimmer has to be period synchronous. Frame RMS aliases against the
    # pitch period (a 30 ms frame holds four or five pulses, and whether it is
    # four or five swings the energy by a tenth), which hits a perfectly
    # regular signal harder than a jittered one and inverts the comparison.
    d_amp: List[float] = []
    energies = [e for _, _, e, _ in voiced]
    if energies:
        gate = 0.6 * float(np.median(energies))
        period = max(8, int(round(sr / float(np.median([f for _, f, _, _ in voiced])))))
        prev_peak = None
        for start in range(0, max(0, x.size - period), period):
            w = x[start:start + period]
            if float(np.sqrt(np.mean(w ** 2))) < gate:
                prev_peak = None
                continue
            peak = float(np.max(np.abs(w)))
            if prev_peak and peak > 0:
                d_amp.append(abs(peak - prev_peak) / (0.5 * (peak + prev_peak)))
            prev_peak = peak

    def mean(v: List[float]) -> float:
        return float(np.mean(v)) if len(v) >= 3 else 0.0

    return {
        "jitter": round(mean(d_f0), 5),
        "shimmer": round(mean(d_amp), 5),
        "periodicity": round(
            float(np.mean([r for _, _, _, r in voiced])) if voiced else 0.0, 5
        ),
        "flatness_var": round(float(np.std(flat)) if flat else 0.0, 5),
        "voiced_frames": len(voiced),
    }


if __name__ == "__main__":
    import time

    sr = TARGET_SR
    text = "Main XYZ Bank se bol raha hoon, aapka OTP 445566 abhi bataiye"
    sustained = "aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa"
    for spk in ("spk_00", "spk_07"):
        voice = make_voice(spk, seed=20230100)
        print(f"\nvoice {spk}: {voice_summary(voice)}")
        for label_text, sample in (("sentence", text), ("sustained", sustained)):
            for synthetic in (False, True):
                t0 = time.time()
                y = synthesize_turn(sample, sr, voice, synthetic=synthetic, seed=1)
                dt = time.time() - t0
                label = "cloned" if synthetic else "human "
                m = _measure(y, sr)
                print(
                    f"  {label_text:9s} {label} {y.size / sr:5.2f}s "
                    f"{dt * 1000:6.1f}ms  jitter={m['jitter']:.4f} "
                    f"shimmer={m['shimmer']:.4f} period={m['periodicity']:.4f} "
                    f"flatvar={m['flatness_var']:.4f}"
                )

    a = synthesize_turn(text, sr, make_voice("spk_00", 1), False, seed=5)
    b = synthesize_turn(text, sr, make_voice("spk_00", 1), False, seed=5)
    print(f"\ndeterministic: {bool(np.array_equal(a, b))}")
    c = synthesize_turn(text, sr, make_voice("spk_03", 1), False, seed=5)
    print(f"different speaker differs: {not np.array_equal(a, c[:a.size])}")
