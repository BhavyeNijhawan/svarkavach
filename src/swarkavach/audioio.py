"""Audio input and output with no hard dependency on soundfile or librosa.

The stdlib `wave` module handles the 16-bit PCM WAV files this project
produces and consumes. soundfile is used when it happens to be installed
because it also opens FLAC and OGG, but nothing here breaks without it.

Resampling is a polyphase-free windowed-sinc implementation in numpy. It is
slower than libsamplerate on a long file and perfectly adequate for the 10 to
60 second calls this system handles.
"""

from __future__ import annotations

import io
import os
import wave
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from .config import TARGET_SR

PathLike = Union[str, os.PathLike, Path]

try:  # optional, widens the set of readable formats
    import soundfile as _sf  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    _sf = None


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def _read_wav_stdlib(path: PathLike) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        n_ch = wf.getnchannels()
        width = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if width == 1:
        # 8-bit WAV is unsigned
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        as_int = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        as_int = np.where(as_int & 0x800000, as_int - 0x1000000, as_int)
        data = as_int.astype(np.float32) / 8388608.0
    else:
        raise ValueError(f"unsupported sample width: {width} bytes")

    if n_ch > 1:
        data = data.reshape(-1, n_ch).mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), sr


def read_audio(
    path: PathLike,
    sr: Optional[int] = TARGET_SR,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """Load an audio file as float32 in [-1, 1].

    Returns (samples, sample_rate). If `sr` is given the signal is resampled
    to it; pass sr=None to keep the file's native rate.
    """
    path = str(path)
    ext = os.path.splitext(path)[1].lower()

    if ext == ".wav":
        try:
            x, native_sr = _read_wav_stdlib(path)
        except (wave.Error, EOFError, ValueError):
            if _sf is None:
                raise
            x, native_sr = _sf.read(path, dtype="float32", always_2d=False)
            x = np.asarray(x, dtype=np.float32)
            if x.ndim > 1 and mono:
                x = x.mean(axis=1)
    else:
        if _sf is None:
            raise RuntimeError(
                f"cannot read {ext} without soundfile installed. "
                "Convert to 16-bit PCM WAV first, or pip install soundfile."
            )
        x, native_sr = _sf.read(path, dtype="float32", always_2d=False)
        x = np.asarray(x, dtype=np.float32)
        if x.ndim > 1 and mono:
            x = x.mean(axis=1)

    if sr is not None and native_sr != sr:
        x = resample(x, native_sr, sr)
        native_sr = sr
    return np.ascontiguousarray(x, dtype=np.float32), int(native_sr)


def read_audio_bytes(
    raw: bytes,
    sr: Optional[int] = TARGET_SR,
) -> Tuple[np.ndarray, int]:
    """Same as read_audio but from an in-memory buffer (browser uploads)."""
    bio = io.BytesIO(raw)
    try:
        with wave.open(bio, "rb") as wf:
            n_ch = wf.getnchannels()
            width = wf.getsampwidth()
            native_sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        if width != 2:
            raise wave.Error("non 16-bit, fall through")
        x = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        if n_ch > 1:
            x = x.reshape(-1, n_ch).mean(axis=1)
    except Exception:
        if _sf is None:
            raise RuntimeError("upload is not 16-bit PCM WAV and soundfile is missing")
        bio.seek(0)
        x, native_sr = _sf.read(bio, dtype="float32", always_2d=False)
        x = np.asarray(x, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)

    if sr is not None and native_sr != sr:
        x = resample(x, native_sr, sr)
        native_sr = sr
    return np.ascontiguousarray(x, dtype=np.float32), int(native_sr)


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def write_wav(path: PathLike, x: np.ndarray, sr: int = TARGET_SR, peak: float = 0.95) -> str:
    """Write float samples as a 16-bit PCM WAV, peak-normalised to `peak`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=np.float32).ravel()
    if x.size == 0:
        x = np.zeros(1, dtype=np.float32)
    m = float(np.max(np.abs(x)))
    if m > 0 and peak is not None:
        x = x * (peak / m)
    pcm = np.clip(x * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())
    return str(path)


def wav_bytes(x: np.ndarray, sr: int = TARGET_SR, peak: float = 0.95) -> bytes:
    """Serialise samples to a WAV byte string, for streaming to the browser."""
    x = np.asarray(x, dtype=np.float32).ravel()
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m > 0 and peak is not None:
        x = x * (peak / m)
    pcm = np.clip(x * 32767.0, -32768, 32767).astype("<i2")
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())
    return bio.getvalue()


# --------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------


def _sinc_kernel(ratio: float, half_width: int = 16, beta: float = 8.6) -> np.ndarray:
    """Kaiser-windowed sinc low-pass, cut off at the lower of the two rates."""
    cutoff = min(1.0, ratio) * 0.98
    n = np.arange(-half_width, half_width + 1, dtype=np.float64)
    h = cutoff * np.sinc(cutoff * n)
    h *= np.kaiser(h.size, beta)
    s = h.sum()
    return (h / s).astype(np.float32) if s else h.astype(np.float32)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample by band-limited interpolation.

    Anti-aliases first when downsampling, then does cubic-free linear
    interpolation on the filtered signal. At the ratios this project uses
    (48k/44.1k/22.05k/16k down to 8k) the residual aliasing sits well below
    the level of the codec artefacts we deliberately add later.
    """
    x = np.asarray(x, dtype=np.float32).ravel()
    if sr_in == sr_out or x.size == 0:
        return x

    ratio = float(sr_out) / float(sr_in)
    if ratio < 1.0:  # downsampling: low-pass before decimating
        h = _sinc_kernel(ratio)
        x = np.convolve(x, h, mode="same").astype(np.float32)

    n_out = int(np.floor(x.size * ratio))
    if n_out <= 1:
        return np.zeros(max(n_out, 1), dtype=np.float32)

    src_idx = np.arange(n_out, dtype=np.float64) / ratio
    i0 = np.floor(src_idx).astype(np.int64)
    i1 = np.minimum(i0 + 1, x.size - 1)
    frac = (src_idx - i0).astype(np.float32)
    y = x[i0] * (1.0 - frac) + x[i1] * frac

    if ratio > 1.0:  # upsampling: smooth the interpolation artefacts
        h = _sinc_kernel(1.0 / ratio)
        y = np.convolve(y, h, mode="same").astype(np.float32)
    return np.ascontiguousarray(y, dtype=np.float32)


# --------------------------------------------------------------------------
# Small helpers used all over the pipeline
# --------------------------------------------------------------------------


def normalize(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return (x * (peak / m)).astype(np.float32) if m > 0 else x.astype(np.float32)


def rms_normalize(x: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    """Loudness-match a signal so codec and detector scores stay comparable."""
    r = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
    if r <= 1e-9:
        return x.astype(np.float32)
    target = 10.0 ** (target_dbfs / 20.0)
    y = x * (target / r)
    m = float(np.max(np.abs(y)))
    if m > 0.99:
        y = y * (0.99 / m)
    return y.astype(np.float32)


def duration_s(x: np.ndarray, sr: int) -> float:
    return float(x.size) / float(sr) if sr else 0.0


def pad_or_trim(x: np.ndarray, n: int) -> np.ndarray:
    """Fixed-length view of a signal, for batched neural inference."""
    if x.size == n:
        return x
    if x.size > n:
        return x[:n]
    reps = int(np.ceil(n / max(x.size, 1)))
    return np.tile(x, reps)[:n].astype(np.float32)


def concat_with_gaps(
    segments: list, sr: int, gap_s: float = 0.25
) -> Tuple[np.ndarray, list]:
    """Join per-turn audio into one call, returning (signal, [(start, end)]).

    The gaps are what make turn timings meaningful, which the streaming
    simulation and the time-to-detection metric both depend on.
    """
    gap = np.zeros(int(round(gap_s * sr)), dtype=np.float32)
    out = []
    spans = []
    cursor = 0.0
    for i, seg in enumerate(segments):
        seg = np.asarray(seg, dtype=np.float32).ravel()
        start = cursor
        out.append(seg)
        cursor += seg.size / float(sr)
        spans.append((round(start, 3), round(cursor, 3)))
        if i != len(segments) - 1:
            out.append(gap)
            cursor += gap.size / float(sr)
    signal = np.concatenate(out) if out else np.zeros(1, dtype=np.float32)
    return signal.astype(np.float32), spans
