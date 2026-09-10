"""Audio rendering for a Call, with pluggable back ends.

The default back end, "sim", is the formant synthesiser in synth.py. It always
works, needs no download and no network, and produces the paired human-sounding
and cloned-sounding audio the anti-spoofing branch trains on. Every other back
end is optional and guarded: if the package is missing the renderer says so and
falls back to "sim" rather than failing the run.

| back end | needs                | offline | notes                          |
|----------|----------------------|---------|--------------------------------|
| sim      | nothing              | yes     | default, always available      |
| sapi     | pyttsx3              | yes     | Windows SAPI5 voices           |
| edge     | edge-tts             | no      | needs network, never automatic |
| xtts     | Coqui XTTS on Colab  | no      | stub, points at the notebook   |

Who is cloned: a fraudster using a voice clone is the caller, and the person
being called is real. So when call.label_voice is "synthetic" the caller turns
are rendered on the cloned path and the callee turns stay on the human path.
That gives the within-call speaker-consistency feature something real to
measure, and the detail dictionary records the flag per turn so a downstream
branch can score caller turns only if it wants to.

Turn timings are written back onto the Call from the real audio spans, so the
streaming simulation and the time-to-detection metric line up with the WAV.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..audioio import concat_with_gaps, read_audio, rms_normalize, write_wav
from ..config import CORPUS_AUDIO_DIR, SETTINGS, TARGET_SR
from ..schema import COERCION_RANK, Call
from .synth import SYNTHETIC_OVERRIDES, make_voice, synthesize_turn, voice_summary

#: Silence inserted between turns. Matches generator.TURN_GAP_S.
TURN_GAP_S = 0.25

XTTS_MESSAGE = (
    "The xtts back end does not run locally in this project. Voice cloning "
    "with Coqui XTTS needs a GPU and a model download, so it lives in "
    "notebooks/05_voice_cloning.ipynb on Colab. Render there, drop the WAVs "
    "into data/corpus/audio/, and load them with load_corpus(with_audio=True). "
    "For a local run use backend='sim'."
)


# --------------------------------------------------------------------------
# Back end discovery
# --------------------------------------------------------------------------


def _has(module: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except Exception:  # pragma: no cover - broken installs
        return False


def AVAILABLE_BACKENDS() -> Dict[str, bool]:
    """Which back ends this machine can actually run right now."""
    return {
        "sim": True,
        "sapi": _has("pyttsx3"),
        "edge": _has("edge_tts"),
        "xtts": False,
    }


def backend_notes() -> Dict[str, str]:
    return {
        "sim": "built-in source-filter synthesiser, offline, always available",
        "sapi": "pyttsx3 over Windows SAPI5, offline, install pyttsx3 to enable",
        "edge": "edge-tts, needs a network connection, never used automatically",
        "xtts": "Coqui XTTS, Colab only, see notebooks/05_voice_cloning.ipynb",
    }


# --------------------------------------------------------------------------
# Back end implementations
# --------------------------------------------------------------------------


def _render_sim(
    call: Call,
    sr: int,
    seed: int,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    caller_voice = make_voice(call.speaker_id, seed)
    callee_voice = make_voice(f"{call.speaker_id}_callee", seed)
    caller_is_synthetic = call.label_voice == "synthetic"

    segments: List[np.ndarray] = []
    per_turn: List[Dict[str, Any]] = []
    for turn in call.turns:
        is_caller = turn.speaker == "caller"
        voice = caller_voice if is_caller else callee_voice
        synthetic = caller_is_synthetic and is_caller
        # How worked up this turn should sound, taken from the dialogue act
        # rather than from the words. Keeping it on the act matters: the
        # prosody-intent mismatch feature builds lexical arousal from the fraud
        # lexicon and the intent model, so driving the rendering from that same
        # lexicon would make the evaluation circular. The act is a related but
        # separate quantity, which is also the relationship a real call has.
        arousal = float(COERCION_RANK.get(turn.act, 0.25))
        if not is_caller:
            arousal = 0.30 if turn.act == "VICTIM_RESIST" else 0.18
        seg = synthesize_turn(
            turn.text,
            sr=sr,
            voice=voice,
            synthetic=synthetic,
            seed=seed + turn.index,
            arousal=arousal,
        )
        segments.append(seg)
        per_turn.append(
            {
                "index": turn.index,
                "speaker": turn.speaker,
                "synthetic": bool(synthetic),
                "arousal": round(arousal, 3),
                "n_samples": int(seg.size),
            }
        )
    # Report the voice as it was actually used: on the cloned path the jitter,
    # shimmer and breath values are overridden inside synthesize_turn.
    effective_caller = dict(caller_voice)
    if caller_is_synthetic:
        effective_caller.update(SYNTHETIC_OVERRIDES)
    detail = {
        "caller_voice": voice_summary(effective_caller),
        "callee_voice": voice_summary(callee_voice),
        "caller_synthetic": bool(caller_is_synthetic),
        "turns": per_turn,
    }
    return segments, detail


def _render_sapi(call: Call, sr: int) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Windows SAPI5 through pyttsx3. Offline, but voices vary by machine."""
    import pyttsx3  # type: ignore

    engine = pyttsx3.init()
    voices = list(getattr(engine, "getProperty")("voices") or [])
    caller_id = voices[0].id if voices else None
    callee_id = voices[1].id if len(voices) > 1 else caller_id

    segments: List[np.ndarray] = []
    tmpdir = Path(tempfile.mkdtemp(prefix="swarkavach_sapi_"))
    try:
        for turn in call.turns:
            if caller_id:
                engine.setProperty(
                    "voice", caller_id if turn.speaker == "caller" else callee_id
                )
            path = tmpdir / f"turn_{turn.index:03d}.wav"
            engine.save_to_file(turn.text, str(path))
            engine.runAndWait()
            if not path.exists():
                raise RuntimeError("pyttsx3 produced no file")
            x, _ = read_audio(path, sr=sr)
            segments.append(rms_normalize(x, target_dbfs=-20.0))
    finally:
        for p in tmpdir.glob("*"):
            try:
                p.unlink()
            except OSError:  # pragma: no cover - Windows file locks
                pass
        try:
            tmpdir.rmdir()
        except OSError:  # pragma: no cover
            pass
    return segments, {"voices": len(voices)}


def _render_edge(call: Call, sr: int, seed: int = 0) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Real neural voices from Microsoft Edge. Needs a network connection.

    This is what makes the audio say the words on the screen. The `sim` back
    end produces the right acoustic statistics but no intelligible speech,
    which is fine for training a detector and useless for a demo.

    The cloned cells run through an LPC vocoder re-synthesis on the CALLER's
    turns only, since it is the caller whose voice is supposedly cloned. That
    imposes the low jitter, low shimmer and high harmonic-to-noise ratio that
    real vocoder-based cloning leaves, while keeping the words intelligible.
    Both classes are machine generated here, so see the module docstring in
    neural_tts.py for what the local anti-spoofing numbers do and do not mean.
    """
    from . import neural_tts as N

    caller_v = N.voice_for_speaker(call.speaker_id, seed)
    callee_v = N.voice_for_speaker(f"{call.speaker_id}_callee", seed)
    caller_is_synthetic = call.label_voice == "synthetic"

    items = []
    for turn in call.turns:
        v = caller_v if turn.speaker == "caller" else callee_v
        items.append((N.prepare_text(turn.text, v["script"]),
                      v["voice"], v["rate"], v["pitch"]))

    rendered = N.synth_batch(items, sr=sr)
    n_failed = sum(1 for r in rendered if r is None)

    segments: List[np.ndarray] = []
    per_turn: List[Dict[str, Any]] = []
    for turn, seg in zip(call.turns, rendered):
        if seg is None:
            # a dropped request must not shift every later turn's timing, so
            # it becomes a short silence rather than disappearing
            seg = np.zeros(int(0.5 * sr), dtype=np.float32)
        is_caller = turn.speaker == "caller"
        vocoded = bool(caller_is_synthetic and is_caller)
        if vocoded:
            seg = N.vocoder_artifacts(seg, sr)
        segments.append(rms_normalize(np.asarray(seg, dtype=np.float32), target_dbfs=-20.0))
        per_turn.append({
            "index": turn.index, "speaker": turn.speaker,
            "synthetic": vocoded, "n_samples": int(segments[-1].size),
        })

    return segments, {
        "caller_voice": {k: caller_v[k] for k in ("voice", "rate", "pitch", "script")},
        "callee_voice": {k: callee_v[k] for k in ("voice", "rate", "pitch", "script")},
        "caller_synthetic": bool(caller_is_synthetic),
        "cloned_via": "lpc_vocoder_resynthesis" if caller_is_synthetic else None,
        "turns": per_turn,
        "n_failed": n_failed,
        "errors": list(N.FAILURES)[:3],
    }


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def render_call_audio(
    call: Call,
    backend: str = "sim",
    out_path: Optional[os.PathLike] = None,
    sr: int = TARGET_SR,
    seed: Optional[int] = None,
    write: bool = True,
    gap_s: float = TURN_GAP_S,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Render a whole call and write the turn timings back onto it.

    Returns (signal, info). info records the back end actually used, any
    fallback and why, the per-turn spans and the voice parameters, so a
    manifest can say exactly how each WAV was produced.
    """
    seed = int(SETTINGS.pipeline.seed if seed is None else seed)
    backend = (backend or "sim").lower()
    available = AVAILABLE_BACKENDS()
    requested = backend
    fallback_reason = None

    if backend == "xtts":
        raise RuntimeError(XTTS_MESSAGE)
    if backend not in available:
        raise ValueError(
            f"unknown backend {backend!r}, expected one of {sorted(available)}"
        )
    if not available[backend]:
        fallback_reason = f"{backend} is not installed"
        backend = "sim"

    detail: Dict[str, Any] = {}
    if backend != "sim":
        try:
            if backend == "sapi":
                segments, detail = _render_sapi(call, sr)
            else:
                segments, detail = _render_edge(call, sr, seed)
        except Exception as exc:  # any optional back end may fail at runtime
            fallback_reason = f"{backend} failed: {exc}"
            backend = "sim"
    if backend == "sim":
        segments, detail = _render_sim(call, sr, seed)

    signal, spans = concat_with_gaps(segments, sr, gap_s=gap_s)
    for turn, (start, end) in zip(call.turns, spans):
        turn.t_start = float(start)
        turn.t_end = float(end)

    path: Optional[Path] = None
    if write:
        path = Path(out_path) if out_path is not None else (
            Path(CORPUS_AUDIO_DIR) / f"{call.call_id}.wav"
        )
        write_wav(path, signal, sr)
        call.audio_path = str(path)
        call.sample_rate = int(sr)
        call.audio_source = backend
        call.meta["timing_source"] = "audio"

    info: Dict[str, Any] = {
        "backend": backend,
        "requested_backend": requested,
        "fallback_reason": fallback_reason,
        "sample_rate": int(sr),
        "duration_s": round(float(signal.size) / float(sr), 3),
        "n_turns": len(call.turns),
        "gap_s": float(gap_s),
        "seed": int(seed),
        "out_path": str(path) if path is not None else None,
        "spans": [(float(a), float(b)) for a, b in spans],
        "detail": detail,
    }
    return signal, info


def render_corpus_audio(
    calls,
    backend: str = "sim",
    sr: int = TARGET_SR,
    seed: Optional[int] = None,
    out_dir: Optional[os.PathLike] = None,
    progress: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Render every call in a list. Returns call_id -> render info."""
    out_dir = Path(out_dir or CORPUS_AUDIO_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Dict[str, Any]] = {}
    for i, call in enumerate(calls):
        _, info = render_call_audio(
            call, backend=backend, out_path=out_dir / f"{call.call_id}.wav",
            sr=sr, seed=seed,
        )
        info.pop("spans", None)
        out[call.call_id] = info
        if progress and (i + 1) % 25 == 0:
            print(f"  rendered {i + 1}/{len(calls)}")
    return out


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------


if __name__ == "__main__":
    import json
    import time

    from .generator import generate_corpus

    print("backends:")
    for name, ok in AVAILABLE_BACKENDS().items():
        print(f"  {name:6s} {'yes' if ok else 'no ':3s}  {backend_notes()[name]}")

    calls = generate_corpus(n_calls=8, write=False)
    pair = [c for c in calls if c.label_voice == "synthetic"][:1]
    pair += [c for c in calls if c.label_voice == "human"][:1]

    tmp = Path(tempfile.gettempdir()) / "swarkavach_tts_selftest"
    tmp.mkdir(parents=True, exist_ok=True)
    for call in pair:
        t0 = time.time()
        x, info = render_call_audio(
            call, backend="sim", out_path=tmp / f"{call.call_id}.wav"
        )
        dt = time.time() - t0
        last = call.turns[-1].t_end
        wav_s = x.size / info["sample_rate"]
        print(
            f"\n{call.call_id} voice={call.label_voice} scam={call.label_scam} "
            f"turns={len(call.turns)}"
        )
        print(f"  rendered in {dt:.2f}s, wav {wav_s:.2f}s, last t_end {last:.2f}s, "
              f"delta {abs(wav_s - last):.3f}s")
        print(f"  caller voice: {json.dumps(info['detail']['caller_voice'])}")
        print(f"  wrote {info['out_path']}")

    try:
        render_call_audio(pair[0], backend="xtts", write=False)
    except RuntimeError as exc:
        print(f"\nxtts raises as designed: {str(exc)[:60]} ...")
