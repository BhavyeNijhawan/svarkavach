"""Flask API behind the SwarKavach console.

Everything heavy lives in swarkavach.pipeline. This file is transport: it
loads the engine once, keeps a small cache of decoded audio, and streams
per-turn verdicts over server-sent events for the live monitor.

Run it with:  swarkavach serve      (or: python server/app.py)
"""

from __future__ import annotations

import json
import io
import os
import sys
import time
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

# make `src` importable when this file is run directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

from swarkavach import config
from swarkavach.audioio import read_audio, wav_bytes, read_audio_bytes, write_wav
from swarkavach.schema import (
    Call, FUSION_FEATURES, FEATURE_GROUPS, FEATURE_LABELS, risk_band,
)

STATIC = Path(__file__).resolve().parent / "static"
app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

# --------------------------------------------------------------------------
# Engine, loaded once and shared
# --------------------------------------------------------------------------

_ENGINE = None
_ENGINE_ERR: Optional[str] = None
_LOCK = threading.Lock()


def engine():
    """Load the pipeline on first use so an import error shows up as JSON."""
    global _ENGINE, _ENGINE_ERR
    if _ENGINE is not None or _ENGINE_ERR is not None:
        if _ENGINE_ERR:
            raise RuntimeError(_ENGINE_ERR)
        return _ENGINE
    with _LOCK:
        if _ENGINE is None and _ENGINE_ERR is None:
            try:
                from swarkavach.pipeline import Pipeline
                _ENGINE = Pipeline.load()
            except Exception as exc:  # surfaced to the browser, not swallowed
                _ENGINE_ERR = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                raise RuntimeError(_ENGINE_ERR) from exc
    if _ENGINE_ERR:
        raise RuntimeError(_ENGINE_ERR)
    return _ENGINE


# --------------------------------------------------------------------------
# Small caches
# --------------------------------------------------------------------------

_AUDIO_CACHE: Dict[str, tuple] = {}
_CACHE_ORDER: list = []
_CACHE_MAX = 24


def get_audio(call_id: str, codec: str = "clean", snr: Optional[float] = None):
    """Decoded, channel-degraded audio for a call. Cached by (id, codec, snr)."""
    key = f"{call_id}|{codec}|{snr}"
    if key in _AUDIO_CACHE:
        return _AUDIO_CACHE[key]

    call = engine().get_call(call_id)
    if call is None or not call.audio_path or not Path(call.audio_path).exists():
        raise FileNotFoundError(f"no audio on disk for {call_id}")

    x, sr = read_audio(call.audio_path, sr=config.TARGET_SR)
    info: Dict[str, Any] = {"codec": "clean", "real_codec": True}
    if codec and codec != "clean":
        from swarkavach.channel import degrade
        x, info = degrade(x, sr, codec=codec, snr_db=snr, seed=config.SETTINGS.pipeline.seed)

    val = (x, sr, info)
    _AUDIO_CACHE[key] = val
    _CACHE_ORDER.append(key)
    while len(_CACHE_ORDER) > _CACHE_MAX:
        _AUDIO_CACHE.pop(_CACHE_ORDER.pop(0), None)
    return val


def _f(v, default=None):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# Static
# --------------------------------------------------------------------------


@app.route("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.after_request
def no_store(resp):
    if request.path.startswith("/static/") or request.path == "/":
        resp.headers["Cache-Control"] = "no-store"
    return resp


# --------------------------------------------------------------------------
# Meta
# --------------------------------------------------------------------------


@app.get("/api/health")
def health():
    out: Dict[str, Any] = {
        "version": config.PROJECT_VERSION,
        "python": sys.version.split()[0],
        "ffmpeg": bool(config.find_ffmpeg()),
        "alert_threshold": config.SETTINGS.pipeline.alert_threshold,
        "seed": config.SETTINGS.pipeline.seed,
        "ready": False,
    }
    try:
        import torch
        out["torch"] = torch.__version__
    except Exception:
        out["torch"] = "not installed"
    try:
        e = engine()
        out.update(e.status())
        out["ready"] = True
    except Exception as exc:
        out["error"] = str(exc)
    return jsonify(out)


@app.get("/api/meta")
def meta():
    feats = [
        {"name": n, "group": FEATURE_GROUPS[n], "label": FEATURE_LABELS[n]}
        for n, _, _, _ in [(f[0], f[1], f[2], f[3]) for f in FUSION_FEATURES]
    ]
    out: Dict[str, Any] = {"fusion_features": feats}
    ref_path = config.RESULTS_DIR / "feature_reference.json"
    if ref_path.exists():
        try:
            out["feature_reference"] = json.loads(ref_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return jsonify(out)


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------


@app.get("/api/corpus")
def corpus():
    try:
        e = engine()
    except Exception as exc:
        return jsonify({"error": str(exc), "calls": [], "stats": None}), 200

    rows = []
    for c in e.calls:
        ents = c.entities()
        rows.append({
            "call_id": c.call_id,
            "scenario": c.scenario,
            "cell": c.cell,
            "split": c.split,
            "speaker_id": c.speaker_id,
            "label_scam": c.label_scam,
            "label_voice": c.label_voice,
            "n_turns": len(c.turns),
            "n_entities": len(ents),
            "duration": round(c.duration, 2),
            "has_audio": bool(c.audio_path and Path(c.audio_path).exists()),
            "cmi": _f(c.meta.get("cmi"), None),
        })
    stats_path = config.RESULTS_DIR / "corpus_stats.json"
    stats = None
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception:
            stats = None
    return jsonify({"calls": rows, "stats": stats})


@app.get("/api/call/<call_id>")
def get_call(call_id):
    try:
        c = engine().get_call(call_id)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    if c is None:
        return jsonify({"error": "call not found"}), 404
    return jsonify(c.to_dict())


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------


@app.get("/api/audio/<call_id>")
def audio(call_id):
    codec = request.args.get("codec", "clean")
    snr = _f(request.args.get("snr"), None)
    try:
        x, sr, _ = get_audio(call_id, codec, snr)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return Response(wav_bytes(x, sr), mimetype="audio/wav",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/waveform/<call_id>")
def waveform(call_id):
    codec = request.args.get("codec", "clean")
    snr = _f(request.args.get("snr"), None)
    try:
        x, sr, info = get_audio(call_id, codec, snr)
    except Exception as exc:
        return jsonify({"error": str(exc), "peaks": [], "duration": 0}), 200

    n_bins = 520
    step = max(1, x.size // n_bins)
    trimmed = x[: step * n_bins]
    peaks = np.abs(trimmed.reshape(-1, step)).max(axis=1) if trimmed.size else np.zeros(1)
    m = float(peaks.max()) or 1.0
    return jsonify({
        "peaks": [round(float(p / m), 4) for p in peaks],
        "duration": round(x.size / sr, 3),
        "sr": sr,
        "channel": info,
    })


@app.get("/api/spectrogram/<call_id>")
def spectrogram(call_id):
    codec = request.args.get("codec", "clean")
    snr = _f(request.args.get("snr"), None)
    try:
        x, sr, _ = get_audio(call_id, codec, snr)
        from swarkavach.dsp.spectral import spectrogram_db
        S = spectrogram_db(x, config.SETTINGS.frame, top_db=80)   # (frames, bins)
    except Exception as exc:
        return jsonify({"error": str(exc), "db": []}), 200

    S = np.asarray(S, dtype=np.float32)
    max_cols = 620
    if S.shape[0] > max_cols:
        idx = np.linspace(0, S.shape[0] - 1, max_cols).astype(int)
        S = S[idx]
    max_rows = 160
    if S.shape[1] > max_rows:
        idx = np.linspace(0, S.shape[1] - 1, max_rows).astype(int)
        S = S[:, idx]

    db = S.T                                # (freq, time), low frequency first
    return jsonify({
        "db": [[round(float(v), 1) for v in row] for row in db],
        "vmin": float(np.min(db)), "vmax": float(np.max(db)),
        "duration": round(x.size / sr, 3), "sr": sr,
    })


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


@app.post("/api/analyze")
def analyze():
    body = request.get_json(silent=True) or {}
    call_id = body.get("call_id")
    codec = body.get("codec", "clean")
    snr = _f(body.get("snr"), None)
    if not call_id:
        return jsonify({"error": "call_id is required"}), 400
    try:
        e = engine()
        call = e.get_call(call_id)
        if call is None:
            return jsonify({"error": "call not found"}), 404
        x = sr = None
        if call.audio_path and Path(call.audio_path).exists():
            x, sr, _ = get_audio(call_id, codec, snr)
        v = e.analyze(call, audio=x, sr=sr, streaming=True)
        out = v.to_dict()
        call_d = call.to_dict()
        out["call"] = call_d
        # the console renders from `turns` directly, so hand it the same list
        # rather than making it dig into `call`
        out["turns"] = call_d["turns"]
        return jsonify(out)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


@app.post("/api/simulate")
def simulate():
    """Generate a brand new call on the spot, so a demo is never a replay."""
    body = request.get_json(silent=True) or {}
    try:
        e = engine()
        call = e.synthesize_call(
            scenario=body.get("scenario"),
            scam=body.get("scam"),
            voice=body.get("voice"),
            with_audio=bool(body.get("audio", True)),
            seed=body.get("seed"),
        )
        ents = call.entities()
        return jsonify({
            "summary": {
                "call_id": call.call_id, "scenario": call.scenario, "cell": call.cell,
                "split": call.split, "speaker_id": call.speaker_id,
                "label_scam": call.label_scam, "label_voice": call.label_voice,
                "n_turns": len(call.turns), "n_entities": len(ents),
                "duration": round(call.duration, 2),
                "has_audio": bool(call.audio_path),
                "cmi": _f(call.meta.get("cmi"), None),
            },
            "call": call.to_dict(),
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


@app.post("/api/upload")
def upload():
    f = request.files.get("audio")
    if f is None:
        return jsonify({"error": "no audio file in the request"}), 400
    try:
        raw = f.read()
        x, sr = read_audio_bytes(raw, sr=config.TARGET_SR)
        e = engine()
        call = e.register_upload(
            x, sr,
            name=os.path.basename(f.filename or "upload.wav"),
            transcript=request.form.get("transcript") or None,
        )
        return jsonify({"call_id": call.call_id, "duration": round(call.duration, 2)})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


# --------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------


def sse(event: str, payload: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


@app.get("/api/stream/<call_id>")
def stream(call_id):
    codec = request.args.get("codec", "clean")
    speed = _f(request.args.get("speed"), 1.0) or 1.0
    snr = _f(request.args.get("snr"), None)

    def gen():
        try:
            e = engine()
            call = e.get_call(call_id)
            if call is None:
                yield sse("failed", {"error": "call not found"})
                return
            x = sr = None
            note = "clean channel"
            if call.audio_path and Path(call.audio_path).exists():
                x, sr, info = get_audio(call_id, codec, snr)
                note = info.get("label", codec)
                if info.get("real_codec") is False:
                    note += " (numpy approximation, ffmpeg not installed)"
            yield sse("meta", {
                "call_id": call_id,
                "duration": round(call.duration, 2),
                "n_turns": len(call.turns),
                "channel_note": note,
                "threshold": config.SETTINGS.pipeline.alert_threshold,
            })

            budget = 0.0
            for ev in e.stream(call, audio=x, sr=sr):
                yield sse("turn", ev)
                if ev.get("alert"):
                    yield sse("alert", ev)
                # a small pace so the browser gets events progressively even
                # when analysis is much faster than the call itself
                pace = min(0.12, max(0.0, (ev.get("dt", 0.0) / max(speed, 0.1)) * 0.25))
                budget += pace
                if pace:
                    time.sleep(pace)
            yield sse("done", {"ok": True})
        except Exception as exc:
            traceback.print_exc()
            yield sse("failed", {"error": f"{type(exc).__name__}: {exc}"})

    return Response(
        stream_with_context(gen()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@app.get("/api/results")
def results_index():
    d = config.RESULTS_DIR
    names = sorted(p.name for p in d.glob("*.json")) if d.exists() else []
    return jsonify({"available": names, "dir": str(d)})


@app.get("/api/results/<name>")
def results_file(name):
    if not name.endswith(".json") or "/" in name or "\\" in name or ".." in name:
        return jsonify({"error": "bad name"}), 400
    p = config.RESULTS_DIR / name
    if not p.exists():
        return jsonify({"error": "not found"}), 404
    return Response(p.read_text(encoding="utf-8"), mimetype="application/json")


# --------------------------------------------------------------------------


def main(host: str = "127.0.0.1", port: int = 7860, debug: bool = False):
    config.ensure_dirs()
    print(f"\n  {config.PROJECT_NAME} console  ->  http://{host}:{port}\n")
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    main(debug=bool(os.environ.get("SWARKAVACH_DEBUG")))
