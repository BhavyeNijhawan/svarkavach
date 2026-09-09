"""End to end checks: corpus, pipeline, streaming, fusion and the HTTP API.

Runs under pytest or as a plain script:

    .venv\\Scripts\\python.exe tests/test_integration.py

The corpus is generated once into a temporary directory so the test never
depends on, or damages, whatever is in data/.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

# Point the whole package at a scratch data directory BEFORE anything imports
# config, otherwise the real corpus gets rewritten by a test run.
_TMP = Path(tempfile.mkdtemp(prefix="swarkavach_it_"))
os.environ["SWARKAVACH_DATA"] = str(_TMP)

import numpy as np  # noqa: E402

from swarkavach import config  # noqa: E402
from swarkavach.schema import (  # noqa: E402
    Call, Turn, Verdict, FUSION_FEATURE_NAMES, decode_bio, risk_band,
)

N_CALLS = 160
_STATE = {}


# --------------------------------------------------------------------------


def test_corpus_generates():
    from swarkavach.corpus.generator import generate_corpus, corpus_stats

    calls = generate_corpus(n_calls=N_CALLS, seed=7, audio=True, audio_backend="sim")
    assert len(calls) >= N_CALLS * 0.9, f"only got {len(calls)} calls"

    stats = corpus_stats(calls)
    assert stats["n_calls"] == len(calls)
    assert stats["n_entities"] > 0, "corpus has no annotated entities"

    cells = stats.get("cells", {})
    for cell in ("humanxbenign", "humanxscam", "clonedxbenign", "clonedxscam"):
        assert cells.get(cell, 0) > 0, f"evaluation cell {cell} is empty"

    _STATE["calls"] = calls
    _STATE["stats"] = stats
    print(f"  corpus: {len(calls)} calls, cells {cells}")


def test_annotations_are_well_formed():
    calls = _STATE["calls"]
    n_ent = 0
    for c in calls:
        for t in c.turns:
            assert len(t.tokens) == len(t.bio) == len(t.lang), \
                f"{c.call_id} turn {t.index}: token/tag length mismatch"
            prev = "O"
            for tag in t.bio:
                if tag.startswith("I-"):
                    etype = tag[2:]
                    assert prev in (f"B-{etype}", f"I-{etype}"), \
                        f"{c.call_id}: dangling {tag} after {prev}"
                prev = tag
            n_ent += len(decode_bio(t.tokens, t.bio, t.index))
    assert n_ent > 0
    print(f"  BIO well formed across {len(calls)} calls, {n_ent} entities")


def test_splits_are_speaker_disjoint():
    calls = _STATE["calls"]
    by_split = {}
    for c in calls:
        by_split.setdefault(c.split, set()).add(c.speaker_id)
    splits = list(by_split)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            overlap = by_split[a] & by_split[b]
            assert not overlap, f"speakers {overlap} appear in both {a} and {b}"
    print(f"  splits speaker-disjoint: { {k: len(v) for k, v in by_split.items()} }")


def test_audio_exists_and_matches_timings():
    from swarkavach.audioio import read_audio

    calls = [c for c in _STATE["calls"] if c.audio_path and Path(c.audio_path).exists()]
    assert calls, "no call rendered audio"
    checked = 0
    for c in calls[:8]:
        x, sr = read_audio(c.audio_path, sr=config.TARGET_SR)
        dur = len(x) / sr
        assert abs(dur - c.duration) < 1.0, \
            f"{c.call_id}: audio {dur:.2f}s but last turn ends at {c.duration:.2f}s"
        checked += 1
    print(f"  audio timings agree with turn timings on {checked} calls "
          f"({len(calls)}/{len(_STATE['calls'])} have audio)")


def test_cmi_definitions_agree():
    """The corpus stats and the fusion feature must measure the same thing.

    CMI is computed in two places: `corpus.generator.code_mixing_index` for the
    corpus report and `text.langid.code_mixing_features` for the fusion vector.
    They were once on different scales (0 to 1 against 0 to 100), which made
    the console's corpus number and the model's feature silently incomparable.
    """
    from swarkavach.corpus.generator import code_mixing_index
    from swarkavach.text.langid import code_mixing_features

    cases = [
        ["hi"] * 10,                                   # monolingual, expect 0
        ["hi"] * 5 + ["en"] * 5,                       # even split, expect 0.5
        ["hi"] * 8 + ["en"] * 2,                       # expect 0.2
        ["hi", "en", "univ", "hi", "en", "univ"],      # universals excluded
        ["univ"] * 4,                                  # no language tokens
    ]
    for langs in cases:
        toks = [f"t{i}" for i in range(len(langs))]
        a = code_mixing_index(langs)
        b = code_mixing_features(toks, langs)["cmi"]
        assert abs(a - b) < 1e-9, f"CMI mismatch on {langs}: {a} vs {b}"
        assert 0.0 <= a <= 1.0, f"CMI out of the [0, 1] range: {a}"
    assert abs(code_mixing_index(["hi"] * 8 + ["en"] * 2) - 0.2) < 1e-9
    print("  both CMI implementations agree on the [0, 1] scale")


def test_pipeline_trains():
    from swarkavach.pipeline import Pipeline

    p = Pipeline.load()
    assert p.calls, "pipeline loaded no corpus"
    report = p.fit(verbose=False)
    assert "fusion" in report, f"fusion did not train: {report}"
    assert p.models.fusion is not None and p.models.fusion.trained
    _STATE["pipeline"] = p
    print(f"  trained: ner={report.get('ner')} intent={report.get('intent')} "
          f"antispoof={report.get('antispoof')}")


def test_analyze_returns_valid_verdict():
    from swarkavach.audioio import read_audio

    p = _STATE["pipeline"]
    test_calls = [c for c in p.calls if c.split == "test"] or p.calls
    call = test_calls[0]
    audio = sr = None
    if call.audio_path and Path(call.audio_path).exists():
        audio, sr = read_audio(call.audio_path, sr=config.TARGET_SR)

    v = p.analyze(call, audio=audio, sr=sr)
    assert isinstance(v, Verdict)
    assert 0.0 <= v.risk <= 1.0, f"risk out of range: {v.risk}"
    assert v.band == risk_band(v.risk)
    assert set(v.features) == set(FUSION_FEATURE_NAMES), "feature set drifted"
    assert all(np.isfinite(list(v.features.values()))), "non-finite feature"
    assert v.evidence.reasons, "verdict has no reasons"
    assert v.timeline, "verdict has no streaming timeline"

    # round trip through JSON, since the API serialises it
    again = Verdict.from_dict(json.loads(json.dumps(v.to_dict())))
    assert abs(again.risk - v.risk) < 1e-9
    print(f"  {call.call_id}: risk {v.risk:.3f} ({v.band}), "
          f"{len(v.evidence.reasons)} reasons, ttd={v.ttd}")


def test_streaming_is_monotonic_in_time():
    p = _STATE["pipeline"]
    call = [c for c in p.calls if c.label_scam][0]
    tl = list(p.stream(call))
    assert len(tl) == len(call.turns)
    times = [e["t_end"] for e in tl]
    assert times == sorted(times), "streaming events are out of order"
    assert all(0.0 <= e["risk"] <= 1.0 for e in tl)
    print(f"  streamed {len(tl)} turns, risk {tl[0]['risk']:.3f} -> {tl[-1]['risk']:.3f}")


def test_fusion_beats_single_branches():
    """The headline claim of the project, asserted as a test.

    If this fails the project has not demonstrated what it says it does, so it
    is a hard assertion rather than a printed number.
    """
    from swarkavach.fusion.featurize import featurize_corpus
    from swarkavach.fusion.model import train_all_arms
    from swarkavach.evaluate import roc_auc

    p = _STATE["pipeline"]
    train = [c for c in p.calls if c.split == "train"]
    test = [c for c in p.calls if c.split == "test"] or p.calls

    Xtr, ytr, _, _ = featurize_corpus(train, models=p.models, with_audio=True)
    Xte, yte, _, _ = featurize_corpus(test, models=p.models, with_audio=True)

    arms = train_all_arms(Xtr, ytr, kind="logreg")
    aucs = {a: roc_auc(m.predict_proba(Xte), yte) for a, m in arms.items()}
    _STATE["aucs"] = aucs
    for a, v in sorted(aucs.items(), key=lambda kv: -kv[1]):
        print(f"    {a:14s} AUC {v:.4f}")

    best_single = max(aucs.get("audio_only", 0.0), aucs.get("text_only", 0.0))
    assert aucs["full"] >= best_single - 1e-9, \
        f"fusion {aucs['full']:.4f} did not match the best single branch {best_single:.4f}"
    assert aucs["full"] > 0.60, f"fused AUC {aucs['full']:.4f} is too low to be meaningful"


def test_evaluation_writes_results():
    from swarkavach.evaluate import run_full_evaluation

    out = run_full_evaluation(codecs=["clean", "g711u"], verbose=False)
    assert out["headline"].get("fused AUC") is not None
    for name in ("ablation_results.json", "ttd_results.json", "provenance.json"):
        p = config.RESULTS_DIR / name
        assert p.exists(), f"{name} was not written"
        json.loads(p.read_text(encoding="utf-8"))
    print(f"  headline: {out['headline']}")


def test_http_api():
    """Exercise every endpoint the console calls, through Flask's test client."""
    from server.app import app

    app.config["TESTING"] = True
    c = app.test_client()

    r = c.get("/")
    assert r.status_code == 200 and b"SwarKavach" in r.data

    r = c.get("/api/health")
    assert r.status_code == 200
    health = r.get_json()
    assert health["ready"] is True, f"engine not ready: {health.get('error')}"

    r = c.get("/api/corpus")
    body = r.get_json()
    assert r.status_code == 200 and body["calls"], "corpus endpoint returned nothing"
    call_id = body["calls"][0]["call_id"]

    assert c.get(f"/api/call/{call_id}").status_code == 200
    assert c.get("/api/meta").status_code == 200

    r = c.get(f"/api/audio/{call_id}")
    assert r.status_code == 200 and r.data[:4] == b"RIFF", "audio endpoint did not return a WAV"

    r = c.get(f"/api/waveform/{call_id}?codec=g711u")
    assert r.status_code == 200 and r.get_json()["peaks"]

    r = c.get(f"/api/spectrogram/{call_id}")
    assert r.status_code == 200 and r.get_json()["db"]

    r = c.post("/api/analyze", json={"call_id": call_id, "codec": "g711u"})
    assert r.status_code == 200, r.get_json()
    v = r.get_json()
    assert 0.0 <= v["risk"] <= 1.0 and v["evidence"]["reasons"]

    r = c.get(f"/api/stream/{call_id}?codec=clean&speed=8")
    assert r.status_code == 200
    text = r.get_data(as_text=True)
    assert "event: meta" in text and "event: turn" in text and "event: done" in text

    r = c.post("/api/simulate", json={"audio": True})
    assert r.status_code == 200, r.get_json()
    assert r.get_json()["summary"]["call_id"]

    r = c.get("/api/results")
    assert r.status_code == 200 and r.get_json()["available"]

    print("  all API endpoints responded")


def test_upload_roundtrip():
    import io
    from swarkavach.audioio import wav_bytes
    from server.app import app

    c = app.test_client()
    x = (0.2 * np.sin(2 * np.pi * 180 * np.arange(8000 * 3) / 8000)).astype(np.float32)
    data = {
        "audio": (io.BytesIO(wav_bytes(x, 8000)), "test.wav"),
        "transcript": "caller: Sir aapka account block ho jayega, OTP batao\ncallee: Main branch jaunga",
    }
    r = c.post("/api/upload", data=data, content_type="multipart/form-data")
    assert r.status_code == 200, r.get_json()
    cid = r.get_json()["call_id"]

    r = c.post("/api/analyze", json={"call_id": cid})
    assert r.status_code == 200, r.get_json()
    v = r.get_json()
    assert 0.0 <= v["risk"] <= 1.0
    print(f"  uploaded audio scored {v['risk']:.3f} ({v['band']})")


# --------------------------------------------------------------------------


TESTS = [
    test_corpus_generates,
    test_annotations_are_well_formed,
    test_splits_are_speaker_disjoint,
    test_audio_exists_and_matches_timings,
    test_cmi_definitions_agree,
    test_pipeline_trains,
    test_analyze_returns_valid_verdict,
    test_streaming_is_monotonic_in_time,
    test_fusion_beats_single_branches,
    test_evaluation_writes_results,
    test_http_api,
    test_upload_roundtrip,
]


if __name__ == "__main__":
    import time
    import traceback

    print(f"scratch data dir: {_TMP}\n")
    failures = 0
    for fn in TESTS:
        t0 = time.time()
        print(f"[{fn.__name__}]")
        try:
            fn()
            print(f"  PASS  ({time.time() - t0:.1f}s)\n")
        except Exception:
            failures += 1
            traceback.print_exc()
            print(f"  FAIL  ({time.time() - t0:.1f}s)\n")
    print("=" * 64)
    print(f"{len(TESTS) - failures}/{len(TESTS)} passed")
    if _STATE.get("aucs"):
        print("ablation AUC:", {k: round(v, 4) for k, v in _STATE["aucs"].items()})
    sys.exit(1 if failures else 0)
