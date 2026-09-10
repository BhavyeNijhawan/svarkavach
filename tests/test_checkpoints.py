"""Drive checkpointing, the robocall transcript reader, and the prosody
contour that separates a cloned caller from a human one.

These three landed together and share one property: they are the parts that
only misbehave on Colab or against a network, so they need a test that does
not require either.

    .venv\\Scripts\\python.exe tests/test_checkpoints.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_TMP = Path(tempfile.mkdtemp(prefix="swarkavach_ckpt_"))
os.environ["SWARKAVACH_DATA"] = str(_TMP)

from swarkavach import colab, config  # noqa: E402
from swarkavach.corpus import neural_tts as N  # noqa: E402


def _seed_local(name: str, n: int = 5) -> Path:
    """Put some plausible files where an artifact expects to find them."""
    d = colab.ARTIFACTS[name]["path"]()
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"f{i}.json").write_text(json.dumps({"i": i, "pad": "x" * 200}),
                                      encoding="utf-8")
    (d / "nested").mkdir(exist_ok=True)
    (d / "nested" / "deep.json").write_text("{}", encoding="utf-8")
    return d


def test_archive_round_trip():
    """Save, wipe, restore. The directory has to come back identical."""
    fake = Path(tempfile.mkdtemp(prefix="fake_drive_"))
    colab._MOUNTED = fake
    try:
        d = _seed_local("results")
        before = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())
        assert colab.save("results", quiet=True)

        import shutil
        shutil.rmtree(d)
        assert not d.exists()

        assert colab.restore("results", quiet=True)
        after = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())
        assert before == after, (before, after)
    finally:
        colab._MOUNTED = None
        import shutil
        shutil.rmtree(fake, ignore_errors=True)


def test_unchanged_artifact_is_not_reuploaded():
    """The second save has to notice nothing moved, or every run re-uploads
    half a gigabyte of speech cache over a network filesystem."""
    fake = Path(tempfile.mkdtemp(prefix="fake_drive_"))
    colab._MOUNTED = fake
    try:
        _seed_local("models")
        assert colab.save("models", quiet=True)
        arc = fake / "models.tar.gz"
        stamp = arc.stat().st_mtime_ns

        assert colab.save("models", quiet=True)
        assert arc.stat().st_mtime_ns == stamp, "archive was rewritten unnecessarily"

        # a change has to be picked up
        (colab.ARTIFACTS["models"]["path"]() / "new.json").write_text("{}", encoding="utf-8")
        assert colab.save("models", quiet=True)
        assert arc.stat().st_mtime_ns != stamp, "a new file did not trigger a save"
    finally:
        colab._MOUNTED = None
        import shutil
        shutil.rmtree(fake, ignore_errors=True)


def test_restoring_nothing_is_not_an_error():
    """First run. Every restore returns False and nothing raises."""
    fake = Path(tempfile.mkdtemp(prefix="fake_drive_"))
    colab._MOUNTED = fake
    try:
        assert colab.restore("pairs", quiet=True) is False
        assert colab.restore("corpus", quiet=True) is False
    finally:
        colab._MOUNTED = None
        import shutil
        shutil.rmtree(fake, ignore_errors=True)


def test_no_drive_is_a_silent_noop():
    """Off Colab the notebook still calls these, so they must do nothing
    quietly rather than raise."""
    colab._MOUNTED = None
    assert colab.save("results") is False
    assert colab.restore("results") is False
    assert colab.save_all() == {}
    assert colab.restore_all() == {}


def test_raw_fingerprint_ignores_the_unused_train_archive():
    """RAW_DIR holds gigabytes of extracted audio and a 2 GB archive nothing
    reads. Only the three core archives may count towards the fingerprint, or
    the artifact reports a change on every single run."""
    raw = config.RAW_DIR
    raw.mkdir(parents=True, exist_ok=True)
    for name in ("GV_Dev_5h.tar.gz", "GV_Eval_3h.tar.gz", "Metadata.tar.gz",
                 "GV_Train_100h.tar.gz"):
        (raw / name).write_bytes(b"0" * 100)
    (raw / "GV_Dev_5h").mkdir(exist_ok=True)
    for i in range(50):
        (raw / "GV_Dev_5h" / f"{i}.wav").write_bytes(b"0" * 10)

    spec = colab.ARTIFACTS["raw"]
    fp = colab._fingerprint(raw, tuple(spec.get("exclude", ())),
                            tuple(spec.get("include", ())))
    assert fp["files"] == 3, fp


def test_robocall_transcripts_reads_the_csv_alone():
    """A synthetic metadata.csv, so this runs without the 1.7 GB clone."""
    from swarkavach.datasets import DATASETS, robocall_transcripts

    root = config.RAW_DIR / DATASETS["robocall"]["dir"]
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.csv").write_text(
        "file_name,language,transcript,case_details,case_pdf\n"
        'audio/a.wav,en,"Your Amazon account has been charged. Press one now.",case-a,p.pdf\n'
        'audio/b.wav,zh,"Ni hao this is a longer line with enough words",case-b,p.pdf\n'
        'audio/c.wav,en,"short",case-c,p.pdf\n',
        encoding="utf-8")

    rows = robocall_transcripts()
    # the one-word row is dropped, the other two survive
    assert len(rows) == 2, rows
    assert rows[0]["id"] == "a"
    assert {r["language"] for r in rows} == {"en", "zh"}
    assert "Amazon" in rows[0]["transcript"]

    assert len(robocall_transcripts(limit=1)) == 1


def test_cloned_caller_holds_a_flat_contour():
    """The whole human against cloned signal in the corpus is this. If a
    cloned caller modulates as much as a human one, the PIM feature has
    nothing to measure and the ablation table means nothing.
    """
    v = N.voice_for_speaker("test_speaker", 0)

    def spread(flat):
        vals = [N.turn_prosody(v, a, flat) for a in (0.0, 0.25, 0.5, 0.75, 1.0)]
        rates = [int(p["rate"].rstrip("%")) for p in vals]
        pitches = [int(p["pitch"].rstrip("Hz")) for p in vals]
        return max(rates) - min(rates), max(pitches) - min(pitches)

    hr, hp = spread(flat=False)
    cr, cp = spread(flat=True)

    assert hr >= 12, f"a human caller barely modulates rate: {hr}"
    assert hp >= 10, f"a human caller barely modulates pitch: {hp}"
    assert cr <= 3, f"a cloned caller modulates rate too much: {cr}"
    assert cp <= 3, f"a cloned caller modulates pitch too much: {cp}"
    assert hr > cr * 3 and hp > cp * 3


def test_volume_is_part_of_the_cache_key():
    """Two renderings that differ only in volume must not collide, or the
    cache silently serves the wrong loudness for every turn after the first.
    """
    a = N._cache_path("same text", "hi-IN-MadhurNeural", "+0%", "+0Hz", "+0%")
    b = N._cache_path("same text", "hi-IN-MadhurNeural", "+0%", "+0Hz", "+8%")
    assert a != b


def test_speakers_get_different_voices():
    """Different conversations have to sound like different people."""
    seen = {}
    for i in range(40):
        v = N.voice_for_speaker(f"spk_{i}", 0)
        seen[(v["voice"], v["rate"], v["pitch"], v["volume"])] = 1
    assert len(seen) >= 30, f"only {len(seen)} distinct voices out of 40 speakers"

    voices = {N.voice_for_speaker(f"spk_{i}", 0)["voice"] for i in range(40)}
    assert len(voices) >= 4, voices


def test_codec_availability_matches_what_ffmpeg_can_actually_do():
    """A build that has ffmpeg but not this encoder must not claim the codec.

    Colab ships an ffmpeg with the native GSM 06.10 encoder and no AMR-NB one,
    because AMR lives in libopencore_amrnb and most distribution builds leave
    it out. Checking only that the binary exists reported AMR-NB as the real
    standard while every encode fell back to the numpy stand-in, so a results
    table would have carried an approximation labelled as the true codec.

    This machine has no ffmpeg at all, so the condition is simulated rather
    than waited for.
    """
    import numpy as np

    from swarkavach import channel

    saved_codec = channel._ffmpeg_codec
    saved_find = channel.find_ffmpeg

    def only_gsm_works(x, sr, kind):
        return (np.asarray(x, dtype=np.float64), 8000) if kind == "gsm" else None

    try:
        channel._ffmpeg_codec = only_gsm_works
        channel.find_ffmpeg = lambda: "/usr/bin/ffmpeg"
        channel._ffmpeg_can_encode.cache_clear()

        x = np.sin(2 * np.pi * 300 * np.arange(8000) / 8000) * 0.3
        for name in ("gsm", "amrnb"):
            _, _, info = channel.apply_codec(x, 8000, name, return_info=True)
            assert info["real_codec"] is channel.codec_available(name), name

        assert channel.codec_available("gsm") is True
        assert channel.codec_available("amrnb") is False

        # and the reason has to be the true one, not "ffmpeg not found"
        _, _, info = channel.apply_codec(x, 8000, "amrnb", return_info=True)
        assert "no amrnb encoder" in info.get("note", ""), info.get("note")
    finally:
        channel._ffmpeg_codec = saved_codec
        channel.find_ffmpeg = saved_find
        channel._ffmpeg_can_encode.cache_clear()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
