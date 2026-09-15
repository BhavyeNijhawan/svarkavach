"""Drive checkpointing, the robocall transcript reader, and the prosody
contour that separates a cloned caller from a human one.

These three landed together and share one property: they are the parts that
only misbehave on Colab or against a network, so they need a test that does
not require either.

    .venv\\Scripts\\python.exe tests/test_checkpoints.py
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

NL = chr(10)

_TMP = Path(tempfile.mkdtemp(prefix="swarkavach_ckpt_"))
os.environ.setdefault("SWARKAVACH_DATA", str(_TMP))

from swarkavach import colab  # noqa: E402
from swarkavach.corpus import neural_tts as N  # noqa: E402


@contextlib.contextmanager
def sandbox(*names: str):
    """Give these artifacts a scratch directory and a scratch Drive.

    Nothing here reads `config`. Setting SWARKAVACH_DATA at module import is
    not enough under pytest, because collection imports every test module and
    the first one to import `swarkavach.config` fixes DATA_DIR for the whole
    process; a module that sets the variable afterwards is talking to itself
    while its tests write into the real `data/`. That happened, and the stub
    archives it left in `data/raw` were enough to make `fetch-data` skip the
    GramVaani download.

    Overriding the artifact paths directly removes the question. These tests
    are about the checkpoint machinery, not about where the project keeps its
    data, so they should not care what DATA_DIR says.
    """
    work = Path(tempfile.mkdtemp(prefix="ckpt_work_"))
    drive = Path(tempfile.mkdtemp(prefix="ckpt_drive_"))
    saved = {n: colab.ARTIFACTS[n]["path"] for n in names}
    saved_mount = colab._MOUNTED
    try:
        for n in names:
            target = work / n
            colab.ARTIFACTS[n]["path"] = (lambda t=target: t)
        colab._MOUNTED = drive
        yield work, drive
    finally:
        for n, fn in saved.items():
            colab.ARTIFACTS[n]["path"] = fn
        colab._MOUNTED = saved_mount
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(drive, ignore_errors=True)


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
    with sandbox("results"):
        d = _seed_local("results")
        before = {p.relative_to(d).as_posix(): p.read_bytes()
                  for p in d.rglob("*") if p.is_file()}
        assert colab.save("results", quiet=True)

        shutil.rmtree(d)
        assert not d.exists()

        assert colab.restore("results", quiet=True)
        after = {p.relative_to(d).as_posix(): p.read_bytes()
                 for p in d.rglob("*") if p.is_file()}
        assert before == after


def test_unchanged_artifact_is_not_reuploaded():
    """The second save has to notice nothing moved, or every run re-uploads
    half a gigabyte of speech cache over a network filesystem.

    Asserted on the recorded fingerprint rather than the archive's mtime. An
    mtime comparison passes for the wrong reason when a previous run left a
    file behind, which is how the leak into the real data directory stayed
    invisible until it broke on Colab.
    """
    with sandbox("models") as (_work, drive):
        d = _seed_local("models")
        assert colab.save("models", quiet=True)
        first = json.loads((drive / "models.meta.json").read_text(encoding="utf-8"))
        assert first["fingerprint"]["files"] == 6, first

        arc = drive / "models.tar.gz"
        stamp = arc.stat().st_mtime_ns
        assert colab.save("models", quiet=True)
        assert arc.stat().st_mtime_ns == stamp, "archive was rewritten unnecessarily"

        # a genuinely new file has to be picked up
        assert not (d / "brand_new.json").exists()
        (d / "brand_new.json").write_text('{"x": 1}', encoding="utf-8")
        assert colab.save("models", quiet=True)
        second = json.loads((drive / "models.meta.json").read_text(encoding="utf-8"))
        assert second["fingerprint"]["files"] == 7, second
        assert second["fingerprint"] != first["fingerprint"]

        # and the new file survives a restore
        shutil.rmtree(d)
        assert colab.restore("models", quiet=True)
        assert (d / "brand_new.json").read_text(encoding="utf-8") == '{"x": 1}'


def test_restoring_nothing_is_not_an_error():
    """First run. Every restore returns False and nothing raises."""
    with sandbox("pairs", "corpus"):
        assert colab.restore("pairs", quiet=True) is False
        assert colab.restore("corpus", quiet=True) is False


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
    the artifact reports a change on every single run.

    Written into a scratch directory, never the real one. An earlier version
    of this test put 100 byte stub archives into `data/raw`, and `fetch-data`
    skips any archive that already exists, so it convinced the downloader it
    had GramVaani when it had zeros.
    """
    with sandbox("raw") as (work, _drive):
        raw = colab.ARTIFACTS["raw"]["path"]()
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
    from swarkavach import datasets as D

    root = Path(tempfile.mkdtemp(prefix="ckpt_rc_"))
    (root / "metadata.csv").write_text(NL.join([
        "file_name,language,transcript,case_details,case_pdf",
        'audio/a.wav,en,"Your Amazon account has been charged. Press one now.",case-a,p.pdf',
        'audio/b.wav,zh,"Ni hao this is a longer line with enough words",case-b,p.pdf',
        'audio/c.wav,en,"short",case-c,p.pdf',
        "",
    ]), encoding="utf-8")

    saved = D.RAW_DIR
    try:
        D.RAW_DIR = root.parent
        D.DATASETS["robocall"] = dict(D.DATASETS["robocall"], dir=root.name)
        rows = D.robocall_transcripts()
        # the one-word row is dropped, the other two survive
        assert len(rows) == 2, rows
        assert rows[0]["id"] == "a"
        assert {r["language"] for r in rows} == {"en", "zh"}
        assert "Amazon" in rows[0]["transcript"]
        assert len(D.robocall_transcripts(limit=1)) == 1
    finally:
        D.RAW_DIR = saved
        shutil.rmtree(root, ignore_errors=True)


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


def test_pickle_aliases_bridge_a_moved_module():
    """A model trained on another scikit-learn has to load here.

    Pickle records a class by module path, and scikit-learn's loss extension
    has reported itself both as bare `_loss` and as `sklearn._loss._loss`
    across releases. A Colab-trained anti-spoofing model raised
    ModuleNotFoundError for a class that was present and identical.
    """
    import importlib
    import sys

    from swarkavach import compat

    for name in ("_loss", "sklearn._loss._loss"):
        sys.modules.pop(name, None)
    compat._installed = False

    done = compat.install_pickle_aliases()
    assert compat._installed
    # whichever spelling is missing here should now resolve
    for missing, target in done.items():
        assert sys.modules[missing] is importlib.import_module(target)
    # at least one spelling has to be reachable, or no model would load
    assert any(n in sys.modules or _importable(n)
               for n in ("_loss", "sklearn._loss._loss"))

    # calling twice is a no-op, and never raises
    compat.install_pickle_aliases()


def _importable(name):
    import importlib
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False
