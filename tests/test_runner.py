"""Tests for the pieces that decide what an expensive run rebuilds, and for
the anti-spoofing front end that a review found reading the recording
pipeline instead of the voice.

Plain asserts, runnable under pytest or directly.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402


# --------------------------------------------------------------------------
# Anti-spoofing front end
# --------------------------------------------------------------------------


def _tone_mix(sr, dur, freqs, amp=0.1):
    t = np.arange(int(sr * dur)) / sr
    return sum(amp * np.sin(2 * np.pi * f * t) for f in freqs).astype(np.float64)


def _band_energy(x, sr, lo, hi):
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    f = np.fft.rfftfreq(x.size, 1.0 / sr)
    return float(spec[(f >= lo) & (f < hi)].sum())


def test_condition_removes_what_leaked():
    """DC, energy outside 300 to 3000 Hz, and trailing silence all go."""
    from swarkavach.antispoof.features import BAND_HIGH_HZ, BAND_LOW_HZ, condition

    sr = 8000
    x = _tone_mix(sr, 2.0, [150, 1000, 3600]) + 0.05          # DC of 0.05
    x = np.concatenate([x, np.zeros(sr)])                     # one second of digital silence
    y = condition(x, sr)

    assert abs(float(np.mean(y))) < 1e-3, "DC survived"
    assert y.size < x.size - sr // 2, "trailing silence was not trimmed"
    inband = _band_energy(y, sr, BAND_LOW_HZ + 50, BAND_HIGH_HZ - 50)
    below = _band_energy(y, sr, 0, BAND_LOW_HZ - 100)
    above = _band_energy(y, sr, BAND_HIGH_HZ + 100, sr / 2)
    assert 10 * np.log10(below / inband) < -40, "energy below the band survived"
    assert 10 * np.log10(above / inband) < -40, "energy above the band survived"


def test_condition_is_deterministic_and_floored():
    """Same input, same output; and nothing is exactly zero any more."""
    from swarkavach.antispoof.features import condition

    sr = 8000
    x = _tone_mix(sr, 1.5, [800])
    a, b = condition(x, sr), condition(x, sr)
    assert np.array_equal(a, b)
    # the dither means an all-zero stop band cannot be told from a tiny one
    spec = np.abs(np.fft.rfft(a)) ** 2
    f = np.fft.rfftfreq(a.size, 1.0 / sr)
    assert spec[f > 3500].min() > 0.0


def test_condition_leaves_short_signals_alone():
    from swarkavach.antispoof.features import condition

    x = np.zeros(20)
    assert condition(x, 8000).size >= 1
    x = np.random.default_rng(0).standard_normal(100) * 0.01
    assert np.all(np.isfinite(condition(x, 8000)))


def test_audit_scores_bandwidth_and_tails():
    """The audit must see a bandwidth or trailing-silence difference."""
    from swarkavach.datasets import AUDIT_STATISTICS, confound_audit

    for k in ("band_share_hi", "band_share_lo", "trailing_quiet_s", "dc_abs",
              "quiet_centroid_hz", "quiet_tilt_db"):
        assert k in AUDIT_STATISTICS

    sr, tmp = 8000, Path(tempfile.mkdtemp())
    try:
        import soundfile as sf

        rng = np.random.default_rng(1)
        items = []
        for i in range(10):
            base = _tone_mix(sr, 2.0, [500, 1200, 2200]) + 0.003 * rng.standard_normal(sr * 2)
            base[: sr // 4] *= 0.05
            real = base.copy()
            fake = base + _tone_mix(sr, 2.0, [3700], amp=0.05)          # content above the band
            fake = np.concatenate([fake, 0.0001 * rng.standard_normal(sr)])  # a silent tail
            for label, sig in ((0, real), (1, fake)):
                p = tmp / f"{label}_{i}.wav"
                sf.write(p, sig, sr)
                items.append({"path": str(p), "label": label})
        out = confound_audit(items, sr)
        raw = out["per_statistic_raw"]
        assert raw["band_share_hi"]["auc_alone"] > 0.95, raw["band_share_hi"]
        assert raw["trailing_quiet_s"]["auc_alone"] > 0.95, raw["trailing_quiet_s"]
        # and after conditioning the detector no longer sees either
        cond = out["per_statistic"]
        assert cond["band_share_hi"]["auc_alone"] < 0.8, cond["band_share_hi"]
        assert cond["trailing_quiet_s"]["auc_alone"] < 0.8, cond["trailing_quiet_s"]
        assert "worst_auc_raw" in out and "worst_auc" in out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pair_voices_read_devanagari():
    from swarkavach.corpus.neural_tts import VOICE_POOL, voice_for_speaker

    deva = {name for name, _, script in VOICE_POOL if script == "deva"}
    assert deva, "no Devanagari voice in the pool"
    for i in range(60):
        v = voice_for_speaker(f"spoof_{i}", 0, script="deva")
        assert v["voice"] in deva, v["voice"]
    seen = {voice_for_speaker(f"spoof_{i}", 0)["voice"] for i in range(60)}
    assert len(seen) > len(deva), "the unrestricted draw should still use every voice"


# --------------------------------------------------------------------------
# What the runner rebuilds
# --------------------------------------------------------------------------


def test_forcing_a_stage_forces_what_depends_on_it():
    from swarkavach.runner import _with_downstream

    assert _with_downstream(["corpus"]) == ["corpus", "train", "evaluate"]
    assert _with_downstream(["pairs"]) == ["pairs", "train", "evaluate"]
    assert _with_downstream(["train"]) == ["train", "evaluate"]
    assert _with_downstream(["evaluate"]) == ["evaluate"]
    assert _with_downstream([]) == []


def test_pairs_staleness_reads_the_build_version():
    from swarkavach import datasets
    from swarkavach.runner import _pairs_stale_reason

    tmp = Path(tempfile.mkdtemp())
    old_raw = datasets.RAW_DIR
    try:
        datasets.RAW_DIR = tmp
        d = tmp / "antispoof_pairs"
        d.mkdir()
        m = d / "manifest.json"
        assert _pairs_stale_reason() is None            # no manifest at all
        m.write_text(json.dumps({"n_pairs": 5}), encoding="utf-8")
        assert "versioned" in (_pairs_stale_reason() or "")
        m.write_text(json.dumps({"pairs_build_version": datasets.PAIRS_BUILD_VERSION}),
                     encoding="utf-8")
        assert _pairs_stale_reason() is None
        m.write_text(json.dumps({"pairs_build_version": datasets.PAIRS_BUILD_VERSION - 1}),
                     encoding="utf-8")
        assert "version" in (_pairs_stale_reason() or "")
    finally:
        datasets.RAW_DIR = old_raw
        shutil.rmtree(tmp, ignore_errors=True)


def test_audit_refresh_rewrites_a_stale_audit_from_disk():
    from swarkavach import datasets
    from swarkavach.datasets import audit_is_current, refresh_confound_audit

    tmp = Path(tempfile.mkdtemp())
    try:
        import soundfile as sf

        d = tmp / "antispoof_pairs"
        (d / "bonafide").mkdir(parents=True)
        (d / "spoof").mkdir()
        sr = 8000
        items = []
        for i in range(6):
            for label, sub in ((0, "bonafide"), (1, "spoof")):
                p = d / sub / f"u{i}.wav"
                sf.write(p, _tone_mix(sr, 1.5, [700 + 50 * label]), sr)
                # recorded under a path that does not exist here, as a
                # manifest restored from another machine would be
                items.append({"path": f"/content/elsewhere/{sub}/u{i}.wav", "label": label})
        stale = {"confound_audit": {"per_statistic": {"noise_floor_db": {}}},
                 "items": items, "sample_rate": sr, "n_pairs": 6}
        (d / "manifest.json").write_text(json.dumps(stale), encoding="utf-8")

        assert not audit_is_current(stale["confound_audit"])
        man = refresh_confound_audit(d)
        assert audit_is_current(man["confound_audit"]), "refresh did not score the new statistics"
        assert man["confound_audit"].get("recomputed_from_disk") is True
        on_disk = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        assert audit_is_current(on_disk["confound_audit"]), "refresh did not write back"
        # and a current audit is left alone
        again = refresh_confound_audit(d)
        assert again["confound_audit"] == man["confound_audit"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_anchor_report_reads_an_old_file_tolerantly():
    """An anchors file from before the tolerance still gets ok / flat / wrong."""
    from swarkavach import config
    from swarkavach.runner import anchor_report

    tmp = Path(tempfile.mkdtemp())
    old = config.MODELS_DIR
    try:
        config.MODELS_DIR = tmp
        (tmp / "pim_anchors.json").write_text(json.dumps({
            "anchors": {"f0_cv": [0.24, 0.34], "rate_std": [0.38, 1.16], "bad": [0.0, 1.0]},
            "medians": {"f0_cv": {"human": 0.29, "synthetic": 0.275},
                        "rate_std": {"human": 0.743, "synthetic": 0.756},
                        "bad": {"human": 0.2, "synthetic": 0.6}},
            "direction_ok": {"f0_cv": True, "rate_std": False, "bad": False},
        }), encoding="utf-8")
        v = anchor_report()
        assert v == {"f0_cv": "ok", "rate_std": "flat", "bad": "wrong"}, v
    finally:
        config.MODELS_DIR = old
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except Exception as exc:  # noqa: BLE001
                fails += 1
                print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    raise SystemExit(1 if fails else 0)
