"""Tests for the acoustic anti-spoofing branch.

Plain asserts, no fixtures, so this runs under pytest and as a script:

    pytest tests/test_antispoof.py -q
    python tests/test_antispoof.py

Everything here uses the demo synthesiser in `antispoof.features`, so the suite
never depends on the corpus generator, on an audio file, or on a trained model.
"""

from __future__ import annotations

import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

# Allow running as a plain script from a checkout that is not pip installed.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from swarkavach.config import SETTINGS, TARGET_SR  # noqa: E402
from swarkavach.schema import BranchScore, FEATURE_GROUPS, FUSION_FEATURE_NAMES  # noqa: E402
from swarkavach.antispoof.embeddings import (  # noqa: E402
    SpeakerEmbedder,
    cosine_consistency,
)
from swarkavach.antispoof.features import (  # noqa: E402
    ANTISPOOF_KEYS,
    FEATURE_SETS,
    POOLING,
    antispoof_feature_dict,
    demo_bonafide,
    demo_spoof,
    frame_features,
    utterance_feature_dim,
    utterance_features,
)
from swarkavach.antispoof.gbm import GBMScorer  # noqa: E402
from swarkavach.antispoof.gmm import GMMScorer  # noqa: E402
from swarkavach.antispoof.scorer import (  # noqa: E402
    VOICE_FEATURES,
    AntiSpoofScorer,
    PlattCalibrator,
    heuristic_score,
)

SR = TARGET_SR
RNG = np.random.default_rng(SETTINGS.pipeline.seed)

SILENCE = np.zeros(2 * SR, dtype=np.float32)
NOISE = (0.2 * RNG.standard_normal(2 * SR)).astype(np.float32)
TONE = (0.4 * np.sin(2 * np.pi * 220.0 * np.arange(2 * SR) / SR)).astype(np.float32)
BONA = demo_bonafide(SR, 3.0, seed=1)
SPOOF = demo_spoof(SR, 3.0, seed=1)


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------


def test_pooled_features_have_the_expected_dimension():
    for name in FEATURE_SETS:
        vec = utterance_features(BONA, SR, name)
        assert vec.ndim == 1, f"{name}: pooled vector must be 1-D, got {vec.shape}"
        assert vec.size == utterance_feature_dim(name), (
            f"{name}: got {vec.size}, expected {utterance_feature_dim(name)}"
        )
    # n_ceps * (static + delta + delta-delta) * eight pooling functions
    cep = SETTINGS.cepstral
    assert utterance_feature_dim("lfcc") == cep.n_ceps * 3 * len(POOLING)
    assert len(POOLING) == 8


def test_pooled_features_are_finite_for_silence_and_noise():
    for label, sig in (("silence", SILENCE), ("noise", NOISE), ("tone", TONE),
                       ("one sample", np.zeros(1, dtype=np.float32)),
                       ("empty", np.zeros(0, dtype=np.float32))):
        for name in FEATURE_SETS:
            vec = utterance_features(sig, SR, name)
            assert np.isfinite(vec).all(), f"{name} on {label} produced a non-finite value"
            assert vec.size == utterance_feature_dim(name)


def test_frame_features_shape_and_normalisation():
    feat = frame_features(BONA, SR, "lfcc")
    cep = SETTINGS.cepstral
    assert feat.ndim == 2
    assert feat.shape[1] == cep.n_ceps * 3
    assert feat.shape[0] > 10
    assert np.isfinite(feat).all()
    # CMVN is on by default, so every live column is centred and scaled.
    assert np.abs(feat.mean(axis=0)).max() < 1e-6
    # and off when asked, which is what the pooled path relies on
    raw = frame_features(BONA, SR, "lfcc", apply_cmvn=False)
    assert np.abs(raw.mean(axis=0)).max() > 1e-6


def test_antispoof_feature_dict_keys_are_finite_floats():
    for label, sig in (("silence", SILENCE), ("noise", NOISE), ("tone", TONE),
                       ("bonafide", BONA), ("spoof", SPOOF)):
        d = antispoof_feature_dict(sig, SR)
        for key in ANTISPOOF_KEYS:
            assert key in d, f"{label}: missing key {key}"
        for key, value in d.items():
            assert isinstance(value, float), f"{label}: {key} is {type(value)}, not float"
            assert math.isfinite(value), f"{label}: {key} is {value}"
            assert not math.isnan(value)
        assert d["as_llr"] == 0.0, "as_llr must stay 0.0 with no GMM loaded"
        extra = set(d) - set(ANTISPOOF_KEYS)
        assert all(k.startswith("dx_") for k in extra), f"undeclared keys: {extra}"


def test_voice_quality_separates_the_demo_signals():
    """The four contract features must point the way the literature says."""
    human = antispoof_feature_dict(BONA, SR)
    machine = antispoof_feature_dict(SPOOF, SR)
    assert human["jitter"] > machine["jitter"] * 3, (human["jitter"], machine["jitter"])
    assert human["shimmer"] > machine["shimmer"] * 2
    assert machine["hnr"] > human["hnr"]
    assert human["spec_flatness_var"] > machine["spec_flatness_var"]


# --------------------------------------------------------------------------
# gmm
# --------------------------------------------------------------------------


def test_gmm_llr_separates_two_distributions():
    """Two obviously different Gaussian clouds: the sign of the LLR must follow."""
    rng = np.random.default_rng(0)
    bona = rng.normal(0.0, 1.0, size=(1200, 8))
    spoof = rng.normal(3.0, 0.5, size=(1200, 8))
    gmm = GMMScorer(feature_set="lfcc").fit(bona, spoof)

    assert gmm.is_fitted
    assert 2 <= gmm.n_components_ <= 64, gmm.n_components_
    # Positive means "looks spoofed", which is the fusion convention.
    assert gmm.llr(rng.normal(3.0, 0.5, size=(300, 8))) > 1.0
    assert gmm.llr(rng.normal(0.0, 1.0, size=(300, 8))) < -1.0
    assert abs(gmm.llr(bona)) <= 10.0, "the LLR must be clipped to the fusion range"


def test_gmm_separates_demo_audio_and_round_trips():
    bona = [demo_bonafide(SR, 2.0, seed=i, f0_base=110.0 + 8.0 * i) for i in range(6)]
    spoof = [demo_spoof(SR, 2.0, seed=200 + i, f0_base=110.0 + 8.0 * i) for i in range(6)]
    gmm = GMMScorer(feature_set="lfcc").fit_audio(bona, spoof, SR)

    held_bona = gmm.llr_audio(demo_bonafide(SR, 2.0, seed=90), SR)
    held_spoof = gmm.llr_audio(demo_spoof(SR, 2.0, seed=90), SR)
    assert held_spoof > held_bona, (held_bona, held_spoof)

    tmp = Path(tempfile.mkdtemp())
    try:
        gmm.save(tmp / "gmm.joblib")
        again = GMMScorer.load(tmp / "gmm.joblib")
        assert again.llr_audio(bona[0], SR) == gmm.llr_audio(bona[0], SR)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gmm_component_count_adapts_to_the_data():
    rng = np.random.default_rng(1)
    small = GMMScorer().fit(rng.normal(size=(240, 6)), rng.normal(1.0, size=(240, 6)))
    big = GMMScorer().fit(rng.normal(size=(9000, 6)), rng.normal(1.0, size=(9000, 6)))
    assert small.n_components_ < big.n_components_
    assert small.n_components_ >= 2


# --------------------------------------------------------------------------
# gbm
# --------------------------------------------------------------------------


def test_gbm_fits_and_round_trips():
    bona = [demo_bonafide(SR, 2.0, seed=i, f0_base=105.0 + 9.0 * (i % 5)) for i in range(8)]
    spoof = [demo_spoof(SR, 2.0, seed=400 + i, f0_base=105.0 + 9.0 * (i % 5)) for i in range(8)]
    gbm = GBMScorer().fit_audio(bona + spoof, [0] * 8 + [1] * 8, SR)

    p_bona = gbm.score_audio(demo_bonafide(SR, 2.0, seed=80), SR)
    p_spoof = gbm.score_audio(demo_spoof(SR, 2.0, seed=80), SR)
    assert 0.0 <= p_bona <= 1.0 and 0.0 <= p_spoof <= 1.0
    assert p_spoof > p_bona, (p_bona, p_spoof)

    tmp = Path(tempfile.mkdtemp())
    try:
        gbm.save(tmp / "gbm.joblib")
        again = GBMScorer.load(tmp / "gbm.joblib")
        assert again.score_audio(bona[0], SR) == gbm.score_audio(bona[0], SR)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# embeddings
# --------------------------------------------------------------------------


def test_embedding_shape_and_determinism():
    emb = SpeakerEmbedder()
    v = emb.embed(BONA, SR)
    assert v.shape == (192,)
    assert np.isfinite(v).all()
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-6
    assert np.array_equal(v, emb.embed(BONA, SR)), "embedding must be deterministic"


def test_cosine_consistency_is_higher_for_a_signal_against_itself():
    emb = SpeakerEmbedder()
    a = demo_bonafide(SR, 3.0, seed=1, f0_base=120.0)
    b = demo_bonafide(SR, 3.0, seed=77, f0_base=195.0)

    same = cosine_consistency(emb.embed_segments(np.concatenate([a, a]), SR, 2))
    different = cosine_consistency(emb.embed_segments(np.concatenate([a, b]), SR, 2))
    assert same > different, (same, different)
    assert same > 0.9, same
    assert -1.0 <= different <= 1.0

    # Degenerate inputs must not explode.
    assert cosine_consistency(np.zeros((1, 192))) == 1.0
    assert cosine_consistency(np.zeros((0, 192))) == 1.0
    assert math.isfinite(cosine_consistency(np.zeros((3, 192))))


def test_consistency_is_higher_for_synthetic_on_long_calls():
    """The reason this feature is in the fusion vector, checked end to end.

    Long calls only: with segments under about a second the measure is noise,
    which is documented in the embeddings module and reflected in the small
    weight the heuristic gives it.
    """
    emb = SpeakerEmbedder()
    human = [emb.consistency(demo_bonafide(SR, 8.0, seed=s, f0_base=110.0 + 13.0 * s), SR)
             for s in range(4)]
    machine = [emb.consistency(demo_spoof(SR, 8.0, seed=s, f0_base=110.0 + 13.0 * s), SR)
               for s in range(4)]
    assert np.mean(machine) > np.mean(human), (human, machine)


# --------------------------------------------------------------------------
# scorer
# --------------------------------------------------------------------------


def test_untrained_scorer_returns_a_valid_branch_score():
    scorer = AntiSpoofScorer()
    assert scorer.active_backend() == "heuristic"

    for label, sig in (("silence", SILENCE), ("noise", NOISE), ("tone", TONE),
                       ("bonafide", BONA), ("spoof", SPOOF)):
        bs = scorer.score(sig, SR)
        assert isinstance(bs, BranchScore)
        assert bs.name == "antispoof"
        assert bs.backend == "heuristic"
        assert 0.0 <= bs.score <= 1.0, f"{label}: score {bs.score} out of range"
        assert math.isfinite(bs.raw)
        assert sorted(bs.features) == sorted(VOICE_FEATURES), label
        for key, value in bs.features.items():
            assert math.isfinite(value), f"{label}: feature {key} is {value}"


def test_voice_features_match_the_schema():
    expected = tuple(n for n in FUSION_FEATURE_NAMES if FEATURE_GROUPS[n] == "voice")
    assert VOICE_FEATURES == expected
    assert set(VOICE_FEATURES) == {
        "as_score", "as_llr", "as_margin", "spk_consistency",
        "jitter", "shimmer", "hnr", "spec_flatness_var",
    }


def test_heuristic_path_separates_human_from_synthetic():
    scorer = AntiSpoofScorer()
    human = [scorer.score(demo_bonafide(SR, 3.0, seed=s, f0_base=105.0 + 9.0 * s), SR).score
             for s in range(5)]
    machine = [scorer.score(demo_spoof(SR, 3.0, seed=s, f0_base=105.0 + 9.0 * s), SR).score
               for s in range(5)]
    assert max(human) < min(machine), (human, machine)
    assert np.mean(human) < 0.35 and np.mean(machine) > 0.65


def test_fitted_scorer_and_save_load_round_trip():
    bona = [demo_bonafide(SR, 2.5, seed=s, f0_base=100.0 + 11.0 * (s % 6)) for s in range(12)]
    spoof = [demo_spoof(SR, 2.5, seed=600 + s, f0_base=100.0 + 11.0 * (s % 6))
             for s in range(12)]
    scorer = AntiSpoofScorer()
    summary = scorer.fit_signals(bona + spoof, [0] * 12 + [1] * 12, sr=SR)

    assert "error" not in summary, summary
    assert scorer.active_backend() in ("gbm", "gmm")
    assert scorer.gmm is not None and scorer.gmm.is_fitted
    assert scorer.gbm is not None and scorer.gbm.is_fitted
    assert scorer.calibrator.fitted, summary["calibration"]

    held = [demo_bonafide(SR, 2.5, seed=900 + s) for s in range(3)] + \
           [demo_spoof(SR, 2.5, seed=950 + s) for s in range(3)]
    before = [scorer.score(x, SR) for x in held]
    assert all(0.0 <= b.score <= 1.0 for b in before)
    assert np.mean([b.score for b in before[:3]]) < np.mean([b.score for b in before[3:]])
    assert before[0].features["as_llr"] != 0.0, "a fitted GMM must fill in as_llr"

    tmp = Path(tempfile.mkdtemp())
    try:
        path = tmp / "scorer.joblib"
        scorer.save(path)
        again = AntiSpoofScorer.load(path)
        assert again.active_backend() == scorer.active_backend()
        after = [again.score(x, SR) for x in held]
        for b, a in zip(before, after):
            assert a.score == b.score, (b.score, a.score)
            assert a.raw == b.raw
            for key in VOICE_FEATURES:
                assert a.features[key] == b.features[key], key
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_calibrator_maps_scores_to_probabilities():
    rng = np.random.default_rng(3)
    raw = np.concatenate([rng.normal(-2.0, 1.0, 40), rng.normal(2.0, 1.0, 40)])
    y = np.array([0] * 40 + [1] * 40)
    cal = PlattCalibrator().fit(raw, y)

    assert cal.fitted
    assert cal.a > 0, "a higher raw score must mean a higher probability"
    assert 0.0 <= cal.transform(-8.0) < cal.transform(0.0) < cal.transform(8.0) <= 1.0

    # A degenerate held-out slice must leave the calibrator alone, not crash.
    empty = PlattCalibrator().fit(np.zeros(4), np.zeros(4, dtype=int))
    assert not empty.fitted
    assert 0.0 <= empty.transform(1.0) <= 1.0


def test_heuristic_terms_point_the_right_way():
    raw_h, terms_h = heuristic_score(antispoof_feature_dict(BONA, SR))
    raw_s, terms_s = heuristic_score(antispoof_feature_dict(SPOOF, SR))
    assert raw_s > raw_h
    assert set(terms_h) >= {"jitter", "shimmer", "hnr", "spec_flatness_var"}
    assert terms_h["jitter"] < terms_s["jitter"]
    assert terms_h["shimmer"] < terms_s["shimmer"]


def test_rawnet_module_is_safe_without_a_checkpoint():
    from swarkavach.antispoof import rawnet as rawnet_mod

    assert rawnet_mod.load_rawnet(Path(tempfile.gettempdir()) / "no_such_rawnet.pt") is None
    if rawnet_mod.TORCH_AVAILABLE:
        net = rawnet_mod.RawNetLite(sr=SR)
        assert net.n_parameters() < 300_000, net.n_parameters()
        p = net.score(BONA, SR)
        assert 0.0 <= p <= 1.0


if __name__ == "__main__":  # pragma: no cover - script mode
    import time

    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = []
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS  {name}  ({time.time() - t0:.1f} s)")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
