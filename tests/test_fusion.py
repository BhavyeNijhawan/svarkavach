"""Tests for the fusion layer: PIM, the calibrated model, the explainers and
the streaming path.

Runs under pytest or as a plain script:

    .venv\\Scripts\\python.exe tests/test_fusion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from swarkavach.schema import (  # noqa: E402
    ABLATION_ARMS, Call, FUSION_FEATURE_NAMES, Turn, feature_vector,
)
from swarkavach.fusion import pim as P  # noqa: E402
from swarkavach.fusion.explain import kernel_shap, linear_shap  # noqa: E402
from swarkavach.fusion.model import FusionModel  # noqa: E402
from swarkavach.fusion.streaming import stream_call, time_to_detection  # noqa: E402

SR = 8000


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _syllables(levels, sr=SR, syl_s=0.18, gap_s=0.06, f0=150.0):
    """A run of syllable-shaped bursts at the given relative levels."""
    out = []
    for lv in levels:
        t = np.arange(int(syl_s * sr)) / sr
        env = np.sin(np.pi * t / syl_s) ** 2
        out.append((lv * env * np.sin(2 * np.pi * f0 * t)).astype(np.float32))
        out.append(np.zeros(int(gap_s * sr), dtype=np.float32))
    return np.concatenate(out)


def _scam_call():
    return Call(
        call_id="t_scam", label_scam=1, label_voice="synthetic",
        scenario="digital_arrest",
        turns=[
            Turn(index=0, speaker="caller", act="GREET",
                 text="Namaste, Sharma ji baat kar rahe hain", t_start=0.0, t_end=2.6),
            Turn(index=1, speaker="caller", act="AUTHORITY_ASSERT",
                 text="Main cyber crime branch se inspector bol raha hoon",
                 t_start=3.0, t_end=6.4),
            Turn(index=2, speaker="caller", act="THREAT",
                 text="Aapka account das minute mein block ho jayega",
                 t_start=6.8, t_end=10.2),
            Turn(index=3, speaker="caller", act="REQUEST_SENSITIVE",
                 text="Turant OTP batao aur kisi ko mat batana",
                 t_start=10.6, t_end=14.0),
        ],
    )


# --------------------------------------------------------------------------
# PIM
# --------------------------------------------------------------------------


def test_syllable_emphasis_var_measures_emphasis():
    """Identical syllables mean no emphasis; alternating levels mean a lot.

    This is the regression test for a real bug: the feature originally used a
    frame-level energy standard deviation, which measures the vowel to
    consonant alternation rather than emphasis, and it measured HIGHER for
    synthetic speech than for human speech.
    """
    flat = P.syllable_emphasis_var(_syllables([0.5] * 8), SR)
    animated = P.syllable_emphasis_var(
        _syllables([0.9, 0.25, 0.8, 0.2, 1.0, 0.3, 0.7, 0.22]), SR)
    assert flat < 1.0, f"flat delivery should show little emphasis, got {flat:.2f}"
    assert animated > 3.0, f"animated delivery should show emphasis, got {animated:.2f}"
    assert animated > flat + 2.0
    print(f"  emphasis_var: flat {flat:.2f} dB, animated {animated:.2f} dB")


def test_prosody_measures_are_finite_on_degenerate_input():
    for name, x in [
        ("silence", np.zeros(SR, dtype=np.float32)),
        ("too short", np.zeros(100, dtype=np.float32)),
        ("dc", np.ones(SR, dtype=np.float32)),
        ("noise", np.random.default_rng(0).normal(0, .1, SR).astype(np.float32)),
    ]:
        v = P.syllable_emphasis_var(x, SR)
        assert np.isfinite(v) and v >= 0.0, f"{name}: {v}"
        c = P.segment_components(x, SR)
        if c is not None:
            assert all(np.isfinite(list(c.values()))), f"{name}: {c}"
    print("  emphasis and components stay finite on silence, DC, noise and short input")


def test_pim_is_high_for_flat_delivery_and_low_for_animated():
    """The core claim, on identical wording."""
    call = _scam_call()
    lex, idx = P.lexical_arousal(call)
    assert lex.max() > 0.5, "the scam turns should carry lexical pressure"

    flat = np.full(len(lex), 0.18)
    animated = np.linspace(0.30, 0.90, len(lex))

    a = P.prosody_intent_mismatch(lex, flat, idx)
    b = P.prosody_intent_mismatch(lex, animated, idx)
    assert a["pim"] > 0.35, f"flat delivery should score high, got {a['pim']:.3f}"
    assert b["pim"] < 0.20, f"animated delivery should score low, got {b['pim']:.3f}"
    assert a["pim"] > 3 * b["pim"]
    print(f"  PIM: flat {a['pim']:.3f} against animated {b['pim']:.3f}")


def test_pim_stays_low_when_there_is_no_pressure():
    """A benign call delivered flatly must not be flagged.

    Without this the feature would fire on every recorded announcement, which
    is exactly the false positive the project says it cares about.
    """
    call = Call(
        call_id="t_benign", label_scam=0, label_voice="synthetic",
        scenario="appointment_reminder",
        turns=[
            Turn(index=0, speaker="caller", act="GREET",
                 text="Namaste, ye ek reminder call hai", t_start=0.0, t_end=2.5),
            Turn(index=1, speaker="caller", act="INFORM",
                 text="Aapka appointment kal subah gyarah baje hai",
                 t_start=3.0, t_end=6.0),
            Turn(index=2, speaker="caller", act="CLOSE",
                 text="Dhanyavaad, aapka din accha rahe", t_start=6.4, t_end=8.8),
        ],
    )
    lex, idx = P.lexical_arousal(call)
    flat = np.full(len(lex), 0.15)
    out = P.prosody_intent_mismatch(lex, flat, idx)
    assert out["pim"] < 0.25, f"benign flat call should stay low, got {out['pim']:.3f}"
    print(f"  benign flat call PIM {out['pim']:.3f} (lex max {lex.max():.2f})")


def test_anchors_are_ordered_and_loadable():
    a = P.anchors()
    assert set(a) == set(P.DEFAULT_ANCHORS)
    for k, (lo, hi) in a.items():
        assert np.isfinite(lo) and np.isfinite(hi) and hi > lo, f"{k}: ({lo}, {hi})"
    assert abs(sum(P.ACOUSTIC_WEIGHTS.values()) - 1.0) < 1e-9, "weights must sum to 1"
    assert set(P.ACOUSTIC_WEIGHTS) == set(P.DEFAULT_ANCHORS)
    print(f"  {len(a)} anchors, weights sum to 1.0")


def test_anchor_override_changes_the_scale():
    original = dict(P.anchors())
    try:
        x = _syllables([0.9, 0.25, 0.8, 0.2, 1.0, 0.3])
        before = P.segment_arousal(x, SR)["arousal"]
        P.set_anchors({k: (v[0] * 0.01, v[1] * 0.02) for k, v in original.items()})
        after = P.segment_arousal(x, SR)["arousal"]
        assert after >= before, "shrinking the anchor range should raise arousal"
    finally:
        P.set_anchors(original)
    print("  anchors actually control the arousal scale")


# --------------------------------------------------------------------------
# explainers
# --------------------------------------------------------------------------


def test_linear_shap_is_exact_and_sums_to_the_logit():
    w = np.array([1.4, -0.8, 0.3, 0.0])
    z = np.array([1.2, 0.5, -2.0, 5.0])
    names = ["as_score", "spk_consistency", "pim", "cmi"]
    out = linear_shap(w, z, names)
    total = sum(c["contribution"] for c in out)
    assert abs(total - float(w @ z)) < 1e-9, f"{total} vs {float(w @ z)}"
    assert abs([c for c in out if c["feature"] == "cmi"][0]["contribution"]) < 1e-12
    print(f"  exact linear SHAP sums to w.z = {total:+.4f}")


def test_kernel_shap_matches_the_closed_form_on_a_linear_model():
    w = np.array([1.4, -0.8, 0.3])
    z = np.array([1.2, 0.5, -2.0])
    names = ["as_score", "spk_consistency", "pim"]

    def f(M):
        return 1.0 / (1.0 + np.exp(-(np.asarray(M) @ w)))

    exact = {c["feature"]: c["contribution"] for c in linear_shap(w, z, names)}
    approx = {c["feature"]: c["contribution"]
              for c in kernel_shap(f, z, np.zeros(3), names, n_samples=800, seed=1)}
    for k in names:
        assert abs(exact[k] - approx[k]) < 0.05, f"{k}: {exact[k]} vs {approx[k]}"
    print("  kernel SHAP agrees with the closed form to within 0.05")


# --------------------------------------------------------------------------
# model and streaming
# --------------------------------------------------------------------------


def test_fusion_model_trains_calibrates_and_round_trips(tmp_path=None):
    import tempfile

    rng = np.random.default_rng(0)
    n, d = 300, len(FUSION_FEATURE_NAMES)
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, d)) * 0.4
    for j, name in enumerate(FUSION_FEATURE_NAMES):
        if name in ("as_score", "intent_score", "pim"):
            X[:, j] += y * 1.3

    m = FusionModel("logreg", "full").fit(X, y)
    p = m.predict_proba(X)
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert float(((p > 0.5).astype(int) == y).mean()) > 0.85

    feats = {k: float(v) for k, v in zip(FUSION_FEATURE_NAMES, X[0])}
    p1 = m.predict_one(feats)
    path = str(Path(tempfile.gettempdir()) / "fusion_test.joblib")
    m.save(path)
    p2 = FusionModel.load(path).predict_one(feats)
    assert abs(p1 - p2) < 1e-9, f"{p1} vs {p2}"

    contrib = m.explain(feats)
    assert len(contrib) == len(m.names)
    print(f"  fusion trains ({m.calibrator}), round trips, explains {len(contrib)} features")


def test_every_ablation_arm_sees_only_its_own_features():
    assert set(ABLATION_ARMS["audio_only"]).isdisjoint(ABLATION_ARMS["text_only"])
    assert set(ABLATION_ARMS["late_fusion"]) == (
        set(ABLATION_ARMS["audio_only"]) | set(ABLATION_ARMS["text_only"]))
    assert set(ABLATION_ARMS["full"]) == set(FUSION_FEATURE_NAMES)
    assert len(ABLATION_ARMS["full"]) > len(ABLATION_ARMS["late_fusion"]), \
        "the cross-modal block must add features beyond late fusion"
    print({k: len(v) for k, v in ABLATION_ARMS.items()})


def test_streaming_is_ordered_and_bounded():
    call = _scam_call()
    tl = list(stream_call(call))
    assert len(tl) == len(call.turns)
    assert [e["t_end"] for e in tl] == sorted(e["t_end"] for e in tl)
    for e in tl:
        assert 0.0 <= e["risk"] <= 1.0
        assert e["turn"]["index"] == e["turn_index"]
    t, k = time_to_detection(tl, threshold=0.0)
    assert t == tl[0]["t_end"] and k == tl[0]["turn_index"], \
        "a zero threshold must alert on the very first turn"
    t2, _ = time_to_detection(tl, threshold=1.01)
    assert t2 is None, "an unreachable threshold must never alert"
    print(f"  streamed {len(tl)} turns, risk {tl[0]['risk']:.3f} to {tl[-1]['risk']:.3f}")


def test_hard_subset_metrics_scores_only_the_hard_calls():
    """The ablation's hard subset picks mild scams and hard negatives, and
    reports every arm on those alone."""
    from types import SimpleNamespace

    from swarkavach.evaluate import hard_subset_metrics

    def call(label, **meta):
        return SimpleNamespace(label_scam=label, meta=meta)

    calls = [
        call(1, mild_scam=True), call(1, mild_scam=True), call(1, mild_scam=True),
        call(1, mild_scam=True), call(1),                 call(1),
        call(0, hard_negative=True), call(0, hard_negative=True),
        call(0, hard_negative=True), call(0, hard_negative=True),
        call(0), call(0),
    ]
    y = np.array([c.label_scam for c in calls])
    # a text arm that is perfect on easy calls and blind on hard ones, and a
    # full arm that is right everywhere
    text = np.array([0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1])
    full = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    out = hard_subset_metrics(calls, y, {"text_only": text, "full": full}, threshold=0.65)
    assert out["n"] == 8 and out["n_scam"] == 4 and out["n_benign"] == 4
    assert out["full"]["auc"] == 1.0 and out["full"]["recall"] == 1.0
    # ties everywhere: chance
    assert abs(out["text_only"]["auc"] - 0.5) < 1e-9
    assert out["text_only"]["recall"] == 0.0

    # too few hard cases, or one class only: a note, not a crash
    few = hard_subset_metrics(calls[:3] + calls[6:7], y[[0, 1, 2, 6]],
                              {"full": full[[0, 1, 2, 6]]}, threshold=0.65)
    assert "note" in few and "full" not in few
    one_class = hard_subset_metrics(calls[:4] + calls[4:6] * 2, np.ones(8),
                                    {"full": np.ones(8)}, threshold=0.65)
    assert "note" in one_class


TESTS = [
    test_hard_subset_metrics_scores_only_the_hard_calls,
    test_syllable_emphasis_var_measures_emphasis,
    test_prosody_measures_are_finite_on_degenerate_input,
    test_pim_is_high_for_flat_delivery_and_low_for_animated,
    test_pim_stays_low_when_there_is_no_pressure,
    test_anchors_are_ordered_and_loadable,
    test_anchor_override_changes_the_scale,
    test_linear_shap_is_exact_and_sums_to_the_logit,
    test_kernel_shap_matches_the_closed_form_on_a_linear_model,
    test_fusion_model_trains_calibrates_and_round_trips,
    test_every_ablation_arm_sees_only_its_own_features,
    test_streaming_is_ordered_and_bounded,
]


if __name__ == "__main__":
    import traceback

    failures = 0
    for fn in TESTS:
        print(f"[{fn.__name__}]")
        try:
            fn()
            print("  PASS\n")
        except Exception:
            failures += 1
            traceback.print_exc()
            print("  FAIL\n")
    print("=" * 60)
    print(f"{len(TESTS) - failures}/{len(TESTS)} passed")
    sys.exit(1 if failures else 0)
