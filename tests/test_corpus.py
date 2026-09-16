"""Tests for the Hinglish fraud-call corpus generator.

Plain asserts, no test framework required. Run either way:

    .venv\\Scripts\\python.exe tests\\test_corpus.py
    .venv\\Scripts\\python.exe -m pytest tests/test_corpus.py

The checks that matter most are the ones a corpus like this normally gets
wrong: that the gold BIO really does correspond to the text the slots
inserted, and that the train / dev / test split is speaker disjoint rather
than call disjoint.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swarkavach.audioio import read_audio  # noqa: E402
from swarkavach.config import DEFAULT_SPLIT, N_SPEAKERS, SETTINGS  # noqa: E402
from swarkavach.corpus import generator, grammar, tts  # noqa: E402
from swarkavach.corpus.lexicon import BENIGN_MARKERS, lexicon_hits  # noqa: E402
from swarkavach.schema import (  # noqa: E402
    BENIGN_SCENARIOS,
    COERCION_RANK,
    DIALOGUE_ACTS,
    ENTITY_TYPES,
    LANG_TAGS,
    SCAM_SCENARIOS,
    decode_bio,
    tokenize,
)

SEED = SETTINGS.pipeline.seed
N_CALLS = 120

_CACHE: dict = {}


def corpus():
    """One corpus, generated once and shared by every test."""
    if "calls" not in _CACHE:
        _CACHE["calls"] = generator.generate_corpus(
            n_calls=N_CALLS, seed=SEED, write=False
        )
    return _CACHE["calls"]


# --------------------------------------------------------------------------
# Grammar
# --------------------------------------------------------------------------


def test_grammar_has_no_broken_templates():
    report = grammar.validate_grammar()
    assert report["problems"] == [], report["problems"][:10]
    assert report["patterns_rendered"] > 300


def test_every_scenario_has_enough_templates():
    counts = grammar.template_counts()
    for scenario in list(SCAM_SCENARIOS) + list(BENIGN_SCENARIOS):
        assert scenario in counts, f"{scenario} missing from the grammar"
        assert counts[scenario] >= 12, f"{scenario} has only {counts[scenario]}"


def test_every_slot_has_enough_fillers():
    for name, spec in grammar.SLOTS.items():
        if name in grammar.GENDERED_SLOTS and name != "NAME":
            continue  # two grammatical genders, by definition
        if name in grammar.FIXED_FORM_SLOTS:
            continue  # deliberately one surface form, see the slot comment
        assert len(spec.options) >= 8, f"slot {name} has {len(spec.options)}"


def test_no_entity_mention_escapes_a_slot():
    """The gold BIO is complete only if literal text never names an entity."""
    leaks = grammar.literal_entity_leaks()
    assert leaks == [], leaks[:10]


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_generation_is_deterministic():
    a = generator.generate_corpus(n_calls=24, seed=SEED, write=False)
    b = generator.generate_corpus(n_calls=24, seed=SEED, write=False)
    assert [c.to_dict() for c in a] == [c.to_dict() for c in b]


def test_different_seed_gives_different_corpus():
    a = generator.generate_corpus(n_calls=24, seed=SEED, write=False)
    b = generator.generate_corpus(n_calls=24, seed=SEED + 1, write=False)
    assert [c.to_dict() for c in a] != [c.to_dict() for c in b]


# --------------------------------------------------------------------------
# Per-turn annotation integrity
# --------------------------------------------------------------------------


def test_turn_fields_are_the_same_length():
    for call in corpus():
        for turn in call.turns:
            n = len(turn.tokens)
            assert n > 0, f"{call.call_id} turn {turn.index} has no tokens"
            assert len(turn.bio) == n, f"{call.call_id} turn {turn.index} bio"
            assert len(turn.lang) == n, f"{call.call_id} turn {turn.index} lang"
            assert tokenize(turn.text) == turn.tokens, (
                f"{call.call_id} turn {turn.index}: text does not re-tokenise "
                f"to the stored tokens"
            )


def test_bio_sequences_are_well_formed():
    valid = {"O"} | {f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")}
    for call in corpus():
        for turn in call.turns:
            prev = "O"
            for tag in turn.bio:
                assert tag in valid, f"{call.call_id}: bad tag {tag}"
                if tag.startswith("I-"):
                    etype = tag[2:]
                    assert prev in (f"B-{etype}", f"I-{etype}"), (
                        f"{call.call_id} turn {turn.index}: {tag} with no B- "
                        f"before it in {turn.text!r}"
                    )
                prev = tag


def test_language_tags_are_valid():
    for call in corpus():
        for turn in call.turns:
            for tag in turn.lang:
                assert tag in LANG_TAGS, f"{call.call_id}: bad lang tag {tag}"


def test_acts_and_speakers_are_valid():
    for call in corpus():
        for turn in call.turns:
            assert turn.act in DIALOGUE_ACTS, f"{call.call_id}: {turn.act}"
            assert turn.speaker in ("caller", "callee")
        assert any(t.speaker == "callee" for t in call.turns), (
            f"{call.call_id} has no callee turns, it is not a two-party call"
        )


def test_entities_decode_to_the_strings_the_slots_inserted():
    """The whole automatic-gold claim, checked end to end.

    Rebuild each call from its recorded seed, collect the slot values the
    grammar actually inserted, and require the decoded BIO spans to be exactly
    those values, in order, with the same entity types.
    """
    checked = 0
    for call in corpus():
        rng = random.Random(generator.derive_seed(SEED, f"call-{call.call_id}"))
        drafts, _ = grammar.build_dialogue(
            call.scenario, rng, call.meta["style"]
        )
        assert len(drafts) == len(call.turns), call.call_id
        for draft, turn in zip(drafts, call.turns):
            assert draft.text == turn.text, f"{call.call_id} turn {turn.index}"
            expected = [
                (etype, " ".join(tokenize(value)).lower())
                for _slot, value, etype in draft.slots
                if etype is not None
            ]
            got = [
                (span.type, span.text.lower())
                for span in decode_bio(turn.tokens, turn.bio, turn.index)
            ]
            assert got == expected, (
                f"{call.call_id} turn {turn.index}: decoded {got} but the "
                f"slots inserted {expected} in {turn.text!r}"
            )
            checked += len(expected)
    assert checked > 100, f"only {checked} entities checked"


def test_timings_increase_monotonically():
    for call in corpus():
        prev_end = -1.0
        for turn in call.turns:
            assert turn.t_end > turn.t_start, f"{call.call_id} turn {turn.index}"
            assert turn.t_start >= prev_end, f"{call.call_id} turn {turn.index}"
            prev_end = turn.t_end
        assert call.duration > 1.0


# --------------------------------------------------------------------------
# Corpus design
# --------------------------------------------------------------------------


def test_splits_are_speaker_disjoint():
    """The single most common way a project like this cheats."""
    by_split: dict = {}
    for call in corpus():
        by_split.setdefault(call.split, set()).add(call.speaker_id)
    splits = sorted(by_split)
    assert len(splits) >= 2, "corpus has fewer than two splits"
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            overlap = by_split[a] & by_split[b]
            assert not overlap, f"speakers in both {a} and {b}: {sorted(overlap)}"
    seen: Counter = Counter()
    for speakers in by_split.values():
        seen.update(speakers)
    assert max(seen.values()) == 1, "a speaker appears in more than one split"


def test_speaker_split_assignment_is_a_partition():
    assignment = generator.assign_speaker_splits(SEED)
    assert len(assignment) == N_SPEAKERS
    assert set(assignment.values()) <= set(DEFAULT_SPLIT)
    counts = Counter(assignment.values())
    for split in DEFAULT_SPLIT:
        assert counts[split] >= 1, f"split {split} has no speakers"


def test_all_four_cells_are_populated():
    cells = Counter(c.cell for c in corpus())
    expected = {"humanxbenign", "humanxscam", "clonedxbenign", "clonedxscam"}
    assert set(cells) == expected, cells
    for cell in expected:
        assert cells[cell] > 0, f"cell {cell} is empty"
    smallest, largest = min(cells.values()), max(cells.values())
    assert largest - smallest <= max(4, 0.25 * largest), f"cells unbalanced: {cells}"

    per_split: dict = {}
    for call in corpus():
        per_split.setdefault(call.split, Counter())[call.cell] += 1
    for split, counter in per_split.items():
        assert set(counter) == expected, f"{split} is missing cells: {counter}"


def test_labels_are_roughly_balanced():
    calls = corpus()
    n_scam = sum(c.label_scam for c in calls)
    n_synth = sum(1 for c in calls if c.label_voice == "synthetic")
    assert abs(n_scam / len(calls) - 0.5) < 0.06, n_scam
    assert abs(n_synth / len(calls) - 0.5) < 0.06, n_synth


def test_every_scenario_appears():
    scenarios = Counter(c.scenario for c in corpus())
    for name in list(SCAM_SCENARIOS) + list(BENIGN_SCENARIOS):
        assert scenarios[name] > 0, f"scenario {name} never generated"


def test_hard_negatives_and_mild_scams_exist():
    calls = corpus()
    hard_negatives = [c for c in calls if c.meta.get("hard_negative")]
    mild = [c for c in calls if c.meta.get("mild_scam")]
    assert len(hard_negatives) >= 0.15 * len(calls), len(hard_negatives)
    assert len(mild) >= 0.05 * len(calls), len(mild)
    # Hard negatives have to actually look like scams to a bag-of-words model.
    with_bank_or_otp = 0
    for call in hard_negatives:
        types = {span.type for span in call.entities()}
        if types & {"BANK_ENTITY", "OTP", "PERSONAL_INFO_REQ"}:
            with_bank_or_otp += 1
    assert with_bank_or_otp >= 0.6 * len(hard_negatives), (
        f"only {with_bank_or_otp}/{len(hard_negatives)} hard negatives mention "
        "bank, OTP or personal-info entities"
    )


def test_benign_calls_carry_anti_fraud_markers():
    """Benign hard negatives must say what a real institution says."""
    hits = 0
    total = 0
    for call in corpus():
        if call.label_scam or not call.meta.get("hard_negative"):
            continue
        total += 1
        if lexicon_hits(call.full_text())["benign"]:
            hits += 1
    assert total > 0
    assert hits >= 0.7 * total, f"only {hits}/{total} hard negatives carry markers"


def test_scam_calls_are_more_coercive_than_benign():
    def mean_rank(call):
        ranks = [COERCION_RANK.get(t.act, 0.0) for t in call.turns]
        return sum(ranks) / len(ranks)

    scam = [mean_rank(c) for c in corpus() if c.label_scam]
    benign = [mean_rank(c) for c in corpus() if not c.label_scam]
    assert scam and benign
    m_scam = sum(scam) / len(scam)
    m_benign = sum(benign) / len(benign)
    assert m_scam > m_benign + 0.10, (scam and m_scam, m_benign)


def test_mild_scams_are_lexically_milder_than_hard_scams():
    """The mild arm exists to defeat the text branch, so prove it is milder."""
    def pressure(call):
        hits = lexicon_hits(call.full_text())
        return sum(w for key in ("threat", "urgency", "isolation")
                   for _term, w in hits[key])

    mild = [pressure(c) for c in corpus() if c.meta.get("style") == "mild"]
    hard = [pressure(c) for c in corpus() if c.meta.get("style") == "hard"]
    assert mild and hard
    assert sum(mild) / len(mild) < sum(hard) / len(hard)


def test_code_mixing_is_not_degenerate():
    stats = generator.corpus_stats(corpus())
    # CMI is kept on the [0, 1] scale here, matching text.langid, so that the
    # fusion vector has exactly one definition of it.
    assert 0.05 < stats["mean_cmi"] < 0.50, stats["mean_cmi"]
    assert stats["lang_tokens"]["hi"] > 0 and stats["lang_tokens"]["en"] > 0
    en_share = stats["lang_tokens"]["en"] / (
        stats["lang_tokens"]["hi"] + stats["lang_tokens"]["en"]
    )
    assert 0.05 < en_share < 0.60, en_share
    assert stats["switch_points"] > stats["n_turns"]


def test_corpus_stats_shape():
    stats = generator.corpus_stats(corpus())
    for key in (
        "n_calls", "n_turns", "n_tokens", "entities", "acts", "scenarios",
        "cells", "splits", "mean_cmi", "mean_coercion", "cells_by_split",
    ):
        assert key in stats, key
    assert stats["n_calls"] == len(corpus())
    assert sum(stats["entities"].values()) == stats["n_entities"]
    assert set(stats["entities"]) == set(ENTITY_TYPES)


# --------------------------------------------------------------------------
# Ethics: nothing real in the generated text
# --------------------------------------------------------------------------

REAL_BANK_NAMES = (
    "sbi", "state bank", "hdfc", "icici", "axis bank", "kotak", "pnb",
    "punjab national", "canara", "yes bank", "indusind", "idfc", "bandhan",
    "union bank", "bank of baroda", "paytm", "phonepe", "google pay", "gpay",
)


def test_no_real_bank_or_payment_brands():
    for call in corpus():
        text = call.full_text().lower()
        for name in REAL_BANK_NAMES:
            assert name not in text, f"{call.call_id} mentions {name!r}"


def test_no_phone_number_shaped_digit_runs():
    for call in corpus():
        for turn in call.turns:
            for token in turn.tokens:
                if token.isdigit():
                    assert len(token) <= 6, (
                        f"{call.call_id}: {token} looks like a real number"
                    )


# --------------------------------------------------------------------------
# Disk round trip
# --------------------------------------------------------------------------


def test_write_and_load_round_trip():
    tmp = Path(tempfile.mkdtemp(prefix="swarkavach_corpus_"))
    try:
        written = generator.generate_corpus(
            n_calls=32, seed=SEED, out_dir=tmp, write=True
        )
        loaded = generator.load_corpus(corpus_dir=tmp)
        assert len(loaded) == len(written)
        by_id = {c.call_id: c for c in loaded}
        for call in written:
            assert by_id[call.call_id].to_dict() == call.to_dict()

        manifest = generator.load_manifest(tmp)
        assert manifest["params"]["seed"] == SEED
        assert manifest["counts"]["calls"] == len(written)
        assert set(manifest["cells"]) == {
            "humanxbenign", "humanxscam", "clonedxbenign", "clonedxscam"
        }
        assert len(manifest["calls"]) == len(written)
        json.dumps(manifest)  # must stay JSON serialisable

        train = generator.load_corpus(split="train", corpus_dir=tmp)
        assert train and all(c.split == "train" for c in train)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------


def test_sim_backend_is_always_available():
    backends = tts.AVAILABLE_BACKENDS()
    assert backends["sim"] is True
    assert set(backends) == {"sim", "sapi", "edge", "xtts"}


def test_xtts_backend_raises_a_useful_message():
    call = corpus()[0]
    try:
        tts.render_call_audio(call, backend="xtts", write=False)
    except RuntimeError as exc:
        assert "Colab" in str(exc) and "sim" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("xtts should refuse to run locally")


def test_audio_duration_matches_turn_timings():
    tmp = Path(tempfile.mkdtemp(prefix="swarkavach_audio_"))
    try:
        calls = generator.generate_corpus(n_calls=8, seed=SEED, write=False)
        for call in calls[:2]:
            path = tmp / f"{call.call_id}.wav"
            signal, info = tts.render_call_audio(
                call, backend="sim", out_path=path, sr=8000
            )
            assert path.exists()
            x, sr = read_audio(path, sr=None)
            assert sr == 8000
            wav_s = x.size / sr
            last = call.turns[-1].t_end
            assert abs(wav_s - last) < 0.5, (wav_s, last)
            assert abs(signal.size / 8000 - wav_s) < 0.01
            assert call.audio_path == str(path)
            assert call.audio_source == "sim"
            assert info["backend"] == "sim"
            assert len(info["spans"]) == len(call.turns)
            for turn, (start, end) in zip(call.turns, info["spans"]):
                assert abs(turn.t_start - start) < 1e-6
                assert abs(turn.t_end - end) < 1e-6
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_human_and_cloned_audio_differ_measurably():
    """The two paths must not be cosmetically different only."""
    from swarkavach.corpus.synth import _measure, make_voice, synthesize_turn

    text = "Aapka account block ho jayega, OTP 445566 abhi bataiye"
    for speaker in ("spk_01", "spk_09"):
        voice = make_voice(speaker, SEED)
        human = synthesize_turn(text, 8000, voice, synthetic=False, seed=3)
        cloned = synthesize_turn(text, 8000, voice, synthetic=True, seed=3)
        assert human.shape[0] > 8000 and cloned.shape[0] > 8000
        mh, mc = _measure(human, 8000), _measure(cloned, 8000)
        assert mc["jitter"] < mh["jitter"], (speaker, mh, mc)
        assert mc["shimmer"] < mh["shimmer"], (speaker, mh, mc)
        assert mc["periodicity"] > mh["periodicity"], (speaker, mh, mc)
        assert mc["flatness_var"] < mh["flatness_var"], (speaker, mh, mc)


def test_voices_are_stable_and_speaker_specific():
    from swarkavach.corpus.synth import make_voice, synthesize_turn

    a1 = make_voice("spk_04", SEED)
    a2 = make_voice("spk_04", SEED)
    b = make_voice("spk_05", SEED)
    assert a1 == a2
    assert a1["f0"] != b["f0"] or a1["vt_scale"] != b["vt_scale"]

    text = "Namaste, main bol raha hoon"
    x1 = synthesize_turn(text, 8000, a1, False, seed=2)
    x2 = synthesize_turn(text, 8000, a2, False, seed=2)
    x3 = synthesize_turn(text, 8000, b, False, seed=2)
    assert (x1 == x2).all(), "synthesis is not deterministic"
    n = min(x1.size, x3.size)
    assert not (x1[:n] == x3[:n]).all(), "two speakers produced the same signal"


# --------------------------------------------------------------------------
# Topic pairing and the neutral pool
#
# These exist because text-only scored AUC 1.000 on the test split with the
# whole fraud lexicon deleted. Scenario determined label exactly, and the two
# classes shared no phrasing at all. Both are checked here so they cannot
# quietly come back.
# --------------------------------------------------------------------------


def test_every_topic_pair_has_one_scenario_per_class():
    from swarkavach.corpus.grammar import TOPIC_PAIRS
    from swarkavach.schema import BENIGN_SCENARIOS, SCAM_SCENARIOS

    for topic, pair in TOPIC_PAIRS.items():
        assert pair["scam"] in SCAM_SCENARIOS, f"{topic}: {pair['scam']} is not a scam scenario"
        assert pair["benign"] in BENIGN_SCENARIOS, f"{topic}: {pair['benign']} is not benign"


def test_label_is_balanced_within_every_paired_topic():
    """The subject of a call must say nothing about its label."""
    from swarkavach.corpus.generator import plan_corpus
    from swarkavach.corpus.grammar import TOPIC_PAIRS, topic_of

    paired = {v for pair in TOPIC_PAIRS.values() for v in pair.values()}
    entries, _ = plan_corpus(n_calls=300, seed=3)
    by_topic: dict = {}
    for e in entries:
        if e["scenario"] in paired:
            by_topic.setdefault(topic_of(e["scenario"]), [0, 0])[e["label_scam"]] += 1
    assert set(by_topic) == set(TOPIC_PAIRS), "every paired topic must be planned"
    for topic, (n_benign, n_scam) in by_topic.items():
        assert n_benign > 0 and n_scam > 0, f"{topic} has only one class"
        share = n_scam / (n_benign + n_scam)
        assert 0.35 <= share <= 0.65, f"{topic} is {share:.0%} scam, that is a leak"


def test_paired_topics_share_their_setup_lines():
    """Both variants of a topic describe the situation in the same words."""
    from swarkavach.corpus.grammar import SCENARIO_ACTS, TOPIC_PAIRS, TOPIC_SPINE

    for topic, acts in TOPIC_SPINE.items():
        scam = SCENARIO_ACTS[TOPIC_PAIRS[topic]["scam"]]
        benign = SCENARIO_ACTS[TOPIC_PAIRS[topic]["benign"]]
        for act, lines in acts.items():
            for line in lines:
                assert line in scam.get(act, ()), f"{topic}/{act} spine line missing from scam side"
                assert line in benign.get(act, ()), f"{topic}/{act} spine line missing from benign side"


def test_no_ordinary_act_belongs_to_one_class():
    """Every dialogue act a benign call can perform, a scam can too, and the
    other way round, except the coercive ones. SMALLTALK used to appear in no
    scam arc, so one chatty turn identified the class."""
    from swarkavach.corpus.grammar import (
        COERCIVE_ACTS, MILD_ARCS, SCENARIO_ARCS, TOPIC_PAIRS, TOPIC_SPINE, parse_arc,
    )
    from swarkavach.schema import SCAM_SCENARIOS

    reach: dict = {}
    for scen, arcs in SCENARIO_ARCS.items():
        cls = "scam" if scen in SCAM_SCENARIOS else "benign"
        for arc in arcs:
            for act, _ in parse_arc(arc):
                reach.setdefault(act, set()).add(cls)
    for arc in MILD_ARCS:
        for act, _ in parse_arc(arc):
            reach.setdefault(act, set()).add("scam")
    lonely = sorted(a for a, c in reach.items() if len(c) == 1 and a not in COERCIVE_ACTS)
    assert not lonely, f"acts only one class can perform: {lonely}"
    for act in COERCIVE_ACTS:
        assert reach.get(act) == {"scam"}, f"{act} must stay scam-only"

    # and the spine's acts must be reachable on both sides of every pair,
    # or the shared lines are shared in name only
    for topic, acts in TOPIC_SPINE.items():
        for side in ("scam", "benign"):
            reachable = {a for arc in SCENARIO_ARCS[TOPIC_PAIRS[topic][side]]
                         for a, _ in parse_arc(arc)}
            missing = sorted(set(acts) - reachable)
            assert not missing, f"{topic}/{side}: spine acts no arc reaches: {missing}"


def test_neutral_pool_reaches_both_classes():
    """A greeting or a close must be available to a scam and a benign call alike."""
    from swarkavach.corpus.grammar import NEUTRAL_CALLER, _pattern_pool

    for act in ("GREET", "CLOSE"):
        scam = {p for _, p in _pattern_pool("kyc_freeze", act, "caller", "hard")}
        benign = {p for _, p in _pattern_pool("bank_reminder", act, "caller", "flat")}
        for line in NEUTRAL_CALLER[act]:
            assert line in scam, f"{act}: neutral line not offered to a hard scam"
            assert line in benign, f"{act}: neutral line not offered to a benign call"


def test_hard_negatives_carry_deadline_and_problem_state():
    """A real bank reminder names a real deadline. Only scams used to."""
    from swarkavach.corpus.grammar import SCENARIO_ACTS, SCENARIO_ARCS

    for scen in ("bank_reminder", "delivery_otp"):
        acts = SCENARIO_ACTS[scen]
        assert "DEADLINE" in acts and acts["DEADLINE"], f"{scen} has no DEADLINE lines"
        assert "PROBLEM_STATE" in acts and acts["PROBLEM_STATE"], f"{scen} has no PROBLEM_STATE lines"
        assert any("DEADLINE" in arc for arc in SCENARIO_ARCS[scen]), f"{scen}: no arc reaches DEADLINE"


def test_pressure_acts_stay_scam_only():
    """The neutral pool must not hand a benign call the coercive lines."""
    from swarkavach.corpus.grammar import _pattern_pool
    from swarkavach.schema import BENIGN_SCENARIOS

    for scen in BENIGN_SCENARIOS:
        for act in ("THREAT", "ISOLATE", "PRESSURE_ESCALATE", "AUTHORITY_ASSERT"):
            got = _pattern_pool(scen, act, "caller", "flat")
            assert not got, f"{scen} can draw {act}: {got[0][1]!r}"


# --------------------------------------------------------------------------
# Grammar fingerprint
#
# A corpus that came back from Drive is only reusable if it was generated
# from the templates this clone carries. A whole session of template fixes
# once never reached a result because the Drive corpus predated them and
# every stage skipped itself.
# --------------------------------------------------------------------------


def test_grammar_fingerprint_is_stable_and_sensitive():
    from swarkavach.corpus import grammar

    a = grammar.grammar_fingerprint()
    b = grammar.grammar_fingerprint()
    assert a == b and len(a) == 16

    pool = grammar.SCENARIO_ACTS["kyc_freeze"]["INFORM"]
    grammar.SCENARIO_ACTS["kyc_freeze"]["INFORM"] = pool + ("Ek naya line.",)
    try:
        assert grammar.grammar_fingerprint() != a, "a template change must change the fingerprint"
    finally:
        grammar.SCENARIO_ACTS["kyc_freeze"]["INFORM"] = pool
    assert grammar.grammar_fingerprint() == a


def test_manifest_records_the_fingerprint_and_runner_trusts_it():
    from swarkavach import config
    from swarkavach.corpus import generator, grammar
    from swarkavach.runner import _corpus_stale_reason

    tmp = Path(tempfile.mkdtemp())
    try:
        old = config.CORPUS_DIR
        config.CORPUS_DIR = tmp
        manifest = tmp / generator.MANIFEST_NAME

        # no manifest at all: nothing to be stale about
        assert _corpus_stale_reason() is None

        # a manifest from before the check
        manifest.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
        assert "predates" in (_corpus_stale_reason() or "")

        # a matching one
        manifest.write_text(json.dumps({"grammar_fingerprint": grammar.grammar_fingerprint()}),
                            encoding="utf-8")
        assert _corpus_stale_reason() is None

        # a different one
        manifest.write_text(json.dumps({"grammar_fingerprint": "0000000000000000"}),
                            encoding="utf-8")
        reason = _corpus_stale_reason() or ""
        assert "0000000000000000" in reason and grammar.grammar_fingerprint() in reason
    finally:
        config.CORPUS_DIR = old
        shutil.rmtree(tmp, ignore_errors=True)


def test_generated_manifest_carries_current_fingerprint():
    from swarkavach.corpus import grammar
    from swarkavach.corpus.generator import build_manifest, generate_corpus

    calls = generate_corpus(n_calls=24, seed=5, audio=False, write=False)
    split = {c.speaker_id: c.split for c in calls}
    m = build_manifest(calls, split, seed=5, n_calls=24, audio=False,
                       audio_backend="sim", sr=16000)
    assert m["grammar_fingerprint"] == grammar.grammar_fingerprint()
    assert "unpaired_topic_share" in m["params"]


def test_corpus_audio_resolves_on_another_machine():
    """A call rendered elsewhere carries the renderer's absolute path. On the
    console machine it must still find its WAV, or the voice branch scores
    nothing without saying so."""
    from swarkavach.corpus.generator import resolve_audio_path

    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "audio").mkdir()
        wav = tmp / "audio" / "call_0007.wav"
        wav.write_bytes(b"RIFF")
        # the path some other machine wrote
        assert resolve_audio_path("/content/svk_repo/data/corpus/audio/call_0007.wav", tmp) == wav
        # a relative path, as the generator now writes
        assert resolve_audio_path("audio/call_0007.wav", tmp) == wav
        # an absolute path that is right here
        assert resolve_audio_path(str(wav), tmp) == wav
        # and one that is nowhere
        assert resolve_audio_path("/content/svk_repo/data/corpus/audio/call_9999.wav", tmp) is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def main() -> int:
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failures = []
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures.append((name, exc))
            print(f"FAIL  {name}\n      {exc}")
        except Exception as exc:  # noqa: BLE001 - report and keep going
            failures.append((name, exc))
            print(f"ERROR {name}\n      {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
