"""Corpus assembly: templates in, annotated Call records out.

What this module guarantees, because the rest of the project depends on it:

1. Every annotation is gold by construction. The BIO tags come from the slots
   that produced the tokens, the dialogue act comes from the arc position, and
   the per-token language tag comes from lexicon.token_language. Nothing here
   is hand annotated, so nothing can drift.
2. The corpus is balanced by design. Roughly half the calls are scams and,
   independently, roughly half carry a synthetic voice, so all four cells of
   the {human, cloned} x {benign, scam} grid are populated in every split.
3. Splits are speaker disjoint. Speakers are partitioned first and calls are
   assigned to speakers afterwards, so a speaker can never appear on both
   sides of a split boundary. Splitting by call instead of by speaker is the
   usual way a project like this accidentally cheats, and tests/test_corpus.py
   asserts against it.
4. Everything is reproducible from the seed recorded in the manifest.

Turn timings are estimated from the token count when audio is off (about 2.6
tokens per second plus a short pause) and overwritten with the real span
boundaries when audio is rendered, so the streaming simulation and the
time-to-detection metric work either way.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import (
    CORPUS_DIR,
    DEFAULT_N_CALLS,
    DEFAULT_SCAM_RATIO,
    DEFAULT_SPLIT,
    DEFAULT_SYNTHETIC_RATIO,
    N_SPEAKERS,
    SETTINGS,
    TARGET_SR,
    ensure_dirs,
)
from ..schema import (
    BENIGN_SCENARIOS,
    COERCION_RANK,
    ENTITY_TYPES,
    SCAM_SCENARIOS,
    SCHEMA_VERSION,
    Call,
    Turn,
)
from . import grammar
from .lexicon import token_language

# --------------------------------------------------------------------------
# Timing model
# --------------------------------------------------------------------------

#: Speaking rate for the estimated timings. Measured against read Hinglish
#: telephone speech, which sits a little slower than conversational English.
SPEAK_RATE_TOKENS_PER_S = 2.6

#: Silence between two turns. Matches the default gap in
#: audioio.concat_with_gaps so estimated and rendered timings agree.
TURN_GAP_S = 0.25

MIN_TURN_S = 0.7
PAUSE_MIN_S = 0.10
PAUSE_MAX_S = 0.45

#: Share of scam calls generated in the lexically mild style. These are the
#: calls the text branch is expected to miss.
MILD_SCAM_RATIO = 0.22

MANIFEST_NAME = "manifest.json"


# --------------------------------------------------------------------------
# Deterministic seeding
# --------------------------------------------------------------------------


def derive_seed(seed: int, tag: str) -> int:
    """A stable child seed for `tag`.

    Python hashes strings with a per-process salt, so hash() cannot be used
    here without breaking reproducibility across runs. blake2b can.
    """
    h = hashlib.blake2b(f"{seed}:{tag}".encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "big")


# --------------------------------------------------------------------------
# Speaker pool and speaker-disjoint splits
# --------------------------------------------------------------------------


def speaker_ids(n_speakers: int = N_SPEAKERS) -> List[str]:
    return [f"spk_{i:02d}" for i in range(n_speakers)]


def _largest_remainder(total: int, weights: Dict[str, float]) -> Dict[str, int]:
    """Split `total` into integer parts by weight, without losing an item."""
    keys = list(weights)
    raw = {k: total * weights[k] / sum(weights.values()) for k in keys}
    out = {k: int(math.floor(raw[k])) for k in keys}
    left = total - sum(out.values())
    order = sorted(keys, key=lambda k: raw[k] - math.floor(raw[k]), reverse=True)
    for i in range(left):
        out[order[i % len(order)]] += 1
    return out


def assign_speaker_splits(
    seed: int,
    n_speakers: int = N_SPEAKERS,
    proportions: Optional[Dict[str, float]] = None,
) -> Dict[str, str]:
    """Partition the speaker pool across splits. One speaker, one split.

    The split boundary is drawn here and nowhere else. Calls inherit their
    split from their speaker, which is what makes the corpus honest.
    """
    proportions = dict(proportions or DEFAULT_SPLIT)
    speakers = speaker_ids(n_speakers)
    rng = random.Random(derive_seed(seed, "speaker-split"))
    rng.shuffle(speakers)
    counts = _largest_remainder(len(speakers), proportions)
    out: Dict[str, str] = {}
    i = 0
    for split in ("train", "dev", "test"):
        for _ in range(counts.get(split, 0)):
            out[speakers[i]] = split
            i += 1
    for spk in speakers[i:]:  # any split name outside the standard three
        out[spk] = "train"
    return out


# --------------------------------------------------------------------------
# Call plan
# --------------------------------------------------------------------------


def _plan_split(
    n_calls: int,
    split: str,
    speakers: Sequence[str],
    rng: random.Random,
    scam_ratio: float,
    synthetic_ratio: float,
    mild_ratio: float,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Lay out one split so all four cells and every scenario are covered."""
    n_scam = int(round(n_calls * scam_ratio))
    n_benign = n_calls - n_scam

    entries: List[Dict[str, Any]] = []
    for i in range(n_scam):
        entries.append(
            {
                "label_scam": 1,
                "scenario": SCAM_SCENARIOS[(i + offset) % len(SCAM_SCENARIOS)],
                "split": split,
            }
        )
    for i in range(n_benign):
        entries.append(
            {
                "label_scam": 0,
                "scenario": BENIGN_SCENARIOS[(i + offset) % len(BENIGN_SCENARIOS)],
                "split": split,
            }
        )

    # Voice label is drawn independently of the scam label, so the four cells
    # stay balanced instead of correlating with the content.
    for group in (
        [e for e in entries if e["label_scam"] == 1],
        [e for e in entries if e["label_scam"] == 0],
    ):
        rng.shuffle(group)
        n_synth = int(round(len(group) * synthetic_ratio))
        for i, entry in enumerate(group):
            entry["label_voice"] = "synthetic" if i < n_synth else "human"

    # The mild style only goes to scenarios that can carry it, so the target
    # count is taken out of the scam calls as a whole but drawn from the
    # capable subset.
    scam_entries = [e for e in entries if e["label_scam"] == 1]
    n_mild = int(round(len(scam_entries) * mild_ratio))
    capable = [
        e for e in scam_entries
        if e["scenario"] in grammar.MILD_CAPABLE_SCENARIOS
    ]
    rng.shuffle(capable)
    for entry in capable[:n_mild]:
        entry["style"] = "mild"
    for entry in entries:
        entry.setdefault("style", "hard" if entry["label_scam"] else "flat")

    rng.shuffle(entries)
    pool = list(speakers)
    rng.shuffle(pool)
    for i, entry in enumerate(entries):
        entry["speaker_id"] = pool[i % len(pool)]
    return entries


def plan_corpus(
    n_calls: int,
    seed: int,
    scam_ratio: float = DEFAULT_SCAM_RATIO,
    synthetic_ratio: float = DEFAULT_SYNTHETIC_RATIO,
    mild_ratio: float = MILD_SCAM_RATIO,
    n_speakers: int = N_SPEAKERS,
    proportions: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Decide what every call in the corpus will be, before generating any."""
    proportions = dict(proportions or DEFAULT_SPLIT)
    speaker_split = assign_speaker_splits(seed, n_speakers, proportions)
    per_split = _largest_remainder(n_calls, proportions)

    plan: List[Dict[str, Any]] = []
    # Each split starts its scenario rotation where the previous one stopped,
    # so the smaller splits do not all pile up on the first few scenarios.
    offset = 0
    for split in ("train", "dev", "test"):
        speakers = [s for s, sp in speaker_split.items() if sp == split]
        if not speakers or per_split.get(split, 0) == 0:
            continue
        rng = random.Random(derive_seed(seed, f"plan-{split}"))
        plan.extend(
            _plan_split(
                per_split[split],
                split,
                sorted(speakers),
                rng,
                scam_ratio,
                synthetic_ratio,
                mild_ratio,
                offset,
            )
        )
        offset += per_split[split]
    for i, entry in enumerate(plan):
        entry["call_id"] = f"call_{i:04d}"
    return plan, speaker_split


# --------------------------------------------------------------------------
# One call
# --------------------------------------------------------------------------


def _estimate_timings(turns: List[Turn], rng: random.Random) -> None:
    """Fill t_start and t_end from the token count when there is no audio."""
    cursor = 0.0
    for turn in turns:
        dur = max(MIN_TURN_S, len(turn.tokens) / SPEAK_RATE_TOKENS_PER_S)
        dur += rng.uniform(PAUSE_MIN_S, PAUSE_MAX_S)
        turn.t_start = round(cursor, 3)
        turn.t_end = round(cursor + dur, 3)
        cursor = turn.t_end + TURN_GAP_S


def build_call(entry: Dict[str, Any], seed: int, sr: int = TARGET_SR) -> Call:
    """Turn one plan entry into a fully annotated Call."""
    rng = random.Random(derive_seed(seed, f"call-{entry['call_id']}"))
    drafts, meta = grammar.build_dialogue(entry["scenario"], rng, entry["style"])

    turns: List[Turn] = []
    for i, draft in enumerate(drafts):
        turns.append(
            Turn(
                index=i,
                speaker=draft.speaker,
                text=draft.text,
                act=draft.act,
                tokens=list(draft.tokens),
                bio=list(draft.bio),
                lang=[token_language(tok) for tok in draft.tokens],
            )
        )
    _estimate_timings(turns, rng)

    meta = dict(meta)
    meta["hard_negative"] = (
        entry["label_scam"] == 0
        and entry["scenario"] in grammar.HARD_NEGATIVE_SCENARIOS
    )
    meta["mild_scam"] = entry["style"] == "mild"
    meta["timing_source"] = "estimated"

    return Call(
        call_id=entry["call_id"],
        turns=turns,
        label_scam=int(entry["label_scam"]),
        label_voice=entry["label_voice"],
        scenario=entry["scenario"],
        speaker_id=entry["speaker_id"],
        split=entry["split"],
        sample_rate=int(sr),
        channel="clean",
        audio_source="none",
        meta=meta,
    )




# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def generate_one(
    scenario: Optional[str] = None,
    scam: Optional[bool] = None,
    voice: Optional[str] = None,
    style: Optional[str] = None,
    speaker_id: Optional[str] = None,
    seed: Optional[int] = None,
    call_id: Optional[str] = None,
    sr: int = TARGET_SR,
) -> Call:
    """Generate a single annotated call outside the corpus planner.

    The console's "synthesise a new call" button uses this so a live demo is
    never a replay of something the model was trained on. Anything left as
    None is drawn at random, and the call is marked split="none" so it can
    never leak into a training or evaluation set by accident.
    """
    if seed is None:
        seed = random.randrange(1, 2 ** 31)
    rng = random.Random(derive_seed(int(seed), "one-off"))

    if scenario is not None:
        is_scam = scenario in SCAM_SCENARIOS
    else:
        is_scam = bool(rng.random() < DEFAULT_SCAM_RATIO) if scam is None else bool(scam)
        scenario = rng.choice(SCAM_SCENARIOS if is_scam else BENIGN_SCENARIOS)

    if voice not in ("human", "synthetic"):
        voice = "synthetic" if rng.random() < DEFAULT_SYNTHETIC_RATIO else "human"

    if style is None:
        if is_scam:
            style = "mild" if (
                scenario in grammar.MILD_CAPABLE_SCENARIOS
                and rng.random() < MILD_SCAM_RATIO
            ) else "hard"
        else:
            style = "flat"

    # The id folds in the resolved configuration, not just the seed, so two
    # calls requested with the same seed but different scenarios do not
    # collide in the console's in-memory call table.
    tag = f"id:{scenario}:{voice}:{style}:{int(is_scam)}"
    entry = {
        "call_id": call_id or f"live_{derive_seed(int(seed), tag) % 10 ** 8:08d}",
        "scenario": scenario,
        "label_scam": int(is_scam),
        "label_voice": voice,
        "style": style,
        "speaker_id": speaker_id or f"spk_live_{rng.randrange(0, 999):03d}",
        "split": "none",
    }
    call = build_call(entry, seed=int(seed), sr=sr)
    call.meta["generated"] = "on demand"
    call.meta["seed"] = int(seed)
    return call


def generate_corpus(
    n_calls: int = DEFAULT_N_CALLS,
    seed: Optional[int] = None,
    out_dir: Optional[Path] = None,
    audio: bool = False,
    audio_backend: str = "sim",
    sr: int = TARGET_SR,
    write: bool = True,
    progress: bool = False,
) -> List[Call]:
    """Generate the corpus and write it to disk.

    Writes one JSON per call to <out_dir>/calls/<call_id>.json plus a
    manifest with the split assignment, the cell counts and every generation
    parameter, including the seed. Set audio=True to render paired audio with
    the tts module; that is much slower, so it is off by default.
    """
    seed = int(SETTINGS.pipeline.seed if seed is None else seed)
    out_dir = Path(out_dir or CORPUS_DIR)
    calls_dir = out_dir / "calls"
    audio_dir = out_dir / "audio"

    plan, speaker_split = plan_corpus(n_calls, seed)

    calls: List[Call] = []
    for i, entry in enumerate(plan):
        call = build_call(entry, seed, sr=sr)
        calls.append(call)
        if progress and (i + 1) % 50 == 0:
            print(f"  generated {i + 1}/{len(plan)} calls")

    if audio:
        from .tts import render_call_audio  # local: keeps numpy out of the
                                            # import path when audio is off

        audio_dir.mkdir(parents=True, exist_ok=True)
        for i, call in enumerate(calls):
            path = audio_dir / f"{call.call_id}.wav"
            _, info = render_call_audio(
                call, backend=audio_backend, out_path=path, sr=sr
            )
            call.meta["audio_info"] = {
                k: v for k, v in info.items() if k != "spans"
            }
            call.meta["timing_source"] = "audio"
            if progress and (i + 1) % 25 == 0:
                print(f"  rendered {i + 1}/{len(calls)} calls")

    if write:
        ensure_dirs()
        calls_dir.mkdir(parents=True, exist_ok=True)
        for path in calls_dir.glob("call_*.json"):
            path.unlink()
        for call in calls:
            call.save(str(calls_dir / f"{call.call_id}.json"))
        manifest = build_manifest(
            calls,
            speaker_split,
            seed=seed,
            n_calls=n_calls,
            audio=audio,
            audio_backend=audio_backend,
            sr=sr,
        )
        with open(out_dir / MANIFEST_NAME, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
    return calls


def build_manifest(
    calls: Sequence[Call],
    speaker_split: Dict[str, str],
    seed: int,
    n_calls: int,
    audio: bool,
    audio_backend: str,
    sr: int,
) -> Dict[str, Any]:
    """Everything needed to reproduce and audit this corpus."""
    per_speaker: Dict[str, Dict[str, Any]] = {
        spk: {"split": split, "n_calls": 0, "n_scam": 0, "n_synthetic": 0}
        for spk, split in sorted(speaker_split.items())
    }
    for call in calls:
        row = per_speaker.setdefault(
            call.speaker_id,
            {"split": call.split, "n_calls": 0, "n_scam": 0, "n_synthetic": 0},
        )
        row["n_calls"] += 1
        row["n_scam"] += int(call.label_scam)
        row["n_synthetic"] += int(call.label_voice == "synthetic")

    cells_by_split: Dict[str, Dict[str, int]] = {}
    for call in calls:
        cells_by_split.setdefault(call.split, Counter())[call.cell] += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "generator": "swarkavach.corpus.generator.generate_corpus",
        "params": {
            "n_calls": int(n_calls),
            "seed": int(seed),
            "scam_ratio": DEFAULT_SCAM_RATIO,
            "synthetic_ratio": DEFAULT_SYNTHETIC_RATIO,
            "mild_scam_ratio": MILD_SCAM_RATIO,
            "split_proportions": dict(DEFAULT_SPLIT),
            "n_speakers": N_SPEAKERS,
            "sample_rate": int(sr),
            "audio": bool(audio),
            "audio_backend": audio_backend if audio else None,
            "speak_rate_tokens_per_s": SPEAK_RATE_TOKENS_PER_S,
            "turn_gap_s": TURN_GAP_S,
        },
        "counts": {
            "calls": len(calls),
            "turns": sum(len(c.turns) for c in calls),
            "tokens": sum(len(t.tokens) for c in calls for t in c.turns),
            "entities": sum(len(c.entities()) for c in calls),
        },
        "splits": dict(Counter(c.split for c in calls)),
        "cells": dict(Counter(c.cell for c in calls)),
        "cells_by_split": {k: dict(v) for k, v in sorted(cells_by_split.items())},
        "scenarios": dict(Counter(c.scenario for c in calls)),
        "speakers": per_speaker,
        "calls": [
            {
                "call_id": c.call_id,
                "split": c.split,
                "speaker_id": c.speaker_id,
                "scenario": c.scenario,
                "label_scam": int(c.label_scam),
                "label_voice": c.label_voice,
                "cell": c.cell,
                "style": c.meta.get("style", ""),
                "hard_negative": bool(c.meta.get("hard_negative", False)),
                "n_turns": len(c.turns),
                "duration_s": round(float(c.duration), 3),
                "audio_path": c.audio_path,
            }
            for c in calls
        ],
    }


def load_corpus(
    split: Optional[str] = None,
    with_audio: bool = False,
    corpus_dir: Optional[Path] = None,
) -> List[Call]:
    """Read the corpus back off disk.

    split filters to one of train / dev / test. with_audio=True keeps only the
    calls whose WAV actually exists, so a caller can ask for the audio subset
    without checking every path itself.
    """
    corpus_dir = Path(corpus_dir or CORPUS_DIR)
    calls_dir = corpus_dir / "calls"
    out: List[Call] = []
    for path in sorted(calls_dir.glob("call_*.json")):
        call = Call.load(str(path))
        if split is not None and call.split != split:
            continue
        if with_audio:
            if not call.audio_path:
                continue
            audio_path = Path(call.audio_path)
            if not audio_path.is_absolute():
                audio_path = corpus_dir / audio_path
            if not audio_path.exists():
                continue
            call.audio_path = str(audio_path)
        out.append(call)
    return out


def load_manifest(corpus_dir: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(corpus_dir or CORPUS_DIR) / MANIFEST_NAME
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def code_mixing_index(langs: Sequence[str]) -> float:
    """Gamback and Das CMI over one token sequence, on the [0, 1] scale.

    1 - max_language_tokens / language_dependent_tokens. Zero when a stretch is
    monolingual, or has no language-bearing tokens at all.

    The original paper states this as a percentage. This project keeps it in
    [0, 1] so it sits on the same scale as the other fusion features, and so
    that there is exactly one definition of CMI in the codebase: `text.langid`
    computes the same quantity for the fusion vector, and the two must not
    drift apart.
    """
    counts = Counter(l for l in langs if l in ("hi", "en"))
    n = sum(counts.values())
    if n == 0:
        return 0.0
    return 1.0 - max(counts.values()) / n


def switch_points(langs: Sequence[str]) -> int:
    """How many times the language changes, ignoring universal tokens."""
    seq = [l for l in langs if l in ("hi", "en")]
    return sum(1 for a, b in zip(seq, seq[1:]) if a != b)


def corpus_stats(calls: Sequence[Call]) -> Dict[str, Any]:
    """Everything the report and the dashboard need about the corpus."""
    entities: Counter = Counter()
    acts: Counter = Counter()
    scenarios: Counter = Counter()
    cells: Counter = Counter()
    splits: Counter = Counter()
    langs: Counter = Counter()
    speakers_by_split: Dict[str, set] = {}
    cells_by_split: Dict[str, Counter] = {}

    n_turns = n_tokens = 0
    cmis: List[float] = []
    switches = 0
    coercion = {"scam": [], "benign": []}
    durations: List[float] = []
    n_hard_neg = n_mild = 0

    # Code-mixing split by class. This is the evidence for the claim that
    # fraud scripts keep the threat in Hindi and the money vocabulary in
    # English, so it is worth reporting separately rather than as one pooled
    # average that hides the whole effect.
    cm = {
        k: {"cmi": [], "switches": 0, "lang_tokens": Counter(),
            "ent_tokens": Counter()}
        for k in ("scam", "benign")
    }

    for call in calls:
        scenarios[call.scenario] += 1
        cells[call.cell] += 1
        splits[call.split] += 1
        speakers_by_split.setdefault(call.split, set()).add(call.speaker_id)
        cells_by_split.setdefault(call.split, Counter())[call.cell] += 1
        durations.append(float(call.duration))
        n_hard_neg += int(bool(call.meta.get("hard_negative")))
        n_mild += int(bool(call.meta.get("mild_scam")))

        key = "scam" if call.label_scam else "benign"
        call_langs: List[str] = []
        ranks: List[float] = []
        for turn in call.turns:
            n_turns += 1
            n_tokens += len(turn.tokens)
            acts[turn.act] += 1
            call_langs.extend(turn.lang)
            for tag in turn.lang:
                langs[tag] += 1
                cm[key]["lang_tokens"][tag] += 1
            ranks.append(COERCION_RANK.get(turn.act, 0.0))
            for span in turn.entities():
                entities[span.type] += 1
                # which language carries this entity, token by token
                for i in range(span.tok_start, min(span.tok_end, len(turn.lang))):
                    cm[key]["ent_tokens"][turn.lang[i]] += 1

        call_cmi = code_mixing_index(call_langs)
        call_switch = switch_points(call_langs)
        cmis.append(call_cmi)
        switches += call_switch
        cm[key]["cmi"].append(call_cmi)
        cm[key]["switches"] += call_switch
        coercion[key].append(sum(ranks) / len(ranks) if ranks else 0.0)

    def mean(xs: Sequence[float]) -> float:
        return float(sum(xs) / len(xs)) if len(xs) else 0.0

    cmi_by_class: Dict[str, Dict[str, float]] = {}
    for key, d in cm.items():
        lang_total = d["lang_tokens"]["hi"] + d["lang_tokens"]["en"]
        ent_total = d["ent_tokens"]["hi"] + d["ent_tokens"]["en"]
        cmi_by_class[key] = {
            "cmi": round(mean(d["cmi"]), 4),
            # per 100 language-bearing tokens, so call length does not matter
            "switch_rate": round(100.0 * d["switches"] / lang_total, 3) if lang_total else 0.0,
            "en_ratio": round(d["lang_tokens"]["en"] / lang_total, 4) if lang_total else 0.0,
            # the novel one: how much of the fraud vocabulary is English
            "ent_lang_align": round(d["ent_tokens"]["en"] / ent_total, 4) if ent_total else 0.0,
            "n_entity_tokens": int(ent_total),
        }

    return {
        # The dashboard reads entities_per_type and cmi_by_class; `entities`
        # is kept as it was so nothing that already reads it breaks.
        "entities_per_type": {t: int(entities.get(t, 0)) for t in ENTITY_TYPES},
        "cmi_by_class": cmi_by_class,
        "scam_scenarios": list(SCAM_SCENARIOS),
        "benign_scenarios": list(BENIGN_SCENARIOS),
        "n_calls": len(calls),
        "n_turns": n_turns,
        "n_tokens": n_tokens,
        "n_entities": int(sum(entities.values())),
        "n_scam": int(sum(1 for c in calls if c.label_scam)),
        "n_benign": int(sum(1 for c in calls if not c.label_scam)),
        "n_hard_negatives": n_hard_neg,
        "n_mild_scams": n_mild,
        "entities": {t: int(entities.get(t, 0)) for t in ENTITY_TYPES},
        "acts": {a: int(n) for a, n in sorted(acts.items())},
        "scenarios": {s: int(n) for s, n in sorted(scenarios.items())},
        "cells": {c: int(n) for c, n in sorted(cells.items())},
        "cells_by_split": {
            k: {c: int(n) for c, n in sorted(v.items())}
            for k, v in sorted(cells_by_split.items())
        },
        "splits": {s: int(n) for s, n in sorted(splits.items())},
        "speakers_per_split": {
            k: len(v) for k, v in sorted(speakers_by_split.items())
        },
        "lang_tokens": {k: int(v) for k, v in sorted(langs.items())},
        "mean_cmi": round(mean(cmis), 3),
        "switch_points": int(switches),
        "mean_turns_per_call": round(mean([len(c.turns) for c in calls]), 3),
        "mean_tokens_per_turn": round(n_tokens / n_turns, 3) if n_turns else 0.0,
        "mean_duration_s": round(mean(durations), 3),
        "total_duration_s": round(float(sum(durations)), 3),
        "mean_coercion": {
            "scam": round(mean(coercion["scam"]), 4),
            "benign": round(mean(coercion["benign"]), 4),
        },
    }


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------


def _print_call(call: Call) -> None:
    print(
        f"\n--- {call.call_id} | {call.scenario} | scam={call.label_scam} "
        f"| voice={call.label_voice} | {call.speaker_id} | {call.split} "
        f"| style={call.meta.get('style')}"
    )
    for turn in call.turns:
        ents = ", ".join(f"{s.text}={s.type}" for s in turn.entities())
        tail = f"   [{ents}]" if ents else ""
        print(
            f"  {turn.t_start:6.2f} {turn.speaker:6s} {turn.act:18s} "
            f"{turn.text}{tail}"
        )


if __name__ == "__main__":
    calls = generate_corpus(n_calls=48, seed=SETTINGS.pipeline.seed, write=False)
    stats = corpus_stats(calls)
    print(json.dumps(stats, indent=2)[:2400])

    by_split: Dict[str, set] = {}
    for c in calls:
        by_split.setdefault(c.split, set()).add(c.speaker_id)
    overlap = any(
        by_split[a] & by_split[b]
        for a in by_split
        for b in by_split
        if a < b
    )
    print(f"\nspeaker overlap between splits: {overlap}")

    scam = next(c for c in calls if c.label_scam and c.meta.get("style") == "hard")
    mild = next(c for c in calls if c.meta.get("style") == "mild")
    hard_neg = next(c for c in calls if c.meta.get("hard_negative"))
    _print_call(scam)
    _print_call(mild)
    _print_call(hard_neg)
