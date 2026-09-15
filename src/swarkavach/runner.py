"""The pipeline stages as the notebook runs them, kept in the repo.

Why this module exists, which is worth explaining because it is not about
Python at all.

A Colab notebook is opened from GitHub or Drive and executed from there. The
clone it makes is a separate thing on disk. So `git pull` inside the clone
updates `src/` and `tests/` and does NOT update the cells being executed. Any
logic written into a cell is frozen at whatever revision the reader opened,
and fixing it means asking them to close the notebook and open it again, which
is not obvious and is easy to forget. We found this the slow way: a change
that cut a 37 minute pre-flight down to a couple of minutes was pulled
successfully and then had no effect at all, twice.

So the cells are thin. Each one calls a function here, and this file is inside
the clone, so a pull genuinely changes what the next run does. The only thing
a reader ever has to re-open the notebook for is a change to the cell
structure itself, and `NOTEBOOK_VERSION` below exists to tell them when that
has happened instead of letting them find out from a strange error.

Each stage knows three things: whether its output already exists and it can be
skipped, how to run itself, and what to checkpoint to Drive afterwards.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import colab

#: The interpreter running the notebook, not whatever "python" resolves to on
#: PATH. On Colab those are the same; locally they are often not.
PY = shlex.quote(sys.executable)

#: Bump ONLY when the notebook's cells must change, not when this file
#: changes. A reader whose notebook declares a lower number is told to open it
#: again; everything else reaches them through a pull.
NOTEBOOK_VERSION = 1

#: Stage name -> the artifacts that stage produces, in save order.
STAGE_ARTIFACTS: Dict[str, Sequence[str]] = {
    "download": ("raw", "robocall"),
    "pairs": ("pairs",),
    "corpus": ("tts_cache", "corpus"),
    "train": ("models",),
    "evaluate": ("results",),
}

_DONE: List[str] = []
_FORCE: List[str] = []


# --------------------------------------------------------------------------
# running commands
# --------------------------------------------------------------------------


def _rule(msg: str) -> None:
    print()
    print("=" * 70)
    print(msg)
    print("=" * 70, flush=True)


def run(cmd: str, *, check: bool = True) -> int:
    """Run one command with its output streaming, and stop on failure.

    Deliberately not the `!` magic. `!` swallows the exit code, so a training
    step that died looks exactly like one that worked and the next cell goes
    on to evaluate models that were never written.
    """
    print(f"$ {cmd}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, shell=True).returncode
    dt = time.time() - t0
    print(f"[{'ok' if rc == 0 else f'FAILED rc={rc}'} in {dt / 60:.1f} min]", flush=True)
    if rc != 0 and check:
        raise SystemExit(f"step failed: {cmd}")
    return rc


# --------------------------------------------------------------------------
# notebook freshness
# --------------------------------------------------------------------------


def check_notebook(declared: Optional[int] = None) -> bool:
    """True when the notebook being executed matches this clone.

    Returns False and says what to do when it does not, rather than letting
    the reader run an old set of cells against new code.
    """
    if declared is None or int(declared) < NOTEBOOK_VERSION:
        print()
        print("!" * 70)
        print("This notebook is older than the code it just pulled.")
        print(f"  notebook declares : {declared}")
        print(f"  clone expects     : {NOTEBOOK_VERSION}")
        print()
        print("A pull updates the clone on disk, not the cells you are running,")
        print("because Colab loaded those separately. Close this notebook and")
        print("open notebooks/00_RUN_EVERYTHING.ipynb from GitHub again.")
        print("!" * 70)
        return False
    return True


# --------------------------------------------------------------------------
# session setup
# --------------------------------------------------------------------------


def begin(force: Iterable[str] = ()) -> List[str]:
    """Mount Drive, pull back what is already done, and report the plan."""
    global _DONE, _FORCE
    _FORCE = _with_downstream([str(f) for f in force])

    mounted = colab.mount()
    if colab.in_colab() and not mounted:
        # Without Drive nothing is restored, every stage looks fresh, and the
        # session spends 60 to 90 minutes rebuilding what Drive already has.
        # That is the most expensive mistake this runner can make, so it is
        # not allowed to happen quietly.
        raise SystemExit("on Colab but Drive did not mount. Fix the mount and "
                         "re-run this cell; do not continue without it.")
    colab.restore_all()

    _DONE = [s for s in colab.what_can_be_skipped() if s not in _FORCE]

    # A corpus from Drive is only reusable if it was generated from the
    # templates this clone carries. Otherwise every stage after it would be
    # trained and scored on text the repo no longer produces, and the fixes in
    # the grammar would never reach a result file. Text the models never see
    # is text that was never fixed.
    for stage, reason in (("pairs", _pairs_stale_reason()),
                          ("corpus", _corpus_stale_reason())):
        if reason and stage in _DONE:
            downstream = [s for s in (stage, *_DOWNSTREAM[stage]) if s in _DONE]
            _DONE = [s for s in _DONE if s not in downstream]
            print()
            print("!" * 70)
            print(f"Drive {stage} does not match the code in this clone.")
            print(f"  {reason}")
            print(f"  rebuilding: {', '.join(downstream)}")
            print("The speech cache is kept, so lines that did not change are not")
            print("rendered again.")
            print("!" * 70)

    print()
    print("already done, will be skipped:", ", ".join(_DONE) if _DONE else
          "nothing, this is a fresh run")
    if _FORCE:
        print("forced to rebuild:", ", ".join(_FORCE))
    print()
    colab.print_status()
    return list(_DONE)


#: What has to be rebuilt when a stage is rebuilt. Forcing "corpus" with the
#: old models kept gives models trained on text the corpus no longer has, and
#: forcing "pairs" alone gives a voice branch trained on the old pairs. The
#: cascade is the same one the grammar-staleness check applies.
_DOWNSTREAM: Dict[str, Tuple[str, ...]] = {
    "download": ("pairs", "train", "evaluate"),
    "pairs": ("train", "evaluate"),
    "corpus": ("train", "evaluate"),
    "train": ("evaluate",),
    "evaluate": (),
}


def _with_downstream(force: List[str]) -> List[str]:
    out: List[str] = []
    for stage in force:
        for s in (stage, *_DOWNSTREAM.get(stage, ())):
            if s not in out:
                out.append(s)
    added = [s for s in out if s not in force]
    if added:
        print(f"forcing {', '.join(force)} also rebuilds {', '.join(added)}, "
              "which depend on it")
    return out


def _pairs_stale_reason() -> Optional[str]:
    """Why the anti-spoofing pairs on disk cannot be reused, or None."""
    from .datasets import PAIRS_BUILD_VERSION, load_antispoof_pairs

    m = load_antispoof_pairs()
    if not m:
        return None
    have = m.get("pairs_build_version")
    if have is None:
        return ("pairs were built before the build was versioned; that build "
                "drew spoof voices from a pool that mostly cannot read "
                "Devanagari, so it lost most of its lines and used two voices")
    if int(have) != PAIRS_BUILD_VERSION:
        return f"pairs build version {have} on Drive, {PAIRS_BUILD_VERSION} in the clone"
    return None


def _corpus_stale_reason() -> Optional[str]:
    """Why the corpus on disk cannot be reused, or None if it can."""
    import json

    from . import config
    from .corpus import generator, grammar

    manifest = config.CORPUS_DIR / generator.MANIFEST_NAME
    if not manifest.exists():
        return None
    try:
        with open(manifest, encoding="utf-8") as fh:
            recorded = json.load(fh).get("grammar_fingerprint")
    except (OSError, ValueError) as exc:
        return f"manifest unreadable ({type(exc).__name__})"
    current = grammar.grammar_fingerprint()
    if recorded is None:
        return "manifest has no grammar fingerprint, it predates the check"
    if recorded != current:
        return f"grammar fingerprint {recorded} on Drive, {current} in the clone"
    return None


def _skip(stage: str) -> bool:
    return stage in _DONE and stage not in _FORCE


def _save(stage: str, quiet: bool = False) -> None:
    for name in STAGE_ARTIFACTS.get(stage, ()):
        colab.save(name, quiet=quiet)


# --------------------------------------------------------------------------
# the stages
# --------------------------------------------------------------------------


def preflight(full: bool = False) -> bool:
    """Run the tests that are worth running before an expensive session.

    `test_integration.py` is excluded unless asked for. It generates 160 calls
    with audio and trains a whole pipeline on them, which measured at 37
    minutes on Colab, and the stages below do exactly that for real on 480
    calls immediately afterwards. Rehearsing it costs a large slice of a free
    session and tells you almost nothing the real run will not.
    """
    _rule("Pre-flight tests")
    cmd = f"{PY} -m pytest tests -q"
    if not full:
        cmd += " --ignore=tests/test_integration.py"
    rc = run(cmd, check=False)
    if rc != 0:
        print()
        print("Tests failed. Stopping here on purpose: the stages below cost")
        print("about an hour, and running them against a broken clone wastes it.")
        raise SystemExit("pre-flight failed")
    return True


def fetch_data(pairs: int = 600, which: str = "core") -> None:
    """Download the real corpora and build the Hindi anti-spoofing pairs."""
    _rule("Data")
    if _skip("download") and _skip("pairs"):
        print("corpora and anti-spoofing pairs came back from Drive, nothing to fetch")
        return

    # Having one and not the other is normal: the pairs are expensive and the
    # archives are not, so they are checkpointed separately.
    n = 0 if _skip("pairs") else int(pairs)
    run(f"{PY} -m swarkavach.cli fetch-data --which {which} --pairs {n}")
    _save("download")
    if n:
        _save("pairs")
        # The pair build renders 600 lines through the same speech cache the
        # corpus uses. Without this, a disconnect between here and the corpus
        # stage throws those renders away and they are paid for twice.
        colab.save("tts_cache")


def gen_corpus(n: int = 480, backend: str = "edge") -> None:
    """Generate the Hinglish call corpus and render its audio."""
    _rule("Corpus")
    if _skip("corpus"):
        print("corpus came back from Drive, skipping generation")
        return
    try:
        run(f"{PY} -m swarkavach.cli gen-corpus "
            f"--n {n} --audio --backend {backend}")
    finally:
        # Save the speech cache even when generation died partway. That cache
        # is the hour you do not want to spend twice, and a half built one is
        # still most of it.
        colab.save("tts_cache")
    colab.save("corpus")


def train() -> None:
    _rule("Train")
    if _skip("train"):
        print("trained models came back from Drive, skipping training")
        anchor_report()
        return
    run(f"{PY} -m swarkavach.cli train")
    _save("train")
    anchor_report()


def anchor_report(flat_tol: float = 0.05) -> Optional[Dict[str, str]]:
    """Print the prosody-intent anchor table with an ok / flat / wrong verdict.

    The verdict is derived here from the stored medians and anchors rather
    than read from the file, so an anchors file written by an older build,
    which recorded only a strict human > synthetic boolean, still gets the
    tolerant reading. The strict rule once printed WRONG WAY for a component
    that separated the two voice cells by 1.6 percent of its span.
    """
    import json

    from . import config

    p = config.MODELS_DIR / "pim_anchors.json"
    if not p.exists():
        print("no anchors file, the pipeline has not trained")
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    anchors = d.get("anchors") or {}
    medians = d.get("medians") or {}
    if not anchors or not medians:
        print(f"anchors file has no medians ({d.get('status')})")
        return None

    verdict: Dict[str, str] = {}
    print()
    print("prosody-intent anchors, per component:")
    print(f"  {'component':14s} {'human':>8s} {'synthetic':>10s} {'gap/span':>9s}   reading")
    for k, (lo, hi) in anchors.items():
        med = medians.get(k) or {}
        h, sy = med.get("human"), med.get("synthetic")
        span = float(hi) - float(lo)
        if h is None or sy is None or span <= 1e-9:
            verdict[k] = "unknown"
            print(f"  {k:14s} {'':>8s} {'':>10s} {'n/a':>9s}   unknown")
            continue
        gap = (float(h) - float(sy)) / span
        verdict[k] = "ok" if gap > flat_tol else ("wrong" if gap < -flat_tol else "flat")
        print(f"  {k:14s} {h:8.4f} {sy:10.4f} {gap:+9.3f}   {verdict[k]}")

    wrong = [k for k, v in verdict.items() if v == "wrong"]
    flat = [k for k, v in verdict.items() if v == "flat"]
    if wrong:
        print(f"  WRONG: {wrong} read clearly higher for synthetic speech than for")
        print("  human, so they fight the feature and need replacing.")
    if flat:
        print(f"  flat: {flat} separate the voice cells by under {flat_tol:.0%} of")
        print("  their span. Expected on the edge backend, where both cells are")
        print("  neural voices; person-or-machine is the anti-spoofing branch's job.")
    return verdict


def evaluate(codecs: str = "clean,g711u,g711a,gsm") -> None:
    _rule("Evaluate")
    if _skip("evaluate"):
        print("results came back from Drive, skipping evaluation")
        return
    run(f'{PY} -m swarkavach.cli evaluate --codecs "{codecs}"')
    _save("evaluate")


def finish() -> None:
    """Push everything to Drive once more, so the zip is a convenience copy."""
    _rule("Saving to Drive")
    colab.save_all()
    print()
    colab.print_status()


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

#: Result file -> the stage that writes it, so a missing one can be explained.
_RESULT_OWNER = {
    "ablation_results.json": "evaluate",
    "antispoof_results.json": "evaluate",
    "ner_results.json": "evaluate",
    "intent_results.json": "evaluate",
    "robocall_ood.json": "evaluate",
    "robustness_results.json": "evaluate",
    "ttd_results.json": "evaluate",
    "calibration.json": "evaluate",
}

CELL_LABEL = {"humanxbenign": "Human/benign", "humanxscam": "Human/scam",
              "clonedxbenign": "Cloned/benign", "clonedxscam": "Cloned/scam"}
ARM_LABEL = {"audio_only": "Audio only", "text_only": "Text only",
             "late_fusion": "Late fusion", "full": "Full (cross-modal)"}


def load_result(name: str) -> Optional[dict]:
    """One results file, or None. Absolute path, so cwd does not matter."""
    import json

    from .config import RESULTS_DIR

    f = RESULTS_DIR / name
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"{name}: unreadable, {type(exc).__name__}: {exc}")
        return None


def _missing(name: str) -> None:
    """Say that a file is absent and what that means.

    The display cells used to be `x = load(...)` then `if x:`, so an absent
    file printed nothing at all. Combined with a stage that was wrongly
    skipped, the result was one table that looked fine and two cells that were
    simply blank, which reads as "nothing to say" rather than "this never ran".
    """
    owner = _RESULT_OWNER.get(name, "a stage")
    print(f"  {name}: MISSING. The {owner} stage did not write it.")


def report() -> None:
    """Print every results table, and name anything that is not there."""
    _rule("Results")

    ab = load_result("ablation_results.json")
    if ab is None:
        _missing("ablation_results.json")
    else:
        cells = ab.get("cells", [])
        print(f"ABLATION, correct decisions per cell, n_test={ab.get('n_test')}, "
              f"threshold {ab.get('threshold')}")
        print(f"arms fitted on: {ab.get('arms_fitted_on', 'not recorded')}")
        print()
        hdr = f"{'arm':20s}" + "".join(f"{CELL_LABEL.get(c, c):>16s}" for c in cells)
        hdr += f"{'AUC':>8s}{'F1':>8s}"
        print(hdr)
        print("-" * len(hdr))
        for arm in ab.get("arms", []):
            row = f"{ARM_LABEL.get(arm, arm):20s}"
            for c in cells:
                v = ab.get("detection_rate", {}).get(arm, {}).get(c)
                row += f"{('n/a' if v is None else format(v, '.0%')):>16s}"
            o = ab.get("overall", {}).get(arm, {})
            row += f"{o.get('auc', 0):>8.3f}{o.get('f1', 0):>8.3f}"
            print(row)

        hs = ab.get("hard_subset") or {}
        print()
        if hs.get("note"):
            print(f"hard subset: {hs['note']}")
        elif hs.get("n"):
            print(f"HARD SUBSET only: {hs.get('n_scam')} lexically mild scams against "
                  f"{hs.get('n_benign')} benign calls written to look like scams.")
            print("The whole-corpus rows above saturate; this is where an arm")
            print("comparison has room to say anything.")
            print()
            print(f"  {'arm':20s} {'AUC':>8s} {'F1':>8s} {'recall':>8s} {'precision':>10s}")
            for arm in ab.get("arms", []):
                v = hs.get(arm)
                if not v:
                    continue
                print(f"  {ARM_LABEL.get(arm, arm):20s} {v['auc']:8.3f} {v['f1']:8.3f} "
                      f"{v['recall']:8.3f} {v['precision']:10.3f}")

    for name, title in (("antispoof_results.json", "ANTI-SPOOFING"),
                        ("ner_results.json", "ENTITY RECOGNITION"),
                        ("intent_results.json", "INTENT"),
                        ("robocall_ood.json", "OUT OF DOMAIN"),
                        ("ttd_results.json", "TIME TO DETECTION"),
                        ("calibration.json", "CALIBRATION"),
                        ("robustness_results.json", "CHANNEL ROBUSTNESS")):
        d = load_result(name)
        print()
        if d is None:
            _missing(name)
            continue
        print(f"{title}")
        _print_one(name, d)


def _print_one(name: str, d: dict) -> None:
    if name == "antispoof_results.json":
        print(f"  evaluated on: {d.get('evaluated_on')}")
        print(f"  {d.get('evaluated_on_note', '')}")
        cond = d.get("default_condition", "clean")
        models = d.get("models", [])
        hdr = f"  {'feature':10s}" + "".join(f"{m.upper():>10s}" for m in models)
        print(hdr)
        for f in d.get("feature_sets", []):
            row = f"  {f.upper():10s}"
            for m in models:
                v = (d.get("eer", {}).get(f, {}).get(m, {}) or {}).get(cond)
                row += f"{('n/a' if v is None else format(v, '.1%')):>10s}"
            print(row)
        if d.get("gmm_converged") is False:
            print("  WARNING: a GMM did not converge, its EER is not a fitted model's EER")
        for fs, v in (d.get("gmm_fit") or {}).items():
            if v.get("converged_bona") is False or v.get("converged_spoof") is False:
                print(f"    {fs}: bona {v.get('converged_bona')}, spoof {v.get('converged_spoof')}")

    elif name == "ner_results.json":
        for model, srcs in (d.get("overall") or {}).items():
            for src, v in srcs.items():
                print(f"  {model.upper():8s} on {src:5s} transcripts   "
                      f"P {v.get('p', 0):.3f}  R {v.get('r', 0):.3f}  F1 {v.get('f1', 0):.3f}")
        if not any("asr" in srcs for srcs in (d.get("overall") or {}).values()):
            print("  no asr row: see the training log for why it could not be measured")

    elif name == "intent_results.json":
        print(f"  threshold {d.get('threshold')}")
        for n, v in (d.get("models") or {}).items():
            print(f"  {n:8s}  AUC {v.get('auc', 0):.3f}  acc {v.get('accuracy', 0):.1%}  "
                  f"F1 {v.get('f1', 0):.3f}")

    elif name == "robocall_ood.json":
        print(f"  {d.get('n_robocalls')} real FTC robocall transcripts, "
              f"languages {d.get('n_by_language')}")
        for n, v in (d.get("models") or {}).items():
            print(f"  {n:8s}  recall at {d.get('threshold')}: "
                  f"{v.get('recall_at_threshold', 0):.1%} "
                  f"({v.get('n_detected')} of {d.get('n_robocalls')})")
            for lang, b in (v.get("by_language") or {}).items():
                print(f"            {lang}: {b['recall']:.1%} of {b['n']}")
        print(f"  {d.get('note', '')}")

    elif name == "ttd_results.json":
        print(f"  median {d.get('median')}s, {d.get('detected')}/{d.get('total')} "
              f"scam calls flagged at threshold {d.get('threshold')}")

    elif name == "calibration.json":
        print(f"  Brier {d.get('brier')}, ECE {d.get('ece')}")

    elif name == "robustness_results.json":
        print(f"  ffmpeg present: {d.get('ffmpeg')}")
        for c in d.get("codecs", []):
            auc = (d.get("auc") or {}).get(c)
            realc = (d.get("real_codec") or {}).get(c)
            nf = (d.get("n_failed") or {}).get(c, 0)
            tag = "" if realc else "  [numpy stand-in, not the real codec]"
            audio = ((d.get("auc_by_arm") or {}).get("audio_only") or {}).get(c)
            if auc is None:
                print(f"  {c:12s} not reported, {nf} calls failed{tag}")
            else:
                print(f"  {c:12s} full AUC {auc:.3f}"
                      + (f"   audio-only AUC {audio:.3f}" if audio is not None else "")
                      + (f"  [{nf} failed]" if nf else "") + tag)
        if d.get("auc_by_arm", {}).get("audio_only"):
            print("  The full arm is carried by text, which no codec touches; read the")
            print("  audio-only column for what the channel does to the voice branch.")


def audit_pairs() -> bool:
    """Refuse to train on an anti-spoofing set that has a channel shortcut.

    Returns True when the set is usable. The previous version of this check
    lived in a notebook cell and read `m.get("n_pairs")` straight into a print,
    so a missing manifest printed "audit: None | worst statistic AUC None",
    the `> 0.75` comparison against None-or-zero was False, and the STOP
    branch never fired. Training then silently fell back to the generated
    corpus, where both classes are machine generated, and the equal error rate
    stopped meaning what the report says it means.
    """
    from .datasets import (audit_is_current, load_antispoof_pairs,
                           refresh_confound_audit)

    _rule("Anti-spoofing set audit")
    m = load_antispoof_pairs()
    if m and not audit_is_current(m.get("confound_audit")):
        # The audit used to run only inside the pair build, so a statistic
        # added afterwards was invisible without re-rendering every line.
        # The WAVs are on disk; re-reading them is seconds.
        print("stored audit predates the colour statistics; re-auditing the")
        print("pair audio already on disk (no rebuild)")
        m = refresh_confound_audit() or m
        colab.save("pairs", quiet=True)
    if not m:
        print("NO PAIRS: data/raw/antispoof_pairs/manifest.json does not exist.")
        print("The voice branch would fall back to the generated corpus, where")
        print("both classes are machine generated. Re-run the data stage.")
        return False

    n = int(m.get("n_pairs") or 0)
    a = m.get("confound_audit") or {}
    worst = a.get("worst_auc")
    print(f"pairs : {n}")
    print(f"audit : {a.get('status', 'not audited')} | worst statistic AUC {worst}")
    print()
    print(f"{'statistic':20s} {'bonafide':>10} {'spoof':>10} {'AUC alone':>10}")
    for k, v in (a.get("per_statistic") or {}).items():
        print(f"{k:20s} {v['bonafide_mean']:10.2f} {v['spoof_mean']:10.2f} "
              f"{v['auc_alone']:10.3f}")

    if n < 50:
        print(f"\nSTOP: {n} pairs is too few to measure anything.")
        return False
    if worst is None:
        print("\nSTOP: the set was never audited, so a channel shortcut cannot "
              "be ruled out. Do not trust an EER from it.")
        return False
    if worst > 0.75:
        print(f"\nSTOP: worst statistic AUC {worst:.3f}. The two classes differ "
              "in recording conditions, so any EER measures the microphone.")
        return False
    if worst > 0.65:
        print(f"\nWarning: worst statistic AUC {worst:.3f} is higher than the "
              "0.62 this set normally reaches. Read the table above.")
    return True


def listen(prefer_cell: str = "clonedxscam", n_turns: int = 10):
    """Print one call's transcript and return its audio for playback.

    Loads real `Call` objects rather than raw JSON. The notebook cell used to
    read the call files itself and print `pick["cell"]`, which is a computed
    property and not a serialised key, so it raised KeyError. Keeping this
    here means that class of mistake is fixed by a pull.

    Returns an IPython Audio widget, or None. Display it with the cell's
    return value.
    """
    from .corpus.generator import load_corpus

    calls = load_corpus()
    if not calls:
        print("no corpus on disk, run the corpus stage first")
        return None

    pick = next((c for c in calls if c.cell == prefer_cell), calls[0])
    print(f"{pick.call_id}  |  {pick.scenario}  |  {pick.cell}  |  split {pick.split}")
    info = (pick.meta or {}).get("audio_info") or {}
    print(f"rendered by: {info.get('backend', pick.audio_source)}"
          + (f"  (fell back: {info['fallback_reason']})"
             if info.get("fallback_reason") else ""))
    det = info.get("detail") or {}
    if det.get("caller_voice"):
        print(f"caller voice: {det['caller_voice'].get('voice')}   "
              f"callee voice: {(det.get('callee_voice') or {}).get('voice')}")
        print(f"cloned via  : {det.get('cloned_via')}")
    print()
    for t in pick.turns[:n_turns]:
        print(f"  {t.speaker:7s} {t.act:18s} {t.text}")
    if len(pick.turns) > n_turns:
        print(f"  ... {len(pick.turns) - n_turns} more turns")

    if not (pick.audio_path and Path(pick.audio_path).exists()):
        print()
        print("no audio file for this call")
        return None
    try:
        from IPython.display import Audio

        return Audio(pick.audio_path)
    except Exception:
        print(f"audio at {pick.audio_path}")
        return None
