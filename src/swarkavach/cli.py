"""Command line entry points.

    swarkavach gen-corpus --n 480 --audio
    swarkavach train
    swarkavach evaluate
    swarkavach serve
    swarkavach demo            # all of the above, in order, then the console
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import config
from .config import SETTINGS

app = typer.Typer(add_completion=False, help="SwarKavach: voice-clone fraud call detection")
console = Console()


def _rule(msg: str) -> None:
    console.rule(f"[bold]{msg}")


@app.command("gen-corpus")
def gen_corpus(
    n: int = typer.Option(config.DEFAULT_N_CALLS, help="number of calls to generate"),
    seed: int = typer.Option(SETTINGS.pipeline.seed, help="random seed"),
    audio: bool = typer.Option(True, help="render audio for every call"),
    backend: str = typer.Option("sim", help="audio backend: sim, sapi, edge"),
):
    """Build the Hinglish scam and benign corpus with gold annotations."""
    from .corpus.generator import generate_corpus, corpus_stats

    _rule(f"Generating {n} calls")
    t0 = time.time()
    calls = generate_corpus(n_calls=n, seed=seed, audio=audio, audio_backend=backend)
    stats = corpus_stats(calls)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "corpus_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")

    t = Table(show_header=False, box=None)
    t.add_row("calls", str(stats.get("n_calls")))
    t.add_row("turns", str(stats.get("n_turns")))
    t.add_row("tokens", f"{stats.get('n_tokens', 0):,}")
    t.add_row("entities", f"{stats.get('n_entities', 0):,}")
    t.add_row("mean CMI", f"{stats.get('mean_cmi', 0):.3f}")
    for cell, k in (stats.get("cells") or {}).items():
        t.add_row(f"cell {cell}", str(k))
    for split, k in (stats.get("splits") or {}).items():
        t.add_row(f"split {split}", str(k))
    console.print(t)

    # Did the audio actually come from the backend that was asked for?
    #
    # render_call_audio falls back to the `sim` backend per call when the
    # requested one throws, and `sim` is speech-shaped noise with no words in
    # it. If the synthesis endpoint is unreachable from this network, every
    # call falls back, gen-corpus exits 0, and you get a corpus that is
    # useless for a demo after an hour of retry backoff. Only the per-call
    # meta recorded the truth and nothing read it.
    if audio and backend != "sim":
        used = {}
        for c in calls:
            b = (c.meta.get("audio_info") or {}).get("backend") or c.audio_source or "?"
            used[b] = used.get(b, 0) + 1
        got = used.get(backend, 0)
        share = got / max(len(calls), 1)
        console.print(f"audio backend: " + ", ".join(f"{k}={v}" for k, v in sorted(used.items())))
        if share < 0.9:
            reasons = {}
            for c in calls:
                r = (c.meta.get("audio_info") or {}).get("fallback_reason")
                if r:
                    reasons[r] = reasons.get(r, 0) + 1
            console.print(f"[red]only {got}/{len(calls)} calls ({share:.0%}) rendered "
                          f"with {backend!r}, the rest fell back to speech-shaped "
                          f"noise with no words in it[/]")
            for r, k in sorted(reasons.items(), key=lambda kv: -kv[1])[:3]:
                console.print(f"  [red]{k}x {r}[/]")
            raise typer.Exit(1)

    console.print(f"[green]done in {time.time() - t0:.1f}s[/] -> {config.CORPUS_DIR}")


@app.command("fetch-data")
def fetch_data(
    which: str = typer.Option("core", help="core | all | a comma separated list"),
    pairs: int = typer.Option(500, help="paired anti-spoof utterances to build, 0 to skip"),
    robocall_audio: bool = typer.Option(
        False, help="also fetch the 1.7 GB robocall WAVs. Only the transcripts are used"),
):
    """Download the real corpora and build the Hindi anti-spoofing pairs.

    Both sources are direct HTTP with no account and no approval form, which
    was a hard constraint on this project. "core" is GramVaani dev and eval
    plus the robocall set, about 800 MB. "all" adds the 2 GB GramVaani
    training archive.
    """
    import shutil
    import subprocess
    import urllib.request

    from . import datasets as D

    config.ensure_dirs()
    names = {
        "core": ["gramvaani_dev", "gramvaani_eval", "robocall"],
        "all": list(D.DATASETS),
    }.get(which, [w.strip() for w in which.split(",") if w.strip()])

    _rule("Fetching data")
    for name in names:
        spec = D.DATASETS.get(name)
        if spec is None:
            console.print(f"[yellow]unknown dataset {name}[/]")
            continue

        if "archive" in spec:
            dest = config.RAW_DIR / spec["archive"]
            if dest.exists():
                # Existence is not the question, plausibility is. A download
                # killed halfway, a Drive restore that ran out of quota, or a
                # test that wrote a stub all leave a file here that this used
                # to accept, and then nothing downloads and the failure shows
                # up much later as an empty corpus. Anything under a tenth of
                # the expected size is not the archive.
                have = dest.stat().st_size
                floor = int(spec["size_mb"] * 1e6 * 0.1)
                if have >= floor:
                    console.print(f"  [dim]{name}: already present, "
                                  f"{have / 1e6:.0f} MB[/]")
                    continue
                console.print(f"  [yellow]{name}: {have / 1e6:.1f} MB on disk but "
                              f"{spec['size_mb']} MB expected, refetching[/]")
                dest.unlink()
                extracted = config.RAW_DIR / spec["archive"].replace(".tar.gz", "")
                if extracted.is_dir():
                    shutil.rmtree(extracted, ignore_errors=True)
            console.print(f"  {name}: {spec['size_mb']} MB from {spec['url']}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            try:
                urllib.request.urlretrieve(spec["url"], tmp)
                got = tmp.stat().st_size
                if got < int(spec["size_mb"] * 1e6 * 0.1):
                    tmp.unlink()
                    raise RuntimeError(
                        f"server returned {got / 1e6:.1f} MB, expected about "
                        f"{spec['size_mb']} MB")
                tmp.rename(dest)
                console.print(f"    [green]done[/] {got / 1e6:.0f} MB")
            except Exception as exc:
                console.print(f"    [red]failed: {exc}[/]")
        else:
            dest = config.RAW_DIR / spec["dir"]
            if dest.exists():
                console.print(f"  [dim]{name}: already cloned[/]")
                continue
            if shutil.which("git") is None:
                console.print("  [red]git not found, cannot clone the robocall set[/]")
                continue
            # The pipeline only reads metadata.csv from this repo, which is
            # 684 KB out of 1.7 GB. A blob-filtered clone fetches the tree and
            # then only the files actually opened, so the default costs
            # seconds instead of minutes and leaves Colab's disk alone. Pass
            # --robocall-audio if you want the WAVs too.
            if robocall_audio:
                console.print(f"  {name}: cloning {spec['git']} with audio, 1.7 GB")
                cmd = ["git", "clone", "--depth", "1", spec["git"], str(dest)]
            else:
                console.print(f"  {name}: cloning {spec['git']}, transcripts only")
                cmd = ["git", "clone", "--depth", "1", "--filter=blob:none",
                       "--no-checkout", spec["git"], str(dest)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and not robocall_audio:
                for c in (["git", "sparse-checkout", "set", "--no-cone",
                           "metadata.csv", "README.md", "LICENSE.md"],
                          ["git", "checkout"]):
                    r = subprocess.run(c, cwd=str(dest), capture_output=True, text=True)
                    if r.returncode != 0:
                        break
                # older git without sparse-checkout: fall back to a plain one
                if r.returncode != 0:
                    console.print("    [dim]sparse checkout unavailable, "
                                  "fetching the whole repo[/]")
                    shutil.rmtree(dest, ignore_errors=True)
                    r = subprocess.run(
                        ["git", "clone", "--depth", "1", spec["git"], str(dest)],
                        capture_output=True, text=True)
            console.print("    [green]done[/]" if r.returncode == 0
                          else f"    [red]failed: {r.stderr[-200:]}[/]")

    # the metadata archive is tiny and carries the gender and accent labels
    meta = config.RAW_DIR / "Metadata.tar.gz"
    if not meta.exists():
        try:
            urllib.request.urlretrieve(f"{D.OPENSLR_118}/Metadata.tar.gz", meta)
            console.print("  [green]metadata (speaker gender and accent labels)[/]")
        except Exception:
            pass

    t = Table(show_header=False, box=None)
    for k, v in D.status().items():
        t.add_row(k, str(v))
    console.print(t)

    if pairs > 0:
        _rule(f"Building {pairs} anti-spoofing pairs")
        out = D.build_antispoof_pairs(n=pairs)
        if out.get("error"):
            console.print(f"[red]{out['error']}[/]")
            raise typer.Exit(1)
        elif int(out.get("n_pairs") or 0) < max(20, pairs // 4):
            console.print(f"[red]only {out.get('n_pairs')} of {pairs} pairs were "
                          f"built, the anti-spoofing branch cannot use this[/]")
            raise typer.Exit(1)
        else:
            console.print(f"[green]{out['n_pairs']} pairs, {out['n_files']} files[/] "
                          f"-> {out.get('manifest')}")


@app.command("train")
def train(
    verbose: bool = typer.Option(True, help="print progress"),
):
    """Train every model from the corpus on disk."""
    from .pipeline import Pipeline

    _rule("Training")
    p = Pipeline.load()
    if not p.calls:
        console.print("[red]No corpus found. Run: swarkavach gen-corpus[/]")
        raise typer.Exit(1)
    t0 = time.time()
    report = p.fit(verbose=verbose)
    console.print_json(json.dumps(report, default=str))

    # Pipeline.fit catches per-stage exceptions so one dead branch does not
    # cost the whole run, records "failed: ..." and returns normally. That is
    # the right behaviour for the library and the wrong exit code for a
    # caller: a run where only the fusion stage worked exited 0 and looked
    # identical to a clean one, and the notebook then checkpointed a half
    # empty models directory to Drive and skipped training ever after.
    failures = [k for k, v in report.items()
                if isinstance(v, str) and v.startswith("failed:")]
    if failures:
        console.print(f"[red]{len(failures)} stage(s) failed: {', '.join(failures)}[/]")
        for k in failures:
            console.print(f"  [red]{k}: {report[k]}[/]")
        raise typer.Exit(1)

    console.print(f"[green]trained in {time.time() - t0:.1f}s[/] -> {config.MODELS_DIR}")


@app.command("evaluate")
def evaluate(
    codecs: str = typer.Option("clean,g711u,g711a,gsm", help="comma separated codec list"),
    skip_robustness: bool = typer.Option(False, help="skip the codec sweep"),
):
    """Run the full evaluation and write data/results/*.json."""
    from .evaluate import run_full_evaluation

    _rule("Evaluating")
    t0 = time.time()
    out = run_full_evaluation(
        codecs=[c.strip() for c in codecs.split(",") if c.strip()],
        skip_robustness=skip_robustness,
    )
    for k, v in (out.get("headline") or {}).items():
        console.print(f"  {k:28s} {v}")
    console.print(f"[green]evaluated in {time.time() - t0:.1f}s[/] -> {config.RESULTS_DIR}")


@app.command("score")
def score(
    path: str = typer.Argument(..., help="a WAV file, or a corpus call id"),
    transcript: Optional[str] = typer.Option(None, help="path to a transcript text file"),
    json_out: bool = typer.Option(False, "--json", help="print the whole verdict as JSON"),
):
    """Score one call and print the verdict."""
    from .pipeline import Pipeline, analyze_file

    p = Pipeline.load()
    call = p.get_call(path)
    if call is not None:
        audio = sr = None
        if call.audio_path and Path(call.audio_path).exists():
            from .audioio import read_audio
            audio, sr = read_audio(call.audio_path, sr=SETTINGS.frame.sr)
        v = p.analyze(call, audio=audio, sr=sr)
    else:
        if not Path(path).exists():
            console.print(f"[red]no such file or call id: {path}[/]")
            raise typer.Exit(1)
        text = Path(transcript).read_text(encoding="utf-8") if transcript else None
        v = analyze_file(path, transcript=text, pipeline=p)

    if json_out:
        console.print_json(json.dumps(v.to_dict()))
        return

    colour = {"low": "green", "elevated": "yellow", "high": "dark_orange", "critical": "red"}
    console.print(
        f"\n[bold]{v.call_id}[/]  risk [bold {colour.get(v.band, 'white')}]"
        f"{v.risk:.1%}[/] ({v.band})")
    console.print(
        f"  voice authenticity {v.authenticity:.2f}   scam intent {v.intent:.2f}   "
        f"prosody mismatch {v.pim:.2f}")
    if v.ttd is not None:
        console.print(f"  alert after {v.ttd:.1f}s (turn {v.ttd_turns + 1})")
    console.print("\n[bold]why[/]")
    for r in v.evidence.reasons:
        console.print(f"  - {r['text']}")


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(7860),
    debug: bool = typer.Option(False),
):
    """Start the console."""
    sys.path.insert(0, str(config.ROOT))
    from server.app import main as run_server

    run_server(host=host, port=port, debug=debug)


@app.command("demo")
def demo(
    n: int = typer.Option(280, help="corpus size for the quick run"),
    port: int = typer.Option(7860),
    skip_existing: bool = typer.Option(True, help="reuse a corpus and models already on disk"),
):
    """Generate, train, evaluate and open the console, in one command."""
    from .runner import _corpus_stale_reason

    have_corpus = any(config.CORPUS_CALLS_DIR.glob("*.json")) if config.CORPUS_CALLS_DIR.exists() else False
    have_models = (config.MODELS_DIR / "fusion_full_logreg.joblib").exists()

    # A corpus generated from older templates is not the corpus this code
    # produces, and models trained on it would never see the current text.
    stale = _corpus_stale_reason() if have_corpus else None
    if stale:
        console.print(f"[yellow]corpus on disk is stale ({stale}), regenerating[/]")
        have_corpus = False
        have_models = False

    if not (skip_existing and have_corpus):
        gen_corpus(n=n, seed=SETTINGS.pipeline.seed, audio=True, backend="sim")
    else:
        console.print("[dim]corpus already on disk, reusing it[/]")

    if not (skip_existing and have_models):
        train(verbose=True)
    else:
        console.print("[dim]models already on disk, reusing them[/]")

    try:
        evaluate(codecs="clean,g711u,gsm", skip_robustness=False)
    except Exception as exc:
        console.print(f"[yellow]evaluation skipped: {exc}[/]")

    serve(port=port)


@app.command("info")
def info():
    """Show what is loaded and what is missing."""
    from .pipeline import Pipeline

    p = Pipeline.load()
    st = p.status()
    t = Table(show_header=False, box=None)
    t.add_row("version", config.PROJECT_VERSION)
    t.add_row("data dir", str(config.DATA_DIR))
    t.add_row("ffmpeg", "found" if config.find_ffmpeg() else "not installed, G.711 in numpy")
    for k in ("n_calls", "antispoof_backend", "ner_backend", "intent_backend",
              "intent_backend", "fusion_model", "asr_backend"):
        t.add_row(k.replace("_", " "), str(st.get(k)))
    console.print(t)
    for note in st.get("notes", []):
        console.print(f"[yellow]  {note}[/]")
    res = sorted(p.name for p in config.RESULTS_DIR.glob("*.json")) if config.RESULTS_DIR.exists() else []
    console.print(f"\nresults on disk: {', '.join(res) if res else 'none'}")


def main():
    app()


if __name__ == "__main__":
    main()
