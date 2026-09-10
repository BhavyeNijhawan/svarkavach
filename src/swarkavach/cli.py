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
    console.print(f"[green]done in {time.time() - t0:.1f}s[/] -> {config.CORPUS_DIR}")


@app.command("fetch-data")
def fetch_data(
    which: str = typer.Option("core", help="core | all | a comma separated list"),
    pairs: int = typer.Option(500, help="paired anti-spoof utterances to build, 0 to skip"),
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
                console.print(f"  [dim]{name}: already present[/]")
                continue
            console.print(f"  {name}: {spec['size_mb']} MB from {spec['url']}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            try:
                urllib.request.urlretrieve(spec["url"], tmp)
                tmp.rename(dest)
                console.print(f"    [green]done[/] {dest.stat().st_size / 1e6:.0f} MB")
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
            console.print(f"  {name}: cloning {spec['git']}")
            r = subprocess.run(["git", "clone", "--depth", "1", spec["git"], str(dest)],
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
    have_corpus = any(config.CORPUS_CALLS_DIR.glob("*.json")) if config.CORPUS_CALLS_DIR.exists() else False
    have_models = (config.MODELS_DIR / "fusion_full_logreg.joblib").exists()

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
