"""Checkpoint the expensive artifacts to Google Drive so a Colab disconnect
does not cost you the whole run.

Colab throws away the local disk when the session ends, and the free tier ends
sessions often. Rebuilding from nothing costs roughly:

    downloads        about 5 minutes     (800 MB over the network)
    TTS cache        about 60 minutes    (thousands of requests, rate limited)
    corpus audio     about 10 minutes    (cheap IF the TTS cache survived)
    training         about 15 minutes
    evaluation       about 15 minutes

So the one that actually hurts is the TTS cache. Everything here exists to
make sure that hour is spent once.

Why not just point SWARKAVACH_DATA at Drive and be done? Because Drive is a
network filesystem with per-file overhead, and the TTS cache is thousands of
small WAVs. Reading them one at a time over Drive is slower than the synthesis
was. So the working directory stays on local disk, and this module moves whole
artifacts across as single archives.

Two transfer modes, chosen per artifact:

    archive   tar.gz the directory. Right for many small files.
    files     copy each file as it is. Right for a few large ones that are
              already compressed, where tarring again would just burn CPU.

Usage in a notebook:

    from swarkavach import colab
    colab.mount()
    colab.restore("raw")        # before the download step
    ...                          # do the work
    colab.save("raw")           # after it
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import CORPUS_DIR, DATA_DIR, MODELS_DIR, RAW_DIR, RESULTS_DIR

#: Where the checkpoints live inside the user's Drive.
DRIVE_MOUNT = Path("/content/drive")
DRIVE_SUBDIR = "swarkavach_checkpoints"

#: name -> what it covers and how to move it.
#: `stage` is only used for the human-readable summary.
ARTIFACTS: Dict[str, Dict[str, Any]] = {
    "raw": {
        "path": lambda: RAW_DIR,
        "mode": "files",
        "stage": "GramVaani core archives, about 160 MB",
        # Named rather than "*.tar.gz" on purpose. GV_Train_100h.tar.gz is 2 GB
        # and the pipeline never opens it, so a "--which all" run would burn an
        # eighth of a free Drive on a file nothing reads.
        "include": ("GV_Dev_5h.tar.gz", "GV_Eval_3h.tar.gz", "Metadata.tar.gz"),
    },
    "robocall": {
        # only metadata.csv is ever read, and it is 700 KB. The clone that
        # produces it is blob-filtered, so this is cheap on both ends.
        "path": lambda: RAW_DIR / "robocall-audio-dataset",
        "mode": "files",
        "stage": "robocall transcripts, about 1 MB",
        "include": ("metadata.csv", "README.md", "LICENSE.md"),
    },
    "tts_cache": {
        "path": lambda: CORPUS_DIR / "tts_cache",
        "mode": "archive",
        "stage": "synthesised speech, the expensive one",
    },
    "pairs": {
        "path": lambda: RAW_DIR / "antispoof_pairs",
        "mode": "archive",
        "stage": "Hindi anti-spoofing pairs",
    },
    "corpus": {
        "path": lambda: CORPUS_DIR,
        "mode": "archive",
        "stage": "generated calls and their audio",
        "exclude": ("tts_cache",),
    },
    "models": {
        "path": lambda: MODELS_DIR,
        "mode": "archive",
        "stage": "trained models",
    },
    "results": {
        "path": lambda: RESULTS_DIR,
        "mode": "archive",
        "stage": "evaluation JSON",
    },
}

_MOUNTED: Optional[Path] = None


# --------------------------------------------------------------------------
# mounting
# --------------------------------------------------------------------------


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except Exception:
        return False


def mount(quiet: bool = False) -> Optional[Path]:
    """Mount Drive and return the checkpoint directory, or None if unavailable.

    Safe to call repeatedly. Off Colab it returns None and every other
    function in this module turns into a no-op, so notebooks and scripts can
    call these unconditionally.
    """
    global _MOUNTED
    if _MOUNTED is not None:
        return _MOUNTED

    if not in_colab():
        if not quiet:
            print("not on Colab, Drive checkpointing is off")
        return None

    try:
        from google.colab import drive

        if not (DRIVE_MOUNT / "MyDrive").exists():
            drive.mount(str(DRIVE_MOUNT))
    except Exception as exc:
        print(f"could not mount Drive: {exc}")
        print("the run will still work, it just will not survive a disconnect")
        return None

    root = DRIVE_MOUNT / "MyDrive" / DRIVE_SUBDIR
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"could not create {root}: {exc}")
        return None

    _MOUNTED = root
    if not quiet:
        print(f"Drive checkpoints: {root}")
    return root


def drive_dir() -> Optional[Path]:
    return _MOUNTED


# --------------------------------------------------------------------------
# fingerprints, so unchanged artifacts are not re-uploaded
# --------------------------------------------------------------------------


def _fingerprint(path: Path, exclude: tuple = (),
                 include: tuple = ()) -> Dict[str, Any]:
    """Cheap signature of a directory: file count and total size.

    Deliberately not a hash. Hashing a few hundred megabytes on every call
    would cost more than the upload it is trying to avoid, and count plus size
    is enough to catch "nothing happened since last time".

    When the artifact only ships some of what is in the directory, `include`
    narrows the count to those files. RAW_DIR is the case that needs it: it
    holds 4 GB of extracted audio but only the source archives are worth
    sending, and a fingerprint over the extracted files would never match after
    a restore and would report a change on every run.
    """
    n, total = 0, 0
    if not path.is_dir():
        return {"files": 0, "bytes": 0}

    if include:
        files = [f for pat in include for f in path.glob(pat)]
    else:
        files = path.rglob("*")

    for f in files:
        if not f.is_file():
            continue
        if any(part in exclude for part in f.parts):
            continue
        n += 1
        try:
            total += f.stat().st_size
        except OSError:
            pass
    return {"files": n, "bytes": total}


def _meta_path(root: Path, name: str) -> Path:
    return root / f"{name}.meta.json"


def _read_meta(root: Path, name: str) -> Dict[str, Any]:
    p = _meta_path(root, name)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


# --------------------------------------------------------------------------
# save and restore
# --------------------------------------------------------------------------


def save(name: str, force: bool = False, quiet: bool = False) -> bool:
    """Copy one artifact from the working directory into Drive."""
    root = drive_dir()
    if root is None:
        return False
    spec = ARTIFACTS.get(name)
    if spec is None:
        print(f"unknown artifact {name!r}")
        return False

    src = spec["path"]()
    if not src.exists():
        return False

    exclude = tuple(spec.get("exclude", ()))
    fp = _fingerprint(src, exclude, tuple(spec.get("include", ())))
    if fp["files"] == 0:
        return False

    old = _read_meta(root, name)
    if not force and old.get("fingerprint") == fp:
        if not quiet:
            print(f"  {name}: unchanged, not re-uploading "
                  f"({fp['files']} files, {fp['bytes'] / 1e6:.0f} MB)")
        return True

    t0 = time.time()
    try:
        if spec["mode"] == "files":
            dest = root / name
            dest.mkdir(parents=True, exist_ok=True)
            patterns = spec.get("include") or ("*",)
            for pat in patterns:
                for f in src.glob(pat):
                    if not f.is_file():
                        continue
                    target = dest / f.name
                    if target.exists() and target.stat().st_size == f.stat().st_size:
                        continue
                    shutil.copy2(f, target)
        else:
            tmp = root / f"{name}.tar.gz.part"
            with tarfile.open(tmp, "w:gz", compresslevel=1) as tf:
                for child in sorted(src.iterdir()):
                    if child.name in exclude:
                        continue
                    tf.add(child, arcname=child.name)
            final = root / f"{name}.tar.gz"
            if final.exists():
                final.unlink()
            tmp.rename(final)
    except Exception as exc:
        print(f"  {name}: save failed: {type(exc).__name__}: {exc}")
        return False

    _meta_path(root, name).write_text(json.dumps({
        "fingerprint": fp, "mode": spec["mode"],
        "saved": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }, indent=2), encoding="utf-8")

    if not quiet:
        print(f"  {name}: saved {fp['files']} files, {fp['bytes'] / 1e6:.0f} MB "
              f"in {time.time() - t0:.0f}s")
    return True


def restore(name: str, quiet: bool = False) -> bool:
    """Copy one artifact out of Drive into the working directory.

    Returns whether anything was restored. A False here is not an error, it
    just means this is the first run and the work still has to be done.
    """
    root = drive_dir()
    if root is None:
        return False
    spec = ARTIFACTS.get(name)
    if spec is None:
        return False

    dest = spec["path"]()
    t0 = time.time()

    try:
        if spec["mode"] == "files":
            src = root / name
            if not src.is_dir():
                return False
            dest.mkdir(parents=True, exist_ok=True)
            n = 0
            for f in src.iterdir():
                if not f.is_file():
                    continue
                target = dest / f.name
                if target.exists() and target.stat().st_size == f.stat().st_size:
                    n += 1
                    continue
                shutil.copy2(f, target)
                n += 1
            if n == 0:
                return False
        else:
            arc = root / f"{name}.tar.gz"
            if not arc.exists():
                return False
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(arc, "r:gz") as tf:
                try:
                    tf.extractall(dest, filter="data")
                except TypeError:      # Python without the filter argument
                    tf.extractall(dest)
    except Exception as exc:
        print(f"  {name}: restore failed: {type(exc).__name__}: {exc}")
        return False

    fp = _fingerprint(dest, tuple(spec.get("exclude", ())),
                      tuple(spec.get("include", ())))
    if not quiet:
        print(f"  {name}: restored {fp['files']} files, {fp['bytes'] / 1e6:.0f} MB "
              f"in {time.time() - t0:.0f}s")
    return fp["files"] > 0


def restore_all(names: Optional[List[str]] = None, quiet: bool = False) -> Dict[str, bool]:
    """Pull back everything that exists in Drive. Call this once at the start."""
    if drive_dir() is None:
        return {}
    out = {}
    if not quiet:
        print("restoring from Drive:")
    for n in (names or list(ARTIFACTS)):
        out[n] = restore(n, quiet=quiet)
    if not quiet and not any(out.values()):
        print("  nothing checkpointed yet, this looks like the first run")
    return out


def save_all(names: Optional[List[str]] = None, quiet: bool = False) -> Dict[str, bool]:
    if drive_dir() is None:
        return {}
    out = {}
    if not quiet:
        print("saving to Drive:")
    for n in (names or list(ARTIFACTS)):
        out[n] = save(n, quiet=quiet)
    return out


# --------------------------------------------------------------------------
# what is already done
# --------------------------------------------------------------------------


def status() -> Dict[str, Any]:
    """What exists locally and what is checkpointed, side by side."""
    root = drive_dir()
    rows = {}
    for name, spec in ARTIFACTS.items():
        local = spec["path"]()
        lfp = _fingerprint(local, tuple(spec.get("exclude", ())),
                           tuple(spec.get("include", ())))
        meta = _read_meta(root, name) if root else {}
        rows[name] = {
            "stage": spec["stage"],
            "local_files": lfp["files"],
            "local_mb": round(lfp["bytes"] / 1e6, 1),
            "in_drive": bool(meta),
            "drive_files": (meta.get("fingerprint") or {}).get("files", 0),
            "drive_saved": meta.get("saved"),
        }
    return rows


def print_status() -> None:
    root = drive_dir()
    print(f"Drive: {root or 'not mounted'}")
    print(f"\n{'artifact':12s} {'local files':>12s} {'local MB':>10s} "
          f"{'in Drive':>10s}  what it is")
    for name, r in status().items():
        print(f"{name:12s} {r['local_files']:>12d} {r['local_mb']:>10.1f} "
              f"{('yes' if r['in_drive'] else 'no'):>10s}  {r['stage']}")


def what_can_be_skipped() -> List[str]:
    """Stages whose output is already present, so the notebook can skip them."""
    done = []
    st = status()
    if st["raw"]["local_files"] >= 2 and st["robocall"]["local_files"] >= 1:
        # the source archives and the transcript CSV, not the extracted audio
        done.append("download")
    if st["pairs"]["local_files"] > 20:
        done.append("pairs")
    if st["corpus"]["local_files"] > 100:
        done.append("corpus")
    if st["models"]["local_files"] >= 5:
        done.append("train")
    if st["results"]["local_files"] >= 3:
        done.append("evaluate")
    return done


if __name__ == "__main__":
    mount()
    print_status()
    print("\nalready done:", what_can_be_skipped() or "nothing")
