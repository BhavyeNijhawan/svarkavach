"""The whole pipeline in one Colab cell, with no notebook to go stale.

Paste the contents of this file into a single Colab cell and run it. That is
the entire workflow. There is nothing else to open, sync or re-download.

Why this exists. A Colab notebook opened from GitHub does not run from the
clone it makes, and Colab writes a copy of it into your Drive the moment you
run it, so "open it again from GitHub" quietly reopens your Drive copy
instead. Three separate sessions ran a pre-flight that had already been fixed,
each costing about 35 minutes, because the fix was in the repo and the cells
were not. A cell that force-syncs the clone and then calls into it cannot have
that problem: the only logic in the cell is the part that fetches the logic.

It is safe to re-run. Every stage checks Drive first and skips itself if its
output is already there, so an interrupted session picks up where it stopped.
"""

REPO_SLUG = "BhavyeNijhawan/svarkavach"
BRANCH = "main"
N_CALLS = 480
N_PAIRS = 600
CODECS = "clean,g711u,g711a,gsm"
FORCE_REBUILD = []        # e.g. ["corpus"] to redo a stage Drive already has
FULL_TESTS = False        # True adds the 35 minute integration test

import os
import subprocess
import sys
from pathlib import Path

CLONE = "/content/svk_repo" if Path("/content").is_dir() else "svk_repo"

# ---------------------------------------------------------------- the token
TOKEN = ""
try:
    from google.colab import userdata

    TOKEN = userdata.get("GH_PAT") or ""
except Exception:
    pass

CLEAN_URL = f"https://github.com/{REPO_SLUG}.git"
AUTH_URL = f"https://{TOKEN}@github.com/{REPO_SLUG}.git" if TOKEN else CLEAN_URL


def _git(*args, cwd=None):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        msg = (r.stderr or r.stdout or "")
        if TOKEN:
            msg = msg.replace(TOKEN, "***")
        raise SystemExit(f"git {' '.join(args[:2])} failed: {msg[-400:]}")
    return r


# --------------------------------------------------------------- get the code
if Path(CLONE, ".git").exists():
    print(f"syncing {CLONE} to origin/{BRANCH}")
    _git("-C", CLONE, "remote", "set-url", "origin", AUTH_URL)
    _git("-C", CLONE, "fetch", "--depth", "1", "origin", BRANCH)
    # Hard reset rather than pull. Runs write results and models into the
    # clone, and a pull then refuses to fast-forward over them. Nothing in
    # the clone is worth keeping: Drive holds the artifacts.
    _git("-C", CLONE, "reset", "--hard", f"origin/{BRANCH}")
    _git("-C", CLONE, "clean", "-fd")
else:
    print(f"cloning into {CLONE}")
    _git("clone", "--branch", BRANCH, AUTH_URL, CLONE)

_git("-C", CLONE, "remote", "set-url", "origin", CLEAN_URL)
os.chdir(CLONE)
head = subprocess.run(["git", "log", "-1", "--format=%h  %s"],
                      capture_output=True, text=True).stdout.strip()
print("at:", head)

# ----------------------------------------------------------------- install
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."])
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "edge-tts", "indic-transliteration", "soundfile", "openai-whisper"])

# Drop anything already imported, so a re-run in the same session picks up the
# code that was just synced rather than the copy sitting in sys.modules.
for m in [m for m in sys.modules if m == "swarkavach" or m.startswith("swarkavach.")]:
    del sys.modules[m]
src = str(Path(CLONE, "src"))
if src not in sys.path:
    sys.path.insert(0, src)

from swarkavach import runner        # noqa: E402

print("runner from:", runner.__file__)

# ------------------------------------------------------------------- run it
runner.preflight(full=FULL_TESTS)
runner.begin(force=FORCE_REBUILD)

runner.fetch_data(pairs=N_PAIRS)
if not runner.audit_pairs():
    raise SystemExit("anti-spoofing set is not usable, see the audit above")

runner.gen_corpus(n=N_CALLS)
runner.train()
runner.evaluate(codecs=CODECS)
runner.report()
runner.finish()

print()
print("done. data/results/*.json and data/models/ are in the clone and in Drive.")
