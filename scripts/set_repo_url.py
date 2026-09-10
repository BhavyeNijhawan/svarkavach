"""Point every Colab notebook at your own GitHub repository.

The notebooks clone the repo before doing anything, so the URL has to be
yours, not the placeholder. Rather than editing five notebooks by hand:

    python scripts/set_repo_url.py https://github.com/<you>/<repo>.git

Run it with no argument to see what the notebooks currently point at.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
# Notebooks 01 to 05 store a full clone URL; notebook 00 stores a user/repo
# slug, because it builds the URL with a token in front for private repos.
PATTERN = re.compile(r'(REPO_URL\s*=\s*)(["\'])([^"\']*)(\2)')
SLUG_PATTERN = re.compile(r'(REPO_SLUG\s*=\s*)(["\'])([^"\']*)(\2)')


def to_slug(url: str) -> str:
    """https://github.com/user/repo.git or git@github.com:user/repo -> user/repo"""
    s = url.strip().rstrip("/")
    if s.endswith(".git"):
        s = s[:-4]
    if "github.com/" in s:
        s = s.split("github.com/", 1)[1]
    elif ":" in s:
        s = s.rsplit(":", 1)[1]
    return s


def current() -> dict:
    found = {}
    for nb_path in sorted(NOTEBOOKS.glob("*.ipynb")):
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        for cell in nb.get("cells", []):
            body = "".join(cell.get("source", []))
            m = PATTERN.search(body) or SLUG_PATTERN.search(body)
            if m:
                found[nb_path.name] = m.group(3)
                break
    return found


def set_url(url: str) -> int:
    if not url.startswith(("https://", "git@")):
        print(f"that does not look like a git remote: {url}")
        return 1
    changed = 0
    for nb_path in sorted(NOTEBOOKS.glob("*.ipynb")):
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        hit = False
        for cell in nb.get("cells", []):
            src = cell.get("source", [])
            slug = to_slug(url)
            for i, line in enumerate(src):
                if PATTERN.search(line):
                    src[i] = PATTERN.sub(
                        lambda m: f"{m.group(1)}{m.group(2)}{url}{m.group(4)}", line)
                    hit = True
                elif SLUG_PATTERN.search(line):
                    src[i] = SLUG_PATTERN.sub(
                        lambda m: f"{m.group(1)}{m.group(2)}{slug}{m.group(4)}", line)
                    hit = True
        if hit:
            # indent=1 matches how the notebooks were written, so the diff
            # stays limited to the lines that actually changed
            nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                               encoding="utf-8")
            changed += 1
            print(f"  updated {nb_path.name}")
    if not changed:
        print("no REPO_URL assignment found in any notebook")
        return 1
    print(f"\n{changed} notebooks now clone {url}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("notebooks currently clone:")
        for name, url in current().items():
            print(f"  {name:34s} {url}")
        print("\nto change them:")
        print("  python scripts/set_repo_url.py https://github.com/<you>/<repo>.git")
        sys.exit(0)
    sys.exit(set_url(sys.argv[1].strip()))
