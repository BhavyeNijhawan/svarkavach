"""Build docs/PROJECT_REPORT.docx from docs/PROJECT_REPORT.md with pandoc.

Word cannot render Mermaid, so each ```mermaid block is replaced by the PNG
in docs/figures/ whose number matches the caption that follows it, and the
Mermaid sources go into an appendix so the diagrams stay editable. Renders
missing PNGs through mermaid.ink.
"""
import base64
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "PROJECT_REPORT.md"
OUT = ROOT / "docs" / "PROJECT_REPORT.docx"
FIG = ROOT / "docs" / "figures"
TMP = ROOT / "docs" / "_report_docx.md"

src = SRC.read_text(encoding="utf-8")
FIG.mkdir(exist_ok=True)
appendix = []


def render(code: str, out: Path) -> None:
    state = json.dumps({"code": code, "mermaid": {"theme": "neutral"}})
    b64 = base64.urlsafe_b64encode(state.encode("utf-8")).decode("ascii")
    url = f"https://mermaid.ink/img/{b64}?type=png&bgColor=ffffff&width=1400"
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=90) as r:
        out.write_bytes(r.read())


def replace_block(m: re.Match) -> str:
    code = m.group(1)
    after = src[m.end():m.end() + 400]
    cap = re.search(r"\*\*Figure (\d+)\.\*\* ([^\n]+)", after)
    n, title = int(cap.group(1)), cap.group(2)
    png = FIG / f"fig{n}.png"
    if "--render" in sys.argv or not png.exists():
        render(code, png)
        print("rendered", png.name)
    appendix.append((n, title, code.rstrip()))
    return f"![]({png.relative_to(ROOT).as_posix()})"


body = re.sub(r"```mermaid\n(.*?)```", replace_block, src, flags=re.S)
appendix.sort()
body += ("\n\n---\n\n# Appendix A. Diagram sources (Mermaid)\n\n"
         "Every figure in this report is a Mermaid diagram. The sources below "
         "reproduce the figures exactly and render in any Mermaid viewer.\n")
for n, title, code in appendix:
    body += f"\n## Figure {n}. {title}\n\n```\n{code}\n```\n"
TMP.write_text(body, encoding="utf-8", newline="\n")

subprocess.run(["pandoc", str(TMP), "-o", str(OUT),
                "--from", "markdown+tex_math_dollars+pipe_tables",
                "--resource-path", str(ROOT)], check=True)
TMP.unlink()
print("wrote", OUT)
