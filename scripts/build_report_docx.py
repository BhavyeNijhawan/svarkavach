"""Build docs/PROJECT_REPORT.docx from docs/PROJECT_REPORT.md.

pandoc does the conversion, because it turns the LaTeX equations into
native Word math. Everything else about the look is set here: a reference
document restyled to Cambria with black headings and narrow margins, and a
post-pass that borders the tables, sizes the images, centres the captions,
adds page numbers and lays out the title page.

Word cannot render Mermaid, so each ```mermaid block is replaced by the PNG
in docs/figures/ whose number matches the caption after it (rendered through
mermaid.ink when missing or with --render), and the sources go into an
appendix so the diagrams stay editable.
"""
import base64
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import docx
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "PROJECT_REPORT.md"
OUT = ROOT / "docs" / "PROJECT_REPORT.docx"
FIG = ROOT / "docs" / "figures"
TMP = ROOT / "docs" / "_report_docx.md"
REF = ROOT / "docs" / "_reference.docx"

FONT = "Cambria"
BLACK = RGBColor(0, 0, 0)
TEXT_WIDTH_IN = 6.2
MAX_HEIGHT_IN = 2.8        # a figure may not own a page          # letter width minus two 0.5 inch margins


# ------------------------------------------------------------- mermaid
def render(code: str, out: Path) -> None:
    state = json.dumps({"code": code, "mermaid": {"theme": "neutral"}})
    b64 = base64.urlsafe_b64encode(state.encode("utf-8")).decode("ascii")
    url = f"https://mermaid.ink/img/{b64}?type=png&bgColor=ffffff&width=1400"
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=90) as r:
        out.write_bytes(r.read())


src = SRC.read_text(encoding="utf-8")
appendix = []


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
# Mermaid sources stay in the markdown, which is in the repository, rather
# than adding pages to a report that has a page budget
# the repository link closes the document
body += "\n\n---\n\n" + (ROOT / "docs" / "_repo_section.md").read_text(encoding="utf-8")
TMP.write_text(body, encoding="utf-8", newline="\n")


# ------------------------------------------------------ reference styles
def set_font(style, size=None, bold=None, italic=None, color=BLACK, name=FONT):
    f = style.font
    f.name = name
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        fonts.set(qn(a), name)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if fonts.get(qn(a)) is not None:
            del fonts.attrib[qn(a)]
    if size is not None:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic
    if color is not None:
        f.color.rgb = color


subprocess.run(["pandoc", "-o", str(REF), "--print-default-data-file", "reference.docx"], check=True)
ref = docx.Document(str(REF))
st = ref.styles
set_font(st["Normal"], 9.5)
st["Normal"].paragraph_format.space_after = Pt(4)
st["Normal"].paragraph_format.line_spacing = 1.0
for name in ("Body Text", "First Paragraph", "Compact"):
    if name in [s.name for s in st]:
        set_font(st[name], 9.5)
        st[name].paragraph_format.space_after = Pt(4)
        st[name].paragraph_format.line_spacing = 1.0
set_font(st["Title"], 19, bold=True)
st["Title"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
h = st["Heading 1"]; set_font(h, 13, bold=True); h.paragraph_format.space_before = Pt(12); h.paragraph_format.space_after = Pt(5); h.paragraph_format.page_break_before = False
h = st["Heading 2"]; set_font(h, 11, bold=True); h.paragraph_format.space_before = Pt(9); h.paragraph_format.space_after = Pt(3)
h = st["Heading 3"]; set_font(h, 10, bold=True, italic=False); h.paragraph_format.space_before = Pt(7); h.paragraph_format.space_after = Pt(2)
for name in ("Heading 4", "Heading 5", "Heading 6"):
    set_font(st[name], 9.5, bold=True, italic=False)
for name in ("Source Code", "Verbatim Char"):
    if name in [s.name for s in st]:
        set_font(st[name], 7.5, name="Consolas")
if "Caption" in [s.name for s in st]:
    set_font(st["Caption"], 9.5, italic=False)
for sec in ref.sections:
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = Inches(0.5)
# kill the theme colour on hyperlinks too, so nothing on the page is blue by default
if "Hyperlink" in [s.name for s in st]:
    st["Hyperlink"].font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
ref.save(str(REF))

subprocess.run(["pandoc", str(TMP), "-o", str(OUT),
                "--from", "markdown+tex_math_dollars+pipe_tables+fenced_divs+raw_tex",
                "--reference-doc", str(REF),
                "--resource-path", str(ROOT)], check=True)


# ------------------------------------------------------------ post-pass
d = docx.Document(str(OUT))


def borders(table):
    tbl_pr = table._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:space"), "0"); e.set(qn("w:color"), "808080")
        b.append(e)
    tbl_pr.append(b)


for t in d.tables:
    borders(t)
    t.autofit = True
    # full text width, so a narrow table does not sit in a corner of the page
    tbl_pr = t._tbl.tblPr
    w = tbl_pr.find(qn("w:tblW"))
    if w is None:
        w = OxmlElement("w:tblW"); tbl_pr.append(w)
    w.set(qn("w:type"), "pct"); w.set(qn("w:w"), "5000")
    for i, row in enumerate(t.rows):
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.line_spacing = 1.0
                for r in p.runs:
                    r.font.size = Pt(8)
                    r.font.name = FONT
                    if i == 0:
                        r.font.bold = True
    # light grey header row
    for cell in t.rows[0].cells:
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "E8E8E8")
        tcPr.append(shd)

# images: fit the text width, centre them; captions: centred, 9.5 pt
emu_per_in = 914400
for p in d.paragraphs:
    pics = p._p.findall(".//" + qn("wp:inline"))
    if pics:
        for inl in pics:
            ext = inl.find(qn("wp:extent"))
            cx, cy = int(ext.get("cx")), int(ext.get("cy"))
            k = min(TEXT_WIDTH_IN * emu_per_in / cx, MAX_HEIGHT_IN * emu_per_in / cy, 1.0)
            if k < 1.0:
                ext.set("cx", str(int(cx * k))); ext.set("cy", str(int(cy * k)))
                for a in inl.iter(qn("a:ext")):
                    a.set("cx", str(int(cx * k))); a.set("cy", str(int(cy * k)))
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
    txt = p.text.strip()
    is_list = p._p.pPr is not None and p._p.pPr.find(qn("w:numPr")) is not None
    is_caption = (re.match(r"^(Figure|Table) (\d+|[A-Z]\d)\.", txt) and not is_list
                  and p.runs and p.runs[0].bold)
    if is_caption:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if txt.startswith("Figure") else WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_after = Pt(7 if txt.startswith("Figure") else 2)
        p.paragraph_format.keep_with_next = txt.startswith("Table")
        for r in p.runs:
            r.font.size = Pt(8.5)

# title page paragraphs (the custom-style divs)
for p in d.paragraphs:
    if p.style.name in ("TitlePage", "TitleBlock"):
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(8)
        if p.style.name == "TitlePage":
            p.paragraph_format.space_before = Pt(140)
        for r in p.runs:
            r.font.name = FONT
            r.font.size = Pt(14 if p.style.name == "TitlePage" else 12)
            if p.style.name == "TitlePage":
                r.font.bold = True
    if p.style.name == "Title":
        p.paragraph_format.space_before = Pt(150)
        p.paragraph_format.space_after = Pt(30)

# the first Heading 1 after the title page must not force another break
first_h1 = next(p for p in d.paragraphs if p.style.name == "Heading 1")
first_h1.paragraph_format.page_break_before = False

# page numbers in the footer
for sec in d.sections:
    fp = sec.footer.paragraphs[0] if sec.footer.paragraphs else sec.footer.add_paragraph()
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp.add_run()
    for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), tag); run._r.append(fc)
        else:
            it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = text; run._r.append(it)
    run.font.size = Pt(9); run.font.name = FONT

d.save(str(OUT))
TMP.unlink()
REF.unlink()
print("wrote", OUT, "|", len(d.paragraphs), "paragraphs |", len(d.tables), "tables |", len(d.inline_shapes), "images")
