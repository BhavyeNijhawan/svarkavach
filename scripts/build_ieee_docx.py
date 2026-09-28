"""Build docs/IEEE_REPORT.docx from docs/IEEE_REPORT.md.

Two-column IEEE conference layout, written directly with python-docx because
the column breaks and the single-column title block are section properties
that a markdown converter will not produce.

Source conventions, all plain text so the markdown stays readable:

    TITLE: ...            one line, 24 pt centred, single column
    AUTHOR: ...           one line under it
    ABSTRACT: ...         one paragraph, 9 pt bold, justified
    INDEX: a, b, c        index terms, 9 pt bold italic lead-in
    ## I. Section         section head, centred, small caps
    ### A. Subsection     subsection head, italic
    FIGURE: path | caption
    TABLE_n_CAPTION: ...  followed by a pipe table
    $$ ... $$ (n)         display equation, kept as plain text
"""
import re
from pathlib import Path

import docx
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "IEEE_REPORT.md"
OUT = ROOT / "docs" / "IEEE_REPORT.docx"

FONT = "Times New Roman"
BODY, SMALL, TINY = 10, 9, 8
COL_WIDTH_IN = 3.4          # (8.5 - 1.24 - 0.2) / 2


def set_cols(section, num, space_twips=288):
    cols = section._sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        section._sectPr.append(cols)
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), str(space_twips))
    cols.set(qn("w:equalWidth"), "1")


def style_font(style, size, bold=False, italic=False):
    style.font.name = FONT
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.italic = italic
    style.font.color.rgb = RGBColor(0, 0, 0)
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for a in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(qn("w:" + a), FONT)
    for a in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        if fonts.get(qn("w:" + a)) is not None:
            del fonts.attrib[qn("w:" + a)]


d = docx.Document()
sec = d.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.left_margin = sec.right_margin = Inches(0.62)
sec.top_margin = Inches(0.75)
sec.bottom_margin = Inches(1.0)
set_cols(sec, 1)

st = d.styles["Normal"]
style_font(st, BODY)
st.paragraph_format.space_after = Pt(0)
st.paragraph_format.line_spacing = 1.0


def para(text="", size=BODY, bold=False, italic=False, align=None,
         before=0, after=0, first_indent=None, keep=False):
    p = d.add_paragraph()
    pf = p.paragraph_format
    pf.space_before, pf.space_after, pf.line_spacing = Pt(before), Pt(after), 1.0
    if first_indent is not None:
        pf.first_line_indent = Inches(first_indent)
    if keep:
        pf.keep_with_next = True
    p.alignment = {"c": WD_ALIGN_PARAGRAPH.CENTER, "j": WD_ALIGN_PARAGRAPH.JUSTIFY,
                   "l": WD_ALIGN_PARAGRAPH.LEFT}.get(align, WD_ALIGN_PARAGRAPH.JUSTIFY)
    if text:
        add_runs(p, text, size, bold, italic)
    return p


def add_runs(p, text, size=BODY, bold=False, italic=False):
    """**bold**, *italic* and $math$ become runs; nothing else is markup."""
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|\$[^$]+\$)", text):
        if not part:
            continue
        b, i = bold, italic
        if part.startswith("**") and part.endswith("**"):
            part, b = part[2:-2], True
        elif part.startswith("*") and part.endswith("*"):
            part, i = part[1:-1], True
        elif part.startswith("$") and part.endswith("$"):
            part, i = part[1:-1].replace("\\", "").replace("_", "").replace("{", "").replace("}", ""), True
        r = p.add_run(part)
        r.bold, r.italic = b, i
        r.font.size = Pt(size)
        r.font.name = FONT


def clean_math(tex):
    """A display equation as readable plain text, since the source has few."""
    s = tex.strip().strip("$").strip()
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\(sum|max|min|log|cos|left|right|cdot|tfrac|text|mid)", " ", s)
    s = s.replace("\\", "").replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def _section(num_cols):
    s = d.add_section(WD_SECTION.CONTINUOUS)
    s.left_margin = s.right_margin = Inches(0.62)
    s.top_margin, s.bottom_margin = Inches(0.75), Inches(1.0)
    set_cols(s, num_cols)
    return s


def table(rows, caption=None, number=None, wide=False):
    # A table with many columns does not fit a 3.4 inch column, so it spans
    # the page inside its own single-column section, which is what IEEE
    # papers do and what the template this follows does for its first table.
    if wide:
        _section(1)
    if caption:
        para(f"TABLE {number}", size=TINY, align="c", before=6, keep=True)
        para(caption, size=TINY, align="c", after=2, keep=True, italic=True)
    t = d.add_table(rows=len(rows), cols=len(rows[0]))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_pr = t._tbl.tblPr
    w = OxmlElement("w:tblW"); w.set(qn("w:type"), "pct"); w.set(qn("w:w"), "5000"); tbl_pr.append(w)
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom", "insideH"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "6"); e.set(qn("w:color"), "000000")
        b.append(e)
    for edge in ("left", "right", "insideV"):
        e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), "none"); b.append(e)
    tbl_pr.append(b)
    for i, row in enumerate(rows):
        for j, text in enumerate(row):
            cell = t.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j else WD_ALIGN_PARAGRAPH.LEFT
            add_runs(p, text, size=TINY, bold=(i == 0))
    d.add_paragraph().paragraph_format.space_after = Pt(4)
    if wide:
        _section(2)


def figure(path, caption, number):
    p = d.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(ROOT / path), width=Inches(COL_WIDTH_IN))
    para(f"Fig. {number}.  {caption}", size=TINY, align="j", after=6)


# --------------------------------------------------------------- parse
lines = SRC.read_text(encoding="utf-8").splitlines()
title = author = abstract = index = None
i = 0
while i < len(lines):
    ln = lines[i].strip(); i += 1
    if ln.startswith("TITLE: "):
        title = ln[7:]
    elif ln.startswith("AUTHOR: "):
        author = ln[8:]
    elif ln.startswith("ABSTRACT: "):
        abstract = ln[10:]
    elif ln.startswith("INDEX: "):
        index = ln[7:]
        break

# --------------------------------------------------- single-column head
para(title, size=18, bold=True, align="c", after=10)
para(author, size=10, align="c", after=14)

# --------------------------------------------------------- two columns
body = d.add_section(WD_SECTION.CONTINUOUS)
body.left_margin = body.right_margin = Inches(0.62)
body.top_margin, body.bottom_margin = Inches(0.75), Inches(1.0)
set_cols(body, 2)

p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p.paragraph_format.space_after = Pt(6)
r = p.add_run("Abstract\u2014"); r.bold = True; r.italic = True; r.font.size = Pt(SMALL); r.font.name = FONT
add_runs(p, abstract, size=SMALL, bold=True)

p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p.paragraph_format.space_after = Pt(8)
r = p.add_run("Index Terms\u2014"); r.bold = True; r.italic = True; r.font.size = Pt(SMALL); r.font.name = FONT
add_runs(p, index, size=SMALL, bold=True)

fig_no = tab_no = 0
pending_caption = None
buf_rows = []

while i < len(lines):
    raw = lines[i]; ln = raw.strip(); i += 1

    if not ln:
        if buf_rows:
            table(buf_rows, *(pending_caption if pending_caption else (None, None, False)))
            buf_rows, pending_caption = [], None
        continue

    if ln.startswith("## "):
        para(ln[3:], size=BODY, align="c", before=10, after=4, keep=True)
        continue
    if ln.startswith("### "):
        para(ln[4:], size=BODY, italic=True, align="l", before=6, after=2, keep=True)
        continue
    if ln.startswith("FIGURE: "):
        path, cap = ln[8:].split(" | ", 1)
        fig_no += 1
        figure(path.strip(), cap.strip(), fig_no)
        continue
    m = re.match(r"TABLE_([IVX]+)(_WIDE)?_CAPTION: (.+)", ln)
    if m:
        tab_no += 1
        pending_caption = (m.group(3), m.group(1), bool(m.group(2)))
        continue
    if ln.startswith("|"):
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):
            continue
        buf_rows.append(cells)
        continue
    if ln.startswith("$$"):
        eq = re.match(r"\$\$(.*)\$\$\s*\((\d+)\)", ln)
        if eq:
            p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(4); p.paragraph_format.space_after = Pt(4)
            r = p.add_run(clean_math(eq.group(1))); r.italic = True; r.font.size = Pt(BODY); r.font.name = FONT
            r2 = p.add_run(f"\t\t({eq.group(2)})"); r2.font.size = Pt(BODY); r2.font.name = FONT
        continue
    if ln.startswith("[") and "]" in ln[:5]:
        para(ln, size=TINY, align="j", after=2)
        continue

    para(ln, size=BODY, align="j", after=4, first_indent=0.18)

if buf_rows:
    table(buf_rows, *(pending_caption if pending_caption else (None, None, False)))

d.save(str(OUT))
print("wrote", OUT, "|", len(d.paragraphs), "paragraphs |", len(d.tables), "tables |", len(d.inline_shapes), "figures")
