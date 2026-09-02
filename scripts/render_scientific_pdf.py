"""Render the generated Markdown paper to a paginated PDF using ReportLab."""

from pathlib import Path
import re
import sys
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    KeepTogether,
    CondPageBreak,
)


def render(source, destination=None):
    source = Path(source)
    destination = Path(destination) if destination else source.with_suffix(".pdf")
    fonts = [Path("/usr/share/fonts/truetype/dejavu"), Path("C:/Windows/Fonts")]
    regular, bold = "Times-Roman", "Times-Bold"
    for path in fonts:
        for normal, strong in [
            ("DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf"),
            ("times.ttf", "timesbd.ttf"),
        ]:
            if (path / normal).exists():
                pdfmetrics.registerFont(TTFont("Paper", str(path / normal)))
                pdfmetrics.registerFont(TTFont("PaperBold", str(path / strong)))
                pdfmetrics.registerFontFamily(
                    "Paper",
                    normal="Paper",
                    bold="PaperBold",
                    italic="Paper",
                    boldItalic="PaperBold",
                )
                regular, bold = "Paper", "PaperBold"
                break
        if regular == "Paper":
            break
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="PaperBody",
            fontName=regular,
            fontSize=10.5,
            leading=14.2,
            spaceAfter=8,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PaperQuote",
            parent=styles["PaperBody"],
            leftIndent=12,
            borderColor=colors.HexColor("#c9d8df"),
            borderWidth=0.5,
            borderPadding=6,
            backColor=colors.HexColor("#f6f8fa"),
        )
    )
    styles.add(ParagraphStyle(name="Cell", fontName=regular, fontSize=8.3, leading=10))
    styles.add(
        ParagraphStyle(
            name="Reference", parent=styles["PaperBody"], fontSize=9, leading=11, spaceAfter=5
        )
    )
    for key, size in [("Title", 24), ("Heading1", 14), ("Heading2", 11)]:
        styles[key].fontName = bold
        styles[key].fontSize = size
        styles[key].leading = size * 1.25
        styles[key].textColor = colors.HexColor("#202b33")
        styles[key].spaceBefore = 13
        styles[key].spaceAfter = 8
        styles[key].keepWithNext = True
    styles["PaperBody"].allowWidows = 0
    styles["PaperBody"].allowOrphans = 0

    def inline(s):
        s = escape(s)
        s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2" color="#176b87">\1</a>', s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", s)
        return re.sub(r"`([^`]+)`", r"\1", s)

    story, lines, i = [], source.read_text(encoding="utf-8").splitlines(), 0

    def kept(block):
        while (
            story
            and isinstance(story[-1], Paragraph)
            and (story[-1].getKeepWithNext() or story[-1].text.rstrip().endswith(":"))
        ):
            block.insert(0, story.pop())
        return KeepTogether(block)

    width = 504
    references = False
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if line == "## References":
            references = True
            story.append(CondPageBreak(320))
        if not line:
            continue
        if line.startswith("!["):
            match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            path = source.parent / match[2]
            if path.exists():
                w, h = ImageReader(str(path)).getSize()
                block = [Image(str(path), width=width, height=width * h / w), Spacer(1, 9)]
                following = i
                while following < len(lines) and not lines[following].strip():
                    following += 1
                if following < len(lines) and re.match(r"Figure \d+\.", lines[following].strip()):
                    block.append(Paragraph(inline(lines[following].strip()), styles["PaperBody"]))
                    i = following + 1
                story.append(kept(block))
            continue
        if line.startswith("|"):
            block = [line]
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            cells = []
            for row in block:
                if re.fullmatch(r"[| :\-]+", row):
                    continue
                cells.append(
                    [
                        Paragraph(inline(v.strip().replace("_", " ")), styles["Cell"])
                        for v in row.strip("|").split("|")
                    ]
                )
            t = Table(
                cells,
                colWidths=[width / len(cells[0])] * len(cells[0]),
                repeatRows=1,
                hAlign="LEFT",
            )
            t.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f0f4")),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.white, colors.HexColor("#f7f9fa")],
                        ),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            if len(cells) <= 9:
                story.append(kept([t, Spacer(1, 10)]))
            else:
                story.extend([t, Spacer(1, 10)])
            continue
        if line.startswith("# "):
            style = styles["Title"]
            line = line[2:]
        elif line.startswith("## "):
            style = styles["Heading1"]
            line = line[3:]
        elif line.startswith("### "):
            style = styles["Heading2"]
            line = line[4:]
        else:
            style = (
                styles["PaperQuote"]
                if line.startswith("> ")
                else styles["Reference" if references else "PaperBody"]
            )
            line = line.removeprefix("> ")
            if line.startswith("**"):
                style = ParagraphStyle(name="EvidenceLabel", parent=style, keepWithNext=True)
        story.append(Paragraph(inline(line), style))

    def page(canvas, doc):
        canvas.setFont(regular, 7)
        canvas.setFillColor(colors.HexColor("#647887"))
        canvas.drawString(54, 28, "SAE features and activation strength")
        canvas.drawRightString(558, 28, str(doc.page))

    doc = SimpleDocTemplate(
        str(destination),
        pagesize=(612, 792),
        leftMargin=54,
        rightMargin=54,
        topMargin=42,
        bottomMargin=48,
        title=source.read_text(encoding="utf-8").splitlines()[0].removeprefix("# "),
        author="Research team",
    )
    doc.build(story, onFirstPage=page, onLaterPages=page)
    print(destination)
    return destination


if __name__ == "__main__":
    render(sys.argv[1])
