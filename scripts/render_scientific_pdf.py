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
)


def render(source, destination=None):
    source = Path(source)
    destination = Path(destination) if destination else source.with_suffix(".pdf")
    fonts = [Path("/usr/share/fonts/truetype/dejavu"), Path("C:/Windows/Fonts")]
    regular, bold = "Helvetica", "Helvetica-Bold"
    for path in fonts:
        for normal, strong in [
            ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
            ("arial.ttf", "arialbd.ttf"),
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
            fontSize=9.2,
            leading=13.4,
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
    styles.add(ParagraphStyle(name="Cell", fontName=regular, fontSize=6.5, leading=8.5))
    for key, size in [("Title", 24), ("Heading1", 14), ("Heading2", 11)]:
        styles[key].fontName = bold
        styles[key].fontSize = size
        styles[key].leading = size * 1.25
        styles[key].textColor = colors.HexColor("#174e63")
        styles[key].spaceBefore = 13
        styles[key].spaceAfter = 8
        styles[key].keepWithNext = True
    styles["PaperBody"].allowWidows = 0
    styles["PaperBody"].allowOrphans = 0

    def inline(s):
        s = escape(s)
        s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2" color="#176b87">\1</a>', s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        return re.sub(r"`([^`]+)`", r"\1", s)

    story, lines, i = [], source.read_text(encoding="utf-8").splitlines(), 0
    width = 504
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line.startswith("!["):
            match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            path = source.parent / match[2]
            if path.exists():
                w, h = ImageReader(str(path)).getSize()
                story.append(Image(str(path), width=width, height=width * h / w))
                story.append(Spacer(1, 9))
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
                story.append(KeepTogether([t, Spacer(1, 10)]))
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
            style = styles["PaperQuote"] if line.startswith("> ") else styles["PaperBody"]
            line = line.removeprefix("> ")
            if line.startswith("**"):
                style = ParagraphStyle(name="EvidenceLabel", parent=style, keepWithNext=True)
        story.append(Paragraph(inline(line), style))

    def page(canvas, doc):
        canvas.setFont(regular, 7)
        canvas.setFillColor(colors.HexColor("#647887"))
        canvas.drawString(
            54, 28, "Activation magnitude and SAE neighborhood structure | Research pilot"
        )
        canvas.drawRightString(558, 28, str(doc.page))

    doc = SimpleDocTemplate(
        str(destination),
        pagesize=(612, 792),
        leftMargin=54,
        rightMargin=54,
        topMargin=42,
        bottomMargin=48,
        title="Activation magnitude and SAE neighborhood structure",
        author="Research team",
    )
    doc.build(story, onFirstPage=page, onLaterPages=page)
    print(destination)
    return destination


if __name__ == "__main__":
    render(sys.argv[1])
