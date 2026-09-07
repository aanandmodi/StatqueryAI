from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.schemas import AnalysisRecord


def build_pdf_report(record: AnalysisRecord, destination: Path) -> Path:
    if record.result is None:
        raise ValueError("Cannot report an analysis without a result")
    destination.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "SatTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=28,
            textColor=colors.HexColor("#10213B"),
            spaceAfter=8 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            "SatBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#33445D"),
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            "SatHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            textColor=colors.HexColor("#243B67"),
            spaceBefore=5 * mm,
            spaceAfter=2 * mm,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            "SatSubHeading",
            parent=styles["SatHeading"],
            fontSize=10.5,
            textColor=colors.HexColor("#2B4C7E"),
            spaceBefore=3 * mm,
            spaceAfter=1.5 * mm,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            "SatBullet",
            parent=styles["SatBody"],
            leftIndent=12,
            firstLineIndent=-8,
            spaceBefore=0.8 * mm,
            spaceAfter=0.8 * mm,
        )
    )
    styles.add(ParagraphStyle("SatSource", parent=styles["SatBody"], keepWithNext=True))
    doc = SimpleDocTemplate(
        str(destination),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"SatQuery analysis {record.id}",
        author="SatQuery",
    )
    result = record.result
    uncalibrated = "uncalibrated" in result.confidence.calibration_version
    confidence_text = (
        f"Evidence quality: {result.confidence.level}; correctness probability is unavailable. "
        if uncalibrated
        else f"{result.confidence.score:.1%} ({result.confidence.level}). "
    ) + result.confidence.meaning
    story = [
        Paragraph("SatQuery analysis report", styles["SatTitle"]),
        Paragraph(f"Analysis ID: {record.id}", styles["SatBody"]),
        Paragraph(f"Created: {record.created_at.isoformat()}", styles["SatBody"]),
        Paragraph("Query", styles["SatHeading"]),
        Paragraph(_escape(record.request.query), styles["SatBody"]),
        *(
            [
                Paragraph("User-supplied location context", styles["SatHeading"]),
                Paragraph(
                    _escape(
                        f"Latitude {record.request.context.latitude}, longitude "
                        f"{record.request.context.longitude}, altitude "
                        + (
                            f"{record.request.context.altitude_m} m"
                            if record.request.context.altitude_m is not None
                            else "not supplied"
                        )
                    ),
                    styles["SatBody"],
                ),
            ]
            if record.request.context
            else []
        ),
        Paragraph("Answer", styles["SatHeading"]),
        *_render_markdown_flowables(result.answer, styles),
        *[
            flowable
            for section in result.sections
            for flowable in [
                Paragraph(_escape(section.title), styles["SatHeading"]),
                Paragraph(_escape(f"Source: {section.source}"), styles["SatSource"]),
                *[
                    sub_flowable
                    for paragraph in section.paragraphs
                    for sub_flowable in _render_markdown_flowables(paragraph, styles)
                ],
            ]
        ],
        Paragraph("Confidence", styles["SatHeading"]),
        Paragraph(_escape(confidence_text), styles["SatBody"]),
        Paragraph("Evidence", styles["SatHeading"]),
    ]
    evidence_rows = [["Label", "Type", "Evidence status", "Asset"]]
    evidence_rows.extend(
        [item.label, item.type, "Candidate; review needed", item.asset_id]
        for item in result.evidence
    )
    story.append(_table(evidence_rows))
    story.extend([Spacer(1, 4 * mm), Paragraph("Execution trace", styles["SatHeading"])])
    trace_rows = [["Step", "Task", "Tool / model", "Status", "Duration"]]
    trace_rows.extend(
        [
            item.step_id,
            str(item.task),
            f"{item.tool} / {item.model_version or 'n/a'}",
            item.status,
            f"{item.duration_ms or 0} ms",
        ]
        for item in result.trace
    )
    story.append(_table(trace_rows, small=True))
    if result.warnings:
        story.extend([Paragraph("Warnings and limitations", styles["SatHeading"])])
        story.extend(
            Paragraph(f"• {_escape(warning)}", styles["SatBody"]) for warning in result.warnings
        )

    def page_footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#66768A"))
        canvas.drawString(18 * mm, 9 * mm, "SatQuery | Candidate evidence, not ground truth")
        canvas.drawRightString(192 * mm, 9 * mm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    return destination


def _table(rows: list[list[str]], *, small: bool = False) -> Table:
    cell_style = ParagraphStyle(
        "Cell", fontName="Helvetica", fontSize=6.8 if small else 8, leading=10, wordWrap="CJK"
    )
    header_style = ParagraphStyle(
        "HeaderCell", parent=cell_style, fontName="Helvetica-Bold", textColor=colors.white
    )
    cells = [
        [
            Paragraph(_escape(str(value)), header_style if index == 0 else cell_style)
            for value in row
        ]
        for index, row in enumerate(rows)
    ]
    table = Table(
        cells, colWidths=[174 * mm / len(rows[0])] * len(rows[0]), repeatRows=1, hAlign="LEFT"
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#10213B")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 6.8 if small else 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6FA")]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CDD6E3")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
    )


def _format_inline_reportlab(text: str) -> str:
    # 1. Escape HTML special characters
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # 2. Convert bold: **text** -> <b>text</b>
    safe = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", safe)
    # 3. Convert italic: *text* -> <i>\1</i>
    safe = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", safe)
    # 4. Convert code: `code` -> <font name="Courier">\1</font>
    safe = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', safe)
    # 5. Status badges: [Verified], [Detected], etc.
    safe = re.sub(
        r"\[(Verified|Confirmed|Passed|Pass|Present)\]",
        r'<b>[<font color="#059669">\1</font>]</b>',
        safe,
        flags=re.I,
    )
    safe = re.sub(
        r"\[(High|Detected)\]",
        r'<b>[<font color="#0284C7">\1</font>]</b>',
        safe,
        flags=re.I,
    )
    safe = re.sub(
        r"\[(Moderate|Warning)\]",
        r'<b>[<font color="#D97706">\1</font>]</b>',
        safe,
        flags=re.I,
    )
    safe = re.sub(
        r"\[(Low|Critical|Fail|Absent)\]",
        r'<b>[<font color="#DC2626">\1</font>]</b>',
        safe,
        flags=re.I,
    )
    # Clean any residual unformatted asterisks
    safe = re.sub(r"\*", "", safe)
    return safe


def _render_markdown_flowables(text: str, styles) -> list:
    flowables = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # 1. Table
        if line.startswith("|") and line.endswith("|") and line.count("|") >= 2:
            table_lines = []
            while (
                i < len(lines)
                and lines[i].strip().startswith("|")
                and lines[i].strip().endswith("|")
            ):
                table_lines.append(lines[i].strip())
                i += 1
            if len(table_lines) >= 2:
                rows = []
                for idx, t_line in enumerate(table_lines):
                    if idx == 1 and re.match(r"^\|(?:\s*:?-+:?\s*\|)+$", t_line):
                        continue
                    cols = [col.strip() for col in t_line.split("|")[1:-1]]
                    rows.append(cols)
                if rows:
                    flowables.append(_table(rows, small=True))
                    flowables.append(Spacer(1, 2 * mm))
                    continue

        # 2. Heading: #, ##, ###
        if line.startswith("###"):
            h_text = line.lstrip("#").strip()
            flowables.append(Paragraph(_format_inline_reportlab(h_text), styles["SatSubHeading"]))
            i += 1
            continue
        if line.startswith("#"):
            h_text = line.lstrip("#").strip()
            flowables.append(Paragraph(_format_inline_reportlab(h_text), styles["SatHeading"]))
            i += 1
            continue

        # 3. Bullet list
        bullet_match = re.match(r"^[-*•+]\s+(.+)$", line)
        if bullet_match:
            b_text = bullet_match.group(1).strip()
            flowables.append(
                Paragraph(f"&bull;&nbsp; {_format_inline_reportlab(b_text)}", styles["SatBullet"])
            )
            i += 1
            continue

        numbered_match = re.match(r"^(\d+)\.\s+(.+)$", line)
        if numbered_match:
            num = numbered_match.group(1)
            b_text = numbered_match.group(2).strip()
            flowables.append(
                Paragraph(
                    f"<b>{num}.</b>&nbsp; {_format_inline_reportlab(b_text)}", styles["SatBullet"]
                )
            )
            i += 1
            continue

        # 4. Key-Value or Bold Label
        kv_match = re.match(
            r"^(?:(?:\*\*([^*:]+)\*\*)|([A-Z][A-Za-z0-9\s/_-]{1,25})):\s*(.+)$", line
        )
        if kv_match:
            key = (kv_match.group(1) or kv_match.group(2)).strip()
            val = kv_match.group(3).strip()
            flowables.append(
                Paragraph(f"<b>{key}:</b> {_format_inline_reportlab(val)}", styles["SatBody"])
            )
            i += 1
            continue

        # 5. Normal paragraph
        flowables.append(Paragraph(_format_inline_reportlab(line), styles["SatBody"]))
        i += 1

    return flowables
