from __future__ import annotations

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
        )
    )
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
        Paragraph(_escape(result.answer), styles["SatBody"]),
        Paragraph("Confidence", styles["SatHeading"]),
        Paragraph(
            f"{result.confidence.score:.1%} ({result.confidence.level}) — "
            f"{_escape(result.confidence.meaning)}",
            styles["SatBody"],
        ),
        Paragraph("Evidence", styles["SatHeading"]),
    ]
    evidence_rows = [["Label", "Type", "Score", "Asset"]]
    evidence_rows.extend(
        [item.label, item.type, f"{item.score:.1%}", item.asset_id] for item in result.evidence
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
    doc.build(story)
    return destination


def _table(rows: list[list[str]], *, small: bool = False) -> Table:
    table = Table(rows, repeatRows=1, hAlign="LEFT")
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
