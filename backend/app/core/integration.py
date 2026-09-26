from __future__ import annotations

from statistics import fmean

from app.core.narrative import audit_visual_narrative
from app.schemas import AssetRecord, Confidence, SpecialistOutput


def integrate_outputs(
    outputs: list[SpecialistOutput], assets: list[AssetRecord], *, simulated: bool
) -> tuple[str, list[dict], list, Confidence, list[str]]:
    if not outputs:
        raise ValueError("At least one specialist output is required")
    outputs = [audit_visual_narrative(output) for output in outputs]

    answer = "\n\n".join(dict.fromkeys(output.text for output in outputs))
    facts = [fact for output in outputs for fact in output.facts]
    evidence = [item for output in outputs for item in output.evidence]
    warnings = list(
        dict.fromkeys(
            [warning for output in outputs for warning in output.warnings]
            + [warning for asset in assets if asset.metadata for warning in asset.metadata.warnings]
        )
    )

    raw_score = fmean(output.raw_score for output in outputs)
    input_quality = fmean(
        asset.metadata.quality_score for asset in assets if asset.metadata is not None
    )
    evidence_coverage = min(1.0, len(evidence) / max(1, len(outputs)))
    # Similar softmax magnitudes do not establish agreement between different tasks.
    # This is only an auditable completeness rubric, never estimated correctness.
    branch_agreement = 0.0
    score = max(
        0.0,
        min(
            1.0,
            0.55 * raw_score
            + 0.20 * input_quality
            + 0.15 * evidence_coverage
            + 0.10 * branch_agreement,
        ),
    )
    has_uncalibrated = any(output.score_kind != "calibrated_probability" for output in outputs)
    if simulated:
        score = min(score, 0.59)
        calibration_version = "demo-uncalibrated"
        meaning = "Integration confidence for simulated plumbing; not a probability of correctness."
    elif has_uncalibrated:
        score = min(score, 0.59)
        calibration_version = "uncalibrated-evidence-quality"
        meaning = (
            "Evidence-quality score only; a calibrated probability of correctness is unavailable."
        )
    else:
        score = min(score, 0.59)
        calibration_version = "uncalibrated-aggregate"
        meaning = (
            "Specialists declared calibrated scores, but no calibration exists for this combined "
            "report. The hand-weighted evidence-quality rubric is not a correctness probability."
        )
    level = "high" if score >= 0.8 else "medium" if score >= 0.6 else "low"
    confidence = Confidence(
        score=score,
        level=level,
        calibration_version=calibration_version,
        factors={
            "specialist_score": raw_score,
            "input_quality": input_quality,
            "evidence_coverage": evidence_coverage,
            "cross_task_agreement_not_measured": branch_agreement,
        },
        meaning=meaning,
    )
    return answer, facts, evidence, confidence, warnings
