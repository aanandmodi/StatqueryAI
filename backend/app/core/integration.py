from __future__ import annotations

from statistics import fmean

from app.schemas import AssetRecord, Confidence, SpecialistOutput


def integrate_outputs(
    outputs: list[SpecialistOutput], assets: list[AssetRecord], *, simulated: bool
) -> tuple[str, list[dict], list, Confidence, list[str]]:
    if not outputs:
        raise ValueError("At least one specialist output is required")

    answer = "\n\n".join(output.text for output in outputs)
    facts = [fact for output in outputs for fact in output.facts]
    evidence = [item for output in outputs for item in output.evidence]
    warnings = list(dict.fromkeys(warning for output in outputs for warning in output.warnings))

    raw_score = fmean(output.raw_score for output in outputs)
    input_quality = fmean(
        asset.metadata.quality_score for asset in assets if asset.metadata is not None
    )
    evidence_coverage = min(1.0, len(evidence) / max(1, len(outputs)))
    branch_agreement = (
        1.0
        if len(outputs) == 1
        else max(0.0, 1.0 - (max(o.raw_score for o in outputs) - min(o.raw_score for o in outputs)))
    )
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
        calibration_version = "specialist-calibration-v1"
        meaning = (
            "Calibrated correctness estimate conditioned on the task and validation distribution."
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
            "branch_agreement": branch_agreement,
        },
        meaning=meaning,
    )
    return answer, facts, evidence, confidence, warnings
