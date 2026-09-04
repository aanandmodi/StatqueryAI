"""Conservative display guard, not a factuality classifier or calibrated verifier."""

from __future__ import annotations

import re

from app.schemas import SpecialistOutput, TaskType

QUANTIFIED_CLAIM = re.compile(
    r"\d[^.!?\n]{0,35}(?:%|\bpercent\b|\bper cent\b|\bhectares?\b|km²|m²|km2|m2|"
    r"\bsquare (?:metres?|meters?|kilometers?|kilometres?)\b|"
    r"\b(?:buildings?|lakes?|rivers?|roads?|trees?|ponds?)\b)",
    re.I,
)
UNSUPPORTED_PHYSICAL_CLAIM = re.compile(
    r"\b(?:moving|flowing)\b[^.!?]{0,70}\b(?:speed|velocity)\b|"
    r"\b(?:suggests?|suggesting|indicates?|indicating)\b[^.!?]{0,70}\bsoil types?\b",
    re.I,
)


def audit_visual_narrative(output: SpecialistOutput) -> SpecialistOutput:
    # Paired analytical specialists already report directly computed quantities.
    if output.task not in {TaskType.CAPTION, TaskType.SINGLE_VQA, TaskType.GROUNDING}:
        return output
    if output.model_version.startswith("demo-simulator"):
        return output
    original = output.text
    sentences = re.split(r"(?<=[.!?])\s+|\n\s*\n", original.strip())
    retained = [
        sentence
        for sentence in sentences
        if not QUANTIFIED_CLAIM.search(sentence) and not UNSUPPORTED_PHYSICAL_CLAIM.search(sentence)
    ]
    warnings = list(output.warnings)
    changed = len(retained) != len(sentences)
    if changed:
        warnings.append(
            "Unmeasured numeric or unsupported physical claims were removed from the "
            "displayed model narrative. "
            "Use the computed mask-measurement section for coverage/area. "
            "Original text is retained "
            "as unverified audit data; this check is not a complete hallucination detector."
        )
    # Long VLM generations can repeat the same sentence under several report headings.
    # Keep the first occurrence so the evidence report remains readable without
    # paraphrasing or manufacturing any new claim.
    unique_retained = []
    seen_sentences = set()
    for sentence in retained:
        key = re.sub(r"\s+", " ", sentence).strip().casefold()
        if key and key in seen_sentences:
            changed = True
            continue
        if key:
            seen_sentences.add(key)
        unique_retained.append(sentence)
    if len(unique_retained) != len(retained):
        warnings.append(
            "Repeated model sentences were collapsed in the displayed narrative; "
            "the original response remains available as unverified audit data."
        )
    retained = unique_retained
    # A long response ending without punctuation commonly hits the generation budget.
    if len(original.split()) >= 70 and original.rstrip()[-1:] not in {".", "!", "?"}:
        warnings.append(
            "The model response appears incomplete; it may have reached its token budget."
        )
        if len(retained) > 1 and retained[-1] == sentences[-1]:
            retained.pop()
            changed = True
    text = "\n\n".join(retained).strip()
    if not text:
        text = (
            "The model supplied no usable qualitative answer after unmeasured numeric "
            "claims were excluded. Inspect the scene and any separately computed mask measurements."
        )
    facts = list(output.facts)
    if changed:
        facts.append(
            {
                "name": "unverified_original_model_text",
                "value": original,
                "display_policy": "unmeasured quantities excluded; not ground truth",
            }
        )
    return output.model_copy(update={"text": text, "warnings": warnings, "facts": facts})
