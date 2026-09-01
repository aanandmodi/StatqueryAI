from __future__ import annotations

from typing import Any


SUPPORTED_TASKS = frozenset({"single_vqa", "caption", "grounding"})
MAX_QUERY_CHARS = 2_000
MAX_NEW_TOKENS = 256


def validate_request(task: str, question: str, max_new_tokens: int | float | str) -> tuple[str, str, int]:
    clean_task = str(task).strip().lower()
    if clean_task not in SUPPORTED_TASKS:
        choices = ", ".join(sorted(SUPPORTED_TASKS))
        raise ValueError(f"Unsupported task '{clean_task}'. Choose one of: {choices}.")

    clean_question = str(question).strip()
    if clean_task == "caption" and not clean_question:
        clean_question = "Describe this satellite image accurately and concisely."
    if not clean_question:
        raise ValueError("question cannot be empty")
    if len(clean_question) > MAX_QUERY_CHARS:
        raise ValueError(f"question exceeds the {MAX_QUERY_CHARS}-character limit")

    try:
        token_limit = int(max_new_tokens)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_new_tokens must be an integer") from exc
    token_limit = max(1, min(token_limit, MAX_NEW_TOKENS))
    return clean_task, clean_question, token_limit


def build_task_prompt(task: str, question: str) -> str:
    if task == "caption":
        return (
            f"{question}\nReturn only a factual remote-sensing scene description. "
            "Do not invent locations, dates, sensors, or confidence values."
        )
    if task == "grounding":
        return (
            f"{question}\nReturn a short answer and every visible target box using coordinates "
            "on a 0..1000 scale in the form <box>(x1,y1),(x2,y2)</box>."
        )
    return (
        f"{question}\nAnswer from visible image evidence only. If the image does not support "
        "the answer, say that the evidence is insufficient."
    )


def make_response(
    *,
    task: str,
    answer: str,
    evidence: list[dict[str, Any]],
    model_version: str,
) -> dict[str, Any]:
    warnings = [
        "The model does not produce a calibrated probability; raw_score is a neutral placeholder.",
        "Outputs were trained on a bounded European Sentinel imagery subset and require validation.",
    ]
    if task == "grounding" and not evidence:
        warnings.append("No machine-readable grounding box could be parsed from the generated text.")
    return {
        "task": task,
        "text": answer,
        "facts": [
            {"name": "execution_mode", "value": "huggingface_zerogpu"},
            {"name": "confidence_semantics", "value": "uncalibrated"},
        ],
        "evidence": evidence,
        "raw_score": 0.5,
        "score_kind": "uncalibrated",
        "model_version": model_version,
        "warnings": warnings,
    }

