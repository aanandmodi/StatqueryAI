from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def normalize_answer(value: str) -> str:
    return " ".join(
        "".join(
            character.lower() if character.isalnum() else " " for character in value
        ).split()
    )


def exact_match(predictions: Sequence[str], references: Sequence[str]) -> float:
    if len(predictions) != len(references) or not predictions:
        raise ValueError(
            "predictions and references must be non-empty and equally sized"
        )
    return float(
        np.mean(
            [
                normalize_answer(prediction) == normalize_answer(reference)
                for prediction, reference in zip(predictions, references, strict=True)
            ]
        )
    )


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    if len(box_a) != 4 or len(box_b) != 4:
        raise ValueError("boxes require four coordinates")
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def grounding_metrics(
    predictions: Sequence[Sequence[float]], references: Sequence[Sequence[float]]
) -> dict[str, float]:
    if len(predictions) != len(references) or not predictions:
        raise ValueError("box collections must be non-empty and equally sized")
    ious = np.array(
        [box_iou(a, b) for a, b in zip(predictions, references, strict=True)]
    )
    return {
        "mean_iou": float(ious.mean()),
        "acc_at_0_5": float((ious >= 0.5).mean()),
        "acc_at_0_7": float((ious >= 0.7).mean()),
    }


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], *, bins: int = 15
) -> float:
    confidence = np.asarray(confidences, dtype=float)
    labels = np.asarray(correct, dtype=float)
    if confidence.size == 0 or confidence.size != labels.size:
        raise ValueError(
            "confidence and correctness arrays must be equally sized and non-empty"
        )
    if np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("confidence values must be in 0..1")
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (confidence > lower) & (confidence <= upper)
        if lower == 0:
            selected |= confidence == 0
        if selected.any():
            result += selected.mean() * abs(
                labels[selected].mean() - confidence[selected].mean()
            )
    return float(result)


@dataclass
class TemperatureScaler:
    temperature: float = 1.0

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        *,
        steps: int = 500,
        lr: float = 0.01,
    ) -> float:
        """Fit scalar temperature on validation data only using bounded gradient descent."""
        if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
            raise ValueError("logits must be [N,C] and labels [N]")
        log_temperature = 0.0
        for _ in range(steps):
            temperature = float(np.exp(log_temperature))
            scaled = logits / temperature
            shifted = scaled - scaled.max(axis=1, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            one_hot = np.eye(logits.shape[1])[labels]
            gradient_t = np.mean(
                np.sum((probabilities - one_hot) * (-logits / temperature**2), axis=1)
            )
            gradient_log_t = gradient_t * temperature
            log_temperature -= lr * float(np.clip(gradient_log_t, -10, 10))
            log_temperature = float(
                np.clip(log_temperature, np.log(0.05), np.log(20.0))
            )
        self.temperature = float(np.exp(log_temperature))
        return self.temperature

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return logits / self.temperature
