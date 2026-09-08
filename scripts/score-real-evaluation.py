from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "ml"))

from satquery_ml.evaluation import (  # noqa: E402
    TemperatureScaler,
    expected_calibration_error,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score raw base-vs-LoRA JSONL from the evaluation notebook without using GPU time."
        )
    )
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/real-evaluation"))
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("Prediction JSONL is empty")
    required = {
        "patch_id",
        "type",
        "base_pred",
        "lora_pred",
        "base_correct",
        "lora_correct",
        "base_sequence_confidence",
        "lora_sequence_confidence",
    }
    for index, row in enumerate(rows, start=1):
        if missing := required - set(row):
            raise ValueError(f"Row {index} is missing fields: {sorted(missing)}")
    return rows


def mean_field(rows: list[dict], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return float(np.mean(values)) if values else float("nan")


def metric_rows(rows: list[dict], task_types: set[str]) -> list[dict]:
    return [row for row in rows if row["type"] in task_types]


def bootstrap_scene_ci(
    rows: list[dict],
    metric: Callable[[list[dict]], float],
    *,
    repeats: int,
    seed: int,
) -> list[float]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["patch_id"])].append(row)
    scenes = sorted(grouped)
    if not scenes:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(repeats):
        sample = rng.choice(scenes, size=len(scenes), replace=True)
        sample_rows = [row for scene in sample for row in grouped[str(scene)]]
        scores.append(metric(sample_rows))
    return [float(value) for value in np.percentile(scores, [2.5, 97.5])]


def summarize(rows: list[dict], prefix: str, repeats: int, seed: int) -> dict:
    definitions = {
        "vqa_exact_match": (
            metric_rows(rows, {"binary", "mcq"}),
            f"{prefix}_correct",
        ),
        "grounding_mean_iou": (
            metric_rows(rows, {"bounding box"}),
            f"{prefix}_iou",
        ),
        "grounding_accuracy_iou_0_5": (
            metric_rows(rows, {"bounding box"}),
            f"{prefix}_correct",
        ),
        "caption_token_f1": (
            metric_rows(rows, {"captioning"}),
            f"{prefix}_caption_token_f1",
        ),
    }
    result = {}
    for name, (selected, field) in definitions.items():
        result[name] = {
            "value": mean_field(selected, field),
            "scene_bootstrap_95_ci": bootstrap_scene_ci(
                selected,
                lambda sample, field=field: mean_field(sample, field),
                repeats=repeats,
                seed=seed,
            ),
            "examples": len(selected),
            "scenes": len({row["patch_id"] for row in selected}),
        }
    return result


def calibration_partition(row: dict) -> str:
    digest = hashlib.sha256(str(row["patch_id"]).encode()).digest()
    return "fit" if digest[0] < 64 else "evaluate"


def calibrate(rows: list[dict], prefix: str) -> dict:
    vqa = metric_rows(rows, {"binary", "mcq"})
    fit = [row for row in vqa if calibration_partition(row) == "fit"]
    evaluate = [row for row in vqa if calibration_partition(row) == "evaluate"]
    if len(fit) < 10 or len(evaluate) < 20:
        return {
            "status": "insufficient_data",
            "fit_examples": len(fit),
            "evaluation_examples": len(evaluate),
        }

    def logits_for(items: list[dict]) -> np.ndarray:
        probability = np.clip(
            np.asarray([row[f"{prefix}_sequence_confidence"] for row in items], dtype=float),
            1e-6,
            1 - 1e-6,
        )
        return np.stack([np.log1p(-probability), np.log(probability)], axis=1)

    fit_labels = np.asarray([int(row[f"{prefix}_correct"]) for row in fit], dtype=int)
    evaluation_labels = np.asarray(
        [int(row[f"{prefix}_correct"]) for row in evaluate], dtype=int
    )
    scaler = TemperatureScaler()
    temperature = scaler.fit(logits_for(fit), fit_labels)
    raw_logits = logits_for(evaluate)
    scaled_logits = scaler.transform(raw_logits)

    def class_one_probability(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        probability = np.exp(shifted)
        probability /= probability.sum(axis=1, keepdims=True)
        return probability[:, 1]

    raw_probability = class_one_probability(raw_logits)
    scaled_probability = class_one_probability(scaled_logits)
    return {
        "status": "within-split scene-held-out diagnostic",
        "temperature": temperature,
        "fit_examples": len(fit),
        "evaluation_examples": len(evaluate),
        "raw_ece": expected_calibration_error(raw_probability, evaluation_labels),
        "temperature_scaled_ece": expected_calibration_error(
            scaled_probability, evaluation_labels
        ),
        "meaning": (
            "Post-hoc correctness calibration derived from real sequence scores. It becomes a "
            "release claim only after the same frozen temperature is evaluated on an untouched "
            "organizer-prescribed test split."
        ),
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_repeats < 200:
        raise ValueError("Use at least 200 scene-bootstrap repeats")
    rows = load_rows(args.predictions)
    args.output.mkdir(parents=True, exist_ok=True)
    base = summarize(rows, "base", args.bootstrap_repeats, args.seed)
    lora = summarize(rows, "lora", args.bootstrap_repeats, args.seed)
    deltas = {
        name: lora[name]["value"] - base[name]["value"]
        for name in base
    }
    scorecard = {
        "source_predictions": str(args.predictions.resolve()),
        "examples": len(rows),
        "base": base,
        "lora": lora,
        "lora_minus_base": deltas,
        "calibration": {
            "base": calibrate(rows, "base"),
            "lora": calibrate(rows, "lora"),
        },
        "scope": (
            "Real recorded inference only. No change/fusion accuracy is reported until their "
            "trained checkpoints and held-out labels exist."
        ),
    }
    target = args.output / "scorecard.json"
    target.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
    print(json.dumps(scorecard, indent=2))
    print(f"PASS: wrote {target.resolve()}")


if __name__ == "__main__":
    main()
