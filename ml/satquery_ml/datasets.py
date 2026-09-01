from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def convert_box(
    box: Sequence[float],
    *,
    source_max: float = 100.0,
    destination_max: int = 1000,
) -> list[int]:
    """Convert `[x1, y1, x2, y2]` between normalized coordinate conventions.

    VRSBench coordinates must be handled through an explicit adapter because the
    upstream convention has changed. The caller must verify `source_max` against
    the pinned dataset revision; this function never guesses from the values.
    """

    if len(box) != 4:
        raise ValueError("box must contain x1, y1, x2, y2")
    if source_max <= 0 or destination_max <= 0:
        raise ValueError("coordinate ranges must be positive")
    x1, y1, x2, y2 = [float(value) for value in box]
    if not (0 <= x1 <= x2 <= source_max and 0 <= y1 <= y2 <= source_max):
        raise ValueError(f"box must be ordered and bounded within 0..{source_max}")
    scale = destination_max / source_max
    return [round(value * scale) for value in (x1, y1, x2, y2)]


def qwen_grounding_answer(label: str, box_0_1000: Sequence[int]) -> str:
    if len(box_0_1000) != 4:
        raise ValueError("grounding answer requires four coordinates")
    return json.dumps(
        {"label": label, "bbox_2d": [int(value) for value in box_0_1000]},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def build_qwen_example(
    *,
    image_path: str,
    prompt: str,
    answer: str,
    task: str,
    sample_id: str,
) -> dict[str, Any]:
    if task not in {"vqa", "caption", "grounding"}:
        raise ValueError(f"unsupported VLM task: {task}")
    return {
        "id": sample_id,
        "image": image_path,
        "task": task,
        "conversations": [
            {"from": "human", "value": f"<image>\n[{task.upper()}] {prompt.strip()}"},
            {"from": "gpt", "value": answer.strip()},
        ],
    }


def balance_task_examples(
    examples: Mapping[str, Sequence[dict[str, Any]]],
    mix: Mapping[str, float],
    total: int,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    if total <= 0:
        raise ValueError("total must be positive")
    if abs(sum(mix.values()) - 1.0) > 1e-6:
        raise ValueError("task weights must sum to 1")
    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for task, weight in mix.items():
        pool = list(examples.get(task, []))
        if not pool:
            raise ValueError(f"no examples available for task {task}")
        count = round(total * weight)
        if count <= len(pool):
            sampled.extend(rng.sample(pool, count))
        else:
            sampled.extend(rng.choice(pool) for _ in range(count))
    rng.shuffle(sampled)
    return sampled[:total]


def assert_image_disjoint_splits(
    splits: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    image_key: str = "image",
) -> None:
    owners: dict[str, str] = {}
    leaks: list[tuple[str, str, str]] = []
    for split, records in splits.items():
        for record in records:
            image = str(record[image_key])
            previous = owners.setdefault(image, split)
            if previous != split:
                leaks.append((image, previous, split))
    if leaks:
        preview = leaks[:5]
        raise ValueError(f"image leakage detected across splits: {preview}")


def stream_vrsbench(*, split: str = "train", revision: str | None = None) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the notebook training requirements first") from exc
    return load_dataset(
        "xiang709/VRSBench",
        name="VRSBench",
        split=split,
        streaming=True,
        revision=revision,
    )


def write_jsonl(
    records: Iterable[Mapping[str, Any]], destination: Path
) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            line = json.dumps(
                record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            encoded = f"{line}\n".encode()
            handle.write(encoded.decode())
            digest.update(encoded)
            count += 1
    return count, digest.hexdigest()


def summarize_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    tasks: Counter[str] = Counter()
    images: set[str] = set()
    count = 0
    for record in records:
        count += 1
        tasks[str(record.get("task", "unknown"))] += 1
        images.add(str(record.get("image", "")))
    return {"records": count, "unique_images": len(images), "tasks": dict(tasks)}
