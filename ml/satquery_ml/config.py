from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ExperimentConfig:
    mode: Literal["demo", "evaluate", "train"] = "demo"
    data_mode: Literal["tiny", "subset", "full"] = "tiny"
    seed: int = 42
    run_vlm: bool = True
    run_change: bool = True
    run_fusion: bool = True
    offline: bool = False
    base_model: str = "Qwen/Qwen3-VL-4B-Instruct"
    base_model_revision: str | None = None
    output_dir: Path = Path("artifacts")
    data_dir: Path = Path("data")
    cache_dir: Path = Path("cache")
    max_samples: int | None = 256
    image_size: int = 448
    vlm_task_mix: dict[str, float] = field(
        default_factory=lambda: {"vqa": 0.50, "grounding": 0.30, "caption": 0.20}
    )

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.data_mode == "full" and self.max_samples is not None:
            raise ValueError("full data mode must not set max_samples")
        if not 128 <= self.image_size <= 2048:
            raise ValueError("image_size must be between 128 and 2048")
        if abs(sum(self.vlm_task_mix.values()) - 1.0) > 1e-6:
            raise ValueError("vlm_task_mix must sum to 1")
        if any(value <= 0 for value in self.vlm_task_mix.values()):
            raise ValueError("every enabled task must have a positive sampling weight")

    def prepare(self) -> None:
        self.validate()
        for path in (self.output_dir, self.data_dir, self.cache_dir):
            path.mkdir(parents=True, exist_ok=True)

    def save(self, destination: Path) -> None:
        payload = asdict(self)
        for key in ("output_dir", "data_dir", "cache_dir"):
            payload[key] = str(payload[key])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
