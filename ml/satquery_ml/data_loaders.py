from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", re.I)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            records.append(value)
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    return records


def build_question_vocabulary(
    questions: list[str], *, min_frequency: int = 2
) -> dict[str, int]:
    from collections import Counter

    counts = Counter(
        token.lower() for question in questions for token in TOKEN_PATTERN.findall(question)
    )
    vocabulary = {"<pad>": 0, "<unk>": 1}
    for token in sorted(token for token, count in counts.items() if count >= min_frequency):
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_question(question: str, vocabulary: dict[str, int], *, max_length: int = 128) -> list[int]:
    unknown = vocabulary.get("<unk>", 1)
    tokens = [
        vocabulary.get(token.lower(), unknown) for token in TOKEN_PATTERN.findall(question)
    ][:max_length]
    return tokens or [unknown]


class ChangeManifestDataset:
    """CDVQA/SECOND adapter over a leakage-audited JSONL pair manifest.

    Each row requires `time_a`, `time_b`, `question`, `answer_index`, and `mask`.
    Paths may be absolute or relative to the manifest's directory.
    """

    def __init__(
        self,
        manifest: Path,
        vocabulary: dict[str, int],
        *,
        image_size: int = 448,
    ) -> None:
        self.root = manifest.parent
        self.records = load_jsonl(manifest)
        self.vocabulary = vocabulary
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        record = self.records[index]
        time_a = self._image(record["time_a"])
        time_b = self._image(record["time_b"])
        mask_path = self._path(record["mask"])
        mask = Image.open(mask_path).convert("L").resize(
            (self.image_size, self.image_size), Image.Resampling.NEAREST
        )
        return {
            "time_a": torch.from_numpy(time_a),
            "time_b": torch.from_numpy(time_b),
            "question_tokens": torch.tensor(
                encode_question(str(record["question"]), self.vocabulary), dtype=torch.long
            ),
            "answer": torch.tensor(int(record["answer_index"]), dtype=torch.long),
            "mask": torch.from_numpy((np.asarray(mask) > 0).astype(np.float32))[None],
        }

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _image(self, value: str) -> np.ndarray:
        image = Image.open(self._path(value)).convert("RGB").resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        return np.moveaxis(np.asarray(image, dtype=np.float32) / 255.0, -1, 0)


def pad_change_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    import torch
    from torch.nn.utils.rnn import pad_sequence

    return {
        "time_a": torch.stack([item["time_a"] for item in items]),
        "time_b": torch.stack([item["time_b"] for item in items]),
        "question_tokens": pad_sequence(
            [item["question_tokens"] for item in items], batch_first=True, padding_value=0
        ),
        "answer": torch.stack([item["answer"] for item in items]),
        "mask": torch.stack([item["mask"] for item in items]),
    }


class FusionManifestDataset:
    """BigEarthNet v2 adapter over pre-aligned NumPy patches.

    Each row requires `.npy` arrays `s1` [2,H,W], `s2` [12,H,W], a multi-hot
    `labels` list, and optionally an integer `mask` array. Dataset-level mean/std
    must be computed from the training split only and supplied explicitly.
    """

    def __init__(
        self,
        manifest: Path,
        *,
        s1_mean: list[float],
        s1_std: list[float],
        s2_mean: list[float],
        s2_std: list[float],
    ) -> None:
        self.root = manifest.parent
        self.records = load_jsonl(manifest)
        self.s1_mean = np.asarray(s1_mean, dtype=np.float32)[:, None, None]
        self.s1_std = np.asarray(s1_std, dtype=np.float32)[:, None, None]
        self.s2_mean = np.asarray(s2_mean, dtype=np.float32)[:, None, None]
        self.s2_std = np.asarray(s2_std, dtype=np.float32)[:, None, None]
        if self.s1_mean.shape[0] != 2 or self.s2_mean.shape[0] != 12:
            raise ValueError("fusion normalization must contain 2 S1 and 12 S2 channels")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        record = self.records[index]
        s1 = np.load(self._path(record["s1"]), allow_pickle=False).astype(np.float32)
        s2 = np.load(self._path(record["s2"]), allow_pickle=False).astype(np.float32)
        if s1.shape[0] != 2 or s2.shape[0] != 12 or s1.shape[-2:] != s2.shape[-2:]:
            raise ValueError("fusion pair must be aligned [2,H,W] S1 and [12,H,W] S2")
        item = {
            "s1": torch.from_numpy((s1 - self.s1_mean) / self.s1_std),
            "s2": torch.from_numpy((s2 - self.s2_mean) / self.s2_std),
            "labels": torch.tensor(record["labels"], dtype=torch.float32),
        }
        if record.get("mask"):
            item["mask"] = torch.from_numpy(
                np.load(self._path(record["mask"]), allow_pickle=False).astype(np.int64)
            )
        return item

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path
