from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def train_change_epoch(
    model: Any,
    batches: Iterable[dict[str, Any]],
    optimizer: Any,
    *,
    device: Any,
    gradient_clip: float = 1.0,
) -> dict[str, float]:
    import torch

    from satquery_ml.models.change import change_loss

    model.train()
    totals = {"loss": 0.0, "answer_loss": 0.0, "mask_loss": 0.0}
    count = 0
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch["time_a"].to(device),
            batch["time_b"].to(device),
            batch["question_tokens"].to(device),
        )
        loss, parts = change_loss(
            output,
            answer_targets=batch["answer"].to(device),
            mask_targets=batch["mask"].to(device),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        totals["loss"] += float(loss.detach())
        totals["answer_loss"] += float(parts["answer_loss"])
        totals["mask_loss"] += float(parts["mask_loss"])
        count += 1
    if count == 0:
        raise ValueError("training loader produced no batches")
    return {key: value / count for key, value in totals.items()}


def train_fusion_epoch(
    model: Any,
    batches: Iterable[dict[str, Any]],
    optimizer: Any,
    *,
    device: Any,
    class_weights: Any | None = None,
) -> dict[str, float]:
    import torch

    from satquery_ml.models.fusion import fusion_loss

    model.train()
    totals = {"loss": 0.0, "classification_loss": 0.0, "segmentation_loss": 0.0}
    count = 0
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["s2"].to(device), batch["s1"].to(device))
        loss, parts = fusion_loss(
            output,
            multi_hot_targets=batch["labels"].to(device),
            mask_targets=batch.get("mask").to(device)
            if batch.get("mask") is not None
            else None,
            class_weights=class_weights,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        totals["loss"] += float(loss.detach())
        totals["classification_loss"] += float(parts["classification_loss"])
        totals["segmentation_loss"] += float(parts.get("segmentation_loss", 0.0))
        count += 1
    if count == 0:
        raise ValueError("training loader produced no batches")
    return {key: value / count for key, value in totals.items()}
