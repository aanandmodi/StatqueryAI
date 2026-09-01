from __future__ import annotations

import math
from dataclasses import dataclass

from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FusionOutput:
    class_logits: Tensor
    mask_logits: Tensor
    patch_features: Tensor


class TerraMindFusionExpert(nn.Module):
    """TerraMind S1/S2 backbone with classification and spatial-evidence heads."""

    def __init__(
        self,
        *,
        num_classes: int,
        backbone_name: str = "terramind_v1_base",
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        try:
            from terratorch.registry import BACKBONE_REGISTRY
        except ImportError as exc:
            raise RuntimeError('Install "terratorch>=1.2.5" to use TerraMind') from exc
        self.backbone = BACKBONE_REGISTRY.build(
            backbone_name,
            pretrained=pretrained,
            modalities=["S2L2A", "S1GRD"],
            merge_method="mean",
        )
        embedding_dim = 768 if backbone_name != "terramind_v1_large" else 1024
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes),
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(embedding_dim, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, s2_l2a: Tensor, s1_grd: Tensor) -> FusionOutput:
        if s2_l2a.shape[1] != 12:
            raise ValueError(
                "TerraMind S2L2A input must contain 12 bands in official order"
            )
        if s1_grd.shape[1] != 2:
            raise ValueError("TerraMind S1GRD input must contain VV and VH channels")
        features = self.backbone({"S2L2A": s2_l2a, "S1GRD": s1_grd})[-1]
        pooled = features.mean(dim=1)
        class_logits = self.classifier(pooled)
        side = math.isqrt(features.shape[1])
        if side * side != features.shape[1]:
            raise ValueError("TerraMind patch tokens do not form a square grid")
        grid = features.transpose(1, 2).reshape(
            features.shape[0], features.shape[2], side, side
        )
        mask_logits = F.interpolate(
            self.mask_head(grid),
            size=s2_l2a.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return FusionOutput(
            class_logits=class_logits,
            mask_logits=mask_logits,
            patch_features=features,
        )


def fusion_loss(
    output: FusionOutput,
    *,
    multi_hot_targets: Tensor,
    mask_targets: Tensor | None,
    class_weights: Tensor | None = None,
    mask_weight: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    classification = F.binary_cross_entropy_with_logits(
        output.class_logits,
        multi_hot_targets.float(),
        pos_weight=class_weights,
    )
    if mask_targets is None:
        return classification, {"classification_loss": classification.detach()}
    segmentation = F.cross_entropy(
        output.mask_logits, mask_targets.long(), ignore_index=-1
    )
    total = classification + mask_weight * segmentation
    return total, {
        "classification_loss": classification.detach(),
        "segmentation_loss": segmentation.detach(),
    }
