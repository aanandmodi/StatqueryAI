from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


@dataclass(frozen=True)
class ChangeOutput:
    answer_logits: Tensor
    mask_logits: Tensor


class ChangeExpert(nn.Module):
    """Shared visual encoder with answer and genuine change-mask heads.

    The language head is intentionally small because CDVQA uses a closed answer
    vocabulary. A downstream synthesizer verbalizes these structured predictions.
    """

    def __init__(
        self,
        *,
        vocabulary_size: int,
        answer_classes: int,
        padding_index: int = 0,
        question_dim: int = 256,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        encoder = resnet18(weights=weights)
        self.stem = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
            encoder.maxpool,
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4,
        )
        channels = 512
        fused_channels = channels * 4
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_channels, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.question_embedding = nn.Embedding(
            vocabulary_size, question_dim, padding_idx=padding_index
        )
        self.question_encoder = nn.GRU(
            input_size=question_dim,
            hidden_size=question_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.answer_head = nn.Sequential(
            nn.Linear(256 + 2 * question_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, answer_classes),
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def forward(
        self, time_a: Tensor, time_b: Tensor, question_tokens: Tensor
    ) -> ChangeOutput:
        if time_a.shape != time_b.shape:
            raise ValueError("time_a and time_b must have identical shapes")
        feature_a = self.stem(time_a)
        feature_b = self.stem(time_b)
        fused = self.fuse(
            torch.cat(
                [
                    feature_a,
                    feature_b,
                    torch.abs(feature_a - feature_b),
                    feature_a * feature_b,
                ],
                dim=1,
            )
        )
        visual = F.adaptive_avg_pool2d(fused, output_size=1).flatten(1)
        question = self.question_embedding(question_tokens)
        _, hidden = self.question_encoder(question)
        question_vector = torch.cat([hidden[-2], hidden[-1]], dim=1)
        answer_logits = self.answer_head(torch.cat([visual, question_vector], dim=1))
        mask_logits = F.interpolate(
            self.mask_head(fused),
            size=time_a.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return ChangeOutput(answer_logits=answer_logits, mask_logits=mask_logits)


def change_loss(
    output: ChangeOutput,
    *,
    answer_targets: Tensor,
    mask_targets: Tensor,
    mask_weight: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    answer = F.cross_entropy(output.answer_logits, answer_targets)
    binary = F.binary_cross_entropy_with_logits(
        output.mask_logits, mask_targets.float()
    )
    probabilities = torch.sigmoid(output.mask_logits)
    intersection = (probabilities * mask_targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + mask_targets.sum(dim=(1, 2, 3))
    dice = 1 - ((2 * intersection + 1) / (denominator + 1)).mean()
    mask = 0.5 * binary + 0.5 * dice
    total = answer + mask_weight * mask
    return total, {"answer_loss": answer.detach(), "mask_loss": mask.detach()}
