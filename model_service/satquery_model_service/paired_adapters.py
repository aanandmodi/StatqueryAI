"""Strict runtime for the v2 cloud-notebook artifacts. No random-weight fallback."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np

from satquery_model_service.contracts import Evidence, SpecialistResponse


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_config(root):
    manifest_path = root / "sha256_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("model.safetensors", "config.json"):
        if manifest.get(name) != sha256(root / name):
            raise ValueError(f"Artifact integrity check failed: {name}")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("artifact_version") != "satquery-pair-v2":
        raise ValueError(
            "Use the corrected v2 training notebook export; legacy architectures are incompatible"
        )
    return config, f"paired-v2:sha256:{sha256(root / 'model.safetensors')}"


def validate_pair(paths):
    import rasterio

    with rasterio.open(paths[0]) as a, rasterio.open(paths[1]) as b:
        if (a.width, a.height, a.crs) != (
            b.width,
            b.height,
            b.crs,
        ) or not a.transform.almost_equals(b.transform):
            raise ValueError(
                "Learned expert requires an identical co-registered source grid"
            )
        if a.width * a.height > 4_194_304:
            raise ValueError(
                "Tile this scene to at most 2048×2048 pixels before learned pair inference"
            )
        for source in (a, b):
            if any("complex" in value for value in source.dtypes):
                raise ValueError(
                    "Complex SAR cannot be converted to real intensity implicitly"
                )
        return a.height, a.width


def require_shared_support(valid):
    if not valid.any():
        raise ValueError("Paired rasters contain no shared valid pixels")


def resample_supported_mask(candidate, valid):
    from PIL import Image

    height, width = valid.shape
    scale = min(1, 1024 / max(height, width))
    shape = (max(1, round(width * scale)), max(1, round(height * scale)))
    candidate = np.asarray(
        Image.fromarray(candidate).resize(shape, Image.Resampling.NEAREST)
    )
    support = np.asarray(Image.fromarray(valid).resize(shape, Image.Resampling.NEAREST))
    # PIL-backed arrays are read-only; an in-place &= would fail on every request.
    return candidate & support


class ChangeAdapter:
    capability = "change"

    def __init__(self, settings):
        import torch
        from safetensors.torch import load_file
        from satquery_ml.models.notebook_experts import ChangeExpert

        root = Path(settings.artifact_dir or "")
        config, self.version = verified_config(root)
        if config.get("architecture") != "shared_resnet18_gru_answer_mask":
            raise ValueError("Wrong change architecture")
        self.vocabulary = config["word_vocab"]
        self.answers = [
            name
            for name, _ in sorted(
                config["answer_vocab"].items(), key=lambda row: row[1]
            )
        ]
        self.size = int(config["config"]["image_size"])
        self.max_tokens = int(config["config"]["max_question_tokens"])
        self.device = torch.device(
            settings.device if torch.cuda.is_available() else "cpu"
        )
        self.model = ChangeExpert(
            len(self.vocabulary), len(self.answers), pretrained=False
        )
        self.model.load_state_dict(load_file(root / "model.safetensors"), strict=True)
        self.model.to(self.device).eval()

    def infer(self, payload, paths):
        import rasterio
        import torch
        from PIL import Image
        from satquery_ml.sensors import visual_indexes

        if payload.step.task != "change_vqa" or len(paths) != 2:
            raise ValueError("ChangeVQA requires two temporal scenes")
        pairs = {
            asset.role: (asset, path)
            for asset, path in zip(payload.assets, paths, strict=True)
        }
        if set(pairs) != {"time_a", "time_b"}:
            raise ValueError("Explicit time_a and time_b roles required")
        ordered = [pairs[role] for role in ("time_a", "time_b")]
        height, width = validate_pair([pair[1] for pair in ordered])
        tensors, valid = [], np.ones((height, width), dtype=bool)
        for asset, path in ordered:
            if asset.modality not in {"optical", "multispectral"}:
                raise ValueError(
                    "The CDVQA/SECOND expert is trained on optical RGB, not SAR"
                )
            with rasterio.open(path) as source:
                indexes = visual_indexes(source)
                if source.count < 3 or any(
                    source.dtypes[index - 1] != "uint8" for index in indexes
                ):
                    raise ValueError(
                        "CDVQA expert expects uint8 RGB previews, not unnormalized spectral DN"
                    )
                raw = source.read(indexes, masked=True)
                valid &= ~np.ma.getmaskarray(raw).any(axis=0) & (
                    source.dataset_mask() > 0
                )
                image = Image.fromarray(np.moveaxis(raw.filled(0), 0, -1)).resize(
                    (self.size, self.size), Image.Resampling.BILINEAR
                )
                array = np.asarray(image).astype(np.float32).transpose(2, 0, 1) / 255.0
                array = (
                    array
                    - np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
                ) / np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
                tensors.append(torch.from_numpy(array).unsqueeze(0).to(self.device))
        require_shared_support(valid)
        tokens = [
            self.vocabulary.get(token, 1)
            for token in re.findall(r"[a-z0-9']+", payload.query.lower())
        ]
        tokens = (tokens[: self.max_tokens] + [0] * self.max_tokens)[: self.max_tokens]
        with torch.inference_mode():
            logits, mask_logits = self.model(
                *tensors, torch.tensor([tokens], device=self.device)
            )
            if (
                not torch.isfinite(logits).all()
                or not torch.isfinite(mask_logits).all()
            ):
                raise ValueError("Non-finite learned output")
            probabilities = logits.softmax(1)[0]
            index, score = int(probabilities.argmax()), float(probabilities.max())
            candidate = mask_logits.sigmoid()[0, 0].cpu().numpy() >= 0.5
        candidate = resample_supported_mask(candidate, valid)
        shape = candidate.shape[::-1]
        stream = io.BytesIO()
        Image.fromarray(candidate.astype(np.uint8) * 255).save(stream, format="PNG")
        return SpecialistResponse(
            task="change_vqa",
            text=f"Learned closed-vocabulary ChangeVQA answer: {self.answers[index]}. "
            "The separate mask predicts generic semantic change, not a target-specific loss or event cause.",
            facts=[
                {"name": "answer_label", "value": self.answers[index]},
                {"name": "execution_mode", "value": "learned_paired_change"},
            ],
            evidence=[
                Evidence(
                    id="ev_learned_change",
                    type="mask",
                    label="Learned semantic-change candidate",
                    score=score,
                    coordinate_space="pixel",
                    asset_id=ordered[1][0].id,
                    geometry={
                        "encoding": "png-base64",
                        "data": base64.b64encode(stream.getvalue()).decode(),
                        "width": shape[0],
                        "height": shape[1],
                        "method": "CDVQA SECOND supervised change mask",
                        "target": "semantic_change",
                        "threshold": 0.5,
                        "status": "candidate",
                        "comparison_asset_id": ordered[0][0].id,
                    },
                )
            ],
            raw_score=score,
            score_kind="uncalibrated",
            model_version=self.version,
            warnings=[
                "Answer softmax is not calibrated. SECOND-domain validation does not establish Cartosat or Nepal-flood accuracy.",
                "Mask boundaries are upsampled model estimates, not source-resolution delineation.",
            ],
        )


class FusionAdapter:
    capability = "fusion"

    def __init__(self, settings):
        import torch
        from safetensors.torch import load_file
        from terratorch.registry import BACKBONE_REGISTRY
        from satquery_ml.models.notebook_experts import FusionExpert

        root = Path(settings.artifact_dir or "")
        config, self.version = verified_config(root)
        if config.get("mask_supervision") != "none; scene labels only":
            raise ValueError("Unknown fusion artifact supervision contract")
        self.labels, self.normalization = config["classes"], config["normalization"]
        self.size = int(config["config"]["image_size"])
        self.device = torch.device(
            settings.device if torch.cuda.is_available() else "cpu"
        )
        backbone = BACKBONE_REGISTRY.build(
            config["backbone"],
            pretrained=False,
            modalities=["S2L2A", "S1GRD"],
            merge_method="mean",
        )
        self.model = FusionExpert(
            backbone, int(config["embedding_dim"]), len(self.labels), self.size
        )
        self.model.load_state_dict(load_file(root / "model.safetensors"), strict=True)
        self.model.to(self.device).eval()

    def infer(self, payload, paths):
        import rasterio
        import torch
        from torch.nn import functional as F
        from satquery_ml.sensors import sentinel_fusion_indexes

        if payload.step.task != "optical_sar_fusion" or len(paths) != 2:
            raise ValueError("Fusion requires exactly two registered modalities")
        pairs = [
            (asset, path) for asset, path in zip(payload.assets, paths, strict=True)
        ]
        optical = next(
            (
                path
                for asset, path in pairs
                if asset.modality in {"optical", "multispectral"}
            ),
            None,
        )
        sar = next((path for asset, path in pairs if asset.modality == "sar"), None)
        if optical is None or sar is None:
            raise ValueError("Declare optical and SAR modalities")
        validate_pair([optical, sar])
        with rasterio.open(optical) as s2, rasterio.open(sar) as s1:
            indexes = sentinel_fusion_indexes(s2, s1)
            inputs = []
            for source, bands, prefix in zip(
                [s2, s1], indexes, ["s2", "s1"], strict=True
            ):
                raw = source.read(bands, masked=True).astype(np.float32)
                if (
                    np.ma.getmaskarray(raw).any()
                    or not np.isfinite(raw.data).all()
                    or not (source.dataset_mask() > 0).all()
                ):
                    raise ValueError(
                        "Use a shared-valid cropped fusion tile; missing pixels cannot be fabricated"
                    )
                array = torch.from_numpy(raw.data)
                array = F.interpolate(
                    array[None],
                    (self.size, self.size),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                mean = torch.tensor(self.normalization[f"{prefix}_mean"])[:, None, None]
                std = torch.tensor(self.normalization[f"{prefix}_std"])[:, None, None]
                inputs.append(((array - mean) / std)[None].to(self.device))
        with torch.inference_mode():
            logits, _ = self.model(s2=inputs[0], s1=inputs[1])
            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite fusion predictions")
            scores = logits.sigmoid()[0].cpu().numpy()
        selected = np.flatnonzero(scores >= 0.5)
        labels = ", ".join(self.labels[index] for index in selected)
        return SpecialistResponse(
            task="optical_sar_fusion",
            text=f"Learned TerraMind optical/SAR scene-label candidates: {labels}."
            if labels
            else "No learned scene label exceeded the default 0.5 threshold.",
            facts=[
                {
                    "name": "scene_class_scores",
                    "value": dict(zip(self.labels, scores.tolist(), strict=True)),
                },
                {"name": "execution_mode", "value": "learned_cross_modal_fusion"},
            ],
            evidence=[],
            raw_score=float(scores.max()),
            score_kind="uncalibrated",
            model_version=self.version,
            warnings=[
                "This artifact has scene-label supervision only: no precision masks are returned.",
                "Sigmoid scores are uncalibrated; Cartosat and RISAT transfer requires separate training/evaluation.",
            ],
        )
