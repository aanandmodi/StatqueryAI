from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import numpy as np

from satquery_model_service.config import ModelSettings
from satquery_model_service.paired_adapters import ChangeAdapter, FusionAdapter
from satquery_model_service.contracts import (
    Evidence,
    InferencePayload,
    SpecialistResponse,
)


class Adapter(Protocol):
    capability: str
    version: str

    def infer(
        self, payload: InferencePayload, paths: list[Path]
    ) -> SpecialistResponse: ...


def _manifest_version(root: Path) -> str:
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise RuntimeError("Model artifact is missing manifest.json")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def _normalized_box(mask: np.ndarray, threshold: float) -> dict[str, float] | None:
    selected = np.argwhere(mask >= threshold)
    if selected.size == 0:
        return None
    y1, x1 = selected.min(axis=0)[-2:]
    y2, x2 = selected.max(axis=0)[-2:]
    height, width = mask.shape[-2:]
    return {
        "x": float(x1 / width),
        "y": float(y1 / height),
        "width": float((x2 - x1 + 1) / width),
        "height": float((y2 - y1 + 1) / height),
    }


def _read_raster(path: Path, bands: list[int], size: int) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling

    with (
        rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"),
        rasterio.open(path) as source,
    ):
        if max(bands) > source.count:
            raise ValueError(
                f"raster {path.name} does not contain required bands {bands}"
            )
        return source.read(
            bands,
            out_shape=(len(bands), size, size),
            resampling=Resampling.bilinear,
        ).astype(np.float32)


class QwenAdapter:
    capability = "vlm"

    def __init__(self, settings: ModelSettings) -> None:
        from satquery_ml.models.vlm import QwenVLRuntime

        self.runtime = QwenVLRuntime(
            settings.base_model,
            revision=settings.base_revision or "",
            adapter_path=settings.adapter_dir or settings.adapter_model,
            adapter_revision=None if settings.adapter_dir else settings.adapter_revision,
            processor_id=settings.adapter_dir or settings.adapter_model,
            processor_revision=None if settings.adapter_dir else settings.adapter_revision,
            four_bit=settings.four_bit,
            max_pixels=settings.max_pixels,
            device=settings.device,
        )
        adapter_version = (
            _manifest_version(settings.adapter_dir)
            if settings.adapter_dir
            else f"{settings.adapter_model}@{settings.adapter_revision}"
        )
        self.version = f"{settings.base_model}@{settings.base_revision}+{adapter_version}"
        self.max_new_tokens = settings.max_new_tokens
        self.max_image_edge = settings.max_image_edge

    def infer(self, payload: InferencePayload, paths: list[Path]) -> SpecialistResponse:
        from satquery_ml.preprocessing import geotiff_to_rgb_preview

        if payload.step.task not in {"single_vqa", "caption", "grounding"}:
            raise ValueError(f"Qwen adapter cannot execute {payload.step.task}")
        if len(paths) != 1:
            raise ValueError("Qwen adapter requires one validated image")
        preview = paths[0].with_suffix(".preview.png")
        geotiff_to_rgb_preview(paths[0], preview, max_size=self.max_image_edge)
        context_note = ""
        if payload.context:
            context_note = (
                " User-supplied context (do not claim it was inferred from pixels): "
                + payload.context.model_dump_json()
            )
        prompts = {
            "single_vqa": (
                "Answer the remote-sensing question from the image only. State uncertainty and "
                f"do not invent sensor facts. Question: {payload.query}{context_note}"
            ),
            "caption": (
                "Describe the land cover, visible objects, spatial relationships, and uncertainty "
                f"in this remote-sensing image. Do not infer raw SAR or hidden bands.{context_note}"
            ),
            "grounding": (
                "Locate the requested feature and return only JSON with keys label and bbox_2d. "
                "bbox_2d must be [x1,y1,x2,y2] in relative coordinates from 0 to 1000. "
                f"Request: {payload.query}{context_note}"
            ),
        }
        response = self.runtime.generate(
            preview,
            prompts[payload.step.task],
            max_new_tokens=self.max_new_tokens,
            expect_json=payload.step.task == "grounding",
        )
        evidence: list[Evidence] = []
        warnings = [
            "Open-ended VLM output is not probability-calibrated; raw_score is evidence quality."
        ]
        if payload.step.task == "grounding":
            box = _grounding_box(response.text, response.parsed)
            if (
                isinstance(box, list)
                and len(box) == 4
                and all(isinstance(value, int | float) for value in box)
            ):
                x1, y1, x2, y2 = [max(0.0, min(1000.0, float(value))) for value in box]
                if x2 > x1 and y2 > y1:
                    evidence.append(
                        Evidence(
                            id=f"ev_{uuid4().hex}",
                            type="box",
                            label=str(response.parsed.get("label", "target"))[:120],
                            score=0.5,
                            coordinate_space="normalized",
                            geometry={
                                "x": x1 / 1000,
                                "y": y1 / 1000,
                                "width": (x2 - x1) / 1000,
                                "height": (y2 - y1) / 1000,
                            },
                            asset_id=payload.assets[0].id,
                        )
                    )
                else:
                    warnings.append(
                        "Grounding JSON contained an empty or reversed box."
                    )
            else:
                warnings.append("Grounding output did not contain a valid bbox_2d.")
        return SpecialistResponse(
            task=payload.step.task,
            text=response.text,
            facts=(
                [
                    {
                        "name": "user_location",
                        "value": {
                            "latitude": payload.context.latitude,
                            "longitude": payload.context.longitude,
                            "altitude_m": payload.context.altitude_m,
                        },
                    }
                ]
                if payload.context
                else []
            ),
            evidence=evidence,
            raw_score=0.5,
            score_kind="evidence_quality",
            model_version=self.version,
            warnings=warnings,
        )


_TAGGED_BOX = re.compile(
    r"<box>\s*\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?\s*,"
    r"\s*\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?\s*</box>",
    re.IGNORECASE,
)


def _grounding_box(text: str, parsed: dict[str, object] | None) -> list[float] | None:
    """Accept the JSON and tagged-box formats used by Qwen-VL checkpoints."""

    if parsed:
        candidate = parsed.get("bbox_2d") or parsed.get("bbox") or parsed.get("box")
        if isinstance(candidate, list) and len(candidate) == 4:
            try:
                return [float(item) for item in candidate]
            except (TypeError, ValueError):
                pass
    match = _TAGGED_BOX.search(text)
    return [float(item) for item in match.groups()] if match else None


def build_adapter(settings: ModelSettings) -> Adapter:
    if settings.capability == "vlm":
        return QwenAdapter(settings)
    if settings.capability == "change":
        return ChangeAdapter(settings)
    return FusionAdapter(settings)
