from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject

from app.errors import ModelUnavailableError
from app.schemas import (
    AssetRecord,
    AssetRole,
    EvidenceItem,
    GeospatialContext,
    Modality,
    PlannedStep,
    SpecialistOutput,
    TaskType,
)
from app.storage import LocalAssetStore

PAIR_TASKS = {TaskType.CHANGE_VQA, TaskType.OPTICAL_SAR_FUSION}


@dataclass(frozen=True)
class ReferenceGrid:
    height: int
    width: int
    transform: Affine
    crs: object | None


def _band_indexes(dataset: rasterio.io.DatasetReader, *, visual: bool) -> list[int]:
    if not visual:
        return list(range(1, min(dataset.count, 2) + 1))
    descriptions = [str(item or "").strip().lower() for item in dataset.descriptions]
    aliases = (("red", "b04", "b4"), ("green", "b03", "b3"), ("blue", "b02", "b2"))
    selected: list[int] = []
    for names in aliases:
        match = next(
            (index for index, description in enumerate(descriptions, 1) if description in names),
            None,
        )
        if match is None:
            selected = []
            break
        selected.append(match)
    if selected:
        return selected
    if dataset.count >= 4:
        return [4, 3, 2]
    if dataset.count >= 3:
        return [1, 2, 3]
    return [1, 1, 1]


def _reference_grid(path: Path, max_edge: int) -> ReferenceGrid:
    with rasterio.open(path) as source:
        scale = min(1.0, max_edge / max(source.width, source.height))
        width = max(1, round(source.width * scale))
        height = max(1, round(source.height * scale))
        transform = source.transform @ Affine.scale(source.width / width, source.height / height)
        return ReferenceGrid(height=height, width=width, transform=transform, crs=source.crs)


def _read_on_grid(path: Path, grid: ReferenceGrid, *, visual: bool) -> np.ndarray:
    with rasterio.open(path) as source:
        indexes = _band_indexes(source, visual=visual)
        if source.crs is None or grid.crs is None:
            data = source.read(
                indexes,
                out_shape=(len(indexes), grid.height, grid.width),
                resampling=Resampling.bilinear,
                masked=True,
            )
            return np.ma.filled(data.astype(np.float32), np.nan)

        output = np.full((len(indexes), grid.height, grid.width), np.nan, np.float32)
        for output_index, band_index in enumerate(indexes):
            reproject(
                source=rasterio.band(source, band_index),
                destination=output[output_index],
                src_transform=source.transform,
                src_crs=source.crs,
                src_nodata=source.nodata,
                dst_transform=grid.transform,
                dst_crs=grid.crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
        return output


def _unit_scale(array: np.ndarray) -> np.ndarray:
    output = np.zeros_like(array, dtype=np.float32)
    for index, band in enumerate(array):
        finite = np.isfinite(band)
        if not finite.any():
            continue
        values = band[finite]
        low, high = np.percentile(values, [2.0, 98.0])
        if high <= low:
            low, high = float(values.min()), float(values.max())
        if high > low:
            output[index] = np.clip((band - low) / (high - low), 0.0, 1.0)
        output[index][~finite] = 0.0
    return output


def _joint_unit_scale(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_out = np.zeros_like(left, dtype=np.float32)
    right_out = np.zeros_like(right, dtype=np.float32)
    for index in range(left.shape[0]):
        left_band = left[index]
        right_band = right[index]
        values = np.concatenate(
            [left_band[np.isfinite(left_band)], right_band[np.isfinite(right_band)]]
        )
        if values.size == 0:
            continue
        low, high = np.percentile(values, [2.0, 98.0])
        if high <= low:
            low, high = float(values.min()), float(values.max())
        if high > low:
            left_out[index] = np.clip((left_band - low) / (high - low), 0.0, 1.0)
            right_out[index] = np.clip((right_band - low) / (high - low), 0.0, 1.0)
        left_out[index][~np.isfinite(left_band)] = 0.0
        right_out[index][~np.isfinite(right_band)] = 0.0
    return left_out, right_out


def _normalized_box(mask: np.ndarray) -> dict[str, float] | None:
    selected = np.argwhere(mask)
    if selected.size == 0:
        return None
    y1, x1 = selected.min(axis=0)
    y2, x2 = selected.max(axis=0)
    height, width = mask.shape
    return {
        "x": float(x1 / width),
        "y": float(y1 / height),
        "width": float((x2 - x1 + 1) / width),
        "height": float((y2 - y1 + 1) / height),
    }


def _evidence(
    *, mask: np.ndarray, label: str, score: float, asset_id: str
) -> EvidenceItem | None:
    geometry = _normalized_box(mask)
    if geometry is None:
        return None
    return EvidenceItem(
        id=f"ev_{uuid4().hex}",
        type="box",
        label=label,
        score=max(0.0, min(1.0, score)),
        coordinate_space="normalized",
        geometry=geometry,
        asset_id=asset_id,
    )


class LocalPairSpecialistGateway:
    """Bounded CPU baselines for the two mandatory paired-image demonstrations.

    These tools intentionally expose evidence-quality scores, not semantic class
    probabilities. They make the paired workflows runnable before learned CDVQA
    and optical/SAR artifacts pass their separate benchmark release gates.
    """

    CHANGE_VERSION = "satquery-spectral-change-tool-v1"
    FUSION_VERSION = "satquery-optical-sar-proxy-tool-v1"

    def __init__(self, asset_store: LocalAssetStore, *, max_edge: int = 512) -> None:
        self.asset_store = asset_store
        self.max_edge = max_edge

    async def infer(
        self,
        step: PlannedStep,
        assets: list[AssetRecord],
        query: str,
        context: GeospatialContext | None = None,
    ) -> SpecialistOutput:
        del context
        if step.task not in PAIR_TASKS:
            raise ModelUnavailableError(
                "The local pair gateway cannot execute a single-image task",
                details={"task": step.task.value},
            )
        if len(assets) != 2:
            raise ModelUnavailableError(
                "Paired analysis requires exactly two validated rasters",
                details={"task": step.task.value, "assets": len(assets)},
            )
        return await asyncio.to_thread(self._infer_sync, step, assets, query)

    def _infer_sync(
        self, step: PlannedStep, assets: list[AssetRecord], query: str
    ) -> SpecialistOutput:
        if step.task == TaskType.CHANGE_VQA:
            return self._change(step, assets, query)
        return self._fusion(step, assets, query)

    def _change(
        self, step: PlannedStep, assets: list[AssetRecord], query: str
    ) -> SpecialistOutput:
        del query
        before = next((item for item in assets if item.role == AssetRole.TIME_A), assets[0])
        after = next((item for item in assets if item.role == AssetRole.TIME_B), assets[1])
        grid = _reference_grid(self.asset_store.resolve(before.id), self.max_edge)
        left_raw = _read_on_grid(self.asset_store.resolve(before.id), grid, visual=True)
        right_raw = _read_on_grid(self.asset_store.resolve(after.id), grid, visual=True)
        left, right = _joint_unit_scale(left_raw, right_raw)
        valid = np.isfinite(left_raw).any(axis=0) & np.isfinite(right_raw).any(axis=0)
        difference = np.mean(np.abs(right - left), axis=0)
        valid_values = difference[valid]
        requested_threshold = step.permitted_params.get("threshold")
        threshold = (
            float(requested_threshold)
            if isinstance(requested_threshold, int | float)
            else max(0.12, float(np.percentile(valid_values, 85.0)))
            if valid_values.size
            else 0.12
        )
        changed = valid & (difference >= threshold)
        changed_fraction = float(changed.sum() / max(1, valid.sum()))
        delta = np.mean(right - left, axis=0)
        mean_delta = float(delta[changed].mean()) if changed.any() else 0.0
        direction = (
            "increased"
            if mean_delta > 0.03
            else "decreased"
            if mean_delta < -0.03
            else "mixed"
        )
        quality = min(0.58, 0.35 + 0.18 * float(valid.mean()) + min(0.05, changed_fraction))
        evidence_item = _evidence(
            mask=changed,
            label=f"spectral change proxy · {changed_fraction:.1%}",
            score=quality,
            asset_id=after.id,
        )
        answer = (
            f"The co-registered pair shows a spectral-change proxy over approximately "
            f"{changed_fraction:.1%} of valid pixels at threshold {threshold:.3f}. "
            f"The changed area has a {direction} mean normalized response. This establishes "
            "where pixel values changed, but it does not by itself prove a semantic class such "
            "as construction, flooding, or deforestation."
        )
        return SpecialistOutput(
            task=TaskType.CHANGE_VQA,
            text=answer,
            facts=[
                {"name": "changed_pixel_fraction", "value": round(changed_fraction, 6)},
                {"name": "normalized_change_threshold", "value": round(threshold, 6)},
                {"name": "mean_response_direction", "value": direction},
                {"name": "valid_pixel_fraction", "value": round(float(valid.mean()), 6)},
            ],
            evidence=[evidence_item] if evidence_item else [],
            raw_score=quality,
            score_kind="evidence_quality",
            model_version=self.CHANGE_VERSION,
            warnings=[
                "Uncalibrated deterministic spectral-change baseline; not a semantic "
                "CDVQA probability.",
                "Use the CDVQA-trained change expert after it passes the prescribed public split.",
            ],
        )

    def _fusion(
        self, step: PlannedStep, assets: list[AssetRecord], query: str
    ) -> SpecialistOutput:
        del query
        optical = next(
            item
            for item in assets
            if item.modality in {Modality.OPTICAL, Modality.MULTISPECTRAL}
        )
        sar = next(item for item in assets if item.modality == Modality.SAR)
        grid = _reference_grid(self.asset_store.resolve(optical.id), self.max_edge)
        rgb = _unit_scale(
            _read_on_grid(self.asset_store.resolve(optical.id), grid, visual=True)
        )
        sar_data = _unit_scale(
            _read_on_grid(self.asset_store.resolve(sar.id), grid, visual=False)
        )
        luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        backscatter = sar_data.mean(axis=0)
        grad_y, grad_x = np.gradient(luminance)
        edges = np.hypot(grad_x, grad_y)
        valid = np.isfinite(luminance) & np.isfinite(backscatter)
        valid_backscatter = backscatter[valid]
        valid_luminance = luminance[valid]
        valid_edges = edges[valid]
        if not valid_backscatter.size:
            raise ModelUnavailableError("Optical/SAR pair contains no shared valid pixels")
        low_sar, high_sar = np.percentile(valid_backscatter, [30.0, 70.0])
        dark_optical = float(np.percentile(valid_luminance, 35.0))
        strong_edge = float(np.percentile(valid_edges, 65.0))
        blue_dominance = rgb[2] > (rgb[0] * 1.05)
        water = valid & (backscatter <= low_sar) & (
            (luminance <= dark_optical) | blue_dominance
        )
        built_up = valid & (backscatter >= high_sar) & (edges >= strong_edge) & (
            luminance >= dark_optical
        )
        masks = {"water proxy": water, "built-up proxy": built_up}
        requested = {str(item).lower() for item in step.permitted_params.get("targets", [])}
        selected_names = [
            name
            for name in masks
            if not requested
            or ("water" in requested and name.startswith("water"))
            or ({"built-up", "urban", "building"} & requested and name.startswith("built-up"))
        ]
        if not selected_names:
            selected_names = list(masks)
        evidence: list[EvidenceItem] = []
        fractions: dict[str, float] = {}
        quality = min(0.58, 0.37 + 0.18 * float(valid.mean()))
        for name in selected_names:
            fraction = float(masks[name].sum() / max(1, valid.sum()))
            fractions[name] = fraction
            item = _evidence(
                mask=masks[name],
                label=f"{name} · {fraction:.1%}",
                score=quality,
                asset_id=optical.id,
            )
            if item:
                evidence.append(item)
        summary = "; ".join(f"{name}: {fraction:.1%}" for name, fraction in fractions.items())
        return SpecialistOutput(
            task=TaskType.OPTICAL_SAR_FUSION,
            text=(
                "The aligned optical/SAR analytical baseline combined visible-context cues with "
                f"relative SAR backscatter and produced these proxy coverages: {summary}. "
                "The boxes identify candidate regions for review, not confirmed land-cover labels."
            ),
            facts=[
                {"name": name.replace(" ", "_"), "value": round(value, 6)}
                for name, value in fractions.items()
            ]
            + [{"name": "valid_pixel_fraction", "value": round(float(valid.mean()), 6)}],
            evidence=evidence,
            raw_score=quality,
            score_kind="evidence_quality",
            model_version=self.FUSION_VERSION,
            warnings=[
                "Uncalibrated optical/SAR proxy baseline; not a TerraMind class probability.",
                "Validate against the prescribed paired benchmark and hidden ISRO/SAC "
                "data before operational use.",
            ],
        )

    async def health(self) -> bool:
        return True

    def versions(self) -> dict[str, str]:
        return {
            TaskType.CHANGE_VQA.value: self.CHANGE_VERSION,
            TaskType.OPTICAL_SAR_FUSION.value: self.FUSION_VERSION,
        }

    def supported_tasks(self) -> list[str]:
        return sorted(task.value for task in PAIR_TASKS)
