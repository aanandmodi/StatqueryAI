from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from app.core.sensors import semantic_indexes, sensor_profile
from app.errors import ModelUnavailableError
from app.models.masks import encode_mask
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
    from app.core.sensors import visual_indexes
    if not visual:
        return list(range(1, min(dataset.count, 2) + 1))
    return visual_indexes(dataset)


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

        # A masked VRT preserves source nodata/internal masks as well as pixels
        # outside the footprint, without reading the full-resolution image.
        with WarpedVRT(
            source,
            crs=grid.crs,
            transform=grid.transform,
            height=grid.height,
            width=grid.width,
            dtype="float32",
            nodata=np.nan,
            resampling=Resampling.bilinear,
            UNIFIED_SRC_NODATA="NO",
        ) as aligned:
            data = aligned.read(indexes, masked=True)
            return np.ma.filled(data.astype(np.float32), np.nan)


def _read_semantic_on_grid(
    path: Path, grid: ReferenceGrid, meanings: list[str]
) -> tuple[np.ndarray, list[int]]:
    """Read verified reflectance bands and apply declared scale/offset on the common grid."""
    with rasterio.open(path) as source:
        indexes = semantic_indexes(source, meanings)
        if indexes is None:
            raise ModelUnavailableError(
                f"Spectral change requires verified {'/'.join(meanings)} bands at both dates"
            )
        if source.crs is None or grid.crs is None:
            data = source.read(
                indexes,
                out_shape=(len(indexes), grid.height, grid.width),
                resampling=Resampling.nearest,
                masked=True,
            )
        else:
            with WarpedVRT(
                source,
                crs=grid.crs,
                transform=grid.transform,
                height=grid.height,
                width=grid.width,
                dtype="float32",
                nodata=np.nan,
                resampling=Resampling.nearest,
                UNIFIED_SRC_NODATA="NO",
            ) as aligned:
                data = aligned.read(indexes, masked=True)
        values = np.ma.filled(data.astype(np.float32), np.nan)
        for plane, index in enumerate(indexes):
            values[plane] = values[plane] * source.scales[index - 1] + source.offsets[index - 1]
        return values, indexes


def _normalized_difference(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    denominator = values[0] + values[1]
    valid = (
        np.isfinite(values).all(axis=0)
        & (values >= 0).all(axis=0)
        & (denominator > 1e-8)
    )
    index = np.divide(
        values[0] - values[1], denominator, out=np.zeros_like(denominator), where=valid
    )
    return index, valid


def _shared_valid(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # Missing one selected channel must not turn into a dark, apparently valid
    # observation after normalization.
    valid = np.isfinite(left).all(axis=0) & np.isfinite(right).all(axis=0)
    if not valid.any():
        raise ModelUnavailableError("Paired rasters contain no shared valid pixels")
    return valid


def _has_spatial_signal(array: np.ndarray, valid: np.ndarray) -> bool:
    for band in array:
        values = band[valid]
        low, high = float(values.min()), float(values.max())
        tolerance = max(1e-6, max(abs(low), abs(high)) * 1e-6)
        if high - low > tolerance:
            return True
    return False


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


def _regional_change_summary(changed: np.ndarray) -> list[dict[str, float | str]]:
    """Describe mask concentration in image coordinates without inventing compass directions."""
    total = int(changed.sum())
    if total == 0:
        return []
    height, width = changed.shape
    vertical = (
        ("upper", 0, height // 3),
        ("middle", height // 3, 2 * height // 3),
        ("lower", 2 * height // 3, height),
    )
    horizontal = (
        ("left", 0, width // 3),
        ("center", width // 3, 2 * width // 3),
        ("right", 2 * width // 3, width),
    )
    rows: list[dict[str, float | str]] = []
    for y_name, y1, y2 in vertical:
        for x_name, x1, x2 in horizontal:
            count = int(changed[y1:y2, x1:x2].sum())
            rows.append(
                {
                    "region": f"{y_name}-{x_name}",
                    "changed_pixel_share": round(count / total, 6),
                }
            )
    return sorted(rows, key=lambda item: float(item["changed_pixel_share"]), reverse=True)[:3]


def _evidence(*, mask: np.ndarray, label: str, score: float, asset_id: str) -> EvidenceItem | None:
    if not mask.any():
        return None
    return EvidenceItem(
        id=f"ev_{uuid4().hex}",
        type="mask",
        label=label,
        score=max(0.0, min(1.0, score)),
        coordinate_space="pixel",
        geometry={
            "encoding": "png-base64",
            "data": encode_mask(mask),
            "width": mask.shape[1],
            "height": mask.shape[0],
            "method": "optical color/texture and relative SAR backscatter proxy",
            "status": "candidate",
        },
        asset_id=asset_id,
    )


class LocalPairSpecialistGateway:
    """Bounded CPU baselines for the two mandatory paired-image demonstrations.

    These tools intentionally expose evidence-quality scores, not semantic class
    probabilities. They make the paired workflows runnable before learned CDVQA
    and optical/SAR artifacts pass their separate benchmark release gates.
    """

    CHANGE_VERSION = "satquery-spectral-change-tool-v2"
    FUSION_VERSION = "satquery-optical-sar-proxy-tool-v2"

    def __init__(self, asset_store: LocalAssetStore, *, max_edge: int = 1536) -> None:
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

    def _change(self, step: PlannedStep, assets: list[AssetRecord], query: str) -> SpecialistOutput:
        before = next((item for item in assets if item.role == AssetRole.TIME_A), assets[0])
        after = next((item for item in assets if item.role == AssetRole.TIME_B), assets[1])
        grid = _reference_grid(self.asset_store.resolve(before.id), self.max_edge)
        left_raw = _read_on_grid(
            self.asset_store.resolve(before.id), grid, visual=before.modality != Modality.SAR
        )
        right_raw = _read_on_grid(
            self.asset_store.resolve(after.id), grid, visual=after.modality != Modality.SAR
        )
        valid = _shared_valid(left_raw, right_raw)
        if not (_has_spatial_signal(left_raw, valid) or _has_spatial_signal(right_raw, valid)):
            raise ModelUnavailableError(
                "Change baseline abstained: shared imagery has insufficient spatial signal"
            )
        left_raw = np.where(valid[None, :, :], left_raw, np.nan)
        right_raw = np.where(valid[None, :, :], right_raw, np.nan)
        left, right = _joint_unit_scale(left_raw, right_raw)
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
        median_difference = float(np.median(valid_values)) if valid_values.size else 0.0
        p95_difference = float(np.percentile(valid_values, 95.0)) if valid_values.size else 0.0
        delta = np.mean(right - left, axis=0)
        mean_delta = float(delta[changed].mean()) if changed.any() else 0.0
        direction = (
            "increased" if mean_delta > 0.03 else "decreased" if mean_delta < -0.03 else "mixed"
        )
        after_brightness = np.mean(right, axis=0)
        after_chroma = np.max(right, axis=0) - np.min(right, axis=0)
        bright_neutral = valid & (after_brightness >= 0.75) & (after_chroma <= 0.18)
        bright_neutral_fraction = float(bright_neutral.sum() / max(1, valid.sum()))
        regional_summary = _regional_change_summary(changed)
        quality = min(0.58, 0.35 + 0.18 * float(valid.mean()) + min(0.05, changed_fraction))
        evidence: list[EvidenceItem] = []
        if changed.any():
            encoded = encode_mask(changed)
            common_geometry = {
                "encoding": "png-base64",
                "data": encoded,
                "width": grid.width,
                "height": grid.height,
                "method": "jointly normalized absolute selected-band difference",
                "threshold": threshold,
                "status": "candidate",
                "comparison_asset_id": after.id,
                "mask_group": "paired_change_candidates",
            }
            evidence.append(
                EvidenceItem(
                    id=f"ev_{uuid4().hex}",
                    type="mask",
                    label=f"change candidates on before scene · {changed_fraction:.1%}",
                    score=quality,
                    coordinate_space="pixel",
                    asset_id=before.id,
                    geometry=common_geometry,
                )
            )
            after_grid = _reference_grid(self.asset_store.resolve(after.id), self.max_edge)
            if (
                after_grid.height == grid.height
                and after_grid.width == grid.width
                and after_grid.transform == grid.transform
                and after_grid.crs == grid.crs
            ):
                evidence.append(
                    EvidenceItem(
                        id=f"ev_{uuid4().hex}",
                        type="mask",
                        label=f"same change candidates on after scene · {changed_fraction:.1%}",
                        score=quality,
                        coordinate_space="pixel",
                        asset_id=after.id,
                        geometry={**common_geometry, "comparison_asset_id": before.id},
                    )
                )
        concentration = (
            ", ".join(
                f"{item['region']} ({float(item['changed_pixel_share']):.1%} of selected pixels)"
                for item in regional_summary
            )
            or "no region exceeded the selected threshold"
        )
        visibility_text = (
            "SAR interpretation: optical cloud visibility cannot be assessed from backscatter. "
            "Speckle, incidence angle, layover, radar shadow and moisture can drive "
            "differences.\n\n"
            if after.modality == Modality.SAR
            else f"Visibility check: {bright_neutral_fraction:.1%} of after-image valid pixels "
            "match a bright-neutral obstruction proxy. Cloud, haze or bright bare surfaces may "
            "therefore dominate parts of the difference mask.\n\n"
        )
        answer = (
            "## Executive finding\n\n"
            f"The registered-grid screening found candidate change in **{changed_fraction:.1%}** "
            f"of the shared valid pixels. The strongest selected changes are concentrated in "
            f"{concentration}. This is a measured image difference, not yet a semantic event "
            "classification.\n\n"
            "## Comparison and magnitude\n\n"
            f"Time A is `{before.original_name}` and time B is `{after.original_name}`. "
            "Pixels were jointly normalized and compared at threshold "
            f"**{threshold:.3f}**. The median absolute "
            f"difference is **{median_difference:.3f}** and the 95th percentile is "
            f"**{p95_difference:.3f}**. Inside the candidate mask, the mean selected-band response "
            f"is **{direction}**.\n\n"
            "## Spatial pattern\n\n"
            f"The regional distribution is {concentration}. Region names describe positions in the "
            "display grid; they are not compass bearings or administrative areas. Toggle the mask "
            "against both originals to separate persistent boundaries from newly appearing or "
            "disappearing texture.\n\n"
            "## Plausible explanations—not conclusions\n\n"
            "Candidate differences may reflect real land-cover change, construction, vegetation "
            "phenology, inundation or disturbance. They can also be produced by cloud/shadow, "
            "illumination, seasonal appearance, sensor response or imperfect registration. The "
            "images alone do not establish which explanation is correct.\n\n"
            f"{visibility_text}"
            "## Verification required\n\n"
            "Confirm acquisition timestamps, sensor/product consistency, co-registration and "
            "cloud/shadow quality. Compare candidate regions with an independently labelled change "
            "map or authoritative event record before reporting flooding, damage, deforestation, "
            "construction or causality."
        )
        result = SpecialistOutput(
            task=TaskType.CHANGE_VQA,
            text=answer,
            facts=[
                {"name": "changed_pixel_fraction", "value": round(changed_fraction, 6)},
                {"name": "normalized_change_threshold", "value": round(threshold, 6)},
                {
                    "name": "threshold_selection",
                    "value": "user-specified"
                    if requested_threshold is not None
                    else "max(0.12, 85th percentile); highlights upper-tail differences, "
                    "not all real changes",
                },
                {"name": "mean_response_direction", "value": direction},
                {"name": "valid_pixel_fraction", "value": round(float(valid.mean()), 6)},
                {"name": "median_normalized_difference", "value": round(median_difference, 6)},
                {"name": "p95_normalized_difference", "value": round(p95_difference, 6)},
                *(
                    [
                        {
                            "name": "after_bright_neutral_proxy_fraction",
                            "value": round(bright_neutral_fraction, 6),
                        }
                    ]
                    if after.modality != Modality.SAR
                    else []
                ),
                {"name": "top_change_regions", "value": regional_summary},
            ],
            evidence=evidence,
            raw_score=quality,
            score_kind="evidence_quality",
            model_version=self.CHANGE_VERSION,
            warnings=[
                "Uncalibrated deterministic spectral-change baseline; not a semantic "
                "CDVQA probability.",
                "The same candidate mask is displayed on each scene only when their analysis grids "
                "match. Cloud, shadow, season, illumination, sensor and registration differences "
                "can all trigger it.",
                "Use the CDVQA-trained change expert after it passes the prescribed public split.",
            ],
        )
        return self._append_dnbr(result, step, before, after, grid)

    def _append_dnbr(
        self,
        output: SpecialistOutput,
        step: PlannedStep,
        before: AssetRecord,
        after: AssetRecord,
        grid: ReferenceGrid,
    ) -> SpecialistOutput:
        targets = {str(item).lower() for item in step.permitted_params.get("targets", [])}
        if not targets & {"burn", "burn-scar", "burned"}:
            return output
        if any(
            asset.modality not in {Modality.OPTICAL, Modality.MULTISPECTRAL}
            for asset in (before, after)
        ):
            return output.model_copy(
                update={
                    "warnings": [
                        *output.warnings,
                        "dNBR unavailable: both dates must be optical/multispectral Sentinel-2.",
                    ]
                }
            )
        with rasterio.open(self.asset_store.resolve(before.id)) as source_before:
            before_platform = sensor_profile(source_before)["platform"]
        with rasterio.open(self.asset_store.resolve(after.id)) as source_after:
            after_platform = sensor_profile(source_after)["platform"]
        if {before_platform, after_platform} != {"sentinel-2"}:
            return output.model_copy(
                update={
                    "warnings": [
                        *output.warnings,
                        "dNBR unavailable: Cartosat-2-series has no SWIR band; both dates need "
                        "verified Sentinel-2 NIR/SWIR2 metadata.",
                    ]
                }
            )

        before_values, before_indexes = _read_semantic_on_grid(
            self.asset_store.resolve(before.id), grid, ["nir", "swir2"]
        )
        after_values, after_indexes = _read_semantic_on_grid(
            self.asset_store.resolve(after.id), grid, ["nir", "swir2"]
        )
        before_nbr, before_valid = _normalized_difference(before_values)
        after_nbr, after_valid = _normalized_difference(after_values)
        valid = before_valid & after_valid
        dnbr = before_nbr - after_nbr
        threshold = float(step.permitted_params.get("threshold", 0.1))
        candidate = valid & (dnbr >= threshold)
        severity_bounds = {
            "unburned_or_regrowth": (-np.inf, 0.1),
            "low": (0.1, 0.27),
            "moderate_low": (0.27, 0.44),
            "moderate_high": (0.44, 0.66),
            "high": (0.66, np.inf),
        }
        severity = {
            name: round(
                float((valid & (dnbr >= lower) & (dnbr < upper)).sum() / max(1, valid.sum())),
                6,
            )
            for name, (lower, upper) in severity_bounds.items()
        }
        evidence = list(output.evidence)
        if candidate.any():
            evidence.append(
                EvidenceItem(
                    id=f"ev_{uuid4().hex}",
                    type="mask",
                    label=(
                        "dNBR burn-severity candidate · "
                        f"{candidate.sum() / max(1, valid.sum()):.1%}"
                    ),
                    score=min(output.raw_score, 0.58),
                    coordinate_space="pixel",
                    asset_id=after.id,
                    geometry={
                        "encoding": "png-base64",
                        "data": encode_mask(candidate),
                        "width": grid.width,
                        "height": grid.height,
                        "method": "dNBR = pre-fire NBR - post-fire NBR",
                        "target": "burn-scar",
                        "threshold": threshold,
                        "status": "candidate",
                        "comparison_asset_id": before.id,
                        "before_band_indexes": before_indexes,
                        "after_band_indexes": after_indexes,
                    },
                )
            )
        return output.model_copy(
            update={
                "text": (
                    output.text
                    + "\n\nSentinel-2 dNBR addendum: candidate severity was computed from verified "
                    f"NIR/SWIR2 bands using threshold {threshold:.3f}. Review class fractions in "
                    "the facts table; these screening classes are not an insurance or damage "
                    "determination."
                ),
                "facts": [
                    *output.facts,
                    {"name": "dnbr_threshold", "value": threshold},
                    {"name": "dnbr_severity_fraction", "value": severity},
                ],
                "evidence": evidence,
                "warnings": [
                    *output.warnings,
                    "dNBR severity thresholds are screening conventions; atmospheric correction, "
                    "phenology, clouds, registration and local validation remain required.",
                ],
            }
        )

    def _fusion(self, step: PlannedStep, assets: list[AssetRecord], query: str) -> SpecialistOutput:
        optical = next(
            item for item in assets if item.modality in {Modality.OPTICAL, Modality.MULTISPECTRAL}
        )
        sar = next(item for item in assets if item.modality == Modality.SAR)
        grid = _reference_grid(self.asset_store.resolve(optical.id), self.max_edge)
        if min(grid.height, grid.width) < 2:
            raise ModelUnavailableError(
                "Optical/SAR baseline needs at least two pixels on each analysis-grid axis"
            )
        optical_raw = _read_on_grid(self.asset_store.resolve(optical.id), grid, visual=True)
        sar_raw = _read_on_grid(self.asset_store.resolve(sar.id), grid, visual=False)
        valid = _shared_valid(optical_raw, sar_raw)
        if not (_has_spatial_signal(optical_raw, valid) and _has_spatial_signal(sar_raw, valid)):
            raise ModelUnavailableError(
                "Optical/SAR baseline abstained: optical and SAR images both need spatial contrast"
            )
        rgb = _unit_scale(np.where(valid[None, :, :], optical_raw, np.nan))
        sar_data = _unit_scale(np.where(valid[None, :, :], sar_raw, np.nan))
        luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        backscatter = sar_data.mean(axis=0)
        grad_y, grad_x = np.gradient(luminance)
        edges = np.hypot(grad_x, grad_y)
        # Do not mistake edges introduced by a nodata boundary for built-up
        # texture. Coverage itself is still measured over the shared valid area.
        edge_valid = valid.copy()
        edge_valid[1:] &= valid[:-1]
        edge_valid[:-1] &= valid[1:]
        edge_valid[:, 1:] &= valid[:, :-1]
        edge_valid[:, :-1] &= valid[:, 1:]
        valid_backscatter = backscatter[valid]
        valid_luminance = luminance[valid]
        valid_edges = edges[valid]
        if not valid_backscatter.size:
            raise ModelUnavailableError("Optical/SAR pair contains no shared valid pixels")
        low_sar, high_sar = np.percentile(valid_backscatter, [30.0, 70.0])
        dark_optical = float(np.percentile(valid_luminance, 35.0))
        strong_edge = float(np.percentile(valid_edges, 65.0))
        blue_dominance = rgb[2] > (rgb[0] * 1.05)
        sar_low_return = valid & (backscatter <= low_sar)
        optical_water_cue = valid & ((luminance <= dark_optical) | blue_dominance)
        water = sar_low_return & optical_water_cue
        water_disagreement = valid & (sar_low_return ^ optical_water_cue)
        sar_high_return = valid & (backscatter >= high_sar)
        optical_structure_cue = edge_valid & (edges > strong_edge) & (
            luminance >= dark_optical
        )
        built_up = (
            sar_high_return & optical_structure_cue
        )
        structure_disagreement = valid & (sar_high_return ^ optical_structure_cue)
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
                item.geometry["comparison_asset_id"] = sar.id
                evidence.append(item)
        valid_count = max(1, valid.sum())
        agreement = {
            "water_joint_fraction": float(water.sum() / valid_count),
            "water_cue_disagreement_fraction": float(
                water_disagreement.sum() / valid_count
            ),
            "built_up_joint_fraction": float(built_up.sum() / valid_count),
            "structure_cue_disagreement_fraction": float(
                structure_disagreement.sum() / valid_count
            ),
        }
        summary = "; ".join(
            f"{name}: {fraction:.1%}" for name, fraction in fractions.items()
        )
        return SpecialistOutput(
            task=TaskType.OPTICAL_SAR_FUSION,
            text=(
                "## Executive finding\n\n"
                "The optical and SAR inputs were compared on the optical reference grid. Their "
                f"joint analytical candidates are **{summary}**. These are proxy coverages, not "
                "confirmed land-cover labels.\n\n"
                "## What each sensor contributed\n\n"
                "The optical branch contributed visible brightness, colour and edge/texture cues. "
                "The SAR branch contributed relative low- and high-return patterns. The system "
                "selected a pixel only where the relevant cues agreed; it did not reconstruct "
                "missing optical information or treat radar intensity as colour.\n\n"
                "## Agreement and disagreement\n\n"
                f"Joint water-like cues cover **{agreement['water_joint_fraction']:.1%}** of valid "
                f"pixels, while the two water cues disagree on "
                f"**{agreement['water_cue_disagreement_fraction']:.1%}**. Joint structure-like "
                f"cues cover **{agreement['built_up_joint_fraction']:.1%}**, with structural-cue "
                f"disagreement on **{agreement['structure_cue_disagreement_fraction']:.1%}**. "
                "Disagreement is useful review evidence, not an error to hide.\n\n"
                "## Interpretation limits\n\n"
                "Low SAR return can represent smooth water, radar shadow or other smooth surfaces. "
                "High return can arise from buildings, rough terrain, vegetation structure, "
                "incidence geometry or speckle. Optical darkness can be water, shadow or dark "
                "material. Therefore the overlay marks candidates only.\n\n"
                "## Verification required\n\n"
                f"The user asked: {query.strip()} Review polarization, calibration, "
                "incidence angle, "
                "acquisition times, co-registration and independent labels. Use a released learned "
                "fusion checkpoint only after it passes held-out optical/SAR segmentation tests."
            ),
            facts=[
                {"name": name.replace(" ", "_"), "value": round(value, 6)}
                for name, value in fractions.items()
            ]
            + [
                {"name": name, "value": round(value, 6)}
                for name, value in agreement.items()
            ]
            + [{"name": "valid_pixel_fraction", "value": round(float(valid.mean()), 6)}],
            evidence=evidence,
            raw_score=quality,
            score_kind="evidence_quality",
            model_version=self.FUSION_VERSION,
            warnings=[
                "Uncalibrated optical/SAR proxy baseline; not a TerraMind class probability.",
                "Masks follow the analytical proxy pixels, not validated semantic boundaries. "
                "Do not treat colors or relative backscatter as a universal land-cover classifier.",
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
