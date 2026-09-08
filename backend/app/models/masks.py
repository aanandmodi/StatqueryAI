"""Bounded binary masks: preserve holes, exclude no-data, and keep evidence on its source grid."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import math
from itertools import pairwise

import numpy as np
import rasterio
from PIL import Image
from rasterio import Affine
from rasterio.enums import Resampling
from rasterio.features import shapes

from app.core.scene_report import pixel_area_m2
from app.core.sensors import semantic_indexes, sensor_profile, visual_indexes
from app.schemas import EvidenceItem, Modality, TaskType

MAX_MASK_EDGE = 2048
MAX_MASK_BYTES = 1_000_000
MAX_VECTOR_POLYGONS = 64
MAX_VECTOR_VERTICES = 8_000

EVIDENCE_PALETTE = {
    "water": "#2dcef2",
    "vegetation": "#63e39a",
    "built-up": "#ffc857",
    "burn-scar": "#f06a6a",
    "change": "#c78cff",
    "unknown": "#8bd0ff",
}


def evidence_class(item: EvidenceItem) -> str:
    declared = str(item.geometry.get("target", "")).lower()
    label = item.label.lower()
    combined = f"{declared} {label}"
    if "water" in combined or "flood" in combined:
        return "water"
    if any(name in combined for name in ("vegetation", "forest", "crop", "ndvi")):
        return "vegetation"
    if any(name in combined for name in ("built", "urban", "building", "ndbi")):
        return "built-up"
    if any(name in combined for name in ("burn", "nbr")):
        return "burn-scar"
    if "change" in combined or "loss" in combined or "gain" in combined:
        return "change"
    return "unknown"


def evidence_color(item: EvidenceItem) -> str:
    return EVIDENCE_PALETTE[evidence_class(item)]


def _ring_area(ring: list[list[float]]) -> float:
    return abs(
        sum(
            left[0] * right[1] - right[0] * left[1]
            for left, right in pairwise(ring)
        )
    ) / 2


def _bounded_ring(ring: list[list[float]], remaining: int) -> list[list[float]]:
    points = ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else ring
    if len(points) < 3 or remaining < 4:
        return []
    allowed = min(512, remaining - 1)
    stride = max(1, math.ceil(len(points) / allowed))
    sampled = points[::stride][:allowed]
    if len(sampled) < 3:
        return []
    return [*sampled, sampled[0]]


def vectorize_mask(
    mask: np.ndarray,
    *,
    class_name: str,
    confidence: float,
    analysis_pixel_area_m2: float | None,
) -> list[dict]:
    """Return a bounded normalized polygon view while retaining the exact binary mask."""
    height, width = mask.shape
    minimum_pixels = max(4.0, mask.size * 0.00002)
    candidates = []
    for geometry, value in shapes(
        mask.astype(np.uint8),
        mask=mask,
        transform=Affine.scale(1 / width, 1 / height),
        connectivity=8,
    ):
        if int(value) != 1 or geometry.get("type") != "Polygon":
            continue
        raw_rings = [
            [[round(float(x), 7), round(float(y), 7)] for x, y in ring]
            for ring in geometry.get("coordinates", [])
        ]
        if not raw_rings:
            continue
        normalized_area = max(
            0.0,
            _ring_area(raw_rings[0]) - sum(_ring_area(ring) for ring in raw_rings[1:]),
        )
        pixel_area = normalized_area * mask.size
        if pixel_area >= minimum_pixels:
            candidates.append((pixel_area, raw_rings))
    candidates.sort(key=lambda item: item[0], reverse=True)

    output = []
    vertices = 0
    for pixel_area, raw_rings in candidates[:MAX_VECTOR_POLYGONS]:
        bounded = []
        for ring in raw_rings:
            clean = _bounded_ring(ring, MAX_VECTOR_VERTICES - vertices)
            if clean:
                bounded.append(clean)
                vertices += len(clean)
        if not bounded:
            break
        output.append(
            {
                "rings": bounded,
                "class": class_name,
                "confidence": round(float(confidence), 6),
                "pixel_area": round(float(pixel_area), 3),
                "area_m2": (
                    round(float(pixel_area * analysis_pixel_area_m2), 3)
                    if analysis_pixel_area_m2 is not None
                    else None
                ),
            }
        )
        if vertices >= MAX_VECTOR_VERTICES:
            break
    return output


SPECTRAL_METHODS = {
    "NDWI (green - NIR) / (green + NIR)": ["green", "nir"],
    "NDVI (NIR - red) / (NIR + red)": ["nir", "red"],
    "NDBI (SWIR1 - NIR) / (SWIR1 + NIR)": ["swir1", "nir"],
    "NBR (NIR - SWIR2) / (NIR + SWIR2)": ["nir", "swir2"],
}


def mask_support(source, shape, method):
    """Shared support for dependent masks and their final reported statistics."""
    height, width = shape
    meanings = SPECTRAL_METHODS.get(method)
    indexes = semantic_indexes(source, meanings) if meanings else visual_indexes(source)
    if indexes is None:
        required = "/".join(meanings or [])
        raise ValueError(f"Spectral comparison requires verified {required} bands at both dates")
    sampling = Resampling.nearest if meanings else Resampling.bilinear
    raw = source.read(
        indexes, out_shape=(len(indexes), height, width), masked=True, resampling=sampling
    ).astype(np.float32)
    values = np.ma.filled(raw, np.nan)
    valid = np.isfinite(values).all(axis=0)
    valid &= source.dataset_mask(out_shape=(height, width), resampling=sampling) > 0
    if meanings:
        for plane, index in enumerate(indexes):
            values[plane] = values[plane] * source.scales[index - 1] + source.offsets[index - 1]
        valid &= (
            np.isfinite(values).all(axis=0)
            & (values >= 0).all(axis=0)
            & (values.sum(axis=0) > 1e-8)
        )
    return valid


def encode_mask(mask: np.ndarray) -> str:
    stream = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def decode_mask(geometry: dict) -> np.ndarray:
    encoded = geometry.get("data", "")
    if (
        geometry.get("encoding") != "png-base64"
        or not isinstance(encoded, str)
        or len(encoded) > MAX_MASK_BYTES * 4 // 3 + 4
    ):
        raise ValueError("Invalid or oversized mask encoding")
    try:
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if (
                image.format != "PNG"
                or image.mode not in {"1", "L"}
                or min(image.size) < 1
                or max(image.size) > MAX_MASK_EDGE
                or list(image.size) != [geometry.get("width"), geometry.get("height")]
            ):
                raise ValueError("Mask must be a bounded binary PNG on the declared grid")
            pixels = np.asarray(image.convert("L"))
            if not np.isin(pixels, [0, 255]).all():
                raise ValueError("Mask must contain only 0 and 255")
            return pixels == 255
    except (binascii.Error, OSError) as exc:
        raise ValueError("Unreadable binary mask") from exc


def spectral_water_output(output, step, assets, asset_store):
    """Replace coarse boxes with a sensor-qualified spectral mask when the query permits it."""
    if (
        step.task != TaskType.GROUNDING
        or len(assets) != 1
        or assets[0].modality == Modality.SAR
    ):
        return output
    asset = assets[0]
    with rasterio.open(asset_store.resolve(asset.id)) as source:
        targets = {str(item).lower() for item in step.permitted_params.get("targets", [])}
        if targets == {"water"}:
            meanings = ["green", "nir"]
            method = "NDWI (green - NIR) / (green + NIR)"
            label = "Water candidate (NDWI)"
            target = "water"
            threshold = float(step.permitted_params.get("water_index_threshold", 0.0))
            comparator = "greater"
        elif targets and targets <= {
            "vegetation",
            "forest",
            "cropland",
            "agriculture",
            "drought",
        }:
            meanings = ["nir", "red"]
            method = "NDVI (NIR - red) / (NIR + red)"
            label = "Vegetation candidate (NDVI)"
            target = "vegetation"
            threshold = float(step.permitted_params.get("threshold", 0.3))
            comparator = "greater"
        elif targets and targets <= {"built-up", "urban", "building"}:
            meanings = ["swir1", "nir"]
            method = "NDBI (SWIR1 - NIR) / (SWIR1 + NIR)"
            label = "Built-up candidate (NDBI)"
            target = "built-up"
            threshold = float(step.permitted_params.get("threshold", 0.0))
            comparator = "greater"
        elif targets and targets <= {"burn", "burn-scar", "burned"}:
            meanings = ["nir", "swir2"]
            method = "NBR (NIR - SWIR2) / (NIR + SWIR2)"
            label = "Low-NBR candidate (single date)"
            target = "burn-scar"
            threshold = float(step.permitted_params.get("threshold", 0.1))
            comparator = "less"
        else:
            return output

        profile = sensor_profile(source)
        if "swir" in " ".join(meanings) and profile["platform"] != "sentinel-2":
            warning = (
                f"{method.split(' ', 1)[0]} unavailable: {profile['platform']} has no verified "
                "SWIR band contract. Cartosat-2-series cannot supply NDBI/NBR."
            )
            return output.model_copy(update={"warnings": [*output.warnings, warning]})
        indexes = semantic_indexes(source, meanings)
        if indexes is None:
            warning = (
                f"{method.split(' ', 1)[0]} unavailable: explicit "
                f"{'/'.join(meanings)} band descriptions are missing; RGB is not NIR/SWIR."
            )
            return output.model_copy(update={"warnings": [*output.warnings, warning]})
        scale = min(1.0, MAX_MASK_EDGE / max(source.width, source.height))
        height, width = max(1, round(source.height * scale)), max(1, round(source.width * scale))
        raw = source.read(
            indexes, out_shape=(2, height, width), masked=True, resampling=Resampling.nearest
        ).astype(np.float32)
        raw = np.ma.filled(raw, np.nan)
        for plane, index in enumerate(indexes):
            raw[plane] = raw[plane] * source.scales[index - 1] + source.offsets[index - 1]
        denominator = raw[0] + raw[1]
        valid = np.isfinite(raw).all(axis=0) & (denominator > 1e-8) & (raw >= 0).all(axis=0)
        index_values = np.divide(
            raw[0] - raw[1], denominator, out=np.zeros_like(denominator), where=valid
        )
        mask = valid & (
            (index_values > threshold) if comparator == "greater" else (index_values < threshold)
        )
    evidence = EvidenceItem(
        id=f"ev_{target.replace('-', '_')}_{method.split(' ', 1)[0].lower()}",
        type="mask",
        label=label,
        score=0.5,
        coordinate_space="pixel",
        asset_id=asset.id,
        geometry={
            "encoding": "png-base64",
            "data": encode_mask(mask),
            "width": width,
            "height": height,
            "method": method,
            "threshold": threshold,
            "comparison": comparator,
            "status": "candidate",
            "target": target,
            "band_indexes": indexes,
            "band_meanings": meanings,
            "validity_basis": "finite nonnegative calibrated spectral bands with positive sum",
        },
    )
    return output.model_copy(
        update={
            "evidence": [evidence],
            "facts": [
                *output.facts,
                {
                    "name": "spectral_mask_method",
                    "value": method.split(" ", 1)[0],
                    "band_meanings": meanings,
                    "band_indexes": indexes,
                    "threshold": threshold,
                },
            ],
            "warnings": [
                *output.warnings,
                f"{method.split(' ', 1)[0]} candidates replace coarse model boxes for this "
                "request. The threshold is a baseline, not a validated universal classifier; "
                "band calibration, atmosphere, shadows, season and land-cover confounders require "
                "review.",
                *(
                    [
                        "Single-date NBR cannot establish a burn scar or severity. Use a "
                        "co-registered pre/post Sentinel-2 pair and dNBR for that claim."
                    ]
                    if target == "burn-scar"
                    else []
                ),
            ],
        }
    )


def materialize_masks(analysis_id, evidence, assets, asset_store, artifacts, api_prefix):
    """Recompute grid statistics and store masks; never follow a model-supplied URL/path."""
    result = []
    source_assets = {asset.id: asset for asset in assets}
    mask_number = 0
    for item in evidence:
        if item.asset_id not in source_assets:
            raise ValueError("Evidence references an asset outside this analysis")
        if item.type != "mask":
            result.append(item.model_copy(update={"artifact_url": None}))
            continue
        mask_number += 1
        if mask_number > 8:
            raise ValueError("At most eight masks are accepted per analysis")
        asset = source_assets[item.asset_id]
        mask = decode_mask(item.geometry)
        height, width = mask.shape
        verified_group = None
        with rasterio.open(asset_store.resolve(asset.id)) as source:
            # Smaller remote grids are allowed, but they must preserve the source aspect ratio.
            if (
                width > source.width
                or height > source.height
                or abs(width / height - source.width / source.height) > 2 / height
            ):
                raise ValueError("Mask dimensions do not match the source raster aspect ratio")
            valid = (
                source.dataset_mask(out_shape=(height, width), resampling=Resampling.nearest) > 0
            )
            spectral_method = item.geometry.get("method")
            if spectral_method in SPECTRAL_METHODS:
                meanings = SPECTRAL_METHODS[spectral_method]
                indexes = item.geometry.get("band_indexes")
                if (
                    not isinstance(indexes, list)
                    or len(indexes) != 2
                    or not all(type(i) is int and 1 <= i <= source.count for i in indexes)
                    or indexes != semantic_indexes(source, meanings)
                ):
                    raise ValueError(
                        f"{spectral_method.split(' ', 1)[0]} support requires verified "
                        f"{'/'.join(meanings)} band indexes"
                    )
                spectral = source.read(
                    indexes,
                    out_shape=(2, height, width),
                    masked=True,
                    resampling=Resampling.nearest,
                ).astype(np.float32)
                spectral = np.ma.filled(spectral, np.nan)
                for plane, index in enumerate(indexes):
                    spectral[plane] = (
                        spectral[plane] * source.scales[index - 1] + source.offsets[index - 1]
                    )
                valid &= (
                    np.isfinite(spectral).all(axis=0)
                    & (spectral >= 0).all(axis=0)
                    & (spectral.sum(axis=0) > 1e-8)
                )
            comparison_id = item.geometry.get("comparison_asset_id")
            if comparison_id is not None:
                if comparison_id not in source_assets or comparison_id == asset.id:
                    raise ValueError("Comparison mask must reference the other analysis asset")
                # Read actual pair bytes using the same band selection, bilinear grid and
                # finite-value policy as the local specialist. Never trust remote counts.
                from rasterio import Affine

                from app.models.pair_tools import ReferenceGrid, _read_on_grid, _shared_valid

                grid = ReferenceGrid(
                    height,
                    width,
                    source.transform @ Affine.scale(source.width / width, source.height / height),
                    source.crs,
                )
                other = source_assets[comparison_id]
                left = _read_on_grid(
                    asset_store.resolve(asset.id), grid, visual=asset.modality != Modality.SAR
                )
                right = _read_on_grid(
                    asset_store.resolve(other.id), grid, visual=other.modality != Modality.SAR
                )
                valid &= _shared_valid(left, right)
                if item.geometry.get("method") == "dependent same-target mask comparison":
                    with rasterio.open(asset_store.resolve(other.id)) as other_source:
                        support_method = item.geometry.get("support_method")
                        valid = mask_support(source, (height, width), support_method)
                        valid &= mask_support(other_source, (height, width), support_method)
            mask &= valid
            if comparison_id is not None:
                # Derive grouping from bytes and the actual grid; never trust a model's group.
                descriptor = json.dumps(
                    [
                        sorted([asset.id, comparison_id]),
                        width,
                        height,
                        list(grid.transform),
                        str(grid.crs),
                    ],
                    sort_keys=True,
                ).encode()
                verified_group = hashlib.sha256(descriptor + mask.tobytes()).hexdigest()
        foreground = int(mask.sum())
        grid_area = pixel_area_m2(asset.metadata)
        analysis_pixel_area = (
            grid_area * asset.metadata.width * asset.metadata.height / mask.size
            if grid_area is not None
            else None
        )
        mask_id = f"ev_mask_{mask_number}"
        path = artifacts.allocate(analysis_id, f"{mask_id}.png")
        Image.fromarray(mask.astype(np.uint8) * 255).save(path, format="PNG")
        geometry = {
            key: value
            for key, value in item.geometry.items()
            if key not in {"data", "encoding", "mask_group"}
        }
        if verified_group:
            geometry["mask_group"] = verified_group
        class_name = evidence_class(item)
        geometry.update(
            {
                "width": width,
                "height": height,
                "foreground_pixels": foreground,
                "valid_pixels": int(valid.sum()),
                "coverage_percent": 100 * foreground / max(1, int(valid.sum())),
                "area_m2": foreground * analysis_pixel_area if analysis_pixel_area else None,
                "analysis_pixel_area_m2": analysis_pixel_area,
                "source_width": asset.metadata.width,
                "source_height": asset.metadata.height,
                "resampled": width != asset.metadata.width or height != asset.metadata.height,
                "status": "candidate",
                "encoding": "binary-png-artifact",
                "class_name": class_name,
                "color_hex": EVIDENCE_PALETTE[class_name],
                "polygons": vectorize_mask(
                    mask,
                    class_name=class_name,
                    confidence=item.score,
                    analysis_pixel_area_m2=analysis_pixel_area,
                ),
            }
        )
        result.append(
            item.model_copy(
                update={
                    "id": mask_id,
                    "geometry": geometry,
                    "artifact_url": f"{api_prefix}/analyses/{analysis_id}/masks/{mask_id}",
                }
            )
        )
    return result
