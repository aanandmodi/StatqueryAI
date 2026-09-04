"""Bounded binary masks: preserve holes, exclude no-data, and keep evidence on its source grid."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling

from app.core.scene_report import pixel_area_m2
<<<<<<< HEAD
from app.core.sensors import semantic_indexes, visual_indexes
=======
>>>>>>> 2f620623f8897788bd2df2ce4f5700cb183d84f8
from app.schemas import EvidenceItem, Modality, TaskType

MAX_MASK_EDGE = 2048
MAX_MASK_BYTES = 1_000_000


<<<<<<< HEAD
def mask_support(source, shape, method):
    """Shared support for dependent masks and their final reported statistics."""
    height, width = shape
    ndwi = method == "NDWI (green - NIR) / (green + NIR)"
    indexes = semantic_indexes(source, ["green", "nir"]) if ndwi else visual_indexes(source)
    if indexes is None:
        raise ValueError("NDWI comparison requires verified green/NIR bands at both dates")
    sampling = Resampling.nearest if ndwi else Resampling.bilinear
    raw = source.read(
        indexes, out_shape=(len(indexes), height, width), masked=True, resampling=sampling
    ).astype(np.float32)
    values = np.ma.filled(raw, np.nan)
    valid = np.isfinite(values).all(axis=0)
    valid &= source.dataset_mask(out_shape=(height, width), resampling=sampling) > 0
    if ndwi:
        for plane, index in enumerate(indexes):
            values[plane] = values[plane] * source.scales[index - 1] + source.offsets[index - 1]
        valid &= (
            np.isfinite(values).all(axis=0)
            & (values >= 0).all(axis=0)
            & (values.sum(axis=0) > 1e-8)
        )
    return valid


=======
>>>>>>> 2f620623f8897788bd2df2ce4f5700cb183d84f8
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
    """Use NDWI ONLY with explicitly named green and NIR bands. Never guess NIR from RGB."""
    if (
        step.task != TaskType.GROUNDING
        or len(assets) != 1
        or step.permitted_params.get("targets", []) != ["water"]
        or assets[0].modality == Modality.SAR
    ):
        return output
    asset = assets[0]
    with rasterio.open(asset_store.resolve(asset.id)) as source:
<<<<<<< HEAD
        indexes = semantic_indexes(source, ["green", "nir"])
        if indexes is None:
=======
        names = [str(name or "").lower().strip() for name in source.descriptions]
        green = [i + 1 for i, name in enumerate(names) if name in {"green", "b03", "b3"}]
        nir = [
            i + 1 for i, name in enumerate(names) if name in {"nir", "near infrared", "b08", "b8"}
        ]
        if len(green) != 1 or len(nir) != 1:
>>>>>>> 2f620623f8897788bd2df2ce4f5700cb183d84f8
            warning = (
                "NDWI unavailable: explicit green and NIR band descriptions are missing; "
                "RGB is not NIR."
            )
            return output.model_copy(update={"warnings": [*output.warnings, warning]})
<<<<<<< HEAD
        green, nir = [indexes[0]], [indexes[1]]
=======
>>>>>>> 2f620623f8897788bd2df2ce4f5700cb183d84f8
        scale = min(1.0, MAX_MASK_EDGE / max(source.width, source.height))
        height, width = max(1, round(source.height * scale)), max(1, round(source.width * scale))
        indexes = [green[0], nir[0]]
        raw = source.read(
            indexes, out_shape=(2, height, width), masked=True, resampling=Resampling.nearest
        ).astype(np.float32)
        raw = np.ma.filled(raw, np.nan)
        for plane, index in enumerate(indexes):
            raw[plane] = raw[plane] * source.scales[index - 1] + source.offsets[index - 1]
        denominator = raw[0] + raw[1]
        valid = np.isfinite(raw).all(axis=0) & (denominator > 1e-8) & (raw >= 0).all(axis=0)
        ndwi = np.divide(raw[0] - raw[1], denominator, out=np.zeros_like(denominator), where=valid)
        threshold = float(step.permitted_params.get("water_index_threshold", 0.0))
        mask = valid & (ndwi > threshold)
    evidence = EvidenceItem(
        id="ev_water_ndwi",
        type="mask",
        label="Water candidate (NDWI)",
        score=0.5,
        coordinate_space="pixel",
        asset_id=asset.id,
        geometry={
            "encoding": "png-base64",
            "data": encode_mask(mask),
            "width": width,
            "height": height,
            "method": "NDWI (green - NIR) / (green + NIR)",
            "threshold": threshold,
            "status": "candidate",
<<<<<<< HEAD
            "target": "water",
=======
>>>>>>> 2f620623f8897788bd2df2ce4f5700cb183d84f8
            "band_indexes": indexes,
            "validity_basis": "finite nonnegative calibrated green/NIR with positive sum",
        },
    )
    return output.model_copy(
        update={
            "evidence": [evidence],
            "facts": [
                *output.facts,
                {
                    "name": "water_mask_method",
                    "value": "NDWI-v1",
                    "green_band": green[0],
                    "nir_band": nir[0],
                    "threshold": threshold,
                },
            ],
            "warnings": [
                *output.warnings,
                "NDWI candidates replace coarse model boxes for this water request. "
                "A threshold of zero is a baseline, not a validated universal classifier. "
                "Band calibration, shadows, snow, buildings and cloud require review.",
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
            if item.geometry.get("method") == "NDWI (green - NIR) / (green + NIR)":
                indexes = item.geometry.get("band_indexes")
<<<<<<< HEAD
=======
                names = [str(name or "").lower().strip() for name in source.descriptions]
>>>>>>> 2f620623f8897788bd2df2ce4f5700cb183d84f8
                if (
                    not isinstance(indexes, list)
                    or len(indexes) != 2
                    or not all(type(i) is int and 1 <= i <= source.count for i in indexes)
<<<<<<< HEAD
                    or indexes != semantic_indexes(source, ["green", "nir"])
=======
                    or names[indexes[0] - 1] not in {"green", "b03", "b3"}
                    or names[indexes[1] - 1] not in {"nir", "near infrared", "b08", "b8"}
>>>>>>> 2f620623f8897788bd2df2ce4f5700cb183d84f8
                ):
                    raise ValueError("NDWI support requires verified green/NIR band indexes")
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
<<<<<<< HEAD
                if item.geometry.get("method") == "dependent same-target mask comparison":
                    with rasterio.open(asset_store.resolve(other.id)) as other_source:
                        support_method = item.geometry.get("support_method")
                        valid = mask_support(source, (height, width), support_method)
                        valid &= mask_support(other_source, (height, width), support_method)
=======
>>>>>>> 2f620623f8897788bd2df2ce4f5700cb183d84f8
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
