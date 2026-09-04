"""Dependent, deterministic measurements of candidate segmentation extent, not water level."""

from __future__ import annotations

import numpy as np
import rasterio

from app.core.scene_report import pixel_area_m2
from app.models.masks import decode_mask, encode_mask, mask_support
from app.schemas import AssetRole, EvidenceItem, SpecialistOutput, TaskType


def compare_mask_extent(step, dependencies, assets, asset_store):
    warnings = [
        "Candidate masks are not ground truth; clouds, shadows and model errors affect counts.",
        "Water surface extent cannot establish water level/depth, timing or physical cause.",
    ]

    def abstain(reason):
        return SpecialistOutput(
            task=TaskType.CHANGE_VQA,
            text=f"Extent measurement withheld: {reason}",
            raw_score=0,
            score_kind="uncalibrated",
            model_version="mask-extent-v1:abstained",
            warnings=[*warnings, reason],
        )

    if len(step.depends_on) != 2 or any(key not in dependencies for key in step.depends_on):
        return abstain("both prior segmentation outputs are required")
    before = next(a for a in assets if a.role == AssetRole.TIME_A)
    after = next(a for a in assets if a.role == AssetRole.TIME_B)
    target = (step.permitted_params.get("targets") or [None])[0]
    masks, methods = [], []
    try:
        for key, asset in zip(step.depends_on, [before, after], strict=True):
            output = dependencies[key]
            items = [
                item
                for item in output.evidence
                if item.type == "mask"
                and item.asset_id == asset.id
                and item.geometry.get("target") == target
            ]
            if not items or not target:
                return abstain("missing mask evidence is not evidence of an empty target")
            method = {item.geometry.get("method") for item in items}
            if len(method) != 1 or None in method:
                return abstain("mask methods must be declared and consistent")
            decoded = [decode_mask(item.geometry) for item in items]
            if len({mask.shape for mask in decoded}) != 1:
                return abstain("target masks use inconsistent grids")
            masks.append(np.logical_or.reduce(decoded))
            thresholds = {item.geometry.get("threshold") for item in items}
            if len(thresholds) != 1:
                return abstain("segmentation thresholds differ within a date")
            methods.append((next(iter(method)), output.model_version, next(iter(thresholds))))
        if methods[0] != methods[1]:
            return abstain("segmentation method/model differs between dates")
        if masks[0].shape != masks[1].shape:
            return abstain("mask grids differ; resample imagery onto one grid before segmentation")
        height, width = masks[0].shape
        with (
            rasterio.open(asset_store.resolve(before.id)) as a,
            rasterio.open(asset_store.resolve(after.id)) as b,
        ):
            if (a.width, a.height, a.crs) != (
                b.width,
                b.height,
                b.crs,
            ) or not a.transform.almost_equals(b.transform):
                return abstain(
                    "source grids are not identical; co-register before measuring pixel change"
                )
            if (
                width > a.width
                or height > a.height
                or abs(width / height - a.width / a.height) > 2 / height
            ):
                return abstain("mask aspect ratio does not match the source")
            valid = mask_support(a, (height, width), methods[0][0]) & mask_support(
                b, (height, width), methods[0][0]
            )
            if not valid.any():
                return abstain("no shared valid pixels")
        left, right = masks[0] & valid, masks[1] & valid
        lost, gained, retained = left & ~right, right & ~left, left & right
        counts = {
            "before": int(left.sum()),
            "after": int(right.sum()),
            "lost": int(lost.sum()),
            "gained": int(gained.sum()),
            "retained": int(retained.sum()),
            "shared_valid": int(valid.sum()),
        }
        counts["net_change"] = counts["after"] - counts["before"]
        source_area = pixel_area_m2(before.metadata)
        grid_pixel_area = (
            source_area * before.metadata.width * before.metadata.height / left.size
            if source_area
            else None
        )
        areas = (
            {name: count * grid_pixel_area for name, count in counts.items()}
            if grid_pixel_area
            else None
        )
        evidence = [
            EvidenceItem(
                id=f"ev_extent_{label}",
                type="mask",
                label=f"{target} candidate {label}",
                score=0.5,
                coordinate_space="pixel",
                asset_id=after.id,
                geometry={
                    "encoding": "png-base64",
                    "data": encode_mask(mask),
                    "width": width,
                    "height": height,
                    "method": "dependent same-target mask comparison",
                    "status": "candidate",
                    "target": target,
                    "support_method": methods[0][0],
                    "comparison_asset_id": before.id,
                },
            )
            for label, mask in [("loss", lost), ("gain", gained)]
        ]
        area_text = (
            f" Estimated gross loss {areas['lost']:,.1f} m²; gain {areas['gained']:,.1f} m²."
            " These are projected-grid estimates."
            if areas
            else " No reliable metric-area grid: counts are pixels, not square metres."
        )
        return SpecialistOutput(
            task=TaskType.CHANGE_VQA,
            text=f"Candidate {target}: {counts['before']:,} before; {counts['after']:,} after. "
            f"Gross loss {counts['lost']:,}; gain {counts['gained']:,}; "
            f"net change {counts['net_change']:+,} pixels." + area_text,
            facts=[
                {
                    "name": "target_extent_change",
                    "target": target,
                    "pixel_counts": counts,
                    "estimated_area_m2": areas,
                    "grid": [width, height],
                    "method": methods[0][0],
                    "units": "analysis-grid pixels",
                    "date_status": "unverified",
                }
            ],
            evidence=evidence,
            raw_score=0.5,
            score_kind="uncalibrated",
            model_version="mask-extent-v1",
            warnings=warnings,
        )
    except (ValueError, TypeError, rasterio.errors.RasterioError):
        return abstain("mask decoding or raster support validation failed")
