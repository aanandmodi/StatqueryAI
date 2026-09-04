"""Deterministic scene context. Never expand short model labels into invented observations."""

from __future__ import annotations

from rasterio.crs import CRS
from rasterio.warp import transform_bounds

from app.schemas import AssetRecord, ReportSection


def pixel_area_m2(metadata) -> float | None:
    """Projected grid area, not surveyed surface area. Geographic degrees are NOT metres."""
    if not metadata or not metadata.crs:
        return None
    crs = CRS.from_user_input(metadata.crs)
    if not crs.is_projected:
        return None
    _, factor = crs.linear_units_factor
    a, b, _, d, e, _ = metadata.transform[:6]
    return abs(a * e - b * d) * factor**2


def build_scene_sections(
    assets: list[AssetRecord], answer: str, evidence: list
) -> list[ReportSection]:
    sections = []
    for asset in assets:
        meta = asset.metadata
        if meta is None:
            continue
        observations = [
            f"{asset.original_name}: {meta.width:,} x {meta.height:,} pixels, "
            f"{meta.count} band(s), {', '.join(sorted(set(meta.dtypes)))} samples.",
            f"Declared modality: {asset.modality.value}. "
            f"Raster CRS: {meta.crs or 'not available'}.",
        ]
        if meta.crs:
            observations.append(
                f"Pixel spacing: {meta.resolution[0]:g} x {meta.resolution[1]:g} "
                "in the raster CRS units. Spatial resolution limits the smallest "
                "distinguishable feature."
            )
        else:
            observations.append(
                "Image coordinates only: this file supplies no CRS or meaningful affine "
                "transform. Pixel positions are not geographic coordinates; "
                "metric distance/area cannot be calculated."
            )
        area = pixel_area_m2(meta)
        if area is not None:
            observations.append(
                f"Raster footprint: {area * meta.width * meta.height / 1e6:,.3f} km² "
                "of projected grid area, including any no-data pixels; not land-cover area."
            )
        if meta.crs:
            try:
                west, south, east, north = transform_bounds(meta.crs, "EPSG:4326", *meta.bounds)
                observations.append(
                    f"Raster-derived bounding extent (WGS84): longitude {west:.5f} to {east:.5f}, "
                    f"latitude {south:.5f} to {north:.5f}. This is file georeferencing, not a "
                    "location guessed by the language model."
                )
            except Exception:
                observations.append("The raster extent could not be transformed to WGS84.")
        sections.append(
            ReportSection(
                title="Scene and geographic extent",
                source="raster metadata",
                paragraphs=observations,
            )
        )
    mask_items = []
    groups = set()
    for item in evidence:
        if item.type != "mask":
            continue
        group = item.geometry.get("mask_group", item.id)
        if not isinstance(group, str) or not group or len(group) > 128:
            group = item.id
        if group not in groups:
            mask_items.append(item)
            groups.add(group)
    if mask_items:
        sections.append(
            ReportSection(
                title="Pixel-mask measurements",
                source="computed from candidate masks",
                paragraphs=[
                    f"{item.label}: {item.geometry.get('foreground_pixels', 0):,} "
                    "selected pixels on "
                    f"a {item.geometry['width']} x {item.geometry['height']} analysis grid; "
                    f"{item.geometry.get('coverage_percent', 0):.2f}% "
                    "of valid analysis-grid pixels. "
                    + (
                        "Estimated selected projected area: "
                        f"{item.geometry['area_m2'] / 1e6:.4f} km². "
                        if item.geometry.get("area_m2") is not None
                        else "Metric area is unavailable. "
                    )
                    + f"Method: {item.geometry.get('method', 'unspecified')}. "
                    "These measurements describe the predicted mask, not verified ground truth. "
                    "A paired mask repeated on both dates is counted once, not summed."
                    for item in mask_items
                ],
            )
        )
    temporal_pair = {asset.role.value for asset in assets} == {"time_a", "time_b"}
    sections.append(
        ReportSection(
            title="Evidence readiness",
            source="input and method audit",
            paragraphs=[
                "Spatial basis: "
                + (
                    "file georeferencing is present; registration still needs scene review."
                    if all(a.metadata and a.metadata.crs for a in assets)
                    else "image coordinates only. Pixel-grid alignment, where used, "
                    "is user-declared, not independently verified."
                ),
                "Input profile: "
                + ", ".join(sorted({a.input_profile.value for a in assets}))
                + ". Acquisition dates, sensor calibration and cloud masks "
                "are not verified by filenames.",
                "Accuracy status: no scene-specific labeled ground truth was supplied. "
                "Model scores are not calibrated correctness probabilities; "
                "measurable mask coverage is not accuracy.",
            ],
        )
    )
    sections.append(
        ReportSection(
            title="Questions this evidence cannot settle",
            source="scope and verification requirements",
            paragraphs=[
                "Vegetation: broad visible cover can be described, but species, forest health "
                "and legal forest classification need appropriate resolution, spectral/seasonal "
                "data and field verification.",
                "Weather and timing: visible cloud or haze is a scene observation, not a record "
                "of rainfall, temperature or the event date. Confirm the acquisition timestamp "
                "and consult sourced weather data.",
                "Cause and impact: flooding, erosion, logging or construction are hypotheses "
                "until supported by comparable dates and independent sources. Do not infer "
                "casualties, ownership or infrastructure damage.",
            ],
        )
    )
    limitations = [
        "The visual interpretation is an unverified model observation, not a measured land-cover "
        "inventory. Numerical area and coverage claims belong only in the computed-mask section.",
        (
            "A before/after pixel difference does not establish the cause or semantic class of "
            "change. Cloud, shadow, season, illumination, sensor and registration differences "
            "require separate evidence."
            if temporal_pair
            else "A single image cannot establish change over time, water depth, drinking-water "
            "quality, pollution, ownership or flood history. These require other evidence."
        ),
        "No pixel-level precision, recall, IoU or boundary accuracy has been established for this "
        "image. Shadows, cloud, vegetation, mixed pixels and out-of-domain imagery "
        "can mislead models.",
    ]
    if len(answer.split()) < 35:
        limitations.insert(
            0,
            "The specialist returned a short observation, not a detailed interpretation. "
            "The application has not invented a longer visual explanation. Use the "
            "updated Kaggle quality cell for the instruction-model narrative.",
        )
    if not evidence:
        limitations.insert(
            0,
            "No spatial evidence was returned. The preview is unmarked; "
            "absence of a mask does not establish absence of the requested feature.",
        )
    elif not mask_items:
        limitations.insert(
            0,
            "Only coarse region boxes are available, not pixel-level boundaries. "
            "Install the Kaggle quality update to enable SAM 2 candidate masks.",
        )
    sections.append(
        ReportSection(
            title="Limits and verification", source="system assessment", paragraphs=limitations
        )
    )
    sections.append(
        ReportSection(
            title="Recommended verification",
            source="review procedure",
            paragraphs=[
                "Compare candidate boundaries with the original-resolution imagery. "
                "For water mapping, prefer correctly labelled green/NIR or green/SWIR bands "
                "and inspect cloud/shadow masks.",
                "Annotate representative held-out scenes, including no-water scenes, "
                "and measure pixel "
                "precision, recall, IoU and boundary errors before using masks for decisions. "
                "Do not infer accuracy from a plausible-looking overlay or from a model's "
                "confidence score.",
            ],
        )
    )
    return sections
