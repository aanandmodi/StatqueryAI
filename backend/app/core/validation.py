from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import rasterio
from PIL import Image, UnidentifiedImageError
from rasterio.errors import RasterioIOError

from app.config import Settings
from app.errors import ValidationFailure
from app.schemas import (
    AssetRecord,
    AssetRole,
    Modality,
    RasterMetadata,
    RegistrationBasis,
    TaskType,
)

TIFF_SIGNATURES = (b"II*\x00", b"MM\x00*")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"
BENCHMARK_DATASETS = {"vrsbench", "rsvqa", "cdvqa", "second"}


@dataclass(frozen=True)
class PairCheck:
    overlap_ratio: float
    resolution_delta: float
    grid_offset_pixels: float


class RasterValidator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def inspect(
        self,
        path: Path,
        *,
        modality: Modality,
        source_dataset: str | None,
        allow_image_grid: bool = False,
        exploration: bool = False,
    ) -> RasterMetadata:
        kind = self._sniff(path)
        benchmark = (source_dataset or "").lower() in BENCHMARK_DATASETS

        if (
            kind != "tiff"
            and not (self.settings.allow_benchmark_images and benchmark and kind in {"png", "jpeg"})
            and not exploration
        ):
            raise ValidationFailure(
                "GeoTIFF/TIFF is required; PNG/JPEG is allowed only for named benchmarks",
                details={"detected_format": kind, "source_dataset": source_dataset},
            )
        if kind not in {"tiff", "png", "jpeg", "webp"}:
            raise ValidationFailure("Unsupported or unrecognized raster file")
        if kind != "tiff":
            if modality != Modality.OPTICAL:
                raise ValidationFailure(
                    "JPG/PNG/WebP exploration accepts optical images only. "
                    "SAR/multispectral analysis needs original sensor TIFF data."
                )
            try:
                with Image.open(path) as image:
                    if image.width * image.height > min(
                        self.settings.max_raster_pixels, 40_000_000
                    ):
                        raise ValidationFailure(
                            "Display image exceeds the 40-megapixel safety limit"
                        )
                    if image.mode not in {"RGB", "L"}:
                        raise ValidationFailure(
                            "Export this image as RGB PNG/JPEG/WebP first. Palette, CMYK and "
                            "alpha-channel images are not accepted because their "
                            "pixel interpretation differs."
                        )
                    if getattr(image, "n_frames", 1) != 1:
                        raise ValidationFailure("Animated or multi-frame images are not supported")
                    if image.getexif().get(274, 1) != 1:
                        raise ValidationFailure(
                            "This image uses EXIF rotation. Export an upright image "
                            "with orientation "
                            "applied before upload so masks and paired pixels align."
                        )
                # PNG EXIF access can load/close its parser; verify on a fresh decoder.
                with Image.open(path) as image:
                    image.verify()
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                raise ValidationFailure("The image could not be decoded safely") from exc
        if modality == Modality.UNKNOWN:
            raise ValidationFailure(
                "Raster modality must be declared; it cannot be inferred safely from .tif"
            )

        try:
            with (
                rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"),
                rasterio.open(path) as dataset,
            ):
                pixel_count = dataset.width * dataset.height
                if pixel_count > self.settings.max_raster_pixels:
                    raise ValidationFailure(
                        "Raster dimensions exceed the configured safety limit",
                        details={
                            "pixels": pixel_count,
                            "limit": self.settings.max_raster_pixels,
                        },
                    )
                if dataset.count < 1 or dataset.count > self.settings.max_raster_bands:
                    raise ValidationFailure(
                        "Raster band count is outside the permitted range",
                        details={"bands": dataset.count},
                    )

                warnings: list[str] = []
                crs = dataset.crs.to_string() if dataset.crs else None
                if kind == "tiff" and not crs:
                    if not allow_image_grid and not exploration:
                        raise ValidationFailure(
                            "TIFF is missing a CRS. For an already aligned before/after pair, "
                            "enable image-grid comparison; otherwise upload a real GeoTIFF."
                        )
                    if not dataset.transform.is_identity:
                        raise ValidationFailure(
                            "TIFF has an affine transform but no CRS; provide complete "
                            "georeferencing instead of declaring an image-grid pair"
                        )
                    warnings.append(
                        "No CRS/geotransform: image coordinates only; geographic coordinates "
                        "and metric area are unavailable. Pair alignment needs "
                        "explicit declaration."
                    )
                if kind == "tiff" and crs and dataset.transform.is_identity:
                    raise ValidationFailure("GeoTIFF is missing a meaningful affine transform")
                geo_numbers = [*tuple(dataset.transform)[:6], *dataset.bounds, *dataset.res]
                if not all(isfinite(value) for value in geo_numbers):
                    raise ValidationFailure("Raster geospatial metadata contains non-finite values")
                if dataset.transform.determinant == 0:
                    raise ValidationFailure("Raster affine transform is not invertible")
                nodata = dataset.nodata
                if dataset.nodata is None:
                    warnings.append("nodata value is not declared")
                elif not isfinite(dataset.nodata):
                    # JSON forbids NaN/Infinity. Actual masking still uses original TIFF bytes.
                    warnings.append(
                        f"Non-finite nodata sentinel ({dataset.nodata}) is represented as null "
                        "in JSON; source raster masks remain authoritative"
                    )
                    nodata = None

                tags = {
                    str(key)[:80]: str(value)[:500]
                    for key, value in list(dataset.tags().items())[:80]
                }
                if exploration:
                    warnings.append(
                        "Exploration input: not a prescribed SIH benchmark. Compressed/display "
                        "imagery does not establish spectral calibration, species, "
                        "event time or cause."
                    )
                if not crs and kind != "tiff":
                    warnings.append(
                        "Image has no georeferencing; evidence uses pixel coordinates only. "
                        "No metric area, historical alignment or location is inferred from it."
                    )
                quality = max(0.0, 1.0 - 0.06 * len(warnings))
                transform = list(tuple(dataset.transform)[:6])
                return RasterMetadata(
                    driver=dataset.driver,
                    width=dataset.width,
                    height=dataset.height,
                    count=dataset.count,
                    dtypes=list(dataset.dtypes),
                    crs=crs,
                    transform=transform,
                    bounds=list(dataset.bounds),
                    resolution=[abs(dataset.res[0]), abs(dataset.res[1])],
                    nodata=nodata,
                    tags=tags,
                    warnings=warnings,
                    quality_score=quality,
                )
        except ValidationFailure:
            raise
        except RasterioIOError as exc:
            raise ValidationFailure(
                "Raster cannot be opened safely", details={"reason": str(exc)}
            ) from exc

    def validate_for_task(self, task: TaskType, assets: list[AssetRecord]) -> list[str]:
        invalid = [asset.id for asset in assets if not asset.valid]
        if invalid:
            raise ValidationFailure(
                "One or more assets failed validation", details={"asset_ids": invalid}
            )

        if task in {TaskType.SINGLE_VQA, TaskType.CAPTION, TaskType.GROUNDING}:
            if len(assets) != 1:
                raise ValidationFailure(f"{task.value} requires exactly one image")
            return []

        if len(assets) != 2:
            raise ValidationFailure(f"{task.value} requires exactly two co-registered images")

        modalities = {asset.modality for asset in assets}
        left, right = assets
        if task == TaskType.CHANGE_VQA:
            roles = {asset.role for asset in assets}
            if roles != {AssetRole.TIME_A, AssetRole.TIME_B}:
                raise ValidationFailure(
                    "Change analysis requires explicit time_a and time_b asset roles"
                )
            if len(modalities) != 1:
                raise ValidationFailure(
                    "Temporal change analysis requires the same declared modality at both dates",
                    details={"modalities": sorted(modality.value for modality in modalities)},
                )
            if left.metadata and right.metadata and left.metadata.count != right.metadata.count:
                raise ValidationFailure(
                    "Temporal change analysis requires matching band counts and band meanings",
                    details={
                        "time_a_bands": left.metadata.count,
                        "time_b_bands": right.metadata.count,
                    },
                )
        if task == TaskType.OPTICAL_SAR_FUSION:
            has_optical = bool(modalities & {Modality.OPTICAL, Modality.MULTISPECTRAL})
            if not has_optical or Modality.SAR not in modalities:
                raise ValidationFailure(
                    "Optical-SAR fusion requires one optical/multispectral and one SAR asset"
                )

        warnings: list[str] = []
        a, b = left.metadata, right.metadata
        if a and b and not a.crs and not b.crs:
            if task == TaskType.CHANGE_VQA and self._is_benchmark_pair(left, right):
                warnings.append(
                    "Benchmark pair has matching dimensions but no geographic metadata; "
                    "co-registration is assumed from the benchmark and not independently verified"
                )
                return warnings
            if task != TaskType.CHANGE_VQA or not self._is_declared_image_grid_pair(left, right):
                raise ValidationFailure(
                    "Unreferenced temporal images require an explicit pixel-grid declaration "
                    "and identical dimensions, bands and sample types"
                )
            warnings.append(
                "Both images lack georeferencing. Pixel-for-pixel alignment was declared by the "
                "user and dimensions/bands match, but registration was not independently verified; "
                "geographic coordinates and metric area are unavailable"
            )
            return warnings

        check = self.check_pair(left, right)
        if check.overlap_ratio < 0.98:
            raise ValidationFailure(
                "Paired rasters do not cover the same area",
                details={"overlap_ratio": round(check.overlap_ratio, 4), "minimum": 0.98},
            )
        if check.resolution_delta > 0.02:
            raise ValidationFailure(
                "Paired rasters have incompatible pixel resolutions",
                details={"relative_delta": round(check.resolution_delta, 4), "maximum": 0.02},
            )
        if check.grid_offset_pixels > 0.25:
            raise ValidationFailure(
                "Paired rasters are not aligned to the same pixel grid",
                details={"grid_offset_pixels": round(check.grid_offset_pixels, 4), "maximum": 0.25},
            )
        if (
            left.metadata
            and right.metadata
            and (
                left.metadata.width != right.metadata.width
                or left.metadata.height != right.metadata.height
            )
        ):
            warnings.append("pair requires bounded reprojection/cropping to a shared grid")
        return warnings

    def check_pair(self, left: AssetRecord, right: AssetRecord) -> PairCheck:
        a = left.metadata
        b = right.metadata
        if not a or not b or not a.crs or not b.crs:
            raise ValidationFailure("Pair is missing CRS metadata")
        if a.crs != b.crs:
            raise ValidationFailure(
                "Paired rasters use different CRS values",
                details={"left_crs": a.crs, "right_crs": b.crs},
            )

        for metadata in (a, b):
            if (
                len(metadata.transform) != 6
                or len(metadata.resolution) != 2
                or not all(isfinite(value) for value in metadata.transform + metadata.resolution)
                or min(metadata.resolution) <= 0
            ):
                raise ValidationFailure("Pair contains invalid affine or resolution metadata")
            transform = metadata.transform
            tolerance = max(metadata.resolution) * 1e-9
            if (
                abs(transform[1]) > tolerance
                or abs(transform[3]) > tolerance
                or transform[0] <= 0
                or transform[4] >= 0
            ):
                raise ValidationFailure(
                    "Pair validation requires north-up rasters; reproject rotated, sheared, "
                    "or mirrored inputs onto a common north-up grid before upload"
                )

        overlap = self._overlap_ratio(a.bounds, b.bounds)
        res_delta = max(
            abs(a.resolution[0] - b.resolution[0]) / max(a.resolution[0], b.resolution[0]),
            abs(a.resolution[1] - b.resolution[1]) / max(a.resolution[1], b.resolution[1]),
        )
        x_offset = abs(a.transform[2] - b.transform[2]) / max(a.resolution[0], 1e-12)
        y_offset = abs(a.transform[5] - b.transform[5]) / max(a.resolution[1], 1e-12)
        fractional_offset = max(abs(x_offset - round(x_offset)), abs(y_offset - round(y_offset)))
        return PairCheck(overlap, res_delta, fractional_offset)

    @staticmethod
    def _sniff(path: Path) -> str:
        with path.open("rb") as handle:
            header = handle.read(12)
        if header.startswith(TIFF_SIGNATURES):
            return "tiff"
        if header.startswith(PNG_SIGNATURE):
            return "png"
        if header.startswith(JPEG_SIGNATURE):
            return "jpeg"
        if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
            return "webp"
        return "unknown"

    @staticmethod
    def _is_benchmark_pair(left: AssetRecord, right: AssetRecord) -> bool:
        left_dataset = (left.source_dataset or "").strip().lower()
        right_dataset = (right.source_dataset or "").strip().lower()
        a, b = left.metadata, right.metadata
        return bool(
            left_dataset in BENCHMARK_DATASETS
            and left_dataset == right_dataset
            and a
            and b
            and not a.crs
            and not b.crs
            and a.width == b.width
            and a.height == b.height
        )

    @staticmethod
    def _is_declared_image_grid_pair(left: AssetRecord, right: AssetRecord) -> bool:
        a, b = left.metadata, right.metadata
        return bool(
            left.registration_basis == RegistrationBasis.PIXEL_GRID
            and right.registration_basis == RegistrationBasis.PIXEL_GRID
            and {left.role, right.role} == {AssetRole.TIME_A, AssetRole.TIME_B}
            and a
            and b
            and not a.crs
            and not b.crs
            and a.width == b.width
            and a.height == b.height
            and a.count == b.count
            and a.dtypes == b.dtypes
            and a.transform == b.transform == [1, 0, 0, 0, 1, 0]
            and left.input_profile == right.input_profile
        )

    @staticmethod
    def _overlap_ratio(a: Iterable[float], b: Iterable[float]) -> float:
        a_left, a_bottom, a_right, a_top = list(a)
        b_left, b_bottom, b_right, b_top = list(b)
        width = max(0.0, min(a_right, b_right) - max(a_left, b_left))
        height = max(0.0, min(a_top, b_top) - max(a_bottom, b_bottom))
        intersection = width * height
        larger_area = max(
            max(0.0, a_right - a_left) * max(0.0, a_top - a_bottom),
            max(0.0, b_right - b_left) * max(0.0, b_top - b_bottom),
        )
        # Both footprints must be covered; a tiny crop inside a large scene is
        # not an aligned full-scene pair even though the crop is fully covered.
        return intersection / larger_area if larger_area else 0.0
