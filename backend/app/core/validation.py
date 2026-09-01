from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import rasterio
from rasterio.errors import RasterioIOError

from app.config import Settings
from app.errors import ValidationFailure
from app.schemas import AssetRecord, Modality, RasterMetadata, TaskType

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
    ) -> RasterMetadata:
        kind = self._sniff(path)
        benchmark = (source_dataset or "").lower() in BENCHMARK_DATASETS

        if kind != "tiff" and not (self.settings.allow_benchmark_images and benchmark):
            raise ValidationFailure(
                "GeoTIFF/TIFF is required; PNG/JPEG is allowed only for named benchmarks",
                details={"detected_format": kind, "source_dataset": source_dataset},
            )
        if kind not in {"tiff", "png", "jpeg"}:
            raise ValidationFailure("Unsupported or unrecognized raster file")
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
                    raise ValidationFailure("GeoTIFF is missing a CRS")
                if kind == "tiff" and dataset.transform.is_identity:
                    raise ValidationFailure("GeoTIFF is missing a meaningful affine transform")
                if dataset.nodata is None:
                    warnings.append("nodata value is not declared")

                tags = {
                    str(key)[:80]: str(value)[:500]
                    for key, value in list(dataset.tags().items())[:80]
                }
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
                    nodata=dataset.nodata,
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
        if task == TaskType.OPTICAL_SAR_FUSION:
            has_optical = bool(modalities & {Modality.OPTICAL, Modality.MULTISPECTRAL})
            if not has_optical or Modality.SAR not in modalities:
                raise ValidationFailure(
                    "Optical-SAR fusion requires one optical/multispectral and one SAR asset"
                )

        warnings: list[str] = []
        left, right = assets
        if self._is_benchmark_pair(left, right):
            warnings.append("benchmark images have no geographic co-registration metadata")
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
        return "unknown"

    @staticmethod
    def _is_benchmark_pair(left: AssetRecord, right: AssetRecord) -> bool:
        return all(
            (asset.source_dataset or "").lower() in BENCHMARK_DATASETS for asset in (left, right)
        )

    @staticmethod
    def _overlap_ratio(a: Iterable[float], b: Iterable[float]) -> float:
        a_left, a_bottom, a_right, a_top = list(a)
        b_left, b_bottom, b_right, b_top = list(b)
        width = max(0.0, min(a_right, b_right) - max(a_left, b_left))
        height = max(0.0, min(a_top, b_top) - max(a_bottom, b_bottom))
        intersection = width * height
        smaller_area = min(
            max(0.0, a_right - a_left) * max(0.0, a_top - a_bottom),
            max(0.0, b_right - b_left) * max(0.0, b_top - b_bottom),
        )
        return intersection / smaller_area if smaller_area else 0.0
