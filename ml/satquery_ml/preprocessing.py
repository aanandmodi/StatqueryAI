from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import ColorInterp, Resampling
from rasterio.windows import Window


@dataclass(frozen=True)
class Tile:
    data: np.ndarray
    transform: tuple[float, ...]
    crs: str
    window: tuple[int, int, int, int]
    valid_fraction: float


def read_tile(
    path: Path,
    *,
    row: int,
    col: int,
    height: int,
    width: int,
    bands: Sequence[int] | None = None,
) -> Tile:
    """Windowed raster read; never loads an arbitrary full scene."""
    if min(row, col, height, width) < 0 or height == 0 or width == 0:
        raise ValueError("window values must be non-negative and dimensions non-zero")
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(path) as src:
            if src.crs is None:
                raise ValueError("raster is missing CRS")
            window = Window(col, row, width, height)
            indexes = list(bands) if bands else list(range(1, src.count + 1))
            data = src.read(indexes, window=window, boundless=False, masked=True)
            valid_fraction = float(1.0 - np.mean(np.ma.getmaskarray(data)))
            transform = src.window_transform(window)
            return Tile(
                data=np.asarray(data.filled(0)),
                transform=tuple(transform)[:6],
                crs=src.crs.to_string(),
                window=(row, col, height, width),
                valid_fraction=valid_fraction,
            )


def percentile_stretch_rgb(
    bands_first: np.ndarray,
    *,
    lower: float = 2.0,
    upper: float = 98.0,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    if bands_first.shape[0] != 3:
        raise ValueError("RGB preview requires exactly three bands")
    if not 0 <= lower < upper <= 100:
        raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")
    raster = np.ma.filled(np.ma.asarray(bands_first).astype(np.float32), np.nan)
    output = np.zeros(raster.shape, dtype=np.uint8)
    for index, band in enumerate(raster):
        valid = np.isfinite(band)
        if mask is not None:
            valid &= mask
        values = band[valid]
        if values.size == 0:
            continue
        lo, hi = np.percentile(values, [lower, upper])
        if hi <= lo:
            lo, hi = float(values.min()), float(values.max())
        if hi <= lo:
            continue
        output[index][valid] = (
            np.clip((band[valid] - lo) / (hi - lo), 0, 1) * 255
        ).round().astype(np.uint8)
    return np.moveaxis(output, 0, -1)


def rgb_band_indexes(source: rasterio.io.DatasetReader) -> list[int]:
<<<<<<< HEAD
    from satquery_ml.sensors import visual_indexes
    return visual_indexes(source)
=======
    """Use the same metadata-first visual band mapping as the Kaggle decoder."""
    interpretations = list(source.colorinterp)
    colors = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
    if all(color in interpretations for color in colors):
        return [interpretations.index(color) + 1 for color in colors]
    descriptions = [str(item or "").lower().strip() for item in source.descriptions]
    aliases = (("red", "b04", "b4"), ("green", "b03", "b3"), ("blue", "b02", "b2"))
    indexes = [
        next((index for index, name in enumerate(descriptions, 1) if name in names), None)
        for names in aliases
    ]
    if all(index is not None for index in indexes):
        return [int(index) for index in indexes]
    return [1, 2, 3] if source.count >= 3 else [1, 1, 1]
>>>>>>> 2f620623f8897788bd2df2ce4f5700cb183d84f8


def geotiff_to_rgb_preview(
    source: Path,
    destination: Path,
    *,
    rgb_bands: tuple[int, int, int] | None = None,
    max_size: int = 1024,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(source) as src:
            indexes = list(rgb_bands) if rgb_bands is not None else rgb_band_indexes(src)
            if len(indexes) != 3 or min(indexes) < 1 or max(indexes) > src.count:
                raise ValueError(
                    f"requested RGB bands {indexes} are invalid for raster band count {src.count}"
                )
            scale = min(1.0, max_size / max(src.width, src.height))
            out_width = max(1, round(src.width * scale))
            out_height = max(1, round(src.height * scale))
            data = src.read(
                indexes,
                out_shape=(3, out_height, out_width),
                resampling=Resampling.bilinear,
                masked=True,
            )
            raster = np.ma.filled(data.astype(np.float32), np.nan)
            if all(src.dtypes[index - 1] == "uint8" for index in indexes):
                rgb = np.moveaxis(np.nan_to_num(raster, nan=0.0), 0, -1)
                rgb = np.clip(rgb, 0, 255).round().astype(np.uint8)
            else:
                rgb = percentile_stretch_rgb(raster)
    Image.fromarray(rgb, mode="RGB").save(destination, format="PNG", optimize=True)
    return destination


def sar_to_db(
    values: np.ndarray, *, input_is_db: bool, floor_db: float = -40.0
) -> np.ndarray:
    values = values.astype(np.float32)
    if input_is_db:
        result = values
    else:
        result = 10.0 * np.log10(np.maximum(values, 1e-10))
    return np.clip(result, floor_db, 20.0)


def tile_windows(
    height: int,
    width: int,
    *,
    tile_size: int,
    overlap: int,
) -> list[tuple[int, int, int, int]]:
    if not 0 <= overlap < tile_size:
        raise ValueError("overlap must be smaller than tile_size")
    stride = tile_size - overlap
    rows = list(range(0, max(1, height - tile_size + 1), stride))
    cols = list(range(0, max(1, width - tile_size + 1), stride))
    last_row = max(0, height - tile_size)
    last_col = max(0, width - tile_size)
    if not rows or rows[-1] != last_row:
        rows.append(last_row)
    if not cols or cols[-1] != last_col:
        cols.append(last_col)
    return [
        (row, col, min(tile_size, height - row), min(tile_size, width - col))
        for row in rows
        for col in cols
    ]
