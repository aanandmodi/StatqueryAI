from __future__ import annotations

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin

from satquery_ml.evaluation import box_iou, exact_match, expected_calibration_error
from satquery_ml.preprocessing import (
    geotiff_to_rgb_preview,
    percentile_stretch_rgb,
    sar_to_db,
    tile_windows,
)


def test_tile_windows_cover_edges():
    windows = tile_windows(1000, 920, tile_size=224, overlap=32)
    assert windows[0] == (0, 0, 224, 224)
    assert max(row + height for row, _, height, _ in windows) == 1000
    assert max(col + width for _, col, _, width in windows) == 920


def test_percentile_stretch_shape_and_range():
    source = np.stack([np.arange(100).reshape(10, 10)] * 3)
    rgb = percentile_stretch_rgb(source)
    assert rgb.shape == (10, 10, 3)
    assert rgb.dtype == np.uint8
    assert rgb.min() == 0
    assert rgb.max() == 255


def test_percentile_stretch_does_not_paint_masked_or_nonfinite_pixels():
    data = np.stack([np.arange(100).reshape(10, 10)] * 3).astype(np.float32)
    data[:, 0, 0] = 99999
    data[:, 1, 1] = np.nan
    mask = np.ones((10, 10), dtype=bool)
    mask[0, 0] = False

    rgb = percentile_stretch_rgb(data, mask=mask)

    assert (rgb[0, 0] == 0).all()
    assert (rgb[1, 1] == 0).all()


def test_percentile_stretch_falls_back_to_range_for_sparse_signal():
    data = np.zeros((3, 10, 10), dtype=np.float32)
    data[:, 9, 9] = 1

    assert (percentile_stretch_rgb(data)[9, 9] == 255).all()


def test_geotiff_preview_preserves_uint8_and_uses_named_rgb_mapping(tmp_path):
    source = tmp_path / "reflectance.tif"
    destination = tmp_path / "preview.png"
    values = [30, 80, 140, 200]
    data = np.stack([np.full((16, 16), value, dtype=np.uint8) for value in values])
    mask = np.full((16, 16), 255, dtype=np.uint8)
    mask[:4] = 0
    with rasterio.open(
        source, "w", driver="GTiff", height=16, width=16, count=4, dtype="uint8",
        crs="EPSG:32644", transform=from_origin(500000, 3200000, 10, 10),
        photometric="MINISBLACK",
    ) as target:
        target.write(data)
        target.colorinterp = (ColorInterp.undefined,) * 4
        target.descriptions = (" B02 ", " B03 ", " B04 ", "nir")
        target.write_mask(mask)

    geotiff_to_rgb_preview(source, destination)

    with Image.open(destination) as image:
        actual = np.array(image)
    assert (actual[:4] == 0).all()
    assert (actual[4:] == [140, 80, 30]).all()


def test_geotiff_preview_keeps_explicit_rgb_band_override(tmp_path):
    source = tmp_path / "scene.tif"
    destination = tmp_path / "preview.png"
    data = np.stack([np.full((8, 8), value, dtype=np.uint8) for value in [20, 80, 140]])
    with rasterio.open(
        source, "w", driver="GTiff", height=8, width=8, count=3, dtype="uint8",
        crs="EPSG:32644", transform=from_origin(500000, 3200000, 10, 10),
    ) as target:
        target.write(data)

    geotiff_to_rgb_preview(source, destination, rgb_bands=(3, 2, 1))

    with Image.open(destination) as image:
        assert (np.array(image) == [140, 80, 20]).all()


def test_sar_conversion_is_bounded():
    values = np.array([0.0, 0.01, 1.0, 1000.0])
    result = sar_to_db(values, input_is_db=False)
    assert result.tolist() == pytest.approx([-40.0, -20.0, 0.0, 20.0])


def test_metrics():
    assert exact_match(["Built-up area"], ["built up area!"]) == 1.0
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert expected_calibration_error(
        [0.9, 0.1], [True, False], bins=2
    ) == pytest.approx(0.1)
