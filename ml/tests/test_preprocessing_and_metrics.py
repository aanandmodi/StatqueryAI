from __future__ import annotations

import numpy as np
import pytest

from satquery_ml.evaluation import box_iou, exact_match, expected_calibration_error
from satquery_ml.preprocessing import percentile_stretch_rgb, sar_to_db, tile_windows


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
