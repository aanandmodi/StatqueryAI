from __future__ import annotations

import ast
import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from PIL import Image, UnidentifiedImageError
from rasterio.enums import ColorInterp, Resampling
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from app.models.gateway import _rgb_indexes, render_rgb_preview
from app.core.sensors import visual_indexes


def _write_image(path, data, *, colors=None, descriptions=None, mask=None):
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2],
        count=data.shape[0], dtype=str(data.dtype), crs="EPSG:32644",
        transform=from_origin(500000, 3200000, 10, 10), photometric="MINISBLACK",
    ) as target:
        target.write(data)
        if colors is not None:
            target.colorinterp = colors
        if descriptions is not None:
            target.descriptions = descriptions
        if mask is not None:
            target.write_mask(mask)


def _decoded_preview(path: Path) -> np.ndarray:
    payload = render_rgb_preview(path, max_edge=448, jpeg_quality=100)
    with Image.open(io.BytesIO(payload)) as image:
        return np.array(image)


def test_rgb_png_keeps_ordinary_uint8_colors(tmp_path):
    path = tmp_path / "rgb.png"
    Image.new("RGB", (32, 32), (21, 70, 130)).save(path)

    actual = _decoded_preview(path)

    assert np.max(np.abs(actual.astype(int) - [21, 70, 130])) <= 2


@pytest.mark.parametrize(
    ("colors", "descriptions", "expected_indexes"),
    [
        (
            (ColorInterp.blue, ColorInterp.red, ColorInterp.undefined, ColorInterp.green),
            ("blue", "red", "nir", "green"),
            [2, 4, 1],
        ),
        (
            (ColorInterp.undefined,) * 4,
            ("blue", "green", "red", "nir"),
            [3, 2, 1],
        ),
        ((ColorInterp.undefined,) * 4, None, [1, 2, 3]),
    ],
)
def test_preview_uses_metadata_first_and_safe_unlabeled_fallback(
    tmp_path, colors, descriptions, expected_indexes
):
    path = tmp_path / "multiband.tif"
    values = [30, 80, 140, 200]
    data = np.stack([np.full((32, 32), value, dtype=np.uint8) for value in values])
    _write_image(path, data, colors=colors, descriptions=descriptions)
    with rasterio.open(path) as dataset:
        assert _rgb_indexes(dataset) == expected_indexes

    expected_rgb = np.array([values[index - 1] for index in expected_indexes])
    assert np.max(np.abs(_decoded_preview(path).astype(int) - expected_rgb)) <= 2


def test_two_band_sar_preview_repeats_first_band(tmp_path):
    path = tmp_path / "sar.tif"
    data = np.stack([np.full((32, 32), value, dtype=np.uint8) for value in [60, 210]])
    _write_image(path, data)

    assert np.max(np.abs(_decoded_preview(path).astype(int) - 60)) <= 1


def test_preview_preserves_internal_mask_as_black(tmp_path):
    path = tmp_path / "masked.tif"
    data = np.full((3, 32, 32), 180, dtype=np.uint8)
    mask = np.full((32, 32), 255, dtype=np.uint8)
    mask[:16] = 0
    _write_image(path, data, mask=mask)

    actual = _decoded_preview(path)

    assert actual[:8].max() <= 1
    assert np.max(np.abs(actual[24:].astype(int) - 180)) <= 1


def test_non_uint8_stretch_matches_finalized_kaggle_decoder(tmp_path):
    # Extract only the three pure decoder functions; never run notebook setup,
    # model downloads, token prompts, server startup, or cloud/tunnel cells.
    notebook = Path(__file__).resolve().parents[2] / "notebooks" / (
        "SatQuery_Qwen3VL_Free_GPU_Server.py"
    )
    names = {"scale_band", "rgb_band_indexes", "decode_uploaded_image"}
    tree = ast.parse(notebook.read_text(encoding="utf-8"))
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(functions) == 3
    namespace = {
        "np": np, "rasterio": rasterio, "Image": Image, "io": io, "Any": Any,
        "ColorInterp": ColorInterp, "Resampling": Resampling, "MemoryFile": MemoryFile,
        "UnidentifiedImageError": UnidentifiedImageError, "MAX_UPLOAD_BYTES": 50_000_000,
        "visual_indexes": visual_indexes,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(notebook), "exec"), namespace)
    path = tmp_path / "reflectance.tif"
    values = np.tile(np.arange(64, dtype=np.uint16) * 100 + 1, (64, 1))
    _write_image(path, np.stack([values, values, values]))

    expected = np.array(namespace["decode_uploaded_image"](path.read_bytes())).astype(int)
    actual = _decoded_preview(path).astype(int)

    assert actual.shape == expected.shape
    assert np.max(np.abs(actual - expected)) <= 1
