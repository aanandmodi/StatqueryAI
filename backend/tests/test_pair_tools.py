from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin

from app.errors import ModelUnavailableError
from app.models.pair_tools import (
    LocalPairSpecialistGateway,
    _band_indexes,
    _read_on_grid,
    _reference_grid,
)
from app.schemas import AssetRole, Modality, PlannedStep, TaskType
from app.storage import LocalAssetStore


def _write_raster(
    path: Path, data: np.ndarray, *, nodata: float | None = None, mask: np.ndarray | None = None
) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[2],
        height=data.shape[1],
        count=data.shape[0],
        dtype=str(data.dtype),
        nodata=nodata,
        crs="EPSG:32644",
        transform=from_origin(500_000, 3_200_000, 10, 10),
    ) as destination:
        destination.write(data)
        if mask is not None:
            destination.write_mask(mask)


async def _infer_pair(
    tmp_path: Path,
    make_asset,
    task: TaskType,
    left_data: np.ndarray,
    right_data: np.ndarray,
    *,
    nodata: float | None = None,
):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    _write_raster(upload_dir / "ast_a__left.tif", left_data, nodata=nodata)
    _write_raster(upload_dir / "ast_b__right.tif", right_data, nodata=nodata)
    temporal = task == TaskType.CHANGE_VQA
    left = make_asset(
        "ast_a", Modality.OPTICAL, AssetRole.TIME_A if temporal else AssetRole.OPTICAL
    )
    right = make_asset(
        "ast_b",
        Modality.OPTICAL if temporal else Modality.SAR,
        AssetRole.TIME_B if temporal else AssetRole.SAR,
    )
    gateway = LocalPairSpecialistGateway(
        LocalAssetStore(upload_dir, max_upload_bytes=1_000_000)
    )
    step = PlannedStep(
        step_id="step-1", task=task, asset_ids=[left.id, right.id], policy_reason="test"
    )
    return await gateway.infer(step, [left, right], "Analyze the pair")


@pytest.mark.asyncio
async def test_local_change_tool_returns_quantified_evidence(
    tmp_path: Path, make_asset
) -> None:
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    before_data = np.full((3, 64, 64), 10, dtype=np.uint16)
    after_data = before_data.copy()
    after_data[:, 16:48, 20:44] = 100
    _write_raster(upload_dir / "ast_a__before.tif", before_data)
    _write_raster(upload_dir / "ast_b__after.tif", after_data)
    before = make_asset("ast_a", Modality.OPTICAL, AssetRole.TIME_A)
    after = make_asset("ast_b", Modality.OPTICAL, AssetRole.TIME_B)
    gateway = LocalPairSpecialistGateway(
        LocalAssetStore(upload_dir, max_upload_bytes=1_000_000)
    )
    step = PlannedStep(
        step_id="step-1",
        task=TaskType.CHANGE_VQA,
        asset_ids=[before.id, after.id],
        policy_reason="test",
    )

    output = await gateway.infer(step, [before, after], "Where did the scene change?")

    assert output.task == TaskType.CHANGE_VQA
    assert output.score_kind == "evidence_quality"
    assert output.model_version == gateway.CHANGE_VERSION
    changed = next(
        fact["value"]
        for fact in output.facts
        if fact["name"] == "changed_pixel_fraction"
    )
    assert changed == pytest.approx(0.1875, abs=0.02)
    assert output.evidence and output.evidence[0].asset_id == before.id
    assert "## Executive finding" in output.text
    assert "## Plausible explanations—not conclusions" in output.text
    assert "## Verification required" in output.text


@pytest.mark.asyncio
async def test_local_fusion_tool_supports_single_band_sar(
    tmp_path: Path, make_asset
) -> None:
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    yy, xx = np.mgrid[:64, :64]
    optical_data = np.stack(
        [80 + xx * 2, 90 + yy * 2, 110 + (63 - xx)], axis=0
    ).astype(np.uint16)
    sar_data = np.full((1, 64, 64), 180, dtype=np.uint16)
    sar_data[:, 8:32, 5:30] = 20
    sar_data[:, 35:58, 35:60] = 230
    _write_raster(upload_dir / "ast_o__optical.tif", optical_data)
    _write_raster(upload_dir / "ast_s__sar.tif", sar_data)
    optical = make_asset("ast_o", Modality.MULTISPECTRAL, AssetRole.OPTICAL)
    sar = make_asset("ast_s", Modality.SAR, AssetRole.SAR)
    gateway = LocalPairSpecialistGateway(
        LocalAssetStore(upload_dir, max_upload_bytes=1_000_000)
    )
    step = PlannedStep(
        step_id="step-1",
        task=TaskType.OPTICAL_SAR_FUSION,
        asset_ids=[optical.id, sar.id],
        permitted_params={"targets": ["water", "built-up"]},
        policy_reason="test",
    )

    output = await gateway.infer(step, [optical, sar], "Find water and built-up areas")

    assert output.task == TaskType.OPTICAL_SAR_FUSION
    assert output.score_kind == "evidence_quality"
    assert output.model_version == gateway.FUSION_VERSION
    assert {fact["name"] for fact in output.facts} >= {
        "water_proxy",
        "built-up_proxy",
        "valid_pixel_fraction",
        "water_cue_disagreement_fraction",
        "structure_cue_disagreement_fraction",
    }
    assert all(item.asset_id == optical.id for item in output.evidence)
    assert "## Agreement and disagreement" in output.text
    assert "## Verification required" in output.text


@pytest.mark.asyncio
@pytest.mark.parametrize("task", [TaskType.CHANGE_VQA, TaskType.OPTICAL_SAR_FUSION])
async def test_pair_tools_reject_no_shared_valid_pixels(tmp_path: Path, make_asset, task) -> None:
    yy, xx = np.mgrid[:16, :16]
    left = np.stack([10 + xx, 10 + yy, 20 + xx + yy]).astype(np.uint16)
    right = np.zeros((3 if task == TaskType.CHANGE_VQA else 1, 16, 16), dtype=np.uint16)

    with pytest.raises(ModelUnavailableError, match="no shared valid pixels"):
        await _infer_pair(tmp_path, make_asset, task, left, right, nodata=0)


@pytest.mark.asyncio
@pytest.mark.parametrize("flat_input", ["optical", "sar"])
async def test_fusion_abstains_on_flat_signal(tmp_path: Path, make_asset, flat_input) -> None:
    yy, xx = np.mgrid[:16, :16]
    optical = np.stack([10 + xx, 10 + yy, 20 + xx + yy]).astype(np.uint16)
    sar = (10 + xx + yy)[None].astype(np.uint16)
    if flat_input == "optical":
        optical[:] = 10
    else:
        sar[:] = 10

    with pytest.raises(ModelUnavailableError, match="both need spatial contrast"):
        await _infer_pair(tmp_path, make_asset, TaskType.OPTICAL_SAR_FUSION, optical, sar)


@pytest.mark.asyncio
async def test_change_abstains_on_flat_pair(tmp_path: Path, make_asset) -> None:
    left = np.full((3, 16, 16), 10, dtype=np.uint16)
    right = np.full((3, 16, 16), 20, dtype=np.uint16)
    with pytest.raises(ModelUnavailableError, match="insufficient spatial signal"):
        await _infer_pair(tmp_path, make_asset, TaskType.CHANGE_VQA, left, right)


@pytest.mark.asyncio
async def test_change_identical_textured_pair_has_no_changed_pixels(tmp_path: Path, make_asset):
    yy, xx = np.mgrid[:16, :16]
    data = np.stack([10 + xx, 10 + yy, 20 + xx + yy]).astype(np.uint16)
    output = await _infer_pair(tmp_path, make_asset, TaskType.CHANGE_VQA, data, data)
    facts = {item["name"]: item["value"] for item in output.facts}
    assert facts["changed_pixel_fraction"] == 0
    assert output.evidence == []


@pytest.mark.asyncio
async def test_fusion_excludes_pixels_missing_any_selected_band(tmp_path: Path, make_asset):
    yy, xx = np.mgrid[:16, :16]
    optical = np.stack([10 + xx, 10 + yy, 20 + xx + yy]).astype(np.uint16)
    optical[1, :4, :] = 0
    sar = (10 + xx + yy)[None].astype(np.uint16)

    output = await _infer_pair(
        tmp_path, make_asset, TaskType.OPTICAL_SAR_FUSION, optical, sar, nodata=0
    )
    facts = {item["name"]: item["value"] for item in output.facts}
    assert facts["valid_pixel_fraction"] == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_fusion_rejects_one_pixel_axis_with_clear_error(tmp_path: Path, make_asset):
    optical = np.arange(3 * 16, dtype=np.uint16).reshape(3, 1, 16)
    sar = np.arange(16, dtype=np.uint16).reshape(1, 1, 16)
    with pytest.raises(ModelUnavailableError, match="at least two pixels"):
        await _infer_pair(tmp_path, make_asset, TaskType.OPTICAL_SAR_FUSION, optical, sar)


def test_reprojection_preserves_internal_dataset_mask(tmp_path: Path) -> None:
    path = tmp_path / "masked.tif"
    data = np.full((3, 16, 16), 80, dtype=np.uint16)
    mask = np.full((16, 16), 255, dtype=np.uint8)
    mask[:4] = 0
    _write_raster(path, data, mask=mask)

    read = _read_on_grid(path, _reference_grid(path, 512), visual=True)

    assert np.isnan(read[:, :4]).all()
    assert np.isfinite(read[:, 4:]).all()


@pytest.mark.parametrize(
    ("colors", "descriptions", "expected"),
    [
        ((ColorInterp.undefined,) * 4, (None,) * 4, [1, 2, 3]),
        (
            (ColorInterp.green, ColorInterp.blue, ColorInterp.undefined, ColorInterp.red),
            ("green", "blue", "nir", "red"),
            [4, 1, 2],
        ),
        ((ColorInterp.undefined,) * 4, ("blue", "green", "red", "nir"), [3, 2, 1]),
    ],
)
def test_pair_visual_bands_match_decoder_metadata_and_fallback(
    tmp_path: Path, colors, descriptions, expected
) -> None:
    path = tmp_path / "four-band.tif"
    _write_raster(path, np.ones((4, 16, 16), dtype=np.uint16))
    with rasterio.open(path, "r+") as dataset:
        dataset.colorinterp = colors
        dataset.descriptions = descriptions
    with rasterio.open(path) as dataset:
        assert _band_indexes(dataset, visual=True) == expected
        assert _band_indexes(dataset, visual=False) == [1, 2]
