from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.models.pair_tools import LocalPairSpecialistGateway
from app.schemas import AssetRole, Modality, PlannedStep, TaskType
from app.storage import LocalAssetStore


def _write_raster(path: Path, data: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[2],
        height=data.shape[1],
        count=data.shape[0],
        dtype=str(data.dtype),
        crs="EPSG:32644",
        transform=from_origin(500_000, 3_200_000, 10, 10),
    ) as destination:
        destination.write(data)


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
    assert output.evidence and output.evidence[0].asset_id == after.id


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
    }
    assert all(item.asset_id == optical.id for item in output.evidence)
