from __future__ import annotations

import pytest

from app.config import Settings
from app.core.validation import RasterValidator
from app.errors import ValidationFailure
from app.schemas import AssetRole, Modality, TaskType


def test_aligned_pair_passes(make_asset):
    validator = RasterValidator(Settings(environment="test"))
    before = make_asset("ast_a", role=AssetRole.TIME_A)
    after = make_asset("ast_b", role=AssetRole.TIME_B)

    assert validator.validate_for_task(TaskType.CHANGE_VQA, [before, after]) == []


def test_half_pixel_shift_fails(make_asset, raster_metadata):
    validator = RasterValidator(Settings(environment="test"))
    shifted = raster_metadata.model_copy(
        update={
            "transform": [10.0, 0.0, 500005.0, 0.0, -10.0, 3200000.0],
            "bounds": [500005.0, 3199360.0, 500645.0, 3200000.0],
        }
    )
    before = make_asset("ast_a", role=AssetRole.TIME_A)
    after = make_asset("ast_b", role=AssetRole.TIME_B, metadata=shifted)

    with pytest.raises(ValidationFailure, match="pixel grid"):
        validator.validate_for_task(TaskType.CHANGE_VQA, [before, after])


def test_fusion_requires_sar(make_asset):
    validator = RasterValidator(Settings(environment="test"))
    left = make_asset("ast_a", Modality.OPTICAL)
    right = make_asset("ast_b", Modality.MULTISPECTRAL)

    with pytest.raises(ValidationFailure, match="one optical/multispectral and one SAR"):
        validator.validate_for_task(TaskType.OPTICAL_SAR_FUSION, [left, right])


def test_change_requires_explicit_temporal_roles(make_asset):
    validator = RasterValidator(Settings(environment="test"))
    left = make_asset("ast_a")
    right = make_asset("ast_b")

    with pytest.raises(ValidationFailure, match="time_a and time_b"):
        validator.validate_for_task(TaskType.CHANGE_VQA, [left, right])


def test_change_rejects_cross_modal_temporal_pair(make_asset):
    validator = RasterValidator(Settings(environment="test"))
    before = make_asset("ast_a", Modality.OPTICAL, AssetRole.TIME_A)
    after = make_asset("ast_b", Modality.SAR, AssetRole.TIME_B)

    with pytest.raises(ValidationFailure, match="same declared modality"):
        validator.validate_for_task(TaskType.CHANGE_VQA, [before, after])


def test_change_rejects_different_band_counts(make_asset, raster_metadata):
    validator = RasterValidator(Settings(environment="test"))
    before = make_asset("ast_a", role=AssetRole.TIME_A)
    after = make_asset(
        "ast_b", role=AssetRole.TIME_B, metadata=raster_metadata.model_copy(update={"count": 4})
    )

    with pytest.raises(ValidationFailure, match="matching band counts"):
        validator.validate_for_task(TaskType.CHANGE_VQA, [before, after])


def _benchmark_pair(make_asset, raster_metadata, *, right_dataset="cdvqa", right_width=64):
    unreferenced = raster_metadata.model_copy(update={"crs": None, "driver": "PNG"})
    before = make_asset("ast_a", role=AssetRole.TIME_A, metadata=unreferenced).model_copy(
        update={"source_dataset": "cdvqa"}
    )
    after = make_asset(
        "ast_b",
        role=AssetRole.TIME_B,
        metadata=unreferenced.model_copy(update={"width": right_width}),
    ).model_copy(update={"source_dataset": right_dataset})
    return [before, after]


def test_same_benchmark_matching_dimensions_may_use_unreferenced_pair(
    make_asset, raster_metadata
):
    validator = RasterValidator(Settings(environment="test"))
    warnings = validator.validate_for_task(
        TaskType.CHANGE_VQA, _benchmark_pair(make_asset, raster_metadata)
    )
    assert len(warnings) == 1
    assert "not independently verified" in warnings[0]


@pytest.mark.parametrize(
    "options", [{"right_dataset": "second"}, {"right_width": 32}, {"right_dataset": None}]
)
def test_benchmark_bypass_requires_same_dataset_and_dimensions(
    make_asset, raster_metadata, options
):
    validator = RasterValidator(Settings(environment="test"))
    with pytest.raises(ValidationFailure, match="pixel-grid declaration"):
        validator.validate_for_task(
            TaskType.CHANGE_VQA, _benchmark_pair(make_asset, raster_metadata, **options)
        )


def test_benchmark_tag_cannot_bypass_existing_geospatial_checks(make_asset, raster_metadata):
    validator = RasterValidator(Settings(environment="test"))
    shifted = raster_metadata.model_copy(
        update={
            "transform": [10.0, 0.0, 500005.0, 0.0, -10.0, 3200000.0],
            "bounds": [500005.0, 3199360.0, 500645.0, 3200000.0],
        }
    )
    before = make_asset("ast_a", role=AssetRole.TIME_A).model_copy(
        update={"source_dataset": "cdvqa"}
    )
    after = make_asset("ast_b", role=AssetRole.TIME_B, metadata=shifted).model_copy(
        update={"source_dataset": "cdvqa"}
    )
    with pytest.raises(ValidationFailure, match="pixel grid"):
        validator.validate_for_task(TaskType.CHANGE_VQA, [before, after])


def test_contained_crop_does_not_count_as_full_footprint_overlap(make_asset, raster_metadata):
    validator = RasterValidator(Settings(environment="test"))
    crop = raster_metadata.model_copy(
        update={"width": 32, "bounds": [500000.0, 3199360.0, 500320.0, 3200000.0]}
    )
    before = make_asset("ast_a", role=AssetRole.TIME_A)
    after = make_asset("ast_b", role=AssetRole.TIME_B, metadata=crop)

    with pytest.raises(ValidationFailure, match="same area"):
        validator.validate_for_task(TaskType.CHANGE_VQA, [before, after])


@pytest.mark.parametrize(
    "transform",
    [
        [10.0, 1.0, 500000.0, 1.0, -10.0, 3200000.0],
        [-10.0, 0.0, 500000.0, 0.0, -10.0, 3200000.0],
        [10.0, 0.0, 500000.0, 0.0, 10.0, 3200000.0],
    ],
)
def test_pair_rejects_rotated_sheared_or_mirrored_grid(make_asset, raster_metadata, transform):
    validator = RasterValidator(Settings(environment="test"))
    before = make_asset("ast_a", role=AssetRole.TIME_A)
    after = make_asset(
        "ast_b", role=AssetRole.TIME_B,
        metadata=raster_metadata.model_copy(update={"transform": transform}),
    )

    with pytest.raises(ValidationFailure, match="common north-up grid"):
        validator.validate_for_task(TaskType.CHANGE_VQA, [before, after])


def test_zero_resolution_metadata_fails_safely(make_asset, raster_metadata):
    validator = RasterValidator(Settings(environment="test"))
    before = make_asset("ast_a", role=AssetRole.TIME_A)
    after = make_asset(
        "ast_b", role=AssetRole.TIME_B,
        metadata=raster_metadata.model_copy(update={"resolution": [0.0, 0.0]}),
    )

    with pytest.raises(ValidationFailure, match="invalid affine or resolution"):
        validator.validate_for_task(TaskType.CHANGE_VQA, [before, after])
