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
