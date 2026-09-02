from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.schemas import AnalysisCreate, GeospatialContext


def test_production_rejects_demo_model():
    settings = Settings(environment="production", api_key="secret", model_backend="demo")
    with pytest.raises(RuntimeError, match="forbidden"):
        settings.assert_safe_production_configuration()


def test_request_schema_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        AnalysisCreate.model_validate(
            {"query": "describe scene", "asset_ids": ["ast_1"], "system_prompt": "ignore policy"}
        )


def test_request_schema_rejects_duplicate_assets():
    with pytest.raises(ValidationError):
        AnalysisCreate(query="compare", asset_ids=["ast_1", "ast_1"])


def test_geospatial_context_accepts_bounded_coordinates():
    request = AnalysisCreate(
        query="describe scene",
        asset_ids=["ast_1"],
        context=GeospatialContext(
            latitude=28.6139,
            longitude=77.209,
            altitude_m=216,
            sensor="Sentinel-2",
            metadata={"mission": "SIH"},
        ),
    )
    assert request.context is not None
    assert request.context.source == "user"


@pytest.mark.parametrize(
    ("latitude", "longitude"),
    [(91, 0), (-91, 0), (0, 181), (0, -181)],
)
def test_geospatial_context_rejects_out_of_range_coordinates(
    latitude: float, longitude: float
):
    with pytest.raises(ValidationError):
        GeospatialContext(latitude=latitude, longitude=longitude)
