from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


@pytest.fixture
def raster_metadata():
    from app.schemas import RasterMetadata

    return RasterMetadata(
        driver="GTiff",
        width=64,
        height=64,
        count=3,
        dtypes=["uint16", "uint16", "uint16"],
        crs="EPSG:32644",
        transform=[10.0, 0.0, 500000.0, 0.0, -10.0, 3200000.0],
        bounds=[500000.0, 3199360.0, 500640.0, 3200000.0],
        resolution=[10.0, 10.0],
        nodata=0,
        tags={"sensor": "Sentinel-2"},
        quality_score=1.0,
    )


@pytest.fixture
def make_asset(raster_metadata):
    from app.schemas import AssetRecord, AssetRole, Modality, utc_now

    def factory(
        asset_id: str,
        modality: Modality = Modality.OPTICAL,
        role: AssetRole = AssetRole.PRIMARY,
        metadata=None,
    ) -> AssetRecord:
        return AssetRecord(
            id=asset_id,
            original_name=f"{asset_id}.tif",
            content_type="image/tiff",
            size_bytes=2048,
            sha256=(asset_id[-1] * 64)[:64],
            role=role,
            modality=modality,
            created_at=utc_now(),
            metadata=metadata or raster_metadata,
        )

    return factory
