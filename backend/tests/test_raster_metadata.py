from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from app.config import Settings
from app.main import create_app


def test_float_nan_nodata_upload_and_metadata_are_json_safe(tmp_path: Path):
    source = tmp_path / "float-sar.tif"
    data = np.arange(256, dtype=np.float32).reshape(1, 16, 16)
    data[:, :2] = np.nan
    with rasterio.open(
        source, "w", driver="GTiff", width=16, height=16, count=1, dtype="float32",
        crs="EPSG:32644", transform=from_origin(500000, 3200000, 10, 10), nodata=np.nan,
    ) as dataset:
        dataset.write(data)
    runtime = tmp_path / "runtime"
    settings = Settings(
        _env_file=None, environment="test", model_backend="demo", data_dir=runtime,
        database_path=runtime / "db.sqlite3", upload_dir=runtime / "uploads",
        artifact_dir=runtime / "artifacts", report_dir=runtime / "reports",
    )
    with TestClient(create_app(settings)) as client:
        with source.open("rb") as handle:
            response = client.post(
                "/v1/assets", files={"file": (source.name, handle, "image/tiff")},
                data={"modality": "sar", "role": "primary"},
            )
        assert response.status_code == 201, response.text
        asset = response.json()
        assert asset["metadata"]["nodata"] is None
        assert any("nodata sentinel (nan)" in item for item in asset["metadata"]["warnings"])
        reread = client.get(f"/v1/assets/{asset['id']}")
        assert reread.status_code == 200
        assert reread.json()["metadata"] == asset["metadata"]
        preview = client.get(f"/v1/assets/{asset['id']}/preview")
        assert preview.status_code == 200
        assert preview.headers["content-type"] == "image/jpeg"
