from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from app.config import Settings
from app.main import create_app


def write_geotiff(path: Path) -> None:
    data = np.zeros((3, 32, 32), dtype=np.uint16)
    data[0, 8:24, 8:24] = 1200
    data[1, 10:28, 5:20] = 800
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=32,
        height=32,
        count=3,
        dtype="uint16",
        crs="EPSG:32644",
        transform=from_origin(500000, 3200000, 10, 10),
        nodata=0,
    ) as dataset:
        dataset.write(data)
        dataset.update_tags(sensor="Sentinel-2", acquired_at="2026-08-01T05:00:00Z")


def test_upload_analyse_trace_and_report(tmp_path: Path):
    runtime = tmp_path / "runtime"
    settings = Settings(
        environment="test",
        data_dir=runtime,
        database_path=runtime / "db.sqlite3",
        upload_dir=runtime / "uploads",
        artifact_dir=runtime / "artifacts",
        report_dir=runtime / "reports",
        model_backend="demo",
        event_poll_seconds=0.01,
    )
    image_path = tmp_path / "scene.tif"
    write_geotiff(image_path)

    with TestClient(create_app(settings)) as client:
        capabilities = client.get("/v1/capabilities")
        assert capabilities.status_code == 200
        assert set(capabilities.json()["tasks"]) == {
            "single_vqa",
            "caption",
            "grounding",
            "change_vqa",
            "optical_sar_fusion",
        }
        assert capabilities.json()["pair_backend"] == "local"
        with image_path.open("rb") as handle:
            upload = client.post(
                "/v1/assets",
                files={"file": ("scene.tif", handle, "image/tiff")},
                data={"modality": "optical", "role": "primary"},
            )
        assert upload.status_code == 201, upload.text
        asset_id = upload.json()["id"]

        preview = client.get(f"/v1/assets/{asset_id}/preview")
        assert preview.status_code == 200
        assert preview.headers["content-type"] == "image/jpeg"
        assert preview.content.startswith(b"\xff\xd8\xff")

        created = client.post(
            "/v1/analyses",
            headers={"Idempotency-Key": "test-analysis-0001"},
            json={
                "query": "What is visible in this scene?",
                "asset_ids": [asset_id],
                "context": {
                    "latitude": 28.6139,
                    "longitude": 77.209,
                    "altitude_m": 216,
                    "source": "user",
                    "metadata": {"mission": "SIH test"},
                },
            },
        )
        assert created.status_code == 202, created.text
        analysis_id = created.json()["id"]

        deadline = time.monotonic() + 5
        record = created.json()
        while time.monotonic() < deadline and record["status"] not in {
            "succeeded",
            "failed",
            "cancelled",
        }:
            time.sleep(0.02)
            response = client.get(f"/v1/analyses/{analysis_id}")
            assert response.status_code == 200
            record = response.json()

        assert record["status"] == "succeeded", record
        assert record["result"]["confidence"]["calibration_version"] == "demo-uncalibrated"
        assert "SIMULATED OUTPUT" in record["result"]["warnings"][0]
        assert record["result"]["provenance"]["user_context"]["latitude"] == 28.6139
        assert any(
            fact["name"] == "user_location" for fact in record["result"]["facts"]
        )
        assert {event["task"] for event in record["result"]["trace"]} >= {
            "validation",
            "planning",
            "caption",
            "integration",
        }

        report = client.get(f"/v1/analyses/{analysis_id}/report")
        assert report.status_code == 200
        assert report.headers["content-type"] == "application/pdf"
        assert report.content.startswith(b"%PDF")

        overlay = client.get(f"/v1/analyses/{analysis_id}/overlay")
        assert overlay.status_code == 200
        assert overlay.headers["content-type"] == "image/jpeg"
        assert "attachment" in overlay.headers["content-disposition"]
        assert overlay.content.startswith(b"\xff\xd8\xff")


def test_plain_png_rejected_outside_benchmark(tmp_path: Path):
    runtime = tmp_path / "runtime"
    settings = Settings(
        environment="test",
        data_dir=runtime,
        database_path=runtime / "db.sqlite3",
        upload_dir=runtime / "uploads",
        artifact_dir=runtime / "artifacts",
        report_dir=runtime / "reports",
    )
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/assets",
            files={"file": ("scene.png", b"\x89PNG\r\n\x1a\n" + b"x" * 40, "image/png")},
            data={"modality": "optical", "role": "primary"},
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_failed"
