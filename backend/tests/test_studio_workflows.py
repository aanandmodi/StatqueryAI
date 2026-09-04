"""Real local raster methods; single-image transport uses an explicitly labeled mock."""

import io
import time

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.io import MemoryFile

from app.config import Settings
from app.main import create_app


@pytest.fixture
def client(tmp_path):
    runtime = tmp_path / "runtime"
    settings = Settings(
        environment="test",
        data_dir=runtime,
        database_path=runtime / "db.sqlite3",
        upload_dir=runtime / "uploads",
        artifact_dir=runtime / "artifacts",
        report_dir=runtime / "reports",
        model_backend="demo",
        pair_backend="local",
        api_key="",
        event_poll_seconds=0.01,
    )
    with TestClient(create_app(settings)) as value:
        yield value


def rgb():
    y, x = np.mgrid[:64, :64]
    return np.stack([60 + x * 2, 40 + y * 2, 180 - x], axis=-1).astype("uint8")


def image_bytes(pixels, format):
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format=format)
    return stream.getvalue()


def tiff(pixels, *, geo=True, nodata=None):
    if pixels.ndim == 2:
        pixels = pixels[None]
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=pixels.shape[2],
            height=pixels.shape[1],
            count=pixels.shape[0],
            dtype=str(pixels.dtype),
            nodata=nodata,
            **(
                {"crs": "EPSG:32644", "transform": rasterio.Affine(10, 0, 500000, 0, -10, 3200000)}
                if geo
                else {}
            ),
        ) as target:
            target.write(pixels)
        return memory.read()


def upload(client, raw, name, role="primary", modality="optical", **extra):
    response = client.post(
        "/v1/assets",
        files={"file": (name, raw, "application/octet-stream")},
        data={"role": role, "modality": modality, **extra},
    )
    assert response.status_code == 201, response.text
    return response.json()


def analyze(client, assets, query):
    response = client.post(
        "/v1/analyses", json={"asset_ids": [a["id"] for a in assets], "query": query}
    )
    assert response.status_code == 202, response.text
    record = response.json()
    deadline = time.monotonic() + 10
    while (
        record["status"] not in {"succeeded", "failed", "cancelled"} and time.monotonic() < deadline
    ):
        time.sleep(0.02)
        record = client.get("/v1/analyses/" + record["id"]).json()
    return record


@pytest.mark.parametrize("format,ext", [("PNG", "png"), ("JPEG", "jpg"), ("WEBP", "webp")])
def test_display_upload_preview_and_single_transport(client, format, ext):
    raw = image_bytes(rgb(), format)
    asset = upload(client, raw, "scene." + ext, input_profile="exploration")
    assert asset["input_profile"] == "exploration" and asset["metadata"]["crs"] is None
    assert client.get(f"/v1/assets/{asset['id']}/preview").status_code == 200
    record = analyze(client, [asset], "What is visible?")
    assert record["status"] == "succeeded", record
    assert (
        record["result"]["provenance"]["model_backend"] == "demo"
    )  # no claim of real VLM accuracy
    assert "exploration" in record["result"]["provenance"]["input_profiles"].values()
    strict = client.post(
        "/v1/assets", files={"file": ("scene." + ext, raw)}, data={"modality": "optical"}
    )
    assert strict.status_code == 422
    rejected_id = strict.json()["error"]["details"]["asset_id"]
    assert client.get(f"/v1/assets/{rejected_id}/preview").status_code == 422


@pytest.mark.parametrize("modality", ["sar", "multispectral"])
@pytest.mark.parametrize("profile", ["strict", "exploration"])
def test_display_cannot_fake_sensor(client, modality, profile):
    response = client.post(
        "/v1/assets",
        files={"file": ("scene.png", image_bytes(rgb(), "PNG"))},
        data={"modality": modality, "input_profile": profile, "source_dataset": "cdvqa"},
    )
    assert response.status_code == 422


@pytest.mark.parametrize("mode,format", [("P", "PNG"), ("RGBA", "PNG"), ("CMYK", "JPEG")])
def test_ambiguous_display_modes_rejected(client, mode, format):
    stream = io.BytesIO()
    Image.new(mode, (32, 32)).save(stream, format=format)
    response = client.post(
        "/v1/assets",
        files={"file": ("image", stream.getvalue())},
        data={"modality": "optical", "input_profile": "exploration"},
    )
    assert response.status_code == 422 and "RGB" in response.text


def test_webp_is_not_a_strict_benchmark_format(client):
    response = client.post(
        "/v1/assets",
        files={"file": ("scene.webp", image_bytes(rgb(), "WEBP"))},
        data={"modality": "optical", "source_dataset": "cdvqa"},
    )
    assert response.status_code == 422


@pytest.mark.parametrize("format", ["PNG", "TIFF"])
def test_user_aligned_temporal_pair_masks_reports_and_casebook(client, format):
    before = rgb()
    after = before.copy()
    after[20:40, 20:40] = 255
    assets = [
        upload(
            client,
            image_bytes(p, format),
            f"{role}.img",
            role,
            input_profile="exploration",
            registration_basis="pixel_grid",
        )
        for p, role in [(before, "time_a"), (after, "time_b")]
    ]
    record = analyze(client, assets, "Compare changes between the two dates")
    assert record["status"] == "succeeded", record
    result = record["result"]
    assert result["provenance"]["model_versions"]["change_vqa"].startswith(
        "satquery-spectral-change-tool"
    )
    assert len(result["evidence"]) == 2
    assert all(e["type"] == "mask" and e["geometry"]["area_m2"] is None for e in result["evidence"])
    assert (
        result["evidence"][0]["geometry"]["coverage_percent"]
        == result["evidence"][1]["geometry"]["coverage_percent"]
    )
    assert client.get(f"/v1/analyses/{record['id']}/report").content.startswith(b"%PDF")
    for item in result["evidence"]:
        assert client.get(item["artifact_url"]).content.startswith(b"\x89PNG")
        assert (
            client.get(
                f"/v1/analyses/{record['id']}/overlay?asset_id={item['asset_id']}"
            ).status_code
            == 200
        )
    assert client.get("/v1/analyses?limit=1").json()["items"][0]["id"] == record["id"]
    assert client.get("/v1/analyses?limit=1&offset=1").json()["items"] == []


def test_unreferenced_pair_without_attestation_rejected(client):
    assets = [
        upload(client, image_bytes(rgb(), "PNG"), "x.png", role, input_profile="exploration")
        for role in ("time_a", "time_b")
    ]
    record = analyze(client, assets, "Compare changes")
    assert record["status"] == "failed" and "pixel-grid" in record["error_message"]


def test_identical_temporal_pair_has_no_invented_mask(client):
    raw = tiff(np.moveaxis(rgb(), -1, 0))
    assets = [upload(client, raw, role + ".tif", role) for role in ("time_a", "time_b")]
    record = analyze(client, assets, "Compare changes")
    assert record["status"] == "succeeded", record
    assert record["result"]["evidence"] == []


@pytest.mark.parametrize("sar", [False, True])
def test_shared_nodata_denominator_matches_pair_report(client, sar):
    before = np.moveaxis(rgb(), -1, 0).copy()
    after = before.copy()
    after[:, 10:20, 10:20] = 250
    after[1, 32:, :] = 0
    assets = [
        upload(client, tiff(data, nodata=0), role + ".tif", role, "sar" if sar else "optical")
        for data, role in [(before, "time_a"), (after, "time_b")]
    ]
    record = analyze(client, assets, "Compare changes")
    assert record["status"] == "succeeded", record
    fraction = next(
        f["value"] for f in record["result"]["facts"] if f["name"] == "changed_pixel_fraction"
    )
    if sar:
        assert "bright-neutral obstruction proxy" not in record["result"]["answer"]
        assert "incidence angle" in record["result"]["answer"]
    for item in record["result"]["evidence"]:
        assert item["geometry"]["valid_pixels"] == 2048
        assert item["geometry"]["coverage_percent"] / 100 == pytest.approx(fraction, abs=1e-6)


def test_real_local_fusion_end_to_end(client):
    optical = np.moveaxis(rgb(), -1, 0)
    sar = np.full((1, 64, 64), 180, dtype="uint8")
    sar[:, 8:32, 5:30] = 20
    sar[:, 35:58, 35:60] = 230
    assets = [
        upload(client, tiff(optical), "optical.tif", "optical"),
        upload(client, tiff(sar), "sar.tif", "sar", "sar"),
    ]
    record = analyze(client, assets, "Find water and built-up regions using optical and SAR")
    assert record["status"] == "succeeded", record
    evidence = record["result"]["evidence"]
    assert evidence and all(
        e["type"] == "mask" and e["asset_id"] == assets[0]["id"] for e in evidence
    )
    for item in evidence:
        assert client.get(item["artifact_url"]).status_code == 200
    assert client.get(f"/v1/analyses/{record['id']}/report").status_code == 200
