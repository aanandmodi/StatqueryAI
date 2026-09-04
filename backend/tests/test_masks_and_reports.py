from __future__ import annotations

import io
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.transform import from_origin

from app.api import _draw_overlay
from app.config import Settings
from app.core.narrative import audit_visual_narrative
from app.core.router import PolicyRouter
from app.core.scene_report import build_scene_sections, pixel_area_m2
from app.main import create_app
from app.models.masks import decode_mask, encode_mask, materialize_masks, spectral_water_output
from app.schemas import EvidenceItem, Modality, PlannedStep, SpecialistOutput, TaskType
from app.storage import LocalArtifactStore


def make_raster(path, *, labelled=True):
    data = np.full((3, 64, 64), 100, dtype=np.uint16)
    data[1] = 200  # NIR > green on land
    data[0, 8:56, 8:56] = 400  # synthetic water patch
    data[0, 24:40, 24:40] = 100  # an island, which must NOT be filled
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=64,
        height=64,
        count=3,
        dtype="uint16",
        crs="EPSG:32644",
        transform=from_origin(500000, 3200000, 10, 10),
    ) as target:
        target.write(data)
        if labelled:
            target.descriptions = ("green", "nir", "red")
    return data[0] > data[1]


def output():
    return SpecialistOutput(
        task=TaskType.GROUNDING,
        text="A test observation.",
        model_version="test-not-a-model",
        raw_score=0.5,
    )


def test_narrative_does_not_promote_hallucinated_percentages():
    item = output().model_copy(
        update={
            "text": "A coastal urban scene is visible. "
            "Urban areas occupy 40% of the area. "
            "Water occupies 50% but the sea is 10% of the area."
        }
    )
    safe = audit_visual_narrative(item)
    assert safe.text == "A coastal urban scene is visible."
    assert safe.facts[-1]["name"] == "unverified_original_model_text"
    assert "40%" in safe.facts[-1]["value"]


def test_narrative_keeps_computed_pair_outputs():
    item = output().model_copy(update={"task": TaskType.CHANGE_VQA, "text": "Changed pixels: 25%."})
    assert audit_visual_narrative(item).text == item.text


def test_report_does_not_claim_water_speed_from_static_pixels():
    item = output().model_copy(
        update={
            "text": "A river channel is visible. "
            "The river appears to be moving at a moderate speed. "
            "Depth cannot be determined from this image."
        }
    )
    safe = audit_visual_narrative(item)
    assert "moderate speed" not in safe.text
    assert "Depth cannot" in safe.text
    assert "moderate speed" in safe.facts[-1]["value"]


def test_narrative_collapses_repeated_vlm_sentences_without_rewriting():
    item = output().model_copy(
        update={
            "text": (
                "Direct answer: Water is visible along the western coast. "
                "The boundary follows the shoreline. "
                "The boundary follows the shoreline."
            )
        }
    )

    safe = audit_visual_narrative(item)

    assert safe.text == (
        "Direct answer: Water is visible along the western coast.\n\n"
        "The boundary follows the shoreline."
    )
    assert any("Repeated model sentences" in warning for warning in safe.warnings)
    assert safe.facts[-1]["name"] == "unverified_original_model_text"


def evidence(mask, asset_id="ast_a"):
    return EvidenceItem(
        id="untrusted-id",
        type="mask",
        label="Test candidate",
        score=0.5,
        coordinate_space="pixel",
        asset_id=asset_id,
        artifact_url="https://untrusted.example/mask.png",
        geometry={
            "data": encode_mask(mask),
            "encoding": "png-base64",
            "width": mask.shape[1],
            "height": mask.shape[0],
            "method": "test",
        },
    )


def test_binary_roundtrip_preserves_holes():
    mask = np.zeros((64, 64), bool)
    mask[8:56, 8:56] = True
    mask[24:40, 24:40] = False
    assert np.array_equal(decode_mask(evidence(mask).geometry), mask)


@pytest.mark.parametrize(
    "change",
    [{"width": 65}, {"data": "not-base64"}, {"encoding": "url"}, {"data": "a" * 1_400_000}],
)
def test_mask_decoder_rejects_bad_payloads(change):
    geometry = evidence(np.ones((32, 32), bool)).geometry | change
    with pytest.raises(ValueError):
        decode_mask(geometry)


def test_ndwi_requires_explicit_bands_and_excludes_island(tmp_path, make_asset):
    path = tmp_path / "labelled.tif"
    expected = make_raster(path)
    store = SimpleNamespace(resolve=lambda _: path)
    step = PlannedStep(
        step_id="step-1",
        task=TaskType.GROUNDING,
        asset_ids=["ast_a"],
        permitted_params={"targets": ["water"]},
        policy_reason="test",
    )
    actual = spectral_water_output(output(), step, [make_asset("ast_a")], store)
    assert np.array_equal(decode_mask(actual.evidence[0].geometry), expected)
    mixed = step.model_copy(update={"permitted_params": {"targets": ["water", "forest"]}})
    assert spectral_water_output(actual, mixed, [make_asset("ast_a")], store) is actual
    make_raster(path, labelled=False)
    rejected = spectral_water_output(output(), step, [make_asset("ast_a")], store)
    assert rejected.evidence == []
    assert any("RGB is not NIR" in text for text in rejected.warnings)
    assert (
        spectral_water_output(output(), step, [make_asset("ast_a", Modality.SAR)], store).evidence
        == []
    )


def test_materialization_recomputes_area_and_replaces_url(tmp_path, make_asset):
    path = tmp_path / "scene.tif"
    mask = make_raster(path)
    asset = make_asset("ast_a")
    # File has no nodata; metadata remains checked independently by the normal upload path.
    items = materialize_masks(
        "anl_ab",
        [evidence(mask)],
        [asset],
        SimpleNamespace(resolve=lambda _: path),
        LocalArtifactStore(tmp_path / "artifacts"),
        "/v1",
    )
    assert items[0].artifact_url == "/v1/analyses/anl_ab/masks/ev_mask_1"
    assert "data" not in items[0].geometry
    assert items[0].geometry["foreground_pixels"] == 2048
    assert items[0].geometry["area_m2"] == 204800
    assert items[0].geometry["coverage_percent"] == 50
    with pytest.raises(ValueError, match="outside"):
        materialize_masks(
            "anl_ab",
            [evidence(mask, "ast_other")],
            [asset],
            SimpleNamespace(),
            LocalArtifactStore(tmp_path / "artifacts"),
            "/v1",
        )


def test_ndwi_coverage_uses_spectral_eligible_support(tmp_path, make_asset):
    path = tmp_path / "spectral.tif"
    make_raster(path)
    with rasterio.open(path, "r+") as source:
        raw = source.read()
        raw[:2, :8, :] = 0  # finite/non-nodata but zero denominator: not usable NDWI
        source.write(raw)
    store = SimpleNamespace(resolve=lambda _: path)
    step = PlannedStep(
        step_id="step-1",
        task=TaskType.GROUNDING,
        asset_ids=["ast_a"],
        permitted_params={"targets": ["water"]},
        policy_reason="test",
    )
    output_masks = spectral_water_output(output(), step, [make_asset("ast_a")], store)
    output_masks.evidence[0].geometry["mask_group"] = []
    result = materialize_masks(
        "anl_ndwi",
        output_masks.evidence,
        [make_asset("ast_a")],
        store,
        LocalArtifactStore(tmp_path / "artifacts"),
        "/v1",
    )
    assert result[0].geometry["valid_pixels"] == 3584
    assert result[0].geometry["coverage_percent"] == pytest.approx(2048 / 3584 * 100)
    assert "mask_group" not in result[0].geometry


def test_geographic_degrees_never_become_square_metres(raster_metadata):
    geographic = raster_metadata.model_copy(update={"crs": "EPSG:4326"})
    assert pixel_area_m2(geographic) is None
    assert pixel_area_m2(raster_metadata) == 100


def test_report_separates_metadata_from_model(make_asset):
    sections = build_scene_sections([make_asset("ast_a")], "water, forest", [])
    text = "\n".join(p for section in sections for p in section.paragraphs)
    assert "0.410 km²" in text
    assert "not invented a longer visual explanation" in text
    assert "No spatial evidence" in text
    assert "drinking-water quality" in text


def test_overlay_tints_only_mask_pixels_and_keeps_holes():
    mask = np.zeros((768, 768), bool)
    mask[50:700, 50:700] = True
    mask[200:500, 200:500] = False
    source = io.BytesIO()
    Image.new("RGB", (768, 768), (100, 100, 100)).save(source, format="PNG")
    item = evidence(mask)
    binary = io.BytesIO()
    Image.fromarray(mask.astype("uint8") * 255).save(binary, format="PNG")
    rendered = _draw_overlay(source.getvalue(), [item], masks={item.id: binary.getvalue()})
    with Image.open(io.BytesIO(rendered)) as result:
        assert all(abs(value - 100) <= 2 for value in result.getpixel((350, 350)))
        assert all(abs(value - 100) <= 2 for value in result.getpixel((10, 10)))
        assert result.getpixel((100, 100))[2] > 145


@pytest.mark.parametrize(
    "query", ["Outline water bodies", "Segment lakes", "Mask the river", "Overlay ponds"]
)
def test_water_overlay_intents(query, tmp_path, make_asset):
    plan = PolicyRouter(Settings(environment="test", data_dir=tmp_path)).plan(
        query, [make_asset("ast_a")], None, {}
    )
    assert plan.steps[0].task == TaskType.GROUNDING
    assert "water" in plan.steps[0].permitted_params["targets"]


def test_overlay_and_report_use_one_gpu_request(tmp_path, make_asset):
    plan = PolicyRouter(Settings(environment="test", data_dir=tmp_path)).plan(
        "Outline water bodies and give a detailed report of this scene",
        [make_asset("ast_a")],
        None,
        {},
    )
    assert [step.task for step in plan.steps] == [TaskType.GROUNDING]


def test_mask_api_report_and_ownership_end_to_end(tmp_path):
    runtime = tmp_path / "runtime"
    settings = Settings(
        environment="test",
        data_dir=runtime,
        database_path=runtime / "db.sqlite3",
        upload_dir=runtime / "uploads",
        artifact_dir=runtime / "artifacts",
        report_dir=runtime / "reports",
        model_backend="demo",
        api_key="",
    )
    path = tmp_path / "scene.tif"
    make_raster(path)
    app = create_app(settings)
    with TestClient(app) as client:
        # Transport mock only; water mask is computed from the actual synthetic raster.
        app.state.container.gateway.infer = AsyncMock(return_value=output())
        upload = client.post(
            "/v1/assets",
            files={"file": ("scene.tif", path.read_bytes(), "image/tiff")},
            data={"modality": "multispectral"},
        )
        assert upload.status_code == 201
        asset_id = upload.json()["id"]
        record = client.post(
            "/v1/analyses", json={"query": "Outline water bodies", "asset_ids": [asset_id]}
        ).json()
        deadline = time.monotonic() + 10
        while record["status"] not in {"succeeded", "failed"} and time.monotonic() < deadline:
            time.sleep(0.03)
            record = client.get(f"/v1/analyses/{record['id']}").json()
        assert record["status"] == "succeeded", record
        result = record["result"]
        assert len(result["sections"]) >= 4
        item = result["evidence"][0]
        assert item["geometry"]["area_m2"] == 204800
        mask = client.get(item["artifact_url"])
        assert mask.status_code == 200
        with Image.open(io.BytesIO(mask.content)) as image:
            assert image.getpixel((30, 30)) == 0
            assert image.getpixel((10, 10)) == 255
        tinted = client.get(item["artifact_url"] + "?colored=true")
        with Image.open(io.BytesIO(tinted.content)) as image:
            assert image.getpixel((30, 30))[3] == 0
            assert image.getpixel((10, 10))[3] > 0
        assert client.get(item["artifact_url"].replace("ev_mask_1", "ev_mask_2")).status_code == 404
        assert client.get(f"/v1/analyses/{record['id']}/overlay").status_code == 200
        report = client.get(result["report_url"])
        assert report.status_code == 200 and report.content.startswith(b"%PDF")
