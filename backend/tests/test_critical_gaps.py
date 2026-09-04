"""CPU regression tests for planning, sensor contracts and dependent measurements."""

import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import numpy as np
import pytest
import rasterio
from pydantic import ValidationError
from rasterio.enums import ColorInterp

from app.config import Settings
from app.core.planner import IntentProposal, propose_intents
from app.core.router import PolicyRouter
from app.core.sensors import semantic_indexes, sensor_profile, sentinel_fusion_indexes
from app.models.mask_comparison import compare_mask_extent
from app.models.gateway import HttpSpecialistGateway
from app.models.masks import encode_mask, materialize_masks
from app.schemas import AssetRole, EvidenceItem, ExecutionPlan, SpecialistOutput, TaskType
from app.storage import LocalArtifactStore

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model_service"))


def write_raster(path, descriptions, tags=None, pixels=None):
    pixels = np.full((len(descriptions), 16, 16), 100, dtype="uint8") if pixels is None else pixels
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=16,
        height=16,
        count=len(descriptions),
        dtype=str(pixels.dtype),
        crs="EPSG:32644",
        transform=rasterio.Affine(10, 0, 500000, 0, -10, 3200000),
    ) as source:
        source.write(pixels)
        source.colorinterp = (ColorInterp.undefined,) * len(descriptions)
        source.descriptions = descriptions
        if tags:
            source.update_tags(**tags)


@pytest.mark.parametrize(
    "platform,names,indexes",
    [
        ("Cartosat-2S", ["B1", "B2", "B3", "B4"], [2, 4]),
        ("Sentinel-2", ["B02", "B03", "B04", "B08"], [2, 4]),
        ("unknown", ["B1", "B2", "B3", "B4"], None),
        ("unknown", ["blue", "green", "red", "nir"], [2, 4]),
    ],
)
def test_sensor_qualified_band_identity(tmp_path, platform, names, indexes):
    path = tmp_path / "not-evidence-of-sentinel2.tif"
    write_raster(path, names, {"SatID": platform})
    with rasterio.open(path) as source:
        assert semantic_indexes(source, ["green", "nir"]) == indexes
        if indexes:
            assert semantic_indexes(source, ["red", "green", "blue"]) == [3, 2, 1]


def test_conflicting_color_and_band_name_rejected(tmp_path):
    path = tmp_path / "conflict.tif"
    write_raster(path, ["red", "green", "blue"])
    with rasterio.open(path, "r+") as source:
        source.colorinterp = (ColorInterp.green, ColorInterp.red, ColorInterp.blue)
    with rasterio.open(path) as source, pytest.raises(ValueError, match="Conflicting"):
        sensor_profile(source)


@pytest.mark.parametrize("pols", [["RH", "RV"], ["HV", "HH"], ["VV", "VH"]])
def test_risat_headers_preserve_polarization_and_rtc(tmp_path, pols):
    path = tmp_path / "risat.tif"
    write_raster(
        path,
        ["", ""],
        {
            "SatID": "EOS-04",
            "Sensor": "SAR",
            "ProductType": "L2B",
            "TxRxPol1": pols[0],
            "TxRxPol2": pols[1],
            "RTC_Apply_Flag": "0",
        },
    )
    with rasterio.open(path) as source:
        profile = sensor_profile(source)
        assert [b["polarization"] for b in profile["bands"]] == pols
        assert profile["rtc_applied"] is False
        with pytest.raises(ValueError, match="Sentinel"):
            sentinel_fusion_indexes(source, source)


def compound_plan(make_asset, query=None):
    assets = [
        make_asset("ast_a", role=AssetRole.TIME_A),
        make_asset("ast_b", role=AssetRole.TIME_B),
    ]
    plan = PolicyRouter(Settings(environment="test", planner_backend="policy")).plan(
        query
        or "Highlight the reservoir, tell me if water level dropped compared to last month, and calculate the lost area",
        assets,
        None,
        {},
    )
    return plan, assets


def test_compound_query_has_focused_dependent_plan(make_asset):
    plan, assets = compound_plan(make_asset)
    assert len(plan.steps) == 4
    assert [step.asset_ids for step in plan.steps[:2]] == [[assets[0].id], [assets[1].id]]
    assert plan.steps[-1].depends_on == [step.step_id for step in plan.steps[:2]]
    assert plan.steps[-1].operation == "measure_mask_change"
    assert "dropped" not in plan.steps[0].query


@pytest.mark.parametrize("mutation", ["duplicate", "future", "wrong_task", "wrong_asset"])
def test_dag_rejects_invalid_dependencies(make_asset, mutation):
    plan, _ = compound_plan(make_asset)
    raw = plan.model_dump()
    if mutation == "duplicate":
        raw["steps"][-1]["depends_on"] = ["ground-before", "ground-before"]
    elif mutation == "future":
        raw["steps"][0]["depends_on"] = ["measure-extent"]
    elif mutation == "wrong_task":
        raw["steps"][0]["task"] = "caption"
    else:
        raw["steps"][-1]["asset_ids"].reverse()
    with pytest.raises(ValidationError):
        ExecutionPlan.model_validate(raw)


def test_missing_history_keeps_present_image_work(make_asset):
    plan = PolicyRouter(Settings(environment="test")).plan(
        "Compare water with last month", [make_asset("ast_a")], None, {}
    )
    assert plan.steps[0].task == TaskType.GROUNDING
    assert "second" in plan.rejected_intents[0]


def test_proposal_cannot_select_arbitrary_tools():
    with pytest.raises(ValidationError):
        IntentProposal.model_validate(
            {"objectives": ["exec"], "target": "water", "url": "https://example.com"}
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,body,learned",
    [
        (
            200,
            {
                "proposal": {"objectives": ["ground", "compare", "measure"], "target": "water"},
                "model_version": "test",
            },
            True,
        ),
        (404, {}, False),
        (200, {"proposal": {"objectives": ["exec"], "target": "water"}}, False),
    ],
)
async def test_learned_planner_and_labelled_fallback(
    monkeypatch, make_asset, status, body, learned
):
    client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(status, json=body))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw))
    proposal, source = await propose_intents(
        Settings(environment="test", model_backend="http"), "show water", [make_asset("ast_a")]
    )
    assert (proposal is not None) == learned
    assert source.startswith("learned-intent" if learned else "deterministic-fallback")


def mask_output(asset, mask, target="water", method="test segmentation", threshold=0.5):
    return SpecialistOutput(
        task=TaskType.GROUNDING,
        text="Candidate water",
        raw_score=0.5,
        model_version="test-model-sha",
        evidence=[
            EvidenceItem(
                id=f"mask_{asset.id}",
                type="mask",
                label="not a trusted semantic label",
                score=0.5,
                coordinate_space="pixel",
                asset_id=asset.id,
                geometry={
                    "encoding": "png-base64",
                    "data": encode_mask(mask),
                    "width": 16,
                    "height": 16,
                    "target": target,
                    "method": method,
                    "threshold": threshold,
                },
            )
        ],
    )


def measurement_setup(tmp_path, make_asset):
    plan, assets = compound_plan(make_asset)
    for asset in assets:
        write_raster(tmp_path / f"{asset.id}.tif", ["red", "green", "blue"])
        asset.metadata.width = asset.metadata.height = 16
    store = SimpleNamespace(resolve=lambda key: tmp_path / f"{key}.tif")
    a, b = np.zeros((16, 16), bool), np.zeros((16, 16), bool)
    a[:4, :4], b[2:6, :4] = True, True
    deps = {"ground-before": mask_output(assets[0], a), "ground-after": mask_output(assets[1], b)}
    return plan.steps[-1], assets, store, deps


def test_loss_gain_and_final_artifact_counts_agree(tmp_path, make_asset):
    step, assets, store, deps = measurement_setup(tmp_path, make_asset)
    output = compare_mask_extent(step, deps, assets, store)
    counts = output.facts[0]["pixel_counts"]
    assert counts["lost"] == counts["gained"] == counts["retained"] == 8
    assert counts["net_change"] == 0
    assert output.facts[0]["estimated_area_m2"]["lost"] == 800
    items = materialize_masks(
        "test", output.evidence, assets, store, LocalArtifactStore(tmp_path / "artifacts"), "/v1"
    )
    assert all(item.geometry["foreground_pixels"] == 8 for item in items)


@pytest.mark.parametrize("failure", ["missing", "target", "threshold", "method", "grid"])
def test_measurement_abstains_without_compatible_evidence(tmp_path, make_asset, failure):
    step, assets, store, deps = measurement_setup(tmp_path, make_asset)
    item = deps["ground-after"].evidence[0]
    if failure == "missing":
        deps["ground-after"].evidence = []
    elif failure == "grid":
        with rasterio.open(store.resolve(assets[1].id), "r+") as source:
            source.transform = rasterio.Affine(10, 0, 500001, 0, -10, 3200000)
    else:
        item.geometry[failure] = {"target": "not water", "threshold": 0.7, "method": ["invalid"]}[
            failure
        ]
    output = compare_mask_extent(step, deps, assets, store)
    assert output.model_version.endswith(":abstained")
    assert output.evidence == []


def test_architecture_exports_match_notebooks_without_loading_torch():
    runtime = ast.parse(
        (ROOT / "ml/satquery_ml/models/notebook_experts.py").read_text(encoding="utf-8")
    )
    for name, notebook in [
        ("ChangeExpert", "SatQuery_ChangeVQA_Training"),
        ("FusionExpert", "SatQuery_TerraMind_Fusion_Training"),
    ]:
        training = ast.parse((ROOT / f"notebooks/{notebook}.py").read_text(encoding="utf-8"))
        a = next(
            node for node in runtime.body if isinstance(node, ast.ClassDef) and node.name == name
        )
        b = next(
            node for node in training.body if isinstance(node, ast.ClassDef) and node.name == name
        )
        assert ast.dump(a) == ast.dump(b)


def test_sensor_runtime_mirror_and_notebooks_compile():
    assert (ROOT / "backend/app/core/sensors.py").read_text() == (
        ROOT / "ml/satquery_ml/sensors.py"
    ).read_text()
    import nbformat

    for name in (
        "SatQuery_Qwen3VL_Free_GPU_Server",
        "SatQuery_ChangeVQA_Training",
        "SatQuery_TerraMind_Fusion_Training",
        "SatQuery_Live_Planning_Upgrade",
        "SatQuery_Optional_Paired_Runtime",
    ):
        notebook = nbformat.read(ROOT / f"notebooks/{name}.ipynb", as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                compile(cell.source, name, "exec")


def test_ndwi_invalid_pixels_are_not_counted_as_target_loss(tmp_path, make_asset):
    step, assets, store, deps = measurement_setup(tmp_path, make_asset)
    method = "NDWI (green - NIR) / (green + NIR)"
    before_pixels = np.full((2, 16, 16), 100, dtype=np.float32)
    after_pixels = before_pixels.copy()
    after_pixels[:, :4, :4] = 0  # zero sum is invalid spectral support, not absent water
    write_raster(store.resolve(assets[0].id), ["green", "nir"], pixels=before_pixels)
    write_raster(store.resolve(assets[1].id), ["green", "nir"], pixels=after_pixels)
    before, after = np.zeros((16, 16), bool), np.zeros((16, 16), bool)
    before[:4, :4] = True
    deps = {
        "ground-before": mask_output(assets[0], before, method=method, threshold=0),
        "ground-after": mask_output(assets[1], after, method=method, threshold=0),
    }
    result = compare_mask_extent(step, deps, assets, store)
    assert result.facts[0]["pixel_counts"]["lost"] == 0
    assert result.facts[0]["pixel_counts"]["shared_valid"] == 240
    items = materialize_masks(
        "test", result.evidence, assets, store, LocalArtifactStore(tmp_path / "artifacts"), "/v1"
    )
    assert all(item.geometry["foreground_pixels"] == 0 for item in items)


def test_learned_change_support_and_readonly_pil_arrays():
    from satquery_model_service.paired_adapters import (
        require_shared_support,
        resample_supported_mask,
    )

    with pytest.raises(ValueError, match="no shared valid"):
        require_shared_support(np.zeros((10, 12), bool))
    valid = np.ones((10, 12), bool)
    valid[:3] = False
    result = resample_supported_mask(np.ones((4, 4), bool), valid)
    assert result.shape == (10, 12) and result.sum() == 7 * 12


def test_split_local_annotation_ids_do_not_define_leakage():
    tree = ast.parse(
        (ROOT / "notebooks/SatQuery_ChangeVQA_Training.py").read_text(encoding="utf-8")
    )
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in {"train_images", "validation_images"}
            for target in node.targets
        )
    ]
    namespace = {
        "train_records": [
            {"image_id": 0, "filename": "10589.png"},
            {"image_id": 1, "filename": "10589.png"},
        ],
        "validation_records": [{"image_id": 0, "filename": "02180.png"}],
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "split_identity", "exec"), namespace)
    assert namespace["train_images"] == {"10589.png"}
    assert not namespace["train_images"] & namespace["validation_images"]


@pytest.mark.asyncio
async def test_pair_webp_preserves_bytes_hash_and_legacy_step_contract(tmp_path, make_asset):
    from PIL import Image

    path = tmp_path / "scene.webp"
    Image.new("RGB", (16, 16), (30, 80, 140)).save(path, lossless=True)
    raw = path.read_bytes()
    plan, assets = compound_plan(make_asset)
    for asset in assets:
        asset.metadata.driver = "WEBP"
        asset.original_name = "scene.webp"
        asset.sha256 = hashlib.sha256(raw).hexdigest()
    gateway = HttpSpecialistGateway(
        Settings(environment="test"),
        SimpleNamespace(resolve=lambda key: path),
        allowed_tasks={TaskType.CHANGE_VQA},
    )
    await gateway.client.aclose()
    response = httpx.Response(
        200,
        json=SpecialistOutput(
            task=TaskType.CHANGE_VQA, text="test", raw_score=0.5, model_version="mock-not-a-model"
        ).model_dump(),
        request=httpx.Request("POST", "http://test"),
    )
    gateway.client = SimpleNamespace(post=AsyncMock(return_value=response))
    await gateway.infer(plan.steps[2], assets, "test query")
    call = gateway.client.post.call_args.kwargs
    # Closed handles are acceptable; multipart descriptors still prove original paths vs PNG buffers.
    assert all(part[1][1].name == str(path) for part in call["files"])
    payload = json.loads(call["data"]["payload"])
    assert all(asset["sha256"] == hashlib.sha256(raw).hexdigest() for asset in payload["assets"])
    assert not {"operation", "depends_on", "query"} & payload["step"].keys()
    assert "sensor_profile" not in payload["assets"][0]["metadata"]
