"""CPU-only contract checks; NOT a claim that SAM/Qwen GPU inference has passed."""

import ast
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from test_notebook_runtime import _functions

from app.models.masks import decode_mask
<<<<<<< HEAD
from app.core.sensors import semantic_indexes
=======
>>>>>>> 2f620623f8897788bd2df2ce4f5700cb183d84f8

PATCH = Path(__file__).resolve().parents[2] / "notebooks/patches/quality_upgrade.py"


def functions(*names, **namespace):
<<<<<<< HEAD
    namespace.setdefault("semantic_indexes", semantic_indexes)
=======
>>>>>>> 2f620623f8897788bd2df2ce4f5700cb183d84f8
    tree = ast.parse(PATCH.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in names
    ]
    for node in nodes:
        node.decorator_list = []
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PATCH), "exec"), namespace)
    return namespace


def test_quality_mask_png_matches_backend_contract():
    import base64

    namespace = functions("quality_png", base64=base64, io=io, np=np, Image=Image)
    mask = np.eye(32, dtype=bool)
    encoded = namespace["quality_png"](mask)
    assert np.array_equal(
        mask, decode_mask({"encoding": "png-base64", "data": encoded, "width": 32, "height": 32})
    )


def test_quality_decode_keeps_grid_and_nodata():
    from rasterio.enums import ColorInterp

    namespace = _functions(
        "scale_band", "rgb_band_indexes", Any=object, np=np, ColorInterp=ColorInterp
    )
    namespace = functions(
        "quality_decode",
        **namespace,
        MemoryFile=MemoryFile,
        QUALITY_IMAGE_EDGE=1024,
        MAX_UPLOAD_BYTES=50 * 1024 * 1024,
        Image=Image,
        Resampling=Resampling,
    )
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=1001,
            height=801,
            count=3,
            dtype="uint8",
            nodata=0,
            crs="EPSG:32644",
            transform=rasterio.Affine(10, 0, 500000, 0, -10, 3200000),
            photometric="RGB",
        ) as source:
            pixels = np.full((3, 801, 1001), 100, dtype="uint8")
            pixels[:, :20, :20] = 0
            source.write(pixels)
        image, valid, info = namespace["quality_decode"](memory.read())
    assert image.size == (1001, 801) and valid.shape == (801, 1001)
    assert not valid[10, 10] and valid[50, 50]
    assert info["declared_rgb"] is True


def test_report_and_masks_have_separate_provenance():
    namespace = functions(
        "quality_analyze",
        QUALITY_VERSION="v2",
        QUALITY_MAX_TOKENS=768,
        QUALITY_MAX_TARGETS=3,
        QUALITY_MAX_BOXES=4,
        MODEL_VERSION="adapter@sha",
        BASE_MODEL="qwen",
        BASE_REVISION="base-sha",
        SAM_REPO="sam",
        SAM_REVISION="sam-sha",
        context_text=lambda _: "",
        quality_decode=lambda _: (
            Image.new("RGB", (64, 64)),
            np.ones((64, 64), bool),
            {"declared_rgb": True},
        ),
        quality_generate=Mock(
            side_effect=["forest, water", "A longer spatial explanation.", "NONE"]
        ),
        parse_boxes=lambda *_: [],
    )
    contract = SimpleNamespace(
        query="Outline water",
        context=None,
        assets=[SimpleNamespace(id="ast_a", modality="optical")],
        step=SimpleNamespace(permitted_params={"targets": ["water"]}),
    )
    result = namespace["quality_analyze"](b"bytes", "grounding", contract)
    assert result["text"] == "A longer spatial explanation."
    assert result["evidence"] == []  # NONE is not a full-image mask
    assert any("no mask was invented" in text for text in result["warnings"])
    assert result["facts"][0]["model"] == "adapter@sha"
    assert result["facts"][1]["adapter_enabled"] is False


def test_updated_notebook_is_valid_and_embeds_identical_patch():
    import json

    root = PATCH.parents[2]
    notebook = json.loads(
        (root / "notebooks/SatQuery_Qwen3VL_Free_GPU_Server.ipynb").read_text(encoding="utf-8")
    )
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    quality = [cell for cell in code if "QUALITY_VERSION =" in "".join(cell["source"])]
    assert len(quality) == 1
    assert "".join(quality[0]["source"]).strip() == PATCH.read_text(encoding="utf-8").strip()
    for cell in code:
        ast.parse("".join(cell["source"]))
        assert not cell.get("outputs") and cell.get("execution_count") is None


def test_metrics_empty_masks_are_not_reported_as_perfect():
    import runpy

    metrics = runpy.run_path(str(PATCH.parents[2] / "scripts/evaluate-water-mask.py"))[
        "mask_metrics"
    ]
    empty = np.zeros((4, 4), bool)
    assert metrics(empty, empty)["iou"] is None
    reference = empty.copy()
    reference[:2] = True
    prediction = reference.copy()
    prediction[0, 0] = False
    prediction[3, 3] = True
    scores = metrics(prediction, reference)
    assert scores["precision"] == 7 / 8
    assert scores["recall"] == 7 / 8
    assert scores["iou"] == 7 / 9


def test_live_route_replacement_preserves_auth_and_multipart_contract(make_asset):
    import asyncio
    import json
    from typing import Annotated, Any, Literal

    from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
    from fastapi.testclient import TestClient
    from pydantic import BaseModel, ConfigDict, Field
    from test_notebook_runtime import SOURCE

    namespace = dict(
        BaseModel=BaseModel,
        ConfigDict=ConfigDict,
        Field=Field,
        Annotated=Annotated,
        Any=Any,
        Literal=Literal,
        Form=Form,
        File=File,
        Header=Header,
        UploadFile=UploadFile,
        HTTPException=HTTPException,
        asyncio=asyncio,
        rasterio=rasterio,
        MAX_UPLOAD_BYTES=50 * 1024 * 1024,
        SUPPORTED_TASKS={"caption", "grounding", "single_vqa"},
    )
    class_names = {"StrictModel", "Step", "Asset", "GeospatialContext", "InferencePayload"}
    nodes = [
        node
        for node in ast.parse(SOURCE).body
        if isinstance(node, ast.ClassDef) and node.name in class_names
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "notebook-schemas", "exec"), namespace)

    def authorize(value):
        if value != "Bearer test-only-credential":
            raise HTTPException(status_code=401, detail="Invalid credential")

    analyze = Mock(
        return_value={"text": "Test response, not actual model inference", "evidence": []}
    )
    namespace = functions(
        "quality_infer",
        **namespace,
        authorize=authorize,
        inference_lock=asyncio.Lock(),
        quality_analyze=analyze,
    )
    app = FastAPI()
    # This has the same original route schema used before the hotfix; changing dependant.call
    # must preserve the already-built dependency tree for multipart and Authorization.
    app.add_api_route("/v1/infer/{task}", namespace["quality_infer"], methods=["POST"])
    route = app.routes[-1]
    replacement = functions("quality_infer", **namespace)["quality_infer"]
    route.endpoint = replacement
    route.dependant.call = replacement
    asset = make_asset("ast_a").model_dump(mode="json")
    contract = {
        "query": "Outline water",
        "assets": [asset],
        "context": None,
        "step": {
            "step_id": "step-1",
            "task": "grounding",
            "asset_ids": ["ast_a"],
            "permitted_params": {"targets": ["water"]},
            "policy_reason": "test",
        },
    }
    with TestClient(app) as client:
        form = {"payload": json.dumps(contract)}
        files = {"assets": ("scene.tif", b"test-raster", "image/tiff")}
        assert client.post("/v1/infer/grounding", data=form, files=files).status_code == 401
        response = client.post(
            "/v1/infer/grounding",
            data=form,
            files=files,
            headers={"Authorization": "Bearer test-only-credential"},
        )
        assert response.status_code == 200, response.text
        analyze.assert_called_once()
        contract["step"]["asset_ids"] = ["ast_other"]
        response = client.post(
            "/v1/infer/grounding",
            data={"payload": json.dumps(contract)},
            files=files,
            headers={"Authorization": "Bearer test-only-credential"},
        )
        assert response.status_code == 422
