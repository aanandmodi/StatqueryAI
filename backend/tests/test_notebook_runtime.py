"""Exercise notebook code without installing torch, downloading weights, or opening a tunnel."""

from __future__ import annotations

import ast
import io
import json
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import rasterio
from PIL import Image, UnidentifiedImageError
from rasterio.enums import ColorInterp, Resampling
from rasterio.io import MemoryFile

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "SatQuery_Qwen3VL_Free_GPU_Server.py"
SOURCE = NOTEBOOK.read_text(encoding="utf-8")
COLAB_MARKERS = (
    "COLAB_RELEASE_TAG", "COLAB_BACKEND_VERSION", "COLAB_JUPYTER_IP", "COLAB_GPU",
)
REQUESTS_TYPES = SimpleNamespace(Response=object)


def _cell(section: int) -> str:
    after_heading = SOURCE.split(f"# ## {section}. ", 1)[1]
    return after_heading.split("# %%\n", 1)[1].split("# %% [markdown]", 1)[0].strip()


def _functions(*names: str, **namespace: Any) -> dict[str, Any]:
    if "rgb_band_indexes" in names:
        names = (*names, "compact", "sensor_profile", "semantic_indexes", "visual_indexes")
        namespace.setdefault("re", re)
    nodes = [
        node for node in ast.parse(SOURCE).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(nodes) == len(names)
    for node in nodes:
        # Testing generate's argument handling needs no torch decorator or GPU.
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


def _tunnel_namespace(environ=None, *, working_exists=True):
    ngrok = Mock()
    ngrok.connect.return_value.public_url = "https://test.ngrok-free.app"
    server = SimpleNamespace(started=True, should_exit=False)
    server_thread = Mock()
    server_thread.is_alive.return_value = True
    return {
        "environ": environ if environ is not None else {
            "KAGGLE_KERNEL_RUN_TYPE": "Interactive", "COLAB_RELEASE_TAG": "inherited-image",
        },
        "Path": Mock(return_value=SimpleNamespace(is_dir=lambda: working_exists)),
        "threading": SimpleNamespace(Timer=Mock()),
        "time": SimpleNamespace(monotonic=lambda: 100.0),
        "Mapping": Mapping,
        "ngrok": ngrok,
        "server": server,
        "server_thread": server_thread,
        "NGROK_AUTHTOKEN": "fake-ngrok-secret-not-for-use",
        "SERVICE_TOKEN": "fake-model-service-secret-not-for-use",
    }


def _run_tunnel_cell(namespace):
    # Inject test doubles instead of importing/running the real network or timer clients.
    tree = ast.parse(_cell(8))
    tree.body = [node for node in tree.body if not isinstance(node, ast.Import | ast.ImportFrom)]
    exec(compile(tree, str(NOTEBOOK), "exec"), namespace)


@pytest.mark.parametrize("marker", COLAB_MARKERS)
def test_kaggle_runtime_wins_over_inherited_colab_base_image_markers(marker):
    namespace = _tunnel_namespace({"KAGGLE_KERNEL_RUN_TYPE": "Interactive", marker: "1"})
    _run_tunnel_cell(namespace)
    assert namespace["runtime"] == "kaggle"
    namespace["ngrok"].connect.assert_called_once_with(
        addr="http://127.0.0.1:8080", proto="http", bind_tls=True,
    )


@pytest.mark.parametrize("marker", COLAB_MARKERS)
def test_actual_colab_is_still_blocked_without_opening_tunnel(marker):
    namespace = _tunnel_namespace({marker: "1"}, working_exists=False)
    with pytest.raises(RuntimeError, match="managed Colab"):
        _run_tunnel_cell(namespace)
    assert namespace["ngrok"].mock_calls == []


@pytest.mark.parametrize(
    ("environ", "working_exists"),
    [({}, False), ({}, True), ({"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}, False)],
)
def test_unverified_runtime_is_blocked(environ, working_exists):
    namespace = _tunnel_namespace(environ, working_exists=working_exists)
    with pytest.raises(RuntimeError, match="Could not verify a Kaggle"):
        _run_tunnel_cell(namespace)
    assert namespace["ngrok"].mock_calls == []


def test_missing_or_stopped_server_has_recovery_instructions():
    namespace = _tunnel_namespace()
    del namespace["server"]
    with pytest.raises(RuntimeError, match="Missing session state"):
        _run_tunnel_cell(namespace)
    assert namespace["ngrok"].mock_calls == []

    namespace = _tunnel_namespace()
    namespace["server"].should_exit = True
    with pytest.raises(RuntimeError, match="sections 6 and 7"):
        _run_tunnel_cell(namespace)
    assert namespace["ngrok"].mock_calls == []


def test_tunnel_keeps_bounded_shutdown_and_does_not_print_credentials(capsys):
    namespace = _tunnel_namespace()
    _run_tunnel_cell(namespace)
    timer_factory = namespace["threading"].Timer
    timer_factory.assert_called_once_with(
        3600, namespace["stop_demo"], args=(
            namespace["server"], namespace["server_thread"], namespace["PUBLIC_MODEL_URL"],
        ),
    )
    assert namespace["DEMO_EXPIRES_AT"] == 3700
    assert timer_factory.return_value.daemon is True
    timer_factory.return_value.start.assert_called_once()
    output = capsys.readouterr().out
    assert namespace["NGROK_AUTHTOKEN"] not in output
    assert namespace["SERVICE_TOKEN"] not in output

    # The scheduled callback owns the original server even after globals are replaced.
    old_server = namespace["server"]
    namespace["server"] = SimpleNamespace(should_exit=False)
    timer_factory.call_args.args[1](*timer_factory.call_args.kwargs["args"])
    assert old_server.should_exit is True
    assert namespace["server"].should_exit is False


def test_failed_reconnect_does_not_leave_stale_public_url():
    namespace = _tunnel_namespace()
    namespace.update(PUBLIC_MODEL_URL="https://old.ngrok-free.app", DEMO_EXPIRES_AT=500)
    previous_timer = namespace["demo_shutdown_timer"] = Mock()
    namespace["ngrok"].connect.side_effect = RuntimeError("ngrok unavailable")
    with pytest.raises(RuntimeError, match="ngrok unavailable"):
        _run_tunnel_cell(namespace)
    namespace["ngrok"].disconnect.assert_called_once_with("https://old.ngrok-free.app")
    previous_timer.cancel.assert_called_once()
    assert "PUBLIC_MODEL_URL" not in namespace
    assert "DEMO_EXPIRES_AT" not in namespace


def test_non_https_tunnel_is_closed_instead_of_exposing_bearer_token():
    namespace = _tunnel_namespace()
    namespace["ngrok"].connect.return_value.public_url = "http://test.ngrok-free.app"
    with pytest.raises(RuntimeError, match="HTTPS tunnel"):
        _run_tunnel_cell(namespace)
    namespace["ngrok"].disconnect.assert_called_once_with("http://test.ngrok-free.app")
    assert "PUBLIC_MODEL_URL" not in namespace


def test_section_9_without_tunnel_explains_order_instead_of_name_error():
    with pytest.raises(RuntimeError, match="Run section 8 successfully"):
        exec(compile(_cell(9), str(NOTEBOOK), "exec"), {})


def test_smoke_geotiff_round_trip_has_no_missing_geo_or_pillow_warnings():
    namespace = _functions(
        "make_smoke_geotiff", "scale_band", "rgb_band_indexes", "decode_uploaded_image",
        Image=Image, np=np, MemoryFile=MemoryFile, Any=Any,
        rasterio=rasterio, io=io, UnidentifiedImageError=UnidentifiedImageError,
        ColorInterp=ColorInterp, Resampling=Resampling, MAX_UPLOAD_BYTES=50_000_000,
    )
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    pixels[:] = (45, 75, 200)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        warnings.simplefilter("error", rasterio.errors.NotGeoreferencedWarning)
        payload = namespace["make_smoke_geotiff"](Image.fromarray(pixels))
        actual = np.asarray(namespace["decode_uploaded_image"](payload))
    np.testing.assert_array_equal(actual, pixels)
    with MemoryFile(payload) as memory, memory.open() as dataset:
        assert dataset.crs.to_epsg() == 32644
        assert dataset.transform == rasterio.Affine(10, 0, 500000, 0, -10, 3200000)
        assert dataset.tags()["synthetic"] == "true"


@pytest.mark.parametrize(("requested", "expected"), [(0, 1), (32, 32), (999, 128)])
def test_generation_overrides_sampling_defaults_without_reloading_model(requested, expected):
    model = Mock(device="cuda:0")
    model.generate.return_value = np.array([[1, 2, 3]])
    processor = Mock()
    tensor = Mock(shape=(1, 2))
    tensor.to.return_value = tensor
    processor.return_value = {"input_ids": tensor}
    processor.batch_decode.return_value = [" Test answer "]
    namespace = _functions(
        "generate", Image=Image, model=model, processor=processor,
        process_vision_info=Mock(return_value=(["image"], None)),
    )
    assert namespace["generate"](Image.new("RGB", (32, 32)), "Question", requested) == "Test answer"
    kwargs = model.generate.call_args.kwargs
    assert kwargs["max_new_tokens"] == expected
    assert kwargs["do_sample"] is False
    assert kwargs["use_cache"] is True
    assert (kwargs["temperature"], kwargs["top_p"], kwargs["top_k"]) == (1.0, 1.0, 50)


def test_http_errors_report_status_before_parsing_html():
    function = _functions("checked_model_response", requests=REQUESTS_TYPES, Any=Any)[
        "checked_model_response"
    ]
    response = Mock(status_code=502)
    response.raise_for_status.side_effect = RuntimeError("502 Bad Gateway")
    with pytest.raises(RuntimeError, match="502"):
        function(response, "Public ngrok HTTP")
    response.json.assert_not_called()


@pytest.mark.parametrize("body", [[], {}, {"text": "  "}, {"text": None}])
def test_empty_or_invalid_model_json_does_not_pass_smoke_test(body):
    function = _functions("checked_model_response", requests=REQUESTS_TYPES, Any=Any)[
        "checked_model_response"
    ]
    response = Mock(status_code=200)
    response.json.return_value = body
    with pytest.raises(RuntimeError, match="non-empty model text"):
        function(response, "test")


def test_html_in_place_of_json_has_actionable_error():
    function = _functions("checked_model_response", requests=REQUESTS_TYPES, Any=Any)[
        "checked_model_response"
    ]
    response = Mock(status_code=200)
    response.json.side_effect = ValueError("HTML instead of JSON")
    with pytest.raises(RuntimeError, match="exact ngrok URL"):
        function(response, "test")


def test_copyable_section_8_matches_notebook():
    replacement = ROOT / "notebooks" / "patches" / "section_8_ngrok.py"
    assert ast.dump(ast.parse(replacement.read_text(encoding="utf-8"))) == ast.dump(
        ast.parse(_cell(8))
    )


def test_ipynb_is_synchronized_compiles_and_contains_no_saved_execution_state():
    notebook = json.loads(NOTEBOOK.with_suffix(".ipynb").read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for index, cell in enumerate(code_cells):
        compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
    combined = "\n\n".join("".join(cell["source"]) for cell in code_cells)
    assert ast.dump(ast.parse(combined)) == ast.dump(ast.parse(SOURCE))
