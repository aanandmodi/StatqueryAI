"""Offline regression tests for the upload-ready pack's data/artifact handoffs."""

import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("pack_support", ROOT / "notebooks/pack_support.py")
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


def artifact(root, role):
    folder = root / role
    folder.mkdir()
    config = {
        "segmentation": {"model_type": "segformer"},
        "change": {"architecture": "shared_resnet18_gru_answer_mask"},
        "fusion": {"architecture": "terramind_s1_s2_pixel_flood_segmentation"},
    }[role]
    (folder / "config.json").write_text(json.dumps(config))
    (folder / "model.safetensors").write_bytes(b"fixture-checkpoint-not-real-weights")
    (folder / "sha256_manifest.json").write_text(
        json.dumps(
            {
                name: support.file_sha256(folder / name)
                for name in ("config.json", "model.safetensors")
            }
        )
    )
    (folder / "training_manifest.json").write_text(json.dumps({"release_candidate": True}))
    (folder / "release_gate.json").write_text(
        json.dumps(
            {
                "validation_gate_passed": True,
                "test_gate_passed": True,
            }
        )
    )
    return folder


def test_server_artifacts_require_all_roles_passing_gates_and_matching_bytes(tmp_path):
    folders = {role: artifact(tmp_path, role) for role in ("segmentation", "change", "fusion")}
    assert support.find_trained_artifacts(tmp_path) == folders
    (folders["change"] / "release_gate.json").write_text("{}")
    with pytest.raises(ValueError, match="passing validation/test"):
        support.find_trained_artifacts(tmp_path)
    (folders["fusion"] / "model.safetensors").write_bytes(b"changed-after-manifest")
    with pytest.raises(ValueError):
        support.find_trained_artifacts(tmp_path)


def test_official_csv_download_maps_channels_and_checks_each_object(tmp_path, monkeypatch):
    objects = {}
    for split in ("train", "valid", "test"):
        identifier = f"Fixture_{split}"
        objects[f"v1.1/splits/flood_handlabeled/flood_{split}_data.csv"] = (
            f"{identifier}_S1Hand.tif,{identifier}_LabelHand.tif\n".encode()
        )
        for layer in ("S1Hand", "S2Hand", "LabelHand"):
            objects[f"v1.1/data/flood_events/HandLabeled/{layer}/{identifier}_{layer}.tif"] = (
                f"fixture-bytes-{identifier}-{layer}".encode()
            )
    calls = []

    def urlopen(url, timeout):
        calls.append(url)
        if "/o/" in url:
            name = unquote(url.split("/o/", 1)[1])
            payload = objects[name]
            return io.BytesIO(
                json.dumps(
                    {
                        "size": str(len(payload)),
                        "generation": "12345",
                        "md5Hash": base64.b64encode(hashlib.md5(payload).digest()).decode(),
                    }
                ).encode()
            )
        assert urlsplit(url).query == "generation=12345"
        name = urlsplit(url).path.split("/sen1floods11/", 1)[1]
        return io.BytesIO(objects[name])

    monkeypatch.setattr(support.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(
        support.shutil, "disk_usage", lambda root: SimpleNamespace(free=10 * 1024**3)
    )
    support.prepare_flood_data(tmp_path)
    assert (tmp_path / "splits/flood_train_data.txt").read_text() == "Fixture_train\n"
    assert (
        tmp_path / "data/S2L1CHand/Fixture_train_S2Hand.tif"
    ).read_bytes() == b"fixture-bytes-Fixture_train-S2Hand"
    assert len(json.loads((tmp_path / "source_objects.json").read_text())) == 12
    initial_media_calls = sum("?generation=" in url for url in calls)
    support.prepare_flood_data(tmp_path)
    assert sum("?generation=" in url for url in calls) == initial_media_calls


def test_pack_is_standalone_and_final_server_orders_artifacts_before_tunnel():
    files = sorted((ROOT / "notebooks/kaggle-run-all").glob("*.ipynb"))
    assert len(files) == 6
    for path in files:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == "code":
                compile(cell.source, str(path), "exec")
                assert cell.execution_count is None and cell.outputs == []
    server = nbformat.read(files[-1], as_version=4)
    sources = [c.source for c in server.cells if c.cell_type == "code"]
    discovery = next(
        i for i, s in enumerate(sources) if "trained_paths = find_trained_artifacts" in s
    )
    quality = next(i for i, s in enumerate(sources) if "QUALITY_VERSION =" in s)
    pairs = next(i for i, s in enumerate(sources) if "async def paired_infer" in s)
    tunnel = next(i for i, s in enumerate(sources) if "ngrok.connect(" in s)
    connection = next(i for i, s in enumerate(sources) if "06_backend_connection.env" in s)
    assert discovery < quality < pairs < tunnel < connection < len(sources) - 1
    assert "SATQUERY_SEGMENTATION_PATH" in sources[quality]
    assert '"change": ""' not in sources[pairs]
