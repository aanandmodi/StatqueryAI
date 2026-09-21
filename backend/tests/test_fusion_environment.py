"""Offline setup-control tests. These do not install CUDA packages or run cloud training."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "notebooks/SatQuery_TerraMind_Fusion_Training.py"


def setup_namespace(versions, return_codes=(0,), modules=None):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "setup_fusion_environment"
    ]
    pins = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "FUSION_PINS" for t in node.targets)
    )
    snapshots = iter(versions)
    codes = iter(return_codes)
    commands = []
    subprocess = SimpleNamespace(
        check_call=lambda command: commands.append(command),
        run=lambda *args, **kwargs: SimpleNamespace(
            returncode=next(codes), stdout="probe", stderr="fixture import failure"
        ),
    )
    namespace = {
        "Path": Path,
        "FUSION_PINS": pins,
        "fusion_package_versions": lambda: next(snapshots),
        "subprocess": subprocess,
        "sys": SimpleNamespace(executable="python", modules=modules or {}),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["setup_fusion_environment"], commands


GPU = {"torch": "2.10.0+cu128", "torchvision": "0.25.0+cu128", "numpy": "2.2.6"}


def test_unchanged_numeric_environment_continues_and_protects_cuda(tmp_path):
    setup, commands = setup_namespace([GPU, GPU])
    setup(tmp_path)
    assert len(commands) == 1
    constraints = (tmp_path / "constraints.txt").read_text()
    assert "torch==2.10.0+cu128" in constraints
    assert "torchvision==0.25.0+cu128" in constraints
    assert "numpy==2.2.6" in constraints and "scipy==1.15.3" in constraints
    assert "transformers==4.57.1" in constraints
    assert "huggingface-hub==0.36.2" in constraints
    assert "tokenizers==0.22.1" in constraints
    assert "diffusers==0.35.1" in constraints
    assert "peft==0.17.1" in constraints and "accelerate==1.7.0" in constraints
    # These must be install targets, not just constraints: pip otherwise leaves an
    # incompatible preinstalled Transformers untouched when TerraTorch doesn't require it.
    for requirement in ("transformers==4.57.1", "huggingface-hub==0.36.2", "peft==0.17.1"):
        assert requirement in commands[0]
    assert not any("--force-reinstall" in cmd for cmd in commands)


def test_changed_environment_intentionally_stops_for_restart(tmp_path):
    setup, _ = setup_namespace([{**GPU, "numpy": "2.0.0"}, GPU])
    with pytest.raises(SystemExit, match="RESTART KERNEL"):
        setup(tmp_path)


def test_downgraded_transformers_requires_restart(tmp_path):
    setup, _ = setup_namespace(
        [{**GPU, "transformers": "5.0.0"}, {**GPU, "transformers": "4.57.1"}]
    )
    with pytest.raises(SystemExit, match="RESTART KERNEL"):
        setup(tmp_path)


def test_corrupt_numeric_install_repaired_once_and_restart_required(tmp_path):
    setup, commands = setup_namespace([GPU, GPU], return_codes=(1, 0))
    with pytest.raises(SystemExit, match="RESTART KERNEL"):
        setup(tmp_path)
    assert len(commands) == 2
    repair = commands[-1]
    assert "--force-reinstall" in repair and "--no-deps" in repair
    assert "--only-binary=:all:" in repair
    assert not any(arg.startswith("torch") for arg in repair)


def test_persistent_disk_corruption_stops_without_restart_loop(tmp_path):
    setup, commands = setup_namespace([GPU], return_codes=(1, 1))
    with pytest.raises(RuntimeError, match="fresh interpreter"):
        setup(tmp_path)
    assert len(commands) == 2


def test_stale_loaded_numpy_requires_restart_even_if_disk_matches(tmp_path):
    setup, _ = setup_namespace([GPU, GPU], modules={"numpy": SimpleNamespace(__version__="2.0.0")})
    with pytest.raises(SystemExit, match="RESTART KERNEL"):
        setup(tmp_path)


def test_missing_cloud_torch_does_not_download_replacement_cuda(tmp_path):
    setup, commands = setup_namespace([{}])
    with pytest.raises(RuntimeError, match="preinstalled"):
        setup(tmp_path)
    assert commands == []


def test_numbered_notebook_preflight_is_before_dataset_download():
    import nbformat

    notebook = nbformat.read(
        ROOT / "notebooks/kaggle-run-all/04_Train_Optical_SAR_Flood_Masks.ipynb", as_version=4
    )
    code = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    preflight = next(i for i, source in enumerate(code) if "FUSION_IMPORT_PROBE =" in source)
    download = next(
        i for i, source in enumerate(code) if "prepare_flood_data(CFG.dataset_root)" in source
    )
    assert preflight < download
    assert "numpy.testing" in code[preflight]
    assert "from terratorch.registry import BACKBONE_REGISTRY" in code[preflight]
