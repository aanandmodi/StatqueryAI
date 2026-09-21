"""CPU tests for bounded metrics and real-data archive handling; not GPU training claims."""

import ast
import hashlib
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def definitions(filename, names, **namespace):
    tree = ast.parse((ROOT / "notebooks" / filename).read_text(encoding="utf-8"))
    selected = [
        n for n in tree.body if isinstance(n, ast.FunctionDef | ast.ClassDef) and n.name in names
    ]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), filename, "exec"), namespace)
    return namespace


def metrics_namespace():
    labels = ["background", "forest", "water"]
    return definitions(
        "SatQuery_SegFormer_LoveDA_Training.py",
        {
            "update_confusion",
            "metrics_from_confusion",
            "StreamingSegmentationMetrics",
            "scene_scores",
        },
        np=np,
        LABELS=labels,
        ID2LABEL=dict(enumerate(labels)),
        EvalPrediction=SimpleNamespace,
    )


def test_streaming_metrics_ignore_nodata_match_full_count_and_reset():
    ns = metrics_namespace()
    metric = ns["StreamingSegmentationMetrics"]()
    references = np.array([[[0, 1], [255, 2]], [[1, 2], [1, 255]]])
    predictions = np.array([[[0, 2], [1, 2]], [[1, 2], [2, 0]]], dtype=np.uint8)
    assert metric(SimpleNamespace(predictions=predictions[:1], label_ids=references[:1])) == {}
    result = metric(SimpleNamespace(predictions=predictions[1:], label_ids=references[1:]), True)
    expected = np.zeros((3, 3), dtype=np.int64)
    ns["update_confusion"](expected, predictions, references)
    assert result == ns["metrics_from_confusion"](expected)
    assert result["pixel_accuracy"] == pytest.approx(4 / 6)
    assert result["iou_forest"] == pytest.approx(1 / 3)
    assert not metric.confusion.any()
    assert metric.confusion.nbytes == 9 * 8
    clean = metric(
        SimpleNamespace(predictions=np.array([[[1]]]), label_ids=np.array([[[1]]])), True
    )
    assert clean["mean_iou"] == 1
    scores = ns["scene_scores"](np.array([[1, 2]]), np.array([[255, 2]]))
    assert scores["iou_forest"] is None and scores["iou_water"] == 1


def second_namespace():
    return definitions(
        "SatQuery_ChangeVQA_Training.py",
        {"sha256_file", "required_second_files", "extract_second_pairs"},
        Path=Path,
        hashlib=hashlib,
        json=json,
        np=np,
    )


def png(role, value=1):
    array = np.full(
        (512, 512) if role.startswith("label") else (512, 512, 3), value, dtype=np.uint8
    )
    out = io.BytesIO()
    Image.fromarray(array).save(out, format="PNG")
    return out.getvalue()


def fixture_zip(path, omit=None, duplicate=False, unsafe=False, bad_label=False):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for role in ("im1", "im2", "label1", "label2"):
            if role == omit:
                continue
            prefix = "../train" if unsafe else "train"
            archive.writestr(f"{prefix}/{role}/00001.png", png(role, 9 if bad_label else 1))
        if duplicate:
            archive.writestr("val/im1/00001.png", png("im1"))
        archive.writestr("val/im1/unrequested.png", b"unused augmentation")


def test_second_extracts_exact_pairs_with_checksums_and_ignores_mirror_split(tmp_path, monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3))
    ns = second_namespace()
    archive = tmp_path / "fixture.zip"
    fixture_zip(archive)
    required = {(role, "00001.png") for role in ("im1", "im2", "label1", "label2")}
    dest = tmp_path / "normalized"
    hashes = ns["extract_second_pairs"](archive, dest, required)
    assert len(hashes) == 4
    assert all(ns["sha256_file"](dest / name) == digest for name, digest in hashes.items())
    assert not (dest / "im1/unrequested.png").exists()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"omit": "label2"}, "missing"),
        ({"duplicate": True}, "duplicate"),
        ({"unsafe": True}, "Unsafe"),
        ({"bad_label": True}, "index labels"),
    ],
)
def test_second_rejects_incomplete_ambiguous_unsafe_or_wrong_labels(
    tmp_path, monkeypatch, kwargs, message
):
    import shutil

    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3))
    archive = tmp_path / "bad.zip"
    fixture_zip(archive, **kwargs)
    required = {(role, "00001.png") for role in ("im1", "im2", "label1", "label2")}
    with pytest.raises(ValueError, match=message):
        second_namespace()["extract_second_pairs"](archive, tmp_path / "extracted", required)


def test_cd_vqa_manifest_refuses_split_leakage_and_path_injection(tmp_path):
    ns = second_namespace()
    for split, filename in (("Train", "00001.png"), ("Val", "00002.png"), ("Test", "00003.png")):
        (tmp_path / f"{split}_images.json").write_text(
            json.dumps({"images": [{"file_name": filename}]})
        )
    assert len(ns["required_second_files"](tmp_path)) == 12
    (tmp_path / "Val_images.json").write_text(json.dumps({"images": [{"file_name": "00001.png"}]}))
    with pytest.raises(ValueError, match="overlap"):
        ns["required_second_files"](tmp_path)
    (tmp_path / "Val_images.json").write_text(json.dumps({"images": [{"file_name": "../bad.png"}]}))
    with pytest.raises(ValueError, match="Unsafe"):
        ns["required_second_files"](tmp_path)


def test_notebook_memory_guards_and_no_manual_second_preflight():
    seg = (ROOT / "notebooks/SatQuery_SegFormer_LoveDA_Training.py").read_text(encoding="utf-8")
    assert (
        "batch_eval_metrics=True" in seg and "compute_metrics=StreamingSegmentationMetrics()" in seg
    )
    assert "train_batch_size: int = 1" in seg and "num_workers: int = 0" in seg
    assert "F.one_hot" not in seg and "trainer.save_model(OUTPUT_DIR, safe_serialization" not in seg
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"' in seg
    builder = (ROOT / "scripts/build-kaggle-pack.py").read_text(encoding="utf-8")
    assert "attach SECOND before/after imagery AND label1/label2 folders first" not in builder
