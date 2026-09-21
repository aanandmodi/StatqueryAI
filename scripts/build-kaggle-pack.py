"""Build ordered, standalone Kaggle notebooks from the repository's maintained sources."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import zipfile
from pathlib import Path

import jupytext
import nbformat

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "notebooks" / "kaggle-run-all"
SUPPORT = (ROOT / "notebooks/pack_support.py").read_text(encoding="utf-8")


def notebook(name):
    source = (ROOT / "notebooks" / f"{name}.py").read_text(encoding="utf-8")
    return jupytext.reads(source.replace("# ruff: noqa: E402\n", ""), fmt="py:percent")


def substitute(nb, old, new):
    count = 0
    for cell in nb.cells:
        if cell.cell_type == "code" and old in cell.source:
            count += cell.source.count(old)
            cell.source = cell.source.replace(old, new)
    if count != 1:
        raise ValueError(f"Expected one substitution for {old!r}, got {count}")


def before_code(nb, marker, extra):
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code" and marker in cell.source:
            nb.cells[i:i] = extra
            return
    raise ValueError(f"Missing insertion point: {marker}")


def intro(title, text):
    return nbformat.v4.new_markdown_cell(f"# {title}\n\n{text}")


def code(source):
    return nbformat.v4.new_code_cell(source.strip() + "\n")


def write(nb, name):
    nb.cells = [c for c in nb.cells if c.source.strip()]
    for index, cell in enumerate(nb.cells):
        cell.id = hashlib.sha256(f"{name}:{index}:{cell.source}".encode()).hexdigest()[
            :12
        ]
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
            ast.parse(cell.source, filename=f"{name}:cell-{index}")
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["kaggle"] = {
        "isInternetEnabled": True,
        "isGpuEnabled": True,
        "language": "python",
    }
    nbformat.validate(nb)
    nbformat.write(nb, DEST / name)


def scorer_bundle():
    files = {
        "ml/satquery_ml/__init__.py": "",
        "ml/satquery_ml/evaluation.py": (
            ROOT / "ml/satquery_ml/evaluation.py"
        ).read_text(encoding="utf-8"),
        "scripts/score.py": (ROOT / "scripts/score-real-evaluation.py").read_text(
            encoding="utf-8"
        ),
    }
    return code(
        "import json, runpy, sys\nfrom pathlib import Path\n"
        "scorer_root = Path('/kaggle/working/satquery-scorer')\n"
        f"scorer_files = {files!r}\n"
        "for name, source in scorer_files.items():\n"
        "    target = scorer_root / name\n"
        "    target.parent.mkdir(parents=True, exist_ok=True)\n"
        "    target.write_text(source, encoding='utf-8')\n"
        "scoring = runpy.run_path(str(scorer_root / 'scripts/score.py'))\n"
    )


def build_evaluations():
    common = notebook("SatQuery_Qwen3VL_Evaluation_Only")
    validation = copy.deepcopy(common)
    validation.cells.insert(
        0,
        intro(
            "01 — Qwen validation + raw predictions + CPU scoring",
            "New Kaggle notebook → GPU T4/P100 → Internet On → Run All. No dataset attachment needed. "
            "The existing released Qwen adapter is evaluated; training is already complete. "
            "At the end, save a version with outputs. Notebook 05 needs this notebook's saved outputs.",
        ),
    )
    validation.cells += [
        intro(
            "CPU scoring and downloadable evidence",
            "This cell computes bootstrap intervals and a validation-only calibration diagnostic. "
            "Save the raw JSONL, summary and scorecard together; it does not change the app's confidence cap.",
        ),
        scorer_bundle(),
        code(
            "prediction_path = OUTPUT_DIR / 'validation_predictions.jsonl'\n"
            "subprocess.check_call([sys.executable, str(scorer_root / 'scripts/score.py'), "
            "str(prediction_path), '--output', str(OUTPUT_DIR / 'scored')])\n"
        ),
        code(SUPPORT),
        code(
            "export_inference_zip(OUTPUT_DIR, '/kaggle/working/01_qwen_validation_evidence.zip')"
        ),
    ]
    write(validation, "01_Qwen_Validation_and_Scorecard.ipynb")

    test = copy.deepcopy(common)
    substitute(test, 'split: str = "validation"', 'split: str = "test"')
    test.cells.insert(
        0,
        intro(
            "05 — Final Qwen test + frozen validation calibration check",
            "Run once after reviewing notebook 01 and freezing model choices. Add Input → Notebook Output "
            "→ the saved output of 01. Enable GPU and Internet, then Run All. It evaluates the same pinned "
            "Qwen checkpoint on test; temperature is read from validation and never fitted to test.",
        ),
    )
    test.cells.insert(
        1,
        code(
            "import json\nfrom pathlib import Path\n"
            "candidates = [p for p in Path('/kaggle/input').rglob('validation_predictions.jsonl') "
            "if (p.parent / 'scored/scorecard.json').is_file()]\n"
            "if len(candidates) != 1:\n"
            "    raise RuntimeError('Attach exactly the saved output of notebook 01 via Add Input → Notebook Output.')\n"
            "validation_raw = candidates[0]\n"
            "frozen_scorecard_path = validation_raw.parent / 'scored/scorecard.json'\n"
            "frozen_scorecard = json.loads(frozen_scorecard_path.read_text())\n"
        ),
    )
    test.cells += [
        scorer_bundle(),
        code(
            "test_rows = scoring['load_rows'](OUTPUT_DIR / 'test_predictions.jsonl')\n"
            "validation_rows = scoring['load_rows'](validation_raw)\n"
            "if {r['patch_id'] for r in test_rows} & {r['patch_id'] for r in validation_rows}:\n"
            "    raise ValueError('Validation/test patch leakage detected.')\n"
            "validation_summary = json.loads((validation_raw.parent / 'validation_summary.json').read_text())\n"
            "for key in ('adapter_revision', 'base_revision', 'dataset_revision', 'image_dataset_revision'):\n"
            "    if validation_summary[key] != summary[key]:\n"
            "        raise ValueError('Validation/test revision mismatch: ' + key)\n"
            "test_scorecard = {p: scoring['summarize'](test_rows, p, 2000, 42) for p in ('base', 'lora')}\n"
            "frozen_results = {}\n"
            "for prefix in ('base', 'lora'):\n"
            "    fitted = frozen_scorecard['calibration'][prefix]\n"
            "    if 'temperature' not in fitted:\n"
            "        frozen_results[prefix] = {'status': 'insufficient validation data; no temperature fitted'}\n"
            "        continue\n"
            "    rows = [r for r in test_rows if r['type'] in {'binary', 'mcq'}]\n"
            "    probabilities = np.clip(np.asarray([r[prefix + '_sequence_confidence'] for r in rows]), 1e-6, 1-1e-6)\n"
            "    labels = np.asarray([int(r[prefix + '_correct']) for r in rows])\n"
            "    temperature = float(fitted['temperature'])\n"
            "    logits = np.stack([np.log1p(-probabilities), np.log(probabilities)], axis=1) / temperature\n"
            "    weights = np.exp(logits - logits.max(axis=1, keepdims=True))\n"
            "    calibrated = (weights / weights.sum(axis=1, keepdims=True))[:, 1]\n"
            "    ece = scoring['expected_calibration_error']\n"
            "    frozen_results[prefix] = {'temperature_from_validation': temperature, 'examples': len(rows),\n"
            "        'raw_ece': ece(probabilities, labels), 'frozen_temperature_test_ece': ece(calibrated, labels)}\n"
            "test_scorecard['calibration_test'] = frozen_results\n"
            "test_scorecard['automatic_probability_release'] = False\n"
            "test_scorecard['scope'] = 'Qwen binary/MCQ correctness diagnostics; excludes mask confidence and ISRO transfer.'\n"
            "(OUTPUT_DIR / 'test_scorecard.json').write_text(json.dumps(test_scorecard, indent=2))\n"
            "print(json.dumps(test_scorecard, indent=2))\n"
        ),
        code(SUPPORT),
        code(
            "export_inference_zip(OUTPUT_DIR, '/kaggle/working/05_qwen_test_evidence.zip')"
        ),
    ]
    write(test, "05_Qwen_Final_Test.ipynb")


def build_training():
    seg = notebook("SatQuery_SegFormer_LoveDA_Training")
    seg.cells.insert(
        0,
        intro(
            "02 — Train your vegetation/water/land-cover mask model",
            "GPU T4/P100, Internet On, then Run All. Official LoveDA data downloads automatically. "
            "Memory-safe profile: 384px crops, batch 1, gradient accumulation 8, streaming validation. "
            "70% of official training data is used; validation is preserved. Save Version with outputs "
            "when finished. Final server notebook 06 loads this model if its validation gate passes.",
        ),
    )
    seg.cells += [
        code(SUPPORT),
        code(
            "export_inference_zip(OUTPUT_DIR, '/kaggle/working/02_segmentation_inference.zip')"
        ),
    ]
    write(seg, "02_Train_Vegetation_Water_Masks.ipynb")

    change = notebook("SatQuery_ChangeVQA_Training")
    change.cells.insert(
        0,
        intro(
            "03 — Train learned temporal change answers and masks",
            "No dataset attachment required. GPU T4/P100, Internet On, then Run All. "
            "CDVQA questions and matching SECOND before/after images and semantic label maps download "
            "automatically from a pinned PerASCD research mirror (3.78 GB archive; allow 10 GiB free). "
            "Files are verified against the archive SHA256 and original CDVQA split filenames. Prototype acceptance targets are "
            "answer accuracy ≥0.60 and mask IoU ≥0.40, declared before this run. They are project targets, "
            "not benchmark guarantees. Test is run after validation; failing artifacts cannot enter notebook 06.",
        ),
    )
    substitute(change, "run_test_once: bool = False", "run_test_once: bool = True")
    substitute(
        change,
        "release_minimum_answer_accuracy: float | None = None",
        "release_minimum_answer_accuracy: float | None = 0.60",
    )
    substitute(
        change,
        "release_minimum_mask_iou: float | None = None",
        "release_minimum_mask_iou: float | None = 0.40",
    )
    change.cells += [
        code(SUPPORT),
        code(
            "(ARTIFACTS / 'release_gate.json').write_text(json.dumps(gate, indent=2))\n"
            "export_inference_zip(ARTIFACTS, '/kaggle/working/03_change_inference.zip')\n"
            "print('CANDIDATE PASSED' if gate['validation_gate_passed'] and gate['test_gate_passed'] "
            "else 'NEEDS IMPROVEMENT — review metrics before a new experiment; do not lower targets to pass.')\n"
        ),
    ]
    write(change, "03_Train_Temporal_Change.ipynb")

    fusion = notebook("SatQuery_TerraMind_Fusion_Training")
    fusion.cells.insert(
        0,
        intro(
            "04 — Train optical/SAR flood masks",
            "GPU T4/P100 and Internet On. Run setup first; if it prints SETUP COMPLETE — RESTART KERNEL, "
            "restart the Python kernel without ending the session, then Run All. The pinned numeric stack "
            "and early import preflight prevent continuing with stale NumPy binaries. "
            "This edition automatically downloads only the "
            "official hand-labelled Sen1Floods11 S1/S2 triplets. The project target is flood IoU ≥0.50 "
            "on validation and test, with fused validation at least as good as either modality alone. "
            "It covers Sentinel flood/water segmentation, not universal land cover or Cartosat/RISAT transfer.",
        ),
    )
    substitute(
        fusion,
        'dataset_root: str = ""',
        'dataset_root: str = "/kaggle/working/sen1floods11-v1.1"',
    )
    substitute(fusion, "run_test_once: bool = False", "run_test_once: bool = True")
    substitute(
        fusion,
        "release_minimum_flood_iou: float | None = None",
        "release_minimum_flood_iou: float | None = 0.50",
    )
    before_code(
        fusion,
        "def looks_like_dataset",
        [code(SUPPORT), code("prepare_flood_data(CFG.dataset_root)\n")],
    )
    fusion.cells += [
        code(
            "shutil.copy2(Path(CFG.dataset_root) / 'source_objects.json', ARTIFACTS / 'source_objects.json')\n"
            "(ARTIFACTS / 'release_gate.json').write_text(json.dumps(gate, indent=2))\n"
            "export_inference_zip(ARTIFACTS, '/kaggle/working/04_fusion_inference.zip')\n"
            "print('CANDIDATE PASSED' if gate['validation_gate_passed'] and gate['test_gate_passed'] "
            "else 'NEEDS IMPROVEMENT — retain metrics and review errors before another experiment.')\n"
        )
    ]
    write(fusion, "04_Train_Optical_SAR_Flood_Masks.ipynb")


def build_server():
    nb = notebook("SatQuery_Qwen3VL_Free_GPU_Server")
    nb.cells[0] = intro(
        "06 — Start all trained models + quality reports + planner + ngrok",
        "Attach the saved outputs of 02, 03 and 04 using Add Input → Notebook Output. Enable GPU T4 "
        "and Internet. Enable Secrets HF_TOKEN, NGROK_AUTHTOKEN and SATQUERY_MODEL_SERVICE_TOKEN. "
        "Then Run All. The first cell verifies checkpoint files before installing/loading models. "
        "The final cell stays running during your attended demo and closes the tunnel after at most "
        "60 minutes. This is the same backend API contract you already use. The timer is our notebook's "
        "limit, not a promise of any provider's quota. No manual quality/paired cells need pasting.",
    )
    nb.cells[1:1] = [
        code(SUPPORT),
        code(
            "import os\nfrom kaggle_secrets import UserSecretsClient\n"
            "for name in ('HF_TOKEN', 'NGROK_AUTHTOKEN', 'SATQUERY_MODEL_SERVICE_TOKEN'):\n"
            "    value = UserSecretsClient().get_secret(name).strip()\n"
            "    if not value:\n        raise RuntimeError('Missing Kaggle Secret: ' + name)\n"
            "    os.environ[name] = value\n"
            "del value\n"
            "trained_paths = find_trained_artifacts('/kaggle/input')\n"
            "SATQUERY_SEGMENTATION_PATH = str(trained_paths['segmentation'])\n"
            "SATQUERY_SEGMENTATION_SHA = file_sha256(trained_paths['segmentation'] / 'model.safetensors')\n"
            "print('Verified attached trained checkpoints:', {k: str(v) for k, v in trained_paths.items()})\n"
        ),
    ]
    substitute(
        nb,
        '"huggingface-hub==0.34.4",',
        '"huggingface-hub==0.36.2",\n    "terratorch>=1.2.5,<2",\n    "setuptools<81",',
    )
    optional = nbformat.read(
        ROOT / "notebooks/SatQuery_Optional_Paired_Runtime.ipynb", as_version=4
    )
    runtime_cells = [copy.deepcopy(c) for c in optional.cells if c.cell_type == "code"]
    for cell in runtime_cells:
        cell.source = cell.source.replace(
            'PAIR_ARTIFACT_DIRS = {"change": "", "fusion": ""}',
            'PAIR_ARTIFACT_DIRS = {key: str(trained_paths[key]) for key in ("change", "fusion")}',
        )
    before_code(
        nb,
        "def checked_model_response",
        [
            intro(
                "6d. Automatically load the trained pair experts",
                "Loads only validated attached exports. The quality cell already selected your own SegFormer "
                "weights. All three evidence paths share the protected FastAPI service.",
            ),
            *runtime_cells,
        ],
    )
    # Place the connection export before the final waiting cell so Run All reaches it immediately.
    position = next(
        i
        for i, c in enumerate(nb.cells)
        if c.cell_type == "markdown" and "## 10." in c.source
    )
    nb.cells[position:position] = [
        code(
            "connection = {\n"
            "    'SATQUERY_MODEL_BACKEND': 'http', 'SATQUERY_PAIR_BACKEND': 'http',\n"
            "    'SATQUERY_MODEL_SERVICE_URL': PUBLIC_MODEL_URL,\n"
            "    'SATQUERY_CHANGE_SERVICE_URL': PUBLIC_MODEL_URL,\n"
            "    'SATQUERY_FUSION_SERVICE_URL': PUBLIC_MODEL_URL,\n"
            "    'SATQUERY_PLANNER_BACKEND': 'auto',\n"
            "}\n"
            "connection_path = Path('/kaggle/working/06_backend_connection.env')\n"
            "connection_path.write_text('\\n'.join(k + '=' + v for k, v in connection.items()) + '\\n')\n"
            "print(connection_path.read_text())\n"
            "print('Merge these settings into the local root .env. Keep the existing matching service token. Restart backend.')\n"
            "print('No secrets were written to the connection file.')\n"
        )
    ]
    write(nb, "06_All_Models_Ngrok_Server.ipynb")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    build_evaluations()
    build_training()
    build_server()
    outputs = sorted(DEST.glob("*.ipynb"))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}
    (DEST / "SHA256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    zip_path = ROOT / "notebooks/SatQuery_Kaggle_Run_All_Pack.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(DEST.iterdir()):
            if path.is_file():
                archive.write(path, path.name)
    print(f"Validated and packaged {len(outputs)} standalone notebooks: {zip_path}")


if __name__ == "__main__":
    main()
