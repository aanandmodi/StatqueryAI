"""Build ordered, standalone Kaggle notebooks from the repository's maintained sources."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import shutil
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


def build_runtime_audit():
    nb = notebook("SatQuery_SegFormer_LoveDA_Training")
    # Reuse the EXACT pinned data, label mapping, splits and metrics used for training.
    end = next(i for i, c in enumerate(nb.cells) if c.cell_type == "code" and "checkpoint_root =" in c.source)
    nb.cells = nb.cells[:end + 1]
    nb.cells[-1].source = nb.cells[-1].source.split("checkpoint_root =", 1)[0]
    nb.cells[0] = intro("00 — Audit existing segmentation runtime (NO training)",
        "Attach exactly one extracted 02 output. GPU + Internet, then Run All. Reproduces full validation "
        "and compares 200 fixed-seed scenes at the source label grid using old versus training-matched input size. "
        "Downloads official LoveDA automatically. Exports real masks, previews and paired-scene intervals. "
        "This never changes the existing model or its release gate. Setup may require one kernel restart.")
    guard = code("from pathlib import Path\nimport json\n"
        "roots = [p.parent for p in Path('/kaggle/input').rglob('training_manifest.json') if (p.parent / 'model.safetensors').is_file()]\n"
        "if len(roots) != 1: raise RuntimeError('Attach exactly one extracted notebook 02 output.')\n"
        "AUDIT_CROP = json.loads((roots[0] / 'training_manifest.json').read_text())['config']['crop_size']\n"
        "if type(AUDIT_CROP) is not int or not 128 <= AUDIT_CROP <= 1024: raise ValueError('Invalid recorded crop size')\n")
    nb.cells.insert(1, guard)
    substitute(nb, "crop_size: int = 384", "crop_size: int = AUDIT_CROP")
    substitute(nb, 'training_revision: str = "r3-incumbent-protected"', 'training_revision: str = "runtime-audit-no-training"')
    tree = ast.parse((ROOT / "notebooks/patches/quality_upgrade.py").read_text(encoding="utf-8"))
    names = {"quality_preprocessing_contract", "quality_semantic_prediction"}
    source = "\n\n".join(ast.unparse(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names)
    nb.cells += [code(SUPPORT), code(source),
                 code("import shutil\n"
                      "audit_target = ROOT / 'runtime-audit'\n"
                      "if audit_target.exists(): raise RuntimeError('Audit output exists. Preserve it and use a fresh session; no output is overwritten.')\n"),
                 code((ROOT / "notebooks/patches/segmentation_runtime_audit.py").read_text(encoding="utf-8"))]
    write(nb, "00_Audit_Runtime_Preprocessing.ipynb")


def build_r4_training():
    """A new declared experiment; leave historical R3 and runtime audit reproducible."""
    seg = notebook("SatQuery_SegFormer_LoveDA_Training")
    seg.cells.insert(0, intro(
        "02 R4 — Full-train, incumbent-guided segmentation refinement",
        "First preserve your running R3 checkpoints with 02B. Then use a NEW Kaggle GPU T4/P100 "
        "notebook, Internet On. Attach exactly ONE extracted 02_segmentation_inference (3)/best-model "
        "output (the R2 weights with SHA256 starting 005a1e99). Do not attach 03/04 yet. Run All; "
        "if setup explicitly requests a kernel restart, restart without ending the session, then Run All. "
        "All official TRAIN scenes are used; official validation is never trained on. This is a new "
        "experiment, NOT a promised passing model. No release thresholds or strict-06 checks are weakened. "
        "Future epoch weights are archived, including rejected candidates. Download BOTH ZIPs and save "
        "the full notebook output before ending the session. Inference ZIP -> 06 only if its final gate passes."
    ))
    before_code(seg, "import subprocess", [code('''
from pathlib import Path
import hashlib
R4_REQUIRED_SHA = "005a1e9947717af516e5c8164c5e896326e854ed9cd31f1bd8ef9da00182b308"
attached = [p.parent for p in Path('/kaggle/input').rglob('training_manifest.json')
            if (p.parent / 'model.safetensors').is_file()]
if len(attached) != 1:
    raise RuntimeError('Attach exactly one EXTRACTED latest 02 inference output, including best-model files. ZIP-only is not sufficient. Do not attach other experts here.')
if hashlib.sha256((attached[0] / 'model.safetensors').read_bytes()).hexdigest() != R4_REQUIRED_SHA:
    raise RuntimeError('Wrong incumbent: attach 02_segmentation_inference (3), not (2), or the matching verified R2 output.')
print('R4 attachment matches the known incumbent. File manifest and label checks follow before training.')
''')])
    replacements = {
        'train_fraction: float = 0.70': 'train_fraction: float = 1.0',
        'training_revision: str = "r3-incumbent-protected"': '''training_revision: str = "r4-full-train-teacher-protected"
    required_incumbent_sha256: str = "005a1e9947717af516e5c8164c5e896326e854ed9cd31f1bd8ef9da00182b308"
    teacher_weight: float = 0.25
    teacher_temperature: float = 2.0
    teacher_confidence: float = 0.8
    training_budget_hours: float = 8.0
    forest_loss_multiplier: float = 1.15
    barren_loss_multiplier: float = 1.10
    decoder_lr_multiplier: float = 2.0''',
        'warm_start_learning_rate: float = 2e-5': 'warm_start_learning_rate: float = 5e-6',
        '"/kaggle/working/satquery-segmentation-r3"': '"/kaggle/working/satquery-segmentation-r4"',
        '"/content/satquery-segmentation-r3"': '"/content/satquery-segmentation-r4"',
        '''    A.RandomScale(scale_limit=(-0.15, 0.35), p=0.75),
    A.PadIfNeeded(min_height=CFG.crop_size, min_width=CFG.crop_size, border_mode=0, fill=0, fill_mask=0),
    A.RandomCrop(height=CFG.crop_size, width=CFG.crop_size),''':
        '    # R4 keeps full-scene scale/context; flips/rotations supply geometric diversity.',
        'A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03, p=0.35)':
        'A.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.01, p=0.25)',
        'class_weights = class_weights / class_weights.mean()': '''class_weights[LABEL2ID["forest"]] *= CFG.forest_loss_multiplier
class_weights[LABEL2ID["barren"]] *= CFG.barren_loss_multiplier
class_weights = class_weights / class_weights.mean()''',
        'self.args.learning_rate * 5': 'self.args.learning_rate * CFG.decoder_lr_multiplier',
        'metric_for_best_model="release_balance"': 'metric_for_best_model="protected_selection"',
        'trainer = DiceCETrainer(': 'trainer = R4DiceCETrainer(',
        'early_stopping_patience=6': 'early_stopping_patience=8',
        '    return metrics\n': '''    metrics["protected_selection"] = protected_checkpoint_score(metrics, PROTECTED_REFERENCE)
    return metrics
''',
        'trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)': '''PROTECTED_REFERENCE = dict(incumbent_metrics)
trainer.add_callback(TrainingTimeBudget())
r4_gradient_preflight(trainer)
trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)''',
    }
    for old, new in replacements.items():
        substitute(seg, old, new)
    before_code(seg, "from transformers import EvalPrediction", [
        intro("R4 — Why this differs from the rejected R3 run",
              "100% official training split, 4x lower encoder warm-start LR and 2x decoder LR; mild "
              "forest/barren loss emphasis; no random scale/crop mismatch. A frozen incumbent supplies "
              "KL consistency ONLY on confident pixels where it agrees with training ground truth. "
              "Teacher mistakes and nodata are excluded. Checkpoint selection now prefers candidates "
              "that preserve all four protected metrics DURING training, not just at final export. "
              "This may reduce forgetting, but improvement must be established on the unchanged full "
              "validation split. Repeated validation tuning is not an independent test result. "
              "The eight-hour training budget leaves time for evidence/export; it is not a runtime guarantee."),
        code((ROOT / "notebooks/patches/segmentation_r4.py").read_text(encoding="utf-8")),
    ])
    # Historical source introduction describes the old recipe. Replace rather than
    # leave contradictory instructions in the standalone R4 notebook.
    for cell in seg.cells[1:]:
        if cell.cell_type == "markdown":
            cell.source = cell.source.replace("70%", "100%").replace("R3", "R4")
        elif cell.cell_type == "code":
            cell.source = cell.source.replace(
                "# ~2.7x the evaluation magnification. Random rescaling supplies nearby scales.",
                "# ~2.7x the evaluation magnification. R4 preserves the validation scene scale."
            )
    seg.cells += [code(SUPPORT), code('''
export_inference_zip(OUTPUT_DIR, '/kaggle/working/02_segmentation_inference_R4.zip')
export_inference_zip(EVAL_DIR, '/kaggle/working/02_R4_validation_evidence.zip')
print('STRICT-06 ELIGIBLE ON THIS SEGMENTATION GATE' if release_candidate else
      'NOT ELIGIBLE: preserve both ZIPs and full Notebook Output for review. Do not change thresholds or run demo mode.')
print('Epoch archives are alternative candidate weights, not additional models to merge or serve simultaneously.')
''')]
    write(seg, "02_R4_Refine_Strict_Segmentation.ipynb")


def build_training():
    rescue = nbformat.v4.new_notebook(cells=[
        intro("02B — Preserve surviving SegFormer training weights (NO training)",
              "Best option: copy the single code cell into the END of your still-running original 02 notebook "
              "and run ONLY that cell. Do not Run All on the training notebook. It archives every surviving "
              "SegFormer checkpoint under /kaggle/working without changing weights or quality gates. "
              "For a new recovery notebook, attach the FULL saved Notebook Output and change SEARCH_ROOT "
              "to /kaggle/input. An inference ZIP cannot restore rejected or already rotated weights. "
              "No GPU is required for copying files; do not stop the original session before preserving it."),
        code((ROOT / "notebooks/patches/preserve_checkpoints.py").read_text(encoding="utf-8")),
    ])
    write(rescue, "02B_Preserve_Training_Checkpoints.ipynb")
    seg = notebook("SatQuery_SegFormer_LoveDA_Training")
    seg.cells.insert(
        0,
        intro(
            "02 — Train your vegetation/water/land-cover mask model",
            "GPU T4/P100, Internet On, then Run All. Official LoveDA data downloads automatically. "
            "R3: training-matched 384px input, lower warm-start LR, streaming validation, incumbent retention on regression. "
            "Optionally attach exactly one extracted previous 02 output for checked weight-only warm start. "
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
            "not benchmark guarantees. R2 adds a multiscale mask decoder, pair-balanced training and cached visual evaluation. "
            "Optionally attach exactly one extracted previous 03 output to retain its learned answer weights. "
            "Test runs only after validation passes; failing artifacts cannot enter notebook 06.",
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
            "(ARTIFACTS / 'sha256_manifest.json').write_text(json.dumps({p.name: file_sha256(p) for p in ARTIFACTS.iterdir() if p.is_file() and p.name not in {'training_state.pt', 'sha256_manifest.json'}}, indent=2))\n"
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
            "R2 uses FP32, a convolutional decoder, workers=0, finite-loss/gradient checks and fresh outputs. "
            "Do NOT attach/resume the old NaN checkpoint. This edition automatically downloads only the "
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
            "(ARTIFACTS / 'sha256_manifest.json').write_text(json.dumps({p.name: file_sha256(p) for p in ARTIFACTS.iterdir() if p.is_file() and p.name not in {'training_state.pt', 'sha256_manifest.json'}}, indent=2))\n"
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
            "SATQUERY_SEGMENTATION_RELEASE_STATUS = 'passed_project_validation_gate'\n"
            "print('Verified attached trained checkpoints:', {k: str(v) for k, v in trained_paths.items()})\n"
        ),
    ]
    substitute(
        nb,
        '"huggingface-hub==0.34.4",',
        '"huggingface-hub==0.36.2",\n    "terratorch==1.2.13",\n    "torchgeo==0.9.0",\n    "numpy==2.2.6",\n    "scipy==1.15.3",\n    "albumentations==2.0.8",\n    "albucore==0.0.24",\n    "opencv-python-headless==4.11.0.86",\n    "diffusers==0.35.1",\n    "tokenizers==0.22.1",\n    "setuptools<81",',
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

    # Separate attended demonstration, NOT a release-gate override in the strict notebook.
    demo = copy.deepcopy(nb)
    demo.cells[0] = intro(
        "06 DEMO — run your current Qwen + segmentation + change + fusion models",
        "Use THIS notebook to test the current project without retraining. Attach exactly the extracted "
        "latest 02, 03 and 04 outputs. Your 03/04 must pass their existing release gates. "
        "This demo explicitly permits your still-failing 02 segmentation checkpoint, labels its masks "
        "EXPERIMENTAL, returns the failed-gate warning in the API, and does not alter any manifest or threshold. "
        "Do not present this as a fully released model. GPU T4 + Internet; enable the same three Secrets. "
        "Run All; restart the kernel ONLY if setup asks, then Run All again. No training, no Hub upload, "
        "no website deployment. The final cell waits during the attended temporary ngrok session."
    )
    substitute(demo, "trained_paths = find_trained_artifacts('/kaggle/input')",
        "trained_paths = find_trained_artifacts('/kaggle/input', allow_experimental_segmentation=True)")
    substitute(demo, "SATQUERY_SEGMENTATION_RELEASE_STATUS = 'passed_project_validation_gate'",
        "segmentation_report = json.loads((trained_paths['segmentation'] / 'training_manifest.json').read_text())\n"
        "segmentation_passed = segmentation_report.get('release_candidate') is True\n"
        "SATQUERY_SEGMENTATION_RELEASE_STATUS = 'passed_project_validation_gate' if segmentation_passed else 'experimental_failed_release_gate'\n"
        "SATQUERY_SEGMENTATION_RELEASE_WARNING = '' if segmentation_passed else (\n"
        "    'EXPERIMENTAL single-image masks: this checkpoint failed the project release gate. '\n"
        "    'Vegetation and other boundaries may be incomplete or incorrect; do not treat them as verified evidence.')\n"
        "print({'segmentation_status': SATQUERY_SEGMENTATION_RELEASE_STATUS, 'checks': segmentation_report.get('release_checks')})\n")
    for cell in demo.cells:
        if cell.cell_type == "markdown" and "6d." in cell.source:
            cell.source = "## 6d. Load passed change/fusion experts\n\nThe paired experts must pass validation/test gates. Single-image masks remain explicitly experimental if notebook 02 failed."
    substitute(demo, "print('Verified attached trained checkpoints:',",
        "print('Integrity-checked demo checkpoints (see individual release status):',")
    # Protect cloud CUDA packages and catch stale numeric imports before downloading models.
    substitute(demo, 'subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", *PACKAGES])',
        "from importlib import metadata\n"
        "def installed_demo_versions():\n"
        "    return {d.metadata['Name'].lower().replace('_', '-'): d.version for d in metadata.distributions() if d.metadata['Name']}\n"
        "before_install = installed_demo_versions()\n"
        "constraints = Path('/kaggle/working/satquery-demo-cuda-constraints.txt')\n"
        "constraints.write_text('\\n'.join(f'{name}=={before_install[name]}' for name in ('torch', 'torchvision', 'torchaudio') if name in before_install))\n"
        "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', '-c', str(constraints), *PACKAGES])\n"
        "after_install = installed_demo_versions()\n"
        "probe_code = 'import numpy, numpy.testing; from scipy import special; from transformers import Qwen3VLForConditionalGeneration; from terratorch.registry import BACKBONE_REGISTRY'\n"
        "probe = subprocess.run([sys.executable, '-c', probe_code], capture_output=True, text=True)\n"
        "if probe.returncode:\n"
        "    raise RuntimeError('Fresh-process demo imports failed: ' + probe.stderr[-6000:])\n"
        "if any(before_install.get(name) != after_install.get(name) for name in ('numpy', 'scipy', 'transformers', 'huggingface-hub', 'terratorch', 'albumentations')):\n"
        "    raise SystemExit('SETUP COMPLETE — RESTART KERNEL (keep session files), then Run All again.')\n"
        "try:\n    exec(probe_code)\n"
        "except Exception as exc:\n    raise RuntimeError('Disk imports pass but kernel is stale: Restart Kernel, then Run All.') from exc\n")
    write(demo, "06_Demo_Current_Models_Ngrok.ipynb")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    build_evaluations()
    build_runtime_audit()
    build_training()
    build_r4_training()
    build_server()
    outputs = sorted(DEST.glob("*.ipynb"))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}
    (DEST / "SHA256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    zip_path = ROOT / "notebooks/SatQuery_Kaggle_Run_All_Pack.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(DEST.iterdir()):
            if path.is_file():
                archive.write(path, path.name)
    # One canonical extracted pack plus the downloadable ZIP. The redundant old
    # mirror was recoverably archived during submission cleanup.
    print(f"Validated and packaged {len(outputs)} standalone notebooks: {zip_path}")


if __name__ == "__main__":
    main()
