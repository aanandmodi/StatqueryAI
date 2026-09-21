# Kaggle training recovery — 2026-09-20

## Notebook 04 numeric import repair — 2026-09-21

Follow-up: the fresh-process traceback subsequently reached Lightning → TorchMetrics → Transformers
and failed importing `is_offline_mode` from Hub. The earlier setup pinned Hub but left Transformers
unconstrained, allowing an incompatible preinstalled version to remain. Notebook 04 now explicitly
installs Transformers 4.57.1, Hub 0.36.2, Tokenizers 0.22.1, Diffusers 0.35.1, PEFT 0.17.1 and
Accelerate 1.7.0 in the same constrained transaction. Published PyPI dependency metadata accepts
these mutual versions. Preflight explicitly imports Transformers/Diffusers/PEFT and includes their
installed versions in failure diagnostics. This is not a model-weight or dataset error.

The reported `_blas_supports_fpe` AttributeError occurs inside NumPy testing utilities loaded by
SciPy/Albumentations during TerraTorch import, before training. It indicates an inconsistent numeric
runtime; the traceback alone cannot distinguish stale in-memory extensions from mixed files on disk.
The previous unbounded TerraTorch dependency install could change NumPy after Kaggle loaded it.

Notebook 04 now pins NumPy 2.2.6, SciPy 1.15.3, TerraTorch 1.2.13, TorchGeo 0.9.0 and the
Albumentations/OpenCV components. The installer constrains the existing torch/torchvision/torchaudio
versions instead of replacing cloud CUDA wheels. A clean subprocess checks NumPy testing and SciPy
operations; only if this fails does setup reinstall the two numeric wheels without dependencies.
Any package changes or repair require a kernel restart before continuing. After restart, both a
subprocess and the notebook must import the full TerraTorch chain and confirm CUDA before data is
downloaded. No checkpoint/data deletion, monkey-patching of private NumPy symbols, automatic process
kill, retraining of notebooks 01–03, or release-threshold change is involved.

Run section 0 first. At `SETUP COMPLETE — RESTART KERNEL`, restart only Python while retaining
session storage; then Run All from the top. Do not retry the old failing section in a stale kernel.
Offline tests cover clean setup, changed versions, stale modules, one-shot numeric repair,
persistent corruption, missing cloud wheels and preflight ordering. Actual Kaggle CUDA training
and resolution of every preinstalled third-party package remain unverified locally.

Sources: [NumPy installation troubleshooting](https://numpy.org/doc/2.0/user/troubleshooting-importerror.html),
[TerraTorch 1.2.13 distribution](https://pypi.org/project/terratorch/1.2.13/).

## Delivered notebooks

- `notebooks/kaggle-run-all/02_Train_Vegetation_Water_Masks.ipynb`
- `notebooks/kaggle-run-all/03_Train_Temporal_Change.ipynb`
- Updated complete bundle: `notebooks/SatQuery_Kaggle_Run_All_Pack.zip`

Import each notebook in its own fresh Kaggle GPU session. Internet must be enabled. Select
**GPU T4/P100**, not TPU v5e-8. These notebooks use CUDA; TPU execution requires a separate
PyTorch/XLA training implementation. Nothing here provisions paid resources or trains locally.

## 02: memory fix

The previous evaluation path accumulated all validation predictions and labels as int64 pixel
maps. `eval_accumulation_steps=1` moved these to CPU but did not bound total RAM. This is a code-level
OOM risk consistent with an epoch-end kernel death; no crash traceback was provided to establish
whether the user's specific crash exhausted host RAM or VRAM.

| Component | Updated implementation |
|---|---|
| Validation | `batch_eval_metrics=True`; persistent state is one 7×7 int64 confusion matrix |
| GPU profile | One GPU, 384px crops, batch 1, accumulation 8, FP16 or BF16 according to hardware |
| Data loading | Zero worker processes; no pinned-memory batches; paths rather than decoded-image cache |
| Dice loss | No full int64 one-hot tensor; FP32 reductions; all-nodata batches handled |
| Correctness | Ignore nodata in per-class metrics as well as pixel accuracy |
| Preflight | Two-scene streaming evaluation before training; full validation remains unchanged |
| Export | Correct Trainer save API; safetensors configured in TrainingArguments |
| Reload | Free model references; bounded 384px image, mixed precision |
| Resume | Require exact profile/configuration/selection manifest; never silently resume an old profile |

The crop change is a memory/scale tradeoff, not a promised accuracy improvement. Training still uses
70% of official LoveDA train, and every official validation image is evaluated at the recorded size.
Quality gates have not been lowered. Preserve the old run before stopping it; a fresh session with
no matching checkpoint starts SegFormer training again. Qwen does not need retraining.

## 03: automatic real dataset download

Source: [PerASCD Dataset Preparation](https://github.com/SathShen/PerASCD), linking to the
[processed SECOND distribution](https://huggingface.co/datasets/SathShen/PerASCD-datasets).
This is a researcher-provided mirror, not the original SECOND publisher. Original dataset/imagery
rights apply; a mirror's license tag alone does not establish commercial permission.

| Artifact | Pinned identifier |
|---|---|
| CDVQA annotations | `YZHJessica/CDVQA@cc5893123dd32326de38745b65d2ffe45055937b` |
| SECOND Hub mirror | `SathShen/PerASCD-datasets@c50fab55c275ffa55c113565af54cfa73e2bd709` |
| Archive | `SECONDbi.zip`, 3,783,431,646 bytes |
| Archive SHA256 / Hub LFS object | `e5d9be06636034bfff526f39b7ee3eb5d2a4bd145171238ed4cb4d1ffb588672` |
| CDVQA membership | Train 1,600 pairs; Val 400 pairs; Test 968 pairs |

Inspected the remote ZIP directory using bounded HTTP range reads, not a multi-GB local download.
All required pair filenames appear once for each of `im1`, `im2`, `label1`, `label2`.
A sampled pair has 512×512 RGB images and 512×512 uint8 index labels. The entire archive and all
selected PNGs are verified by the notebook on Kaggle before training; local inspection did not
download or validate every pixel.

The mirror includes other files and uses different split folder names. The importer selects exact
CDVQA filenames, normalizes the directory layout and retains **CDVQA's split membership**. Test2
reuses Test images and is not added as independent test scenes. No generated or estimated masks
replace missing labels. Download/extraction is streamed, SHA256 checked, and extraction rejects
unsafe paths, duplicates, missing members and unexpected label formats. A source manifest records
per-file hashes. Reruns verify/reuse complete extracted data. Allow 10 GiB free disk before download.

## Existing Qwen evidence

Read the user's preserved 200-row validation JSONL under
`upgrade/01_qwen_validation_evidence/outputs` and rescored it locally. Results match the supplied
scorecard: 100 VQA examples (LoRA exact-match 0.57), 50 grounding examples (mean IoU 0.38072956;
IoU≥0.5 accuracy 0.42), and 50 captions (token F1 0.36517592). Scene-bootstrap intervals and
validation-only calibration diagnostics are preserved. These are not new inference results and
not final-test or India/ISRO accuracy claims. Notebook 01 need not be rerun for the same evidence.

## Verification boundaries

CPU regression tests exercise streaming metrics/reset/nodata handling and archive selection,
checksums, missing labels, duplicate members, unsafe paths and split leakage. Generated notebooks
are schema validated and every code cell is compiled. Full CUDA training/peak-memory measurements
remain a Kaggle task; passing CPU checks is not a claim that an entire cloud training run passed.

References: [Trainer batch-evaluation API](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer),
[PyTorch/XLA devices](https://docs.pytorch.org/xla/master/learn/pytorch-on-xla-devices.html),
[CDVQA authors](https://github.com/YZHJessica/CDVQA).
