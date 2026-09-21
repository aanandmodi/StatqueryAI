# Training revision R2 — evidence, changes, and next run

Status: revised training code, not newly trained or released weights. All GPU work stays on Kaggle.
No paid endpoint, hosting change, or automatic Hub upload is introduced.

## Recorded evidence

| Notebook | Recorded result | Current decision |
|---|---|---|
| 01 | Pinned Qwen validation, 200 records | Preserve; no rerun |
| 02 | mean IoU 0.35003; forest 0.24929 | Fails unchanged 0.45 mean / 0.35 forest gates |
| 03 | validation answer 0.71334; mask IoU 0.39661 | Fails unchanged 0.40 mask gate; test 0.41653 cannot override validation |
| 04 | NaN loss in every recorded epoch; non-finite head weights; flood IoU 0 | Invalid numerical run; never resume or serve |
| 05 | Qwen test VQA 0.61 (100), grounding mean IoU 0.42267 (50), caption F1 0.40145 (50) | Preserve frozen results; not proof of segmentation quality |

Notebook 05 frozen-temperature test ECE is 0.13041 versus raw 0.15073. It does **not** authorize
automatic probability promotion: the scorecard still says `automatic_probability_release: false`.
All evaluation summaries and provenance are in `upgrade/`. Large weights and temporal JSONL files
remain on this computer; Git stores summaries/hashes, not a complete downloadable checkpoint.

## What changed

| Notebook | Revision | What it addresses |
|---|---|---|
| 02 | Match training scene scale to validation; moderate scale jitter; decoder LR multiplier; cosine warmup; up to 20 epochs with patience 6 | Old native 384px crops and resized full scenes had different object scales; mean-only selection could hide forest failure |
| 02 | Optional hash-checked old weight initialization; streaming metrics and batch 1 retained; FP32 on T4/P100 | Reuses progress without stale optimizer state; avoids previous RAM-heavy evaluation |
| 03 | Stride 4/8/16 multiscale difference features feeding a detail mask decoder | Old mask came only from an 8×8 feature grid for a 256×256 image |
| 03 | Four QA samples per pair per epoch, rotating deterministically; mask loss weight 2; 16 epochs; select weakest normalized validation gate | Prevents question-rich pairs dominating mask supervision and answer accuracy hiding weak masks |
| 03 | Cache visual features only for the current sorted evaluation pair | Avoids recomputing the visual encoder for each question without accumulating the validation dataset in GPU memory |
| 04 | Finite raw-band support masks; replace invalid **inputs** with normalization means before interpolation; ignore affected labels | Invalid values must not enter attention; masking labels alone cannot neutralize NaN inputs |
| 04 | FP32; batch 2; workers 0; gradient/loss/state checks; all-ignored batches skipped; real training-batch preflight | Stops numerical failure at its source; avoids notebook multiprocessing cleanup errors |
| 04 | Fixed TerraMind tiny, convolutional refinement decoder, lower backbone LR, two frozen-backbone epochs, up to 20 epochs | More spatial capacity than a linear token classifier with bounded free-GPU use |
| 03/04 | New output roots, strict compatible resume, strict fresh reload, test only after validation passes | Old broken states cannot silently contaminate a new candidate |
| 06 | Explicit decoder version loading; finite input treatment matches fusion training | Keeps serving architecture aligned with revised exports; retains legacy compatibility |

The old fusion export establishes numerical corruption, not its unique trigger. Invalid input
values, all-ignored CE batches, and reduced-precision reductions are guarded independently; the
next real run must confirm finite learning. PyTorch documents FP16 numerical-range limitations
in its [AMP documentation](https://github.com/pytorch/pytorch/blob/main/docs/source/amp.md).
Normalization follows the published [TerraMind Sen1Floods11 reference](https://github.com/IBM/terramind/blob/main/configs/terramind_v1_base_tim_lulc_sen1floods11.yaml).

## Exact run order

1. Import `notebooks/kaggle-run-all/02_Train_Vegetation_Water_Masks.ipynb` into a **new** Kaggle notebook.
   Select a GPU, enable Internet, optionally attach the previous extracted 02 output, then Run All.
   If dependency setup requests it, restart only the kernel and Run All again (do not end the session).
   Preserve `02_segmentation_inference.zip` and full saved output. Required: mean IoU ≥0.45 and
   water/forest/agricultural IoU each ≥0.35. A validation-only prototype gate is not an independent test.
2. Import `03_Train_Temporal_Change.ipynb` in a separate GPU session. Optionally attach the old extracted
   03 output, not 04. Dataset downloads automatically. Require validation and test answer ≥0.60,
   mask IoU ≥0.40. Preserve its inference ZIP and full saved output.
3. Import `04_Train_Optical_SAR_Flood_Masks.ipynb` in another fresh GPU session. No dataset attachment
   or old checkpoint is needed. If setup requests kernel restart, restart **kernel**, keep files,
   then Run All. Require finite preflight/loss history, validation/test flood IoU ≥0.50, and fused
   validation IoU not below either single-modality ablation. Preserve ZIP plus full saved output.
4. Only after all pass, import the updated `06_All_Models_Ngrok_Server.ipynb`. Attach the three new
   extracted passing outputs, enable the existing three secrets, Run All, and connect the emitted
   URL to the local backend. This remains an attended temporary demo with a bounded timer.

Do not run notebooks 02–04 simultaneously on the same GPU session. T4/P100 are CUDA GPUs; TPU is
not a drop-in replacement. Dataset download/network time and convergence are not guaranteed to
fit one session. Keep full outputs/checkpoints before stopping; inference ZIPs exclude optimizers.
Use only matching R2 full state for interrupted-run resume, copied into the same R2 working root.
Never overwrite the original `upgrade/02...`, `03...`, `04...` folders with new experiments.

## Scientific limits

- Gates are unchanged, not replaced by automatically favourable thresholds. Failed models stay blocked.
- More epochs and decoder capacity are testable hypotheses, not promises of precision or release.
- Validation is used for selection; official public tests were already inspected. Do not describe a
  repeated public-test run as a newly untouched benchmark or tune using its errors.
- LoveDA cannot establish vegetation species, weather, cause of damage, or all-domain accuracy.
- SECOND mask supervision establishes semantic change, not why a flood happened.
- Sen1Floods11 establishes only its measured Sentinel-domain flood/water task, not Cartosat/RISAT transfer.
- Runtime scores remain uncalibrated unless a separate applicable calibration protocol is accepted.

## Verification boundary

The pack builder parses every code cell and validates notebook schemas. Repository tests verify
data handling and source/runtime architecture parity. `scripts/test-training-numerics.py` runs tiny
CPU shape/gradient/reload/NaN tests with no downloads or training dataset. These checks do not
substitute for a full Kaggle run or claim new IoU, accuracy, latency or GPU-memory measurements.

Local verification for this revision: 290 backend tests, 13 ML utility tests and four CPU numerical
tests passed. Qwen 05 raw predictions were re-scored locally and reproduce the reported aggregates.
