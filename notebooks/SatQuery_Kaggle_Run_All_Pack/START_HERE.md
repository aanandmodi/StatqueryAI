# SatQuery — numbered Kaggle Run All pack

## R2 recovery — use this with the completed 01/05 evidence

**Do not rerun 01 or 05.** Their frozen Qwen evidence is already preserved. Re-run only 02 → 03 → 04,
in separate fresh GPU sessions, and preserve new exports in `upgrade/r2/` without replacing originals.
Read [the revision guide](TRAINING_R2.md) for the recorded failures and changes.

- **02:** optionally attach exactly one previous extracted `02_segmentation_inference` output via
  Add Input. The notebook verifies its weights, reuses them, and starts a new optimizer/schedule.
  Without this input it trains from the pinned encoder. Training data still downloads automatically.
- **03:** optionally attach exactly one previous extracted `03_change_inference` output. Its finite,
  hash-checked weights initialize the existing layers; the new multiscale decoder trains from scratch.
  Without this input the whole specialist trains afresh from the ImageNet encoder.
- **04:** attach **no previous failed checkpoint**. The old NaN export is invalid. Start fresh;
  setup may request **Restart Kernel**, then Run All again. This is a deliberate dependency boundary.
- **06:** import the **new** notebook after all three revised exports actually pass. Attach exactly
  one passing export per specialist. No 01/05 ZIP is required by 06. Changed decoder versions are
  loaded explicitly; an old 06 notebook cannot load the new architectures.

R2 keeps all release thresholds unchanged. A completed run can still fail a quality gate. Tests
are evaluated only after validation gates pass; previous public test results have already been
seen, so these are confirmation runs, not a new untouched benchmark. No pass is guaranteed.

Use one new Kaggle notebook/session for each numbered file. The training notebooks and evaluation
notebooks do not run your website. The last notebook serves your models to the local website.
All Python code needed by each notebook is embedded; no Git clone or manual code-cell patching is needed.

## First-time setup

1. Extract `SatQuery_Kaggle_Run_All_Pack.zip` on your laptop.
2. In Kaggle, create a notebook and use **File → Import Notebook** to select the numbered `.ipynb`.
   If the UI offers **Upload Notebook** instead, use that. Import the notebook file, not the ZIP.
3. Set **Accelerator → GPU T4** (P100 is also suitable for the smaller profiles), and **Internet → On**.
   Check your account's remaining GPU quota. Free GPU availability and runtime duration vary.
4. For notebook 06, enable these three entries in **Add-ons → Secrets**:

   | Secret name | Value |
   |---|---|
   | `HF_TOKEN` | Your Hugging Face read token; use a current token from your account |
   | `NGROK_AUTHTOKEN` | Your ngrok agent authtoken |
   | `SATQUERY_MODEL_SERVICE_TOKEN` | The same random bearer token stored in the local backend `.env` |

   Keep secret values out of notebook code, exported files and screenshots. Notebooks 01–05 do not
   upload models to the Hub by default and need no write token.
5. Attach the inputs in the table below **before Run All**. Then click **Run All** once.
   A genuinely missing dataset or failed quality gate stops the notebook with a specific message.
   Provider outages, quota exhaustion, out-of-memory errors or incompatible preinstalled packages
   can still interrupt a run; Run All is not a guarantee that every cloud environment succeeds.

## Execution order and handoff

| Order | File | What happens automatically | Input you must attach | Preserve after completion |
|---|---|---|---|---|
| 01 | `01_Qwen_Validation_and_Scorecard.ipynb` | Download pinned Qwen/LoRA + bounded data; run base/LoRA comparison; raw JSONL, bootstrap intervals, validation calibration diagnostic | None | `01_qwen_validation_evidence.zip` and saved notebook output |
| 02 | `02_Train_Vegetation_Water_Masks.ipynb` | Download official LoveDA; train SegFormer B0 on 70% of official train; validate/reload; export masks model | None | `02_segmentation_inference.zip` and saved output |
| 03 | `03_Train_Temporal_Change.ipynb` | Download pinned CDVQA annotations and SECOND research-mirror images/labels; verify checksums and original splits; train answer+mask expert; validate/test | None (Internet; allow 10 GiB free disk) | `03_change_inference.zip` and saved output |
| 04 | `04_Train_Optical_SAR_Flood_Masks.ipynb` | Download official Sen1Floods11 hand-labelled triplets with checksums; train TerraMind tiny on a T4; validation/test and modality ablations | None | `04_fusion_inference.zip` and saved output |
| 05 | `05_Qwen_Final_Test.ipynb` | Evaluate pinned Qwen on test; test the temperature frozen by 01; write scorecard | Saved **Notebook Output from 01** | `05_qwen_test_evidence.zip` and saved output |
| 06 | `06_All_Models_Ngrok_Server.ipynb` | Verify/load the three trained exports; load existing Qwen; add quality+planner+pair routes; test HTTP; open ngrok; wait for demo | Saved **Notebook Outputs from 02, 03 and 04**, plus Secrets | `06_backend_connection.env`; keep this session running during demo |

Your Qwen LoRA already exists. Notebook 01 evaluates it; this pack does not repeat Qwen training.
More VQA training alone cannot teach precise pixel masks; notebooks 02–04 train the relevant heads.
The fusion head specifically learns flood/water, not arbitrary vegetation/buildings on optical/SAR.

If you still have the complete raw output from the earlier 200-row evaluation, preserve it instead
of rerunning 01 just for the same numbers. Notebook 05 requires 01's raw JSONL, summary and scored
directory together. The aggregate pasted into chat does not contain those artifacts.

### How to preserve a finished run

For 01–05, wait for the final output/ZIP path. **Save Version** and select the option that saves the
current session/output if available. Avoid choosing **Save & Run All** if it would retrain or rerun
the untouched test unnecessarily. Download the named ZIP as a backup before stopping the GPU.
If your Kaggle UI only preserves outputs through a committed run, configure that run once at the
start instead of completing an interactive run and then launching the same run a second time.

For 05/06, use **Add Input → Notebook Output** to attach the previous saved version. Kaggle exposes
its files under `/kaggle/input`, and the notebooks find them automatically. Alternatively, extract
the downloaded ZIP and upload that folder as a private Kaggle Dataset. **A ZIP sitting unextracted
inside an input is not enough for automatic model discovery.** Attach exactly one candidate per role.

Inference ZIPs omit optimizer state to keep downloads smaller. For resuming interrupted training,
preserve the full notebook output, including `training_state.pt` or Trainer checkpoints. Resume
needs the matching dataset/configuration and state copied back into that notebook's working path.
Do not use a different run's optimizer state or change the split while resuming.

### Notebook 02: recovery from the earlier memory crash

Download the updated notebook and start a **fresh GPU session**. Do not run training in the same
kernel as Qwen/server inference. Select **GPU T4 or P100, not TPU v5e-8**, and enable Internet.
Run All. The fixed profile uses one GPU, 384px crops, microbatch 1, accumulation 8 and zero loader
workers. Validation accumulates only a 7×7 confusion matrix, not thousands of masks in RAM.
All official validation scenes are still evaluated; the 70% training split has not been reduced.
Metrics are measured at 384px and are not directly comparable to a full-resolution benchmark.

Before ending an old session, preserve any available checkpoints/output. Older-profile checkpoints
are not silently resumed under this changed configuration. New-profile checkpoints can resume in
the same working folder, together with `checkpoints/resume_config.json`. If copied from a saved
Kaggle output, preserve the full checkpoint files and that manifest. A fresh session without those
files starts training again; it does not retrain Qwen. Save the new output and ZIP when finished.

### Notebook 03: automatic SECOND imagery and semantic labels

The [CDVQA authors' repository](https://github.com/YZHJessica/CDVQA) supplies annotations. The notebook
now automatically downloads `SECONDbi.zip` from the
[PerASCD researchers' processed distribution](https://github.com/SathShen/PerASCD).
This is a research mirror, not the original SECOND publisher. The archive is about **3.78 GB**;
allow **10 GiB free disk** for downloading, extraction and checkpoints. No HF token, manual image
upload or synthetic label generation is needed. Original imagery rights still apply; the mirror's
license tag does not replace them.

The exact Hub revision and LFS SHA256 are pinned. All 2,968 original CDVQA train/validation/test pair
filenames were checked against the remote ZIP directory. The downloader selects only those filenames,
rejects ambiguous/missing entries, checks PNG dimensions and index-label format, and records per-file
SHA256 hashes. CDVQA assigns the splits, not the mirror's differently named train/val folders.
An existing manually attached dataset is still supported with this layout:

```text
SECOND/
  im1/       before images (or A/)
  im2/       after images  (or B/)
  label1/    before semantic labels
  label2/    after semantic labels
```

Split subfolders `train/`, `val/`, `test/` containing the same four folders are also supported.
Missing test labels prevent completing the test gate; they cannot be replaced by generated labels.

### Quality checks and GPU time

**Notebook 04 dependency repair (2026-09-21):** run its setup cell first. If it prints
`SETUP COMPLETE — RESTART KERNEL`, restart the Python kernel **without ending/deleting the session**,
then Run All again from the top. Setup pins NumPy 2.2.6, SciPy 1.15.3 and TerraTorch 1.2.13,
constrains torch/torchvision to Kaggle's installed CUDA wheels, and verifies the import chain before
any data download. Do not hot-reload NumPy or rerun section 3 in the old process. Dataset downloads,
checkpoints and release thresholds are unchanged. This is a necessary setup pause, not a failed
training epoch. Preserve files if your chosen Kaggle restart action would also reset storage.

The updated 04 also explicitly installs Transformers 4.57.1, Hub 0.36.2, Tokenizers 0.22.1,
Diffusers 0.35.1, PEFT 0.17.1 and Accelerate 1.7.0 together. This fixes the
`cannot import name 'is_offline_mode' from 'huggingface_hub'` mismatch caused by leaving a newer
preinstalled Transformers beside the older Hub pin. Repeat setup and its requested kernel restart
after importing this revision. Do not independently upgrade Hub or bypass the preflight.

Notebook 02 uses its existing per-class validation gates. Notebook 03 declares project prototype
targets before training: answer accuracy ≥0.60 and change-mask IoU ≥0.40. Notebook 04 declares flood
IoU ≥0.50 with fused validation at least as good as both single-modality ablations. These are initial
project acceptance targets, not published benchmark values or guarantees of precision.

Both paired notebooks report untouched-test results after choosing the validation checkpoint. Do
not repeatedly tune against test or lower thresholds just to get PASS. A failed gate still writes
evidence and an archive; notebook 06 refuses that candidate until an improved, properly evaluated
experiment passes. Better checkpoints can require multiple free sessions; total completion time
cannot be promised within one Kaggle quota window.

Our notebooks' nominal epoch counts are starting profiles, not measured runtime guarantees. If
memory is exhausted, lower batch size in that notebook's Config cell before a fresh run. A smaller
batch trades time for memory. Do not run all training notebooks simultaneously.

## Final serving notebook: use 06 after training

1. Import `06_All_Models_Ngrok_Server.ipynb` into a **fresh** Kaggle session.
2. Enable the three Secrets, GPU and Internet; attach the passing outputs of 02, 03, 04.
3. Click **Run All**. Do not paste separate quality/planning/pair cells; they are already included.
4. It verifies hashes/gates first, loads your own SegFormer weights instead of the old transfer
   checkpoint, and loads the two paired experts plus the existing Qwen LoRA.
5. Wait for the local HTTP and public tunnel PASS messages. A generated answer in these smoke tests
   demonstrates connectivity; evaluate real imagery separately for mask/report quality.
6. Download `06_backend_connection.env`, printed **before** the final waiting cell. It contains only
   URLs and backend mode settings. Copy those settings into `D:\Projects\Sih-2026\.env`.
7. Keep the existing `SATQUERY_MODEL_SERVICE_TOKEN` in that `.env` equal to the Kaggle Secret. Do not
   copy the HF or ngrok agent token into frontend configuration.
8. Leave the last cell running and keep the session attended while demonstrating the website.
   The notebook closes its tunnel after at most **60 minutes**, measured from tunnel creation.
   This limit is implemented by this project; it is not an ngrok plan limit or a 24/7 hosting offer.
9. For another attended demo, start the serving notebook again with the same saved inputs. **Do
   not rerun training.** Copy the newly printed connection settings if the URL changes.

## Start the backend and frontend on your laptop

Model computation runs in Kaggle. Only the controller/UI run on the laptop.

First verify `.env` has the settings printed by 06, for example:

```dotenv
SATQUERY_MODEL_BACKEND=http
SATQUERY_PAIR_BACKEND=http
SATQUERY_MODEL_SERVICE_URL=https://YOUR-CURRENT-NGROK-URL
SATQUERY_CHANGE_SERVICE_URL=https://YOUR-CURRENT-NGROK-URL
SATQUERY_FUSION_SERVICE_URL=https://YOUR-CURRENT-NGROK-URL
SATQUERY_PLANNER_BACKEND=auto
SATQUERY_MODEL_SERVICE_TOKEN=YOUR_EXISTING_MATCHING_SECRET
```

Do not replace your whole `.env`: preserve the local API key/database settings already in it.
If services are already running, stop them with Ctrl+C in their own terminals before restarting.

PowerShell terminal 1 (existing project `.venv`):

```powershell
Set-Location D:\Projects\Sih-2026
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

PowerShell terminal 2:

```powershell
Set-Location D:\Projects\Sih-2026
npm run dev
```

Open **http://localhost:3000**. Check backend readiness in another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/v1/health/ready
```

`model_gateway: false` means the model connection still needs attention; verify the current tunnel,
matching bearer token and the notebook's HTTP tests. Do not use an ngrok URL left over from a
stopped session. The local health endpoint alone does not validate every paired model's accuracy.

Use the files in `D:\Projects\Sih-2026\demo-assets` for the existing demo workflows. The SAR display
proxy in that folder is not a calibrated Sentinel-1 product: it exercises the analytical fallback,
not the newly trained fusion expert. For learned fusion, supply real co-registered 13-band S2 L1C
and S1 VV/VH dB with correct embedded band/platform/representation metadata. Plain JPG/PNG cannot
supply spectral bands or SAR measurements that do not exist in those images.

## What remains after all six notebooks finish

Review per-class/per-scene errors, frozen calibration on test, real paired HTTP requests and the
three website evidence workflows. Preserve raw JSONL, hashes and checkpoints for judging. Single
scene VQA calibration does not calibrate masks or prove India/Cartosat/RISAT accuracy. The app keeps
its existing confidence rules until a matching calibration artifact and domain evaluation are
reviewed and integrated. Completion of Run All alone does not close those scientific gaps.

Source references: [official Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11),
[TerraMind flood configuration](https://github.com/IBM/terramind/blob/main/configs/terramind_v1_base_sen1floods11.yaml),
[LoveDA publisher](https://zenodo.org/records/5706578).
