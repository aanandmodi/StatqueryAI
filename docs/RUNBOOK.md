# SatQuery runbook

## Choose the path

| Goal | Command / notebook | Notes |
|---|---|---|
| Real-model jury demo | `06_Demo_Current_Models_Ngrok.ipynb` + `scripts/run-local.py --mode remote` | Recommended; website/controller remain local |
| Strict all-model gate | `06_All_Models_Ngrok_Server.ipynb` | Refuses a failed single-mask release; currently not the demo path |
| UI/controller without VLM | `scripts/run-local.py --mode demo` | Simulated single-scene plumbing; never present as model output |
| Local GPU model | `scripts/run-local.py --mode local` | Optional; not for the target low-spec laptop |

## One-time local setup

```powershell
Set-Location D:\Projects\Sih-2026
Copy-Item .env.example .env
Copy-Item .env.local.example .env.local
./scripts/setup-local.ps1
npm install
```

Linux/macOS:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r backend/requirements.txt -r backend/requirements-dev.txt
npm install
```

## Kaggle secrets

Create these notebook secrets. Never paste values into a code cell, screenshot or commit.

| Name | Value |
|---|---|
| `HF_TOKEN` | Hugging Face read token if authenticated download is required |
| `NGROK_AUTHTOKEN` | ngrok agent authtoken |
| `SATQUERY_MODEL_SERVICE_TOKEN` | random 32+ character bearer secret; same value in local `.env` |

Generate the service token:

```powershell
$bytes = New-Object byte[] 32
[Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
[Convert]::ToBase64String($bytes)
```

If any secret was posted publicly or committed, revoke and rotate it.

## Canonical notebook order

The files in `notebooks/kaggle-run-all/` are self-contained and upload-ready.

| Order | Notebook | Purpose | Required input |
|---:|---|---|---|
| 00 | `00_Audit_Runtime_Preprocessing.ipynb` | Verify runtime and preprocessing | none |
| 01 | `01_Qwen_Validation_and_Scorecard.ipynb` | Base-vs-LoRA validation | none |
| 02 | `02_Train_Vegetation_Water_Masks.ipynb` | Train SegFormer | data downloads automatically |
| 02 R4 | `02_R4_Refine_Strict_Segmentation.ipynb` | Protected full-train refinement | selected 02 artifact |
| 02B | `02B_Preserve_Training_Checkpoints.ipynb` | Preserve interrupted checkpoints | saved notebook output |
| 03 | `03_Train_Temporal_Change.ipynb` | Train CDVQA/SECOND expert | data downloads automatically |
| 04 | `04_Train_Optical_SAR_Flood_Masks.ipynb` | Train TerraMind flood expert | data downloads automatically |
| 05 | `05_Qwen_Final_Test.ipynb` | Frozen Qwen test | saved output from 01 |
| 06 strict | `06_All_Models_Ngrok_Server.ipynb` | Enforce every release gate and serve | one 02, 03 and 04 export |
| 06 demo | `06_Demo_Current_Models_Ngrok.ipynb` | Serve selected models; label 02 experimental | one 02, 03 and 04 export |

Training is already complete for the current demonstration. Use **06 demo**. The selected artifacts are under `upgrade/`; upload those extracted folders as private Kaggle datasets or attach saved Notebook Outputs. Do not attach raw ZIPs unless a notebook explicitly extracts them.

Select a T4/P100 GPU, not TPU, and enable Internet. Run all cells once. Wait for local smoke tests, the printed HTTPS URL, a public 200 verification and the attended keep-alive cell. The tunnel exists only while that notebook session is alive.

## Connect the local controller

Copy the generated values into root `.env`:

```dotenv
SATQUERY_MODEL_BACKEND=http
SATQUERY_MODEL_SERVICE_URL=https://your-current-domain.ngrok-free.dev
SATQUERY_MODEL_SERVICE_TOKEN=the-same-secret-stored-in-kaggle
SATQUERY_PAIR_BACKEND=http
```

Keep `.env.local` as:

```dotenv
SATQUERY_BACKEND_URL=http://127.0.0.1:8000
SATQUERY_BACKEND_API_KEY=
```

The frontend points to the local controller, never directly to ngrok.

```powershell
./.venv/Scripts/python.exe scripts/run-local.py --mode remote
```

Expected order: remote `/ready` passes, controller starts on `127.0.0.1:8000`, then the UI starts on `localhost:3000`.

## Demo inputs

Use [`input/README.md`](../input/README.md). Recommended order:

1. `01_single/02_india_water_multispectral.tif` — “Show me all water bodies in this region.”
2. Nepal `BEFORE` then `AFTER` — “What changed? Show the changed regions.”
3. India `OPTICAL` then `SAR` — “Use both sensors to identify flood/water candidates.”

Never upload a file from `04_reference_labels_DO_NOT_UPLOAD` as inference input.

## Troubleshooting

### Controller cannot be reached

- Keep `run-local.py` open.
- Check `http://127.0.0.1:8000/v1/health/ready`.
- Confirm `.env.local` points to `http://127.0.0.1:8000`.
- Restart after changing environment files.

### `Failed to fetch`

The controller stopped, the tunnel expired, or the browser bypassed the same-origin proxy. Re-run the tunnel section of notebook 06, update `.env`, and restart locally.

### ngrok returns 404

The URL is stale or server cells were not run in order. Confirm `/ready` first, then use the route printed by the same notebook session.

### GeoTIFF validation fails

Pair workflows require compatible dimensions, transform and CRS in strict mode. A `.tif` extension alone does not guarantee a valid GeoTIFF. Exploration mode is acceptable only when its scientific limitations remain visible.

### Training runs out of memory

Use a fresh GPU session, microbatch 1, configured accumulation, no Qwen model in memory and `num_workers=0` where specified. Do not use TPU for these CUDA notebooks.

### Stop safely

Press `Ctrl+C` once locally, then stop the Kaggle session from its power control. Do not leave free GPU sessions unattended.
