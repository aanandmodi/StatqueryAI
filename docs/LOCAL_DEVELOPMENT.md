# Local development runbook

## Supported execution profiles

| Profile | Website | Controller | Model | Cost | Use when |
|---|---|---|---|---:|---|
| `local-gpu` | Local | Local | Local RTX GPU, 4-bit | ₹0 | The model fits the laptop reliably |
| `free-gpu-bridge` | Local | Local | Temporary Kaggle/Colab GPU | ₹0 | The laptop model is slow or out of memory |
| `demo` | Local | Local | Deterministic simulator | ₹0 | UI/API development only; never model validation |

No profile in this runbook publishes the website. Persistent deployment packages and deployment
scripts have been removed from the repository.

## Machine assessment

The audited machine has an Intel i5-11260H, 7.7 GiB RAM, and an RTX 2050 with 4 GiB VRAM. A normal
BF16 2B model consumes roughly 4 GiB for weights before vision activations and framework overhead,
so BF16 is not a viable laptop profile. NF4 4-bit is attempted with the following guardrails:

| Control | Value | Reason |
|---|---:|---|
| Quantization | NF4 + double quantization | Reduce weight memory |
| Compute dtype | FP16 | RTX 2050-compatible inference |
| Image edge | 448 px | Bound visual tokens and activation memory |
| New tokens | 96 default, 128 hard notebook cap | Bound KV cache and latency |
| Concurrency | One model request | Prevent simultaneous VRAM spikes |
| Model revisions | Two immutable 40-character SHAs | Reproducible and auditable runtime |

## Profile A: all services on the laptop

### One-time setup

Open PowerShell:

```powershell
Set-Location D:\Projects\Sih-2026
& .\scripts\setup-local.ps1 -WithLocalModel
```

The script creates `%USERPROFILE%\.venvs\satquery`, installs the pinned runtime, installs frontend
packages, and verifies that CUDA sees the NVIDIA device. It deliberately avoids the D: project
drive because that drive has limited free space.

### Start

Close games, video editors, and other GPU-heavy programs, then run:

```powershell
Set-Location D:\Projects\Sih-2026
& "$env:USERPROFILE\.venvs\satquery\Scripts\python.exe" .\scripts\run-local.py
```

Startup order is model `:8080` → controller `:8000` → website `:3000`. The first start downloads
the model, so `:3000` is intentionally not started until the model readiness check passes.

### Verify

```powershell
Invoke-RestMethod http://127.0.0.1:8080/ready
Invoke-RestMethod http://127.0.0.1:8000/v1/health/ready
Start-Process http://localhost:3000
```

The two JSON responses must report `ready`/`ok`; the backend check `model_gateway` must be `true`.

### Stop

Press `Ctrl+C` in the single SatQuery terminal. The runner terminates all child services. No cloud
job continues after local shutdown.

## Profile B: local app with a temporary free GPU

1. Upload [`SatQuery_Qwen3VL_Free_GPU_Server.ipynb`](../notebooks/SatQuery_Qwen3VL_Free_GPU_Server.ipynb)
   to Kaggle or Colab.
2. Run `scripts/setup-local.ps1` once on the laptop (without `-WithLocalModel`).
3. Enable a free GPU and Internet, then run the cells in order.
4. Confirm the model, local HTTP and public ngrok smoke tests print `PASS`.
5. Copy the printed `https://...ngrok-free.app` URL. Keep cell 10 running.
6. In `D:\Projects\Sih-2026`, create an ignored `.env`:

```dotenv
SATQUERY_ENVIRONMENT=development
SATQUERY_MODEL_BACKEND=http
SATQUERY_MODEL_SERVICE_URL=https://YOUR-NGROK-DEV-DOMAIN.ngrok-free.app
SATQUERY_MODEL_SERVICE_TOKEN=THE_SAME_LONG_RANDOM_KAGGLE_SECRET
SATQUERY_API_KEY=
SATQUERY_ALLOWED_ORIGINS=http://localhost:3000
SATQUERY_MODEL_TIMEOUT_SECONDS=300
```

7. Start the local backend and frontend in separate PowerShell windows:

```powershell
# Terminal 1
Set-Location D:\Projects\Sih-2026\backend
& "$env:USERPROFILE\.venvs\satquery\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

```powershell
# Terminal 2
Set-Location D:\Projects\Sih-2026
$env:SATQUERY_BACKEND_URL = "http://127.0.0.1:8000"
npm run dev
```

The ngrok URL is a temporary model bridge, not the website. It expires when the notebook session
stops and requires the separate bearer token. Do not upload confidential imagery because payloads
traverse third-party tunnel infrastructure.

## Manual three-process start

Use this when debugging one layer:

```powershell
# Model service
$python = "$env:USERPROFILE\.venvs\satquery\Scripts\python.exe"
Set-Location D:\Projects\Sih-2026\model_service
& $python -m uvicorn satquery_model_service.main:app --host 127.0.0.1 --port 8080
```

```powershell
# Controller
$env:SATQUERY_MODEL_BACKEND = "http"
$env:SATQUERY_MODEL_SERVICE_URL = "http://127.0.0.1:8080"
Set-Location D:\Projects\Sih-2026\backend
& "$env:USERPROFILE\.venvs\satquery\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

```powershell
# Frontend
Set-Location D:\Projects\Sih-2026
$env:SATQUERY_BACKEND_URL = "http://127.0.0.1:8000"
npm run dev
```

## Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| `CUDA is unavailable` | CPU PyTorch or driver issue | Run `scripts/verify-local-runtime.py`; reinstall from setup or use Profile B |
| CUDA out of memory | Other GPU use or 4 GiB limit | Close GPU apps, restart; if repeated, use Profile B |
| Frontend says services offline | Backend readiness is degraded | Check `:8000/v1/health/ready`, then `:8080/ready` |
| Analysis fails safely | Model returned 4xx/5xx or timed out | Read controller/model terminals and analysis error code |
| Text exists but no box | Grounding output was not machine-parseable | The system returns text + warning; it intentionally creates no fake overlay |
| Kaggle URL stopped | Notebook ended or ngrok URL changed | Rerun cells 8–10 and update `.env` |
| First model start is very slow | Pinned weights are downloading | Keep the terminal open; later starts use the local HF cache |

## Local data and reset

Runtime state lives under `data/runtime/` relative to the process working directory. Uploads,
SQLite records, reports, and overlays are local and Git-ignored. Deleting them is not part of any
startup script. Remove them manually only after confirming that no evidence or report must be kept.
