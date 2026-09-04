# Local development runbook

The canonical setup and connection instructions are in
[NGROK_KAGGLE_RUNBOOK.md](NGROK_KAGGLE_RUNBOOK.md). All modes keep the frontend, controller,
database, uploads, reports and overlays local. None deploys the website.

## Execution modes

| Runner mode | Single-image specialist | Pair workflows | Appropriate use |
|---|---|---|---|
| `--mode remote` (default, preferred) | Existing LoRA on a temporary Kaggle GPU | Local CPU analytical tools | Attended real-model integration demo within free quotas |
| `--mode demo` | Explicit deterministic simulator; no neural inference | Local CPU analytical tools | UI/API plumbing only, never model acceptance |
| `--mode local` | Opt-in 4-bit CUDA model on this computer | Local CPU analytical tools | Only when hardware has passed a real loading/generation test |

The remote mode is not production hosting: its ngrok tunnel has a maximum 60-minute demo window,
no automatic renewal and no uptime guarantee. Managed Colab reverse-proxy serving is not supported.
Do not bypass platform limits or upgrade to paid infrastructure to keep a demonstration running.

## One-time setup

Install Python 3.12 and Node.js 22 LTS or newer, then reopen PowerShell. `python --version` and
`node --version` must work. The old `satquery` virtualenv may point to a missing Python install;
setup preserves it and uses the separate `satquery-cloud` environment.

```powershell
Set-Location D:\Projects\Sih-2026
& .\scripts\setup-local.ps1
```

This installs the controller/test dependencies and frontend packages, not CUDA or model weights.
For a Python installation outside PATH, pass `-PythonExecutable 'C:\actual\path\to\python.exe'`.
Resolve any setup error before starting services; native installation failures stop setup.

## Preferred: Kaggle model with local backend and frontend

1. Follow the canonical runbook to configure both Kaggle secrets and run inference sections 1–9.
   No fine-tuning rerun or training dataset is required.
2. Require the direct-model, local-HTTP and public-tunnel smoke checks to pass. These synthetic
   checks establish generation and transport, not remote-sensing accuracy.
3. Keep section 10 running during attended testing.
4. Create root configuration files only if absent:

```powershell
Set-Location D:\Projects\Sih-2026
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
if (-not (Test-Path .env.local)) { Copy-Item .env.local.example .env.local }
notepad .env
```

Set the actual section-8 URL and identical Kaggle service secret in root `.env`:

```dotenv
SATQUERY_ENVIRONMENT=development
SATQUERY_MODEL_BACKEND=http
SATQUERY_PAIR_BACKEND=local
SATQUERY_MODEL_SERVICE_URL=https://YOUR-ACTUAL-DEV-DOMAIN.ngrok-free.app
SATQUERY_MODEL_SERVICE_TOKEN=YOUR_SAME_RANDOM_SECRET
SATQUERY_API_KEY=
SATQUERY_ALLOWED_ORIGINS=["http://localhost:3000"]
SATQUERY_MAX_UPLOAD_BYTES=52428800
SATQUERY_MODEL_TIMEOUT_SECONDS=300
```

Keep `.env.local` pointed at the local controller:

```dotenv
SATQUERY_BACKEND_URL=http://127.0.0.1:8000
SATQUERY_BACKEND_API_KEY=
```

Start both local services with one command:

```powershell
Set-Location D:\Projects\Sih-2026
& "$env:USERPROFILE\.venvs\satquery-cloud\Scripts\python.exe" .\scripts\run-local.py --mode remote
```

The runner checks model JSON readiness, starts the controller on `127.0.0.1:8000`, then starts
the frontend at `http://localhost:3000`. HTTP 200 containing HTML is not accepted as model readiness.

### Separate terminals instead

Use these instead of the runner, not at the same time. Start from the repository root so runtime
data paths are consistent. The backend always reads the root `.env`.

```powershell
# Terminal 1: controller
Set-Location D:\Projects\Sih-2026
& "$env:USERPROFILE\.venvs\satquery-cloud\Scripts\python.exe" -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

```powershell
# Terminal 2: frontend
Set-Location D:\Projects\Sih-2026
npm run dev
```

Check readiness and capabilities, then run real-image acceptance:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/v1/health/ready
Invoke-RestMethod http://127.0.0.1:8000/v1/capabilities
```

Require `status: ok` and healthy checks. Follow [SIH_DEMO_RUNBOOK.md](SIH_DEMO_RUNBOOK.md), using
the strict default model checks and `--auto-route`. A readiness response alone does not prove
that the uploaded satellite image receives a correct answer.

### Optimized frontend, still running locally

Keep the manually started backend running. Stop any frontend dev server first, then:

```powershell
Set-Location D:\Projects\Sih-2026
npm run build
npm run start
```

`npm run start` explicitly binds the built frontend to `127.0.0.1:3000`. This is a local
production-build check, not a website deployment or proof of production model quality. It still
loads `.env.local` for the backend URL; do not start it alongside the one-command dev runner.

## Offline UI/API development

```powershell
Set-Location D:\Projects\Sih-2026
& "$env:USERPROFILE\.venvs\satquery-cloud\Scripts\python.exe" .\scripts\run-local.py --mode demo
```

No ngrok credentials, GPU or model download are required. Single-image output is visibly simulated;
the two pair tools still analyze pixels on the CPU. The acceptance script rejects simulator output
unless you explicitly pass `--allow-simulated` for a plumbing-only check.

## Optional local GPU profile

This laptop's previously recorded RTX 2050 has 4 GiB VRAM. Full BF16 weights leave insufficient
room for vision activations; even NF4 fit must be tested, not assumed. Prefer Kaggle for this machine.
If explicitly testing locally, close other GPU-heavy software and install the optional dependencies:

```powershell
Set-Location D:\Projects\Sih-2026
& .\scripts\setup-local.ps1 -WithLocalModel
& "$env:USERPROFILE\.venvs\satquery-cloud\Scripts\python.exe" .\scripts\run-local.py --mode local
```

This mode downloads the immutable base/adapter and starts model `:8080`, controller `:8000`, then
frontend `:3000`. It uses bounded images, 4-bit weights and short generation; these controls do
not guarantee hardware fit. Stop on repeated OOM and return to the preferred remote profile.

## Stop and restart

- Press `Ctrl+C` in the runner, or stop both manually started terminals.
- Interrupt Kaggle section 10, then stop the Kaggle session to release GPU quota. Stopping local
  services alone does **not** stop the remote notebook.
- If model variables remain after section 10 ends, rerun sections **6–10**. Section 10 shuts down
  FastAPI as well as ngrok. A full kernel restart requires every inference section again.
- After a tunnel restart, update the URL in root `.env` if it changed and restart local services.
- Never automate notebook restarts or activity to evade provider limits.

## Troubleshooting and local data

| Symptom | Action |
|---|---|
| `python` missing or old environment cannot launch | Install Python 3.12 and rerun setup using `satquery-cloud` |
| Model API 401 | Match the Kaggle and root `.env` service secrets; the ngrok authtoken is a different secret |
| ngrok offline/502 | Check notebook section 7, then rerun sections 8–10; if API stopped, start at section 6 |
| Frontend reports offline | Inspect controller readiness, then the notebook/runner errors |
| No grounding box | Read the warning; do not substitute a fabricated box |
| Pair rejected | Check declared modalities, time roles, CRS, overlap, resolution, grid and bands |
| CUDA OOM | Stop local-model attempts; use the quota-limited Kaggle profile |

When launched from the root, local state is under `data/runtime/` and remains Git-ignored. Older
runs launched inside `backend/` may have separate state under `backend/data/runtime/`; preserve it
until reviewed. Startup does not delete evidence or reports. Do not send confidential imagery
through a third-party notebook/tunnel without appropriate permission.
