# SatQuery AI

SatQuery is an auditable remote-sensing assistant for single-scene visual questions, captions,
grounding, bi-temporal change analysis, and optical/SAR fusion. The controller validates raster
inputs, routes only to a closed set of specialist tools, integrates evidence, labels uncertainty,
records an execution trace, and creates a downloadable report.

The project is designed for a **strict zero-cost demo deployment**. It never provisions Hugging
Face Dedicated Inference Endpoints or other billed GPU instances.

## Current release status

| Component | Status | Evidence |
|---|---|---|
| Qwen3-VL BigEarthNet.txt LoRA | Released | Public, pinned Hub adapter; fresh reload passed |
| Single-image VQA/caption/grounding | Ready to deploy | ZeroGPU Gradio package and API contract complete |
| FastAPI controller | Implemented | Upload validation, policy routing, queue bridge, cache, trace, report |
| Evaluation-only workflow | Ready to run | Pinned validation/test notebook; no retraining |
| Change-VQA | Training workflow ready | Requires the SECOND pixels attached in Kaggle |
| Optical/SAR fusion | Training workflow ready | Uses the bounded paired S1/S2 LMDB subset and TerraMind |
| Frontend | Visual prototype | API wiring and final QA intentionally follow backend/model completion |

Published adapter: [aanandmodi/satquery-qwen3vl-bigearthnet-txt-lora](https://huggingface.co/aanandmodi/satquery-qwen3vl-bigearthnet-txt-lora), pinned by the deployment code at `ed12e59e0def9468bdf4a226789fc1b77c7900e7`.

## Repository map

```text
app/                              Vinext/React frontend
backend/                          FastAPI controller and tests
deploy/huggingface_space/         Qwen3-VL Gradio ZeroGPU service
deploy/huggingface_backend/       Free CPU Docker Space container
ml/                               Reusable specialist-model utilities
model_service/                    Long-lived GPU service for non-Space deployments
notebooks/                        Kaggle/Colab training and evaluation notebooks
scripts/                          Audited deployment helpers
docs/                             Architecture, PRD, security and runbooks
```

## Run locally against the free model Space

The model Space must be deployed first; see [MODEL_DEPLOYMENT.md](docs/MODEL_DEPLOYMENT.md).

```powershell
Copy-Item .env.example .env
# Edit .env with your Space URL and a random SATQUERY_API_KEY.
.\.venv\Scripts\python.exe -m pip install -r backend\requirements-dev.txt
Set-Location backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

In another terminal:

```powershell
npm install
npm run dev
```

API liveness is `GET http://localhost:8000/v1/health/live`; readiness additionally tests the
configured model gateway.

## Notebook order

1. `SatQuery_Qwen3VL_Training_v2.ipynb` — already completed and released.
2. `SatQuery_Qwen3VL_Evaluation_Only.ipynb` — run validation, then one final test pass.
3. `SatQuery_ChangeVQA_Training.ipynb` — run in a fresh GPU session after attaching SECOND.
4. `SatQuery_TerraMind_Fusion_Training.ipynb` — run in a separate fresh GPU session.

Each notebook has an explicit final PASS gate and a safe stopping instruction. Do not combine the
three training workloads in one GPU session.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest backend\tests -q
.\.venv\Scripts\python.exe -m pytest deploy\huggingface_space\tests -q
.\.venv\Scripts\python.exe -m ruff check backend\app backend\tests deploy scripts
npm run build
```

See [ARCHITECTURE.md](docs/ARCHITECTURE.md), [PRD.md](docs/PRD.md),
[FREE_TIER_RUNBOOK.md](docs/FREE_TIER_RUNBOOK.md), and [PROGRESS.md](docs/PROGRESS.md).

