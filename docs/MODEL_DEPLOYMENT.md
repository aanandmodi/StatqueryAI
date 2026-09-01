# Free model and API deployment

## What is already complete

The public adapter is valid and pinned. Its final release gate confirmed immutable base revision,
SafeTensors, manifests, Hub upload and clean reload. The remaining deployment uses two public Spaces:

- `aanandmodi/satquery-qwen3vl-space`: Gradio + ZeroGPU inference.
- `aanandmodi/satquery-api`: Docker + free CPU FastAPI controller.

## Account prerequisites

Create a fine-grained Hugging Face token that can write only to the two Space repositories. Store
it as `HF_TOKEN` in the terminal/notebook secret store. Do not paste it into a `.py`, `.ipynb`,
`.env.example`, browser bundle, screenshot, or chat.

ZeroGPU eligibility currently requires a free personal account in good standing. Quotas and
availability can change; inspect the Space hardware screen before choosing anything.

## 1. Upload the model Space

```powershell
$env:HF_TOKEN = Read-Host "HF token"
.\.venv\Scripts\python.exe scripts\deploy_zero_gpu_space.py
Remove-Item Env:HF_TOKEN
```

Then open the Space settings and choose **ZeroGPU**. This is the only mandatory manual hardware
selection because the deployment script intentionally refuses to request hardware. Do not choose
T4, L4, A10G, A100 or an Inference Endpoint.

Wait for `Running`, open the Space UI, upload a small RGB satellite image, and verify all three task
modes. The backend machine endpoint is `/gradio_api/call/analyze` and uses the standard two-step
Gradio queue protocol.

## 2. Deploy the CPU API Space

Generate an application API key locally and keep it for the frontend's server-side environment:

```powershell
$bytes = New-Object byte[] 32
[Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
$env:SATQUERY_API_KEY = [Convert]::ToBase64String($bytes)
$env:HF_TOKEN = Read-Host "HF token"
.\.venv\Scripts\python.exe scripts\deploy_free_backend_space.py `
  --allowed-origin "https://YOUR-SITES-DOMAIN"
Remove-Item Env:HF_TOKEN
```

The helper uploads only the backend package, configures public variables, stores the application
key as a write-only Space secret, and leaves hardware on CPU Basic. Save the generated application
key in the frontend host's server-side secret manager, then remove it from the terminal.

Verify:

```text
https://aanandmodi-satquery-api.hf.space/v1/health/live
https://aanandmodi-satquery-api.hf.space/v1/health/ready
```

`live=ok` and `ready=degraded` means the API runs but the model Space is sleeping/building or the
URL is wrong. It does not mean the controller should fabricate output.

## 3. Backend configuration

For local development copy `.env.example` to `.env`. Important settings:

| Variable | Meaning |
|---|---|
| `SATQUERY_MODEL_BACKEND=space` | Use the Gradio queue gateway |
| `SATQUERY_SPACE_URL` | Public `*.hf.space` model URL |
| `SATQUERY_SPACE_TOKEN` | Optional server-only HF token for the deployment account quota |
| `SATQUERY_API_KEY` | Required in production; frontend proxy sends `X-API-Key` |
| `SATQUERY_ALLOWED_ORIGINS` | Exact frontend origin, not `*` |

## 4. Quota-aware behavior

- Same asset hash + task + query + parameters is cached for 24 hours per API process.
- ZeroGPU requests are queued at concurrency one and bounded to 60 GPU seconds / 256 output tokens.
- 429/502/503/504 responses receive a small bounded retry; failures surface as 503.
- Free CPU Space files are ephemeral. Users must keep originals and download reports.
- Dedicated Inference Endpoints remain disabled; they are not a free fallback.

Official references: [ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu),
[Gradio Space APIs](https://huggingface.co/docs/hub/spaces-api-endpoints), and
[Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker).

