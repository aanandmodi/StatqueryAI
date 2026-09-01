# Free model and API deployment

## What is already complete

The public adapter is valid and pinned. Its final release gate confirmed immutable base revision,
SafeTensors, manifests, Hub upload and clean reload. The zero-cost public deployment needs one Space:

- `aanandmodi/satquery-qwen3vl-space`: Gradio + ZeroGPU inference.

The Sites frontend includes a same-origin edge API, so no second hosted controller is required.
`aanandmodi/satquery-api` remains an optional Python deployment for older/eligible accounts.

## Account prerequisites

Create a fine-grained Hugging Face token that can write only to the two Space repositories. Store
it as `HF_TOKEN` in the terminal/notebook secret store. Do not paste it into a `.py`, `.ipynb`,
`.env.example`, browser bundle, screenshot, or chat.

ZeroGPU eligibility currently requires a free personal account in good standing. New accounts can
receive a `402` requiring 30 days of account age. In that case run `scripts/request_zero_gpu_grant.py`;
it publishes a free static holding Space plus the complete source and opens the Community request.

## 1. Upload the model Space

```powershell
$env:HF_TOKEN = Read-Host "HF token"
.\.venv\Scripts\python.exe scripts\deploy_zero_gpu_space.py
Remove-Item Env:HF_TOKEN
```

The helper requests **ZeroGPU** explicitly. It never substitutes T4, L4, dedicated A10G, A100 or
an Inference Endpoint.

Wait for `Running`, open the Space UI, upload a small RGB satellite image, and verify all three task
modes. The backend machine endpoint is `/gradio_api/call/analyze` and uses the standard two-step
Gradio queue protocol.

## 2. Optional Python API Space

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

This is not required by the public Sites build. The helper uploads the complete Python controller,
configures public variables, stores the application key as a write-only Space secret, and requests
only ZeroGPU. Use it after the account becomes eligible when audit PDFs and server-side raster
validation are preferred over keeping the second free slot for a specialist.

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
- Free Space files are ephemeral. Users must keep originals and download reports/marked images.
- Dedicated Inference Endpoints remain disabled; they are not a free fallback.

Official references: [ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu),
[Gradio Space APIs](https://huggingface.co/docs/hub/spaces-api-endpoints), and
[Gradio Server mode](https://www.gradio.app/guides/server-mode).
