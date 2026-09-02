# Zero-cost execution runbook

## Zero cost is not unlimited hosting

| Option | Compute lifetime | Public endpoint | Main limit |
|---|---|---:|---|
| Laptop 4-bit | While local process runs | No | 4 GiB VRAM / 7.7 GiB RAM |
| Kaggle/Colab free GPU | While notebook session runs | Temporary ngrok HTTPS URL | Session/quota/idle limits |
| Hugging Face ZeroGPU | While eligible Space runs | Stable Space URL | Account eligibility and daily GPU quota |
| HF model repository | Durable file hosting | Download URL only | No compute |
| HF Inference Providers | Provider-managed | API | Model support, free credit, and possible billing |

There is no honest way to guarantee a permanent, unlimited, production-grade custom VLM endpoint
at zero cost. The project therefore uses explicit finite profiles and fails safely when compute is
absent.

## Recommended SIH demo profile

Use the local frontend and controller. Prefer laptop 4-bit only after its real smoke passes;
otherwise use the temporary Kaggle/Colab model notebook. Cache one tested demo input/result for
rehearsal evidence, but never present cached output as a new inference.

## Quota and failure handling

| Event | Required response |
|---|---|
| ngrok URL expires/changes | Mark model unavailable; rerun cells 8–10 and update backend URL |
| ZeroGPU quota exhausted | Return `503`; wait for reset; no paid fallback |
| Laptop CUDA OOM | Stop model process, clear other GPU usage, retry once, then use notebook |
| Provider asks for payment method | Stop; do not provision |
| Token limit/rate limit | Reduce generation cap or authenticate within free allowance; do not claim unlimited use |

## Cost guardrails

- Persistent deployment helpers have been removed from the repository.
- Dedicated Inference Endpoints are never created automatically.
- No cloud resource is selected because it is merely described as a trial.
- A future provider must support a hard zero-spend or explicit spend cap before use.
- Account tokens are stored only in provider secret stores or server environments.
