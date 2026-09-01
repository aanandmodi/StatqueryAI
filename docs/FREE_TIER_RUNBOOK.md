# Zero-cost operations runbook

## Non-negotiable cost controls

- Never enable `provision_paid_endpoint` in any notebook.
- Never call `request_space_hardware` or select a paid Space accelerator.
- Model Space hardware must read **ZeroGPU**; API Space hardware must read **CPU Basic**.
- Do not add paid persistent storage.
- Stop Kaggle/Colab immediately after a final PASS cell and save/download artifacts first.
- Keep training, evaluation, change and fusion in separate sessions.

## Daily demo preparation

1. Open both Space pages early enough for cold start/build completion.
2. Check API `/v1/health/live`, then `/v1/health/ready`.
3. Run one small cached single-image query.
4. Keep one known-good RGB/GeoTIFF demo asset locally; do not depend on a live dataset download.
5. Keep screenshots/video of a valid run only as presentation backup, clearly labelled recorded.

## Expected free-tier failures

| Symptom | Meaning | Action |
|---|---|---|
| Space sleeping/building | Normal cold start | Wait and retry readiness |
| 429 / quota message | Daily/shared ZeroGPU quota | Stop retries; use cached result or wait for reset |
| API 503 model unavailable | Remote queue, build or specialist absent | Read `details.reason`; do not switch to demo output |
| Lost upload/job after restart | Ephemeral CPU filesystem | Re-upload; users retain originals |
| Long queue | Shared GPU contention | Use the bounded demo image/query; avoid parallel requests |

## Kaggle stop points

- Qwen training: stop only after section 9 prints the complete PASS dictionary.
- Evaluation: stop after section 7 writes predictions/summary and prints PASS.
- Change: stop after section 5 hashes weights/config and prints PASS.
- Fusion: stop after section 5 records all three ablations and prints PASS.

Close the browser tab only after selecting **Save Version** and stopping the session from Kaggle's
session control. Closing a tab alone does not reliably terminate a cloud GPU.

## Before presenting

- Confirm no secrets appear in notebook outputs or browser developer tools.
- Confirm the UI says `uncalibrated` when appropriate.
- Explain the European Sentinel → Indian Cartosat/RISAT domain gap.
- Demonstrate the execution trace and a rejected incompatible request.
- Have the current quota/cold-start limitation ready as an honest engineering trade-off.

