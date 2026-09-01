# Project progress

Last updated: 2026-09-01

## Completed

- Parsed the SIH project guide as specification, not executable instructions.
- Fine-tuned Qwen3-VL 2B with balanced BigEarthNet.txt binary/MCQ/caption/grounding examples.
- Published a public, ungated adapter with immutable base revision and SHA-256 manifest.
- Passed fresh-session Hub reload and the final Kaggle release gate.
- Built the Gradio ZeroGPU inference package with pinned revisions and bounded API.
- Added grounding parsing, payload validation and uncalibrated score semantics.
- Added a FastAPI ZeroGPU gateway with RGB previewing, queue polling, retry and TTL/LRU cache.
- Added a free CPU Docker Space deployment package and configuration helper.
- Added standalone evaluation-only, Change-VQA and TerraMind fusion `.ipynb` notebooks.
- Added backend/Space tests and architecture, PRD, security, deployment, free-tier and UI docs.

## Verified locally

- Backend tests: 15 passed.
- Space helper tests: 4 passed.
- Python source/notebook scripts compile.
- Ruff passes on backend, deployment package and scripts after the current changes are finalized.

## External/account-bound steps remaining

- Upload the model Space with `scripts/deploy_zero_gpu_space.py` using the owner's `HF_TOKEN`.
- Select ZeroGPU in that Space's hardware settings; never select paid hardware.
- Deploy/configure the API CPU Space with `scripts/deploy_free_backend_space.py`.
- Run the evaluation-only notebook and retain pinned validation/test output.
- Attach SECOND and run the Change-VQA notebook; publish only after its PASS gate.
- Run the TerraMind fusion notebook; publish only after all modality ablations pass.

## Product work remaining

- Wire the existing frontend to the real asset/analysis/events/report APIs.
- Replace all static answer/confidence/evidence examples with truthful state from the backend.
- Complete accessibility, responsive and reduced-motion QA.
- Build and publish the frontend, then run one public end-to-end smoke test.

## Known constraints

- No Hugging Face token is available in this local session, so account-owned Space creation cannot
  be completed without the owner supplying the secret.
- Free ZeroGPU is quota-limited and has no SLA; caching and explicit 503 states are required.
- Change training requires SECOND imagery, which CDVQA annotations do not redistribute.
- European Sentinel benchmark performance does not establish Indian Cartosat/RISAT performance.

