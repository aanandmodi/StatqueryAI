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
- Added a Gradio Server API deployment using the second free ZeroGPU slot.
- Added standalone evaluation-only, Change-VQA and TerraMind fusion `.ipynb` notebooks.
- Added backend/Space tests and architecture, PRD, security, deployment, free-tier and UI docs.
- Wired the frontend to real uploads, analysis polling, RGB/marked-image outputs, evidence, warnings,
  trace and reports.
- Added the zero-cost Sites edge inference route and in-browser GeoTIFF preview/overlay export, so
  the public demo needs only one model Space.
- Published the free static holding Space with full ZeroGPU source and submitted Community grant #1.
- Removed fabricated answer/confidence examples and added explicit free-tier sleeping/quota states.

## Verified locally

- Backend tests: 15 passed.
- ML utility tests: 9 passed.
- Space helper tests: 4 passed.
- Python source/notebook scripts compile.
- Ruff and frontend lint pass; the production Vinext/Sites build completes.

## External/account-bound steps remaining

- Wait for Hugging Face to approve Community grant #1 (or for the account to reach 30 days), then
  run `scripts/deploy_zero_gpu_space.py` to activate the already-published Gradio source.
- Run the evaluation-only notebook and retain pinned validation/test output.
- Attach SECOND and run the Change-VQA notebook; publish only after its PASS gate.
- Run the TerraMind fusion notebook; publish only after all modality ablations pass.

## Product work remaining

- Run one public model/API query after the grant activates ZeroGPU; the Sites edge route already
  targets the permanent Space subdomain and needs no token.

## Known constraints

- The Hugging Face account is too new for its free ZeroGPU allocation; the platform returned 402.
- Free ZeroGPU is quota-limited and has no SLA; caching and explicit 503 states are required.
- Change training requires SECOND imagery, which CDVQA annotations do not redistribute.
- European Sentinel benchmark performance does not establish Indian Cartosat/RISAT performance.
