# Project progress

## Semantic-mask and report-quality pass — 2026-09-08

- Quality-v4 replaces the failed Qwen-box-first path for supported land-cover classes with a
  whole-scene LoveDA SegFormer transfer baseline. Qwen + SAM remains a bounded fallback for
  unsupported prompted objects. The checkpoint is pinned, but its author provides no model card or
  declared license; it is not a release artifact and no India/ISRO accuracy is claimed.
- Added an upload-ready free-Kaggle training notebook for a user-owned SegFormer B0 checkpoint. It
  uses 70% of LoveDA's official training split, stratified by urban/rural domain, while preserving
  the complete official validation split. It computes real per-class IoU/Dice, saves manifests and
  hashes, resumes checkpoints, reloads safetensors and gates optional Hub upload.
- Replaced terse temporal and optical/SAR prose with structured, measured reports. Change reports
  separate observed differences from possible explanations; fusion reports expose cue agreement
  and disagreement instead of presenting analytical proxies as semantic truth.
- Generated and syntax-checked the updated server, live-patch and segmentation-training notebooks.
  Verification: **277 backend/ML tests and 10 proxy tests passed**; frontend lint, TypeScript and
  production build passed. Quality-v4 still requires a live Kaggle GPU run and labelled-image
  evaluation before any accuracy claim.
- After restarting the local controller, a real-model five-workflow rerun passed using actual
  JPG/PNG jury fixtures. Temporal case `anl_6a5862bbd8ac40e5a35a659dfc0b4fcc` and fusion case
  `anl_a017754008e84cda96b0a57e940e1562` contain the new structured reports. The remote service was
  still quality-v3 during this run, so single-image case `anl_df956101f0b1482aac51a23bc92512f0`
  is transport evidence only and must not be used to claim the v4 mask improvement.

Next actions: apply quality-v4 in the active Kaggle kernel, run a real single-image smoke test, then
run the segmentation-training notebook in a separate Kaggle session. See
[Model quality roadmap](MODEL_QUALITY_ROADMAP.md).

## Evidence-integrity and spectral-overlay pass — 2026-09-08

- Removed the random/hard-coded benchmark generator and all reports derived from it. No replacement
  score is claimed.
- The evaluation-only Kaggle notebook now compares the exact same loaded base and LoRA model via
  `disable_adapter()`, and exports raw per-example JSONL. A local CPU scorer calculates real task
  metrics, scene-cluster bootstrap intervals and a clearly labelled calibration diagnostic.
- Added sensor-qualified NDVI, Sentinel-2-only NDBI/NBR and temporal dNBR. Cartosat requests that
  require SWIR explicitly refuse instead of guessing a band.
- Candidate masks are bounded-vectorized into class-labelled polygons and rendered as multi-colour
  SVG overlays, with the original raster mask retained as a fallback/download.
- Demo simulator prose now says that no pixels or models were inspected; it cannot be mistaken for
  measured evidence.
- Verification: **275 backend/ML tests passed**, Python/frontend lint passed, TypeScript passed and
  the production frontend build passed. Learned ChangeVQA and pixel-supervised fusion checkpoints
  remain open release gates.

## Critical-gap implementation — 2026-09-04

- Added optional learned Qwen intent proposals plus bounded dependency planning, focused step
  prompts, mask-loss/gain measurement, exact target/method/threshold/grid checks and abstention.
- Cartosat numeric bands are sensor-qualified; RISAT/EOS-04 polarizations, product mode and RTC
  declarations are preserved. Complex/conflicting inputs reject safely; ISRO transfer remains unvalidated.
- Corrected paired training/serving architecture, normalization, mask representation and artifact
  hashes. Fixed mixed-resolution Sentinel stacking, LMDB transaction-buffer lifetime, and evaluation
  of the wrong epoch. Optional paired runtime shares the existing Kaggle/ngrok service.
- Real case `anl_8ab89f87856a487088742d9a6c43dd76` completed via localhost proxy → Kaggle quality-v3
  Qwen/SAM for both Nepal dates → local baseline comparison → dependent extent tool. Six candidate
  masks were returned. Source images lacked georeferencing, so metric area was correctly withheld.
  Candidate counts are not verified ground truth; visual/class precision is still an open gate.
- `/v1/plan` returned 404 at the latest live check; this case correctly records **deterministic
  fallback**. User must run the provided live upgrade cell for learned planning. No new paired
  checkpoint was trained or released; no learned paired GPU-forward or accuracy claim is made.
- Local backend and frontend restarted; stale frontend PID lock pointed at an unrelated NVIDIA
  process and was archived without stopping that process. Local readiness and website returned 200.
- UI refined with local font stack, balanced headings, consistent panel/control corners and subtle
  depth. Updated architecture/system/pipeline/PRD/design and [gap audit](CRITICAL_GAPS.md).
- Verification initially passed 265 Python tests, 20 Node tests, frontend lint/type check and build;
  final additional regression results are recorded in [Testing](TESTING.md).
- No deployment, paid resources, token changes, local training or GitHub push.

Next cloud actions: [paired expert runbook](PAIRED_EXPERT_RUNBOOK.md).

## Studio delivery — 2026-09-04

- Common optical formats, explicit profiles, PNG verification and WebP/remote-v1 compatibility
  fixed; rejected assets cannot bypass validation through preview.
- Temporal masks/shared-valid counts/source-locked panes fixed. NDWI uses spectral-eligible support;
  mixed-target requests no longer lose nonwater masks. SAR reports omit optical cloud diagnostics.
- New studio pages, persistent cases, opacity controls, presets and bounded history/weather APIs.
- Actual JPEG analysis `anl_1665fb17f3d74ed59c4a8ebb30ad9e50` succeeded: one ~36s Qwen/SAM request,
  water mask and no duplicated full report. Remote quality-v2 was live. V3 prompts are prepared,
  not yet GPU-applied; no retraining required.
- Visual inspection of that JPEG overlay showed suspected forested-hillside false positives.
  Mask transport/rendering passed; water recognition did not achieve a professional accuracy gate.
  No accuracy improvement is claimed from the UI/report changes alone.
- Nepal pair `anl_d14431afdf644382a94e58a28a6fba02` produced candidate difference masks. Bright
  cloud-like regions dominate strong differences; not verified flood mapping.
- Live Sentinel catalog and NASA context worked via local proxy; weather table rendered in browser.
  Automatic historical download/crop/registration and attachment to reports remain unimplemented.
- 239 backend/ML tests and 20 frontend/proxy tests passed; lint, TypeScript and production build
  passed. PDF pages visually checked; heading/source pagination improved.
- No deployment, payment, token change, retraining or GitHub push. User data and existing edits
  preserved. Scientific accuracy and learned-pair release gates remain open.

Current usage: [Studio guide](STUDIO_GUIDE.md). Quality gates:
[Model evaluation plan](MODEL_EVALUATION_PLAN.md). Older dated records below remain historical.

Last updated: 2026-09-03. Nothing was deployed during this implementation pass.

## Report and mask quality upgrade (2026-09-03)

- Follow-up routing fix: the actual request "Show me all the water bodies in this region"
  (`anl_cc91a807294b4e67a201573a0edafae0`) used `caption` despite an active quality-v2 Kaggle
  handler. Display/find requests naming supported features now route to `grounding`; report-only
  requests and explicitly selected caption tasks keep their original behavior. Regression tests
  include the exact user query. Live rerun awaits backend restart after an approval-system failure;
  no successful new water mask is claimed yet.

- Local masks: bounded binary PNG validation, asset ownership, nodata exclusion, transparent
  cyan overlays with holes preserved, original/overlay toggle, binary download, computed coverage
  and projected-grid area. RGB-only rasters cannot masquerade as green/NIR products.
- Local reports: readable narrative paragraphs, raster footprint/extent/grid, mask measurements,
  method limits and verification guidance. PDF tables now wrap long model IDs within page bounds.
- A live test on the user's `sample.tif` exposed contradictory percentages and a truncated answer
  from the old handler. A conservative display guard now excludes unmeasured numeric claims and
  retains original model text explicitly as unverified audit data.
- Kaggle: maintained `notebooks/patches/quality_upgrade.py` is embedded as section 6b. It adds
  a 768-token base-instruction narrative (adapter temporarily disabled, with explicit provenance),
  released-adapter observations and SAM 2.1 tiny box-refined candidate masks on the free GPU.
- **Not yet verified:** the user must apply section 6b to the running Kaggle session and run real
  satellite inference. No SAM GPU result, real-scene IoU/precision or complete water inventory is
  claimed from local CPU/mocked tests. See [quality runbook](ANALYSIS_QUALITY.md).
- Verification: 189 Python tests, 18 frontend/proxy tests, backend lint, frontend lint/type-check
  and production build passed. The user TIFF completed another live caption request through the
  local frontend and Kaggle (`anl_7787e77f510a466295f6a101c825111f`); unmeasured percentages were
  excluded from its main narrative. Two-page synthetic mask-report PDF rendering was inspected.

## Evidence-backed status

| Workstream | Status | Evidence / next gate |
|---|---|---|
| Released Qwen LoRA | Prior training PASS reported and pinned runtime provided | Adapter/base revisions, manifests and fresh reload recorded in training artifacts |
| Kaggle inference server | Live public-tunnel connection verified | Readiness returned pinned Qwen adapter; real inference completed through the local frontend and backend |
| Local controller and frontend | Integration path exercised | All five automatic routes completed through the real local frontend proxy and backend |
| Single-image neural inference | Live laptop-to-Kaggle single-VQA transport passed | Actual pinned Qwen response on synthetic pixels; remaining task and satellite-quality gates stay open |
| Bi-temporal workflow | Real CPU analytical baseline runs | Shared-valid-pixel spectral-change fraction and reference-grid candidate box, not semantic CDVQA |
| Optical/SAR workflow | Real CPU analytical baseline runs | Relative-backscatter/optical candidate proxies, not a trained/calibrated semantic classifier |
| Evidence and downloads | Asset-aware | Source selector, uncropped preview, correct reference-grid overlay, PDF and structured trace |
| Geospatial validation | Hardened | Nodata/masks, same temporal modality/band count, north-up pair grid and coverage checks |
| Laptop launch tooling | Repaired code path and documented setup | Root dotenv/CORS/blank secrets handled; broken old venv preserved; fresh `satquery-cloud` setup |
| Public benchmark/hidden-set quality | Not established | Prescribed datasets and held-out metrics still required; no claim of perfection |

## What changed in this pass

- The default runner starts only the local frontend/controller and checks the remote Kaggle model.
  Local GPU loading requires explicit `--mode local`; `--mode demo` is visibly simulated.
- Configuration reads the root `.env` consistently. Comma/JSON CORS lists work, and blank optional
  secrets in the example configuration mean unset. Invalid requests return safe 422 responses.
- Model readiness validates JSON, capability and dependency checks, not only HTTP 200.
- The frontend streams multipart/JSON POST requests correctly in Node, has bounded proxy timeouts,
  shows degraded/offline status, and keeps selected-source evidence consistent.
- Automatic pair routing retains both images. Raster tools preserve internal masks and reject empty
  or uninformative shared inputs; boxes enclose candidate pixels, not dense segmentations.
- The Kaggle notebook has rerun-safe listener/tunnel cleanup and a 60-minute attended-demo cap.
  Managed Colab is explicitly not a supported tunnel host. No paid fallback is provisioned.
- Added strict acceptance checking: simulator output fails unless `--allow-simulated` is explicitly
  supplied for plumbing tests. `--auto-route` verifies input/query-based orchestration.

## Verification ledger

| Gate | Result / meaning |
|---|---|
| Python tests | **165 passed** across backend and ML, including 28 notebook regression checks; remaining warnings in older Rasterio fixtures/dependencies |
| Frontend transport/client tests | **17 passed** including the actual Vinext multipart preflight, 6 MiB upload, bounded rejection, proxy transport and readable errors |
| Static checks | Ruff, frontend lint, TypeScript and production build passed |
| Notebook | Valid nbformat; 23 cells / 11 code cells compile; synchronized source; no outputs |
| Live local five-workflow integration | Passed through dev and built frontend; synthetic GeoTIFFs; single tasks simulated, pair tools actually executed |
| Real Kaggle Qwen run | **User screenshots: T4 load, direct text generation, local FastAPI HTTP 200/PASS** |
| Public ngrok/laptop-to-Kaggle run | **Passed for single VQA through frontend proxy → local controller → ngrok → pinned Qwen; image/PDF downloads completed** |
| Satellite benchmark accuracy/localization | **Pending licensed/prescribed evaluation data and metric artifacts** |

Latest local integration outputs are under ignored `outputs/smoke-20260903/acceptance-production-build/`: five JSON
records, five JPEGs, five PDFs and a summary explicitly stating simulator-allowed integration
scope. These synthetic fixtures are not evidence of remote-sensing accuracy.

## Kaggle screenshot error repair — 2026-09-03

- Corrected the false Colab detection: Kaggle's runtime marker plus working directory takes
  precedence over inherited Colab image variables. The notebook still refuses actual Colab and
  unverified hosts. No manual platform override or quota workaround was added.
- Removed deprecated Pillow `fromarray(mode=...)` arguments and explicitly neutralized the
  checkpoint's sampling-only settings for deterministic generation.
- Replaced the unreferenced synthetic TIFF with a labelled test GeoTIFF; this does not assert a
  real location or grant georeferencing to user uploads. Domain/calibration warnings remain.
- Added HTTP-before-JSON checks, clear missing-tunnel prerequisites, HTTPS validation and stale
  URL cleanup after failed reconnects. Retained the 60-minute timer and server-side secrets.
- Added a copyable section 8 patch and recovery instructions that preserve the running model.
  Regression tests use injected tunnel/timer/model doubles, not live ngrok or local GPU weights.
- Nothing deployed, no training repeated, no cloud resources provisioned, and no credentials
  changed during the repair. The subsequent live connection check is recorded below.

## Live local launch and Kaggle connection — 2026-09-03

- Started the frontend at `http://localhost:3000` and controller at `http://127.0.0.1:8000`.
  Updated only the ignored local model URL setting to the user's current ngrok HTTPS URL;
  the service secret was preserved and never printed.
- Frontend HTML returned HTTP 200. Backend readiness through the frontend proxy returned
  `status=ok`, `database=true`, `model_gateway=true`, and `model_backend=http`.
- Uploaded the existing explicitly synthetic `single.tif` through the frontend proxy and asked
  the real model to describe its colors. Analysis `anl_747d25bfeafb4ed19c61e7a6ea40d197`
  succeeded with adapter revision `ed12e59e0def` and answer
  "blue, purple, red, orange, yellow". No simulator was enabled.
- JSON, overlay JPEG and PDF artifacts were saved under ignored
  `outputs/live-kaggle-20260903/anl_747d25bfeafb4ed19c61e7a6ea40d197/`.
- This verifies one real-model transport path, not satellite accuracy or valid localization.
  The remaining tasks still need live acceptance and domain-specific quality evaluation.
  No website deployment or paid resource was created; the connection lasts only while the
  user's bounded Kaggle/ngrok session is active.

## User TIFF upload repair — 2026-09-03

- Inspected the supplied `sample.tif`: 6,018,396 bytes, 1001×1001 pixels, three uint16 bands,
  EPSG:32631 and a valid 10 m affine mapping. Direct backend upload returned HTTP 201.
- Reproduced HTTP 413 at the frontend before the proxy handler. The installed Vinext version
  runs multipart forms through a 1 MiB server-action preflight even for App Router API routes;
  early connection closure surfaced as the browser's unhelpful `Failed to fetch` message.
- Set `experimental.serverActions.bodySizeLimit` to `51mb` in `next.config.ts`: 50 MiB file
  plus 1 MiB bounded multipart overhead. UI and backend file limits remain 50 MiB. No dependency
  patch, unbounded upload, token change, CORS relaxation or paid service was introduced.
- Added client diagnostics for connection failures, plain-text HTTP 413 and invalid JSON.
  Regression checks include the real framework preflight, not just direct route invocation.
- Restarted the frontend and verified the exact file through both the development server and
  a temporary local production-build server. Production upload returned HTTP 201; that temporary
  server was stopped after the check. Frontend lint, TypeScript, build and all 17 Node tests passed.
- In the actual browser, the original question completed at 100%, displayed its preview, and
  automatically selected the caption specialist. Pinned Qwen returned
  "coastal, urban, agricultural, forest". Analysis `anl_3cdd7ab67a4b40089081f51b5ece64f5`
  has nonempty JSON, JPEG and PDF artifacts under
  `outputs/live-kaggle-upload-fix-20260903/anl_3cdd7ab67a4b40089081f51b5ece64f5/`.
- This is a successful real-model application test, not independently verified land-cover
  accuracy. Caption mode returned no boxes; localization quality remains a separate gate.

## Remaining gates

```mermaid
flowchart LR
    Secrets[Kaggle secrets + free GPU] --> Notebook[Run inference sections 1–9]
    Notebook --> Connect[Copy tunnel URL + secret into local env]
    Connect --> Query[Real satellite image query through local UI]
    Query --> Five[Five-workflow acceptance without simulator]
    Five --> Train[Train learned pair experts with valid datasets]
    Train --> Bench[Public split metrics and localization gates]
    Bench --> Hidden[Organizer hidden-set evaluation]
```

The next user-operated steps are in [NGROK_KAGGLE_RUNBOOK.md](NGROK_KAGGLE_RUNBOOK.md).
Learned change/fusion evaluation is a requirement for strong SIH semantic claims, not an optional
cosmetic upgrade. The existing analytical tools are useful baselines, not proof of that quality.

## Limits and deployment state

- Free GPU availability, temporary tunnels and traffic quotas cannot guarantee continuous hosting.
- BigEarthNet European Sentinel training does not establish Cartosat/RISAT performance.
- A user-supplied coordinate is context; it does not georeference an ungeoreferenced image or prove
  that the scene depicts that location. TIFF uploads require a valid CRS and transform.
- Grounding can return no valid box. The app reports that limitation instead of inventing one.
- Prior records say Hugging Face Spaces are private archives and the former ChatGPT Site is
  owner-only; that external state was not re-audited this pass. Local deployment bindings/helpers
  remain removed. No public upload, hosting change, paid resource or GitHub push was performed.
