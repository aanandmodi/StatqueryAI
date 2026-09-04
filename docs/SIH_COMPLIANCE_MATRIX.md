# SIH 2026 problem-statement compliance

This document traces the supplied **SatQuery AI — Interactive Vision-Language Assistant for
Multimodal Remote Sensing Image Analysis through Text Queries** problem statement to executable
repository evidence. It distinguishes feature completeness from model-quality claims.

## Verdict

The architecture is feasible as a local prototype, with executable paths for all five workflows.
This is **not yet a fully validated solution to every semantic requirement**. The released
BigEarthNet.txt-adapted Qwen specialist handles single-image work; bounded local analytical tools
make bi-temporal and optical/SAR pair workflows runnable. Those tools cannot answer arbitrary
semantic change questions or establish land-cover classes reliably. Learned expert training and
evaluation remain required before claiming those capabilities. The current Kaggle inference run,
prescribed public-split evaluation, and undisclosed ISRO/SAC evaluation remain separate gates.

```mermaid
flowchart LR
    Q[Question + validated evidence set] --> R{Closed-set router}
    R -->|one image| V[Adapted Qwen VQA/caption/grounding]
    R -->|time A + time B| C[CPU spectral-change tool]
    R -->|optical + SAR| F[CPU cross-modal proxy tool]
    V --> I[Schema integrator]
    C --> I
    F --> I
    I --> O[Text + evidence + confidence semantics + trace + downloads]
```

## Requirement traceability

| Problem-statement requirement | Status | Executable evidence | Qualification |
|---|---|---|---|
| Remote-sensing fine-tuning/domain adaptation | Implemented | `notebooks/SatQuery_Qwen3VL_Training_v2.ipynb`; public LoRA at immutable revision `ed12e59...` | Adapted on BigEarthNet.txt-derived image/question records |
| One optical, multispectral, **or SAR** image | Implemented | UI modality declaration; raster validator; Qwen one/two-band preview fix | SAR is rendered as a repeated grayscale visual channel; raw backscatter semantics are not invented |
| Single-image VQA | Implemented | `single_vqa` registry task and Qwen adapter | Public-split accuracy still requires the prescribed VRSBench/RSVQA evaluation run |
| Captioning or grounding | Implemented beyond minimum | `caption` and `grounding`; strict box parser | A missing/malformed generated box becomes a warning, never fake geometry |
| Bi-temporal change description/change VQA | Implemented baseline | `LocalPairSpecialistGateway`; automatic `change_vqa` route; UI time-A/time-B upload | Current CPU output is a quantified spectral-change proxy, not a learned semantic CDVQA probability |
| Spatial change evidence | Implemented baseline | Changed-pixel fraction and enclosing candidate box on the time-A reference grid | A box is not a dense mask; pixels inside it are not all necessarily changed |
| Co-registered optical/multispectral + SAR analysis | Implemented baseline | Local optical-context/SAR-backscatter proxy tool; automatic fusion route | Candidate water/built-up regions are uncalibrated proxies, explicitly labelled |
| Query/task classification | Implemented | Pattern classifier plus validated-input fallback in `PolicyRouter` | No arbitrary LLM-selected code, URL, path, or model |
| Check number, modality, format, metadata, compatibility | Implemented | `RasterValidator` and task validation | Pair gate checks north-up grids, CRS, ≥98% coverage of the larger footprint, ≤2% resolution delta and ≤0.25-pixel grid offset; temporal pairs require equal declared modality and band count |
| Predefined model/tool registry | Implemented | `TaskType` enum, hybrid specialist gateway | Five closed tasks only |
| Configure permitted parameters only | Implemented | Router accepts bounded targets, threshold, and bbox; drops unknown fields | Threshold clamped to 0.05–0.95 |
| Select, sequence, and execute workflow automatically | Implemented | Input-aware policy plan and async job service | UI defaults to **Auto route**; explicit route remains available for judging |
| Combine textual/spatial outputs and confidence | Implemented | Structured integrator and `Confidence` contract | Open-ended/proxy outputs are evidence-quality scores, not correctness percentages |
| Auditable execution summary | Implemented | Trace includes task, tool, model version, permitted parameters, reason, status, timing | Internal chain-of-thought is neither stored nor exposed |
| GeoTIFF/TIFF support | Implemented | Magic-byte and Rasterio inspection | File extension alone is not trusted |
| PNG/JPEG only for named public benchmarks | Implemented | Allowlist: VRSBench, RSVQA, CDVQA, SECOND | Production benchmark import requires a server-controlled manifest |
| Interactive GUI/web app | Implemented | `app/page.tsx` | Local-first; no website deployment required |
| Textual and visual results | Implemented | Asset selector, uncropped preview, per-asset evidence and marked JPEG | Pair boxes remain attached to the correct reference image; missing model geometry stays missing |
| Downloadable report | Implemented | Deterministic PDF report endpoint | Includes request, trace, warnings, and provenance |
| Code, models, tests, demonstration | Implemented in repository | Automated tests, notebooks, public adapter, `scripts/sih-acceptance.py` | Five-workflow real-model run still needs an active free GPU session |

## Evaluation readiness

| Evaluation source | Current readiness | Required action before final submission |
|---|---|---|
| BigEarthNet.txt | Adapter trained, hashed, uploaded, fresh-reloaded | Preserve the notebook output and exact split manifest with the submission |
| VRSBench | Loader/task path documented; no final prescribed-split score recorded | Run VQA/caption/grounding metrics on the exact organizer-prescribed test split |
| RSVQA | Same | Record normalized VQA accuracy and per-question-type results |
| CDVQA | Learned training notebook exists; CPU baseline runs now | Train/evaluate the learned change expert and record answer plus localization metrics |
| ISRO/SAC Cartosat-2S + RISAT hidden set | Interface intended for these data; compatibility is not yet verified | Validate actual sensor bands, units, co-registration and annotations when released; cannot pre-score hidden data |

## Feasibility constraints

| Constraint | Feasible approach | Limitation that must be stated |
|---|---|---|
| Zero monetary cost | Local UI/controller/storage; CPU pair tools; quota-limited Kaggle GPU for Qwen | Temporary tunnelling is subject to current platform rules; no uptime SLA or paid fallback |
| Low-spec laptop | No local Qwen load required; pair tools are bounded to a 512-pixel long edge | Large originals are inspected but analysis uses a bounded aligned working grid |
| Public demonstration | Start the protected Kaggle tunnel, then run the local app | Re-check the printed URL after every restart; session/traffic limits apply |
| Production-like behavior | Strict validation, auth boundary, immutable hashes, timeouts, trace, reports | A genuinely public multi-user production service cannot promise permanent ₹0 compute |

## Definition of done for judging

Run all static/unit gates, then execute `scripts/sih-acceptance.py` with licensed co-registered
fixtures and `--auto-route`, without `--allow-simulated`. The generated directory must contain five JSON records, five marked images, five PDFs,
and a summary whose model provenance shows Qwen for single tasks and the two bounded pair tools for
paired tasks. Finally, attach the prescribed benchmark score files; do not replace them with manual
demo impressions.

## Pair-input restrictions

- Rotated/sheared/mirrored pairs must be externally co-registered onto a north-up grid first.
- Equal band count does not prove equal physical units or band order. Supply matched sensor products
  with consistent radiometric preprocessing; do not compare reflectance directly with digital numbers.
- Shared valid pixels exclude nodata and internal masks in every selected band. Empty or spatially
  uninformative pairs fail clearly rather than producing confident proxy evidence.
- The ungeoreferenced benchmark exception requires the same allowlisted dataset and pixel dimensions
  for both images. Declaring a benchmark never bypasses geospatial checks on georeferenced images.
