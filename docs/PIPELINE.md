# Model and inference pipelines

## Released Qwen pipeline

```mermaid
flowchart LR
    D[BigEarthNet.txt metadata and questions] --> P[Schema cleaning and task balancing]
    I[Matching Sentinel RGB previews] --> P
    P --> S[Leakage-safe train/validation/test split]
    S --> Q[Qwen3-VL 2B QLoRA fine-tuning]
    Q --> V[Validation and task metrics]
    V --> A[Adapter SafeTensors + manifests]
    A --> H[Public immutable Hub revision]
    H --> R[Fresh-session reload smoke test]
    R -->|PASS| Release[Released single-image specialist]
```

| Release item | Value |
|---|---|
| Base model | `Qwen/Qwen3-VL-2B-Instruct` |
| Base revision | `89644892e4d85e24eaac8bacfd4f463576704203` |
| Adapter | `aanandmodi/satquery-qwen3vl-bigearthnet-txt-lora` |
| Adapter revision | `ed12e59e0def9468bdf4a226789fc1b77c7900e7` |
| Technique | LoRA/QLoRA, rank 16, alpha 32 |
| Target modules | Q/K/V/O and gate/up/down projections |
| Artifact | `adapter_model.safetensors` plus processor/config/manifests |
| Release gate | Immutable revisions, hashes, evaluation summary, Hub upload, fresh reload |

The training notebook remains the evidence source for exact sample counts, epochs, optimizer,
metrics, and final PASS output. The runtime never silently substitutes another base or adapter.

## Inference dataflow

```mermaid
sequenceDiagram
    autonumber
    participant U as User/browser
    participant W as Web server
    participant A as FastAPI controller
    participant F as Local stores
    participant M as Hybrid specialists
    U->>W: GeoTIFF + question + lat/lon/altitude metadata
    W->>A: POST /v1/assets
    A->>F: Stream immutable upload + SHA-256
    A->>A: Inspect TIFF, CRS, transform, bands, pixels
    A-->>W: Asset record
    W->>A: POST /v1/analyses
    A->>A: Closed-set plan and compatibility checks
    A->>M: Bearer-authenticated contract + raster + user context
    M->>M: Qwen generation or bounded aligned pair analysis
    M-->>A: Text + facts + optional box + revision + warnings
    A->>A: Validate and integrate result
    A->>F: Persist result, trace, PDF, optional overlay
    W->>A: Poll GET /v1/analyses/{id}
    A-->>W: Completed structured result
    W-->>U: Text, image overlay, warning, trace, downloads
```

## Preprocessing contract

| Step | Rule | Why |
|---|---|---|
| Decode | Rasterio for TIFF; PIL fallback only for allowed benchmark imagery | Preserve geospatial validation |
| RGB bands | Descriptions `red/green/blue` or B04/B03/B02; 4/3/2 fallback for multispectral; grayscale repeat for fewer than 3 bands | Deterministic optical and SAR preview |
| Stretch | Per-band 2nd–98th percentile | Make reflectance imagery viewable without altering source |
| Resize | Long edge ≤ 448 for laptop VLM | Bound visual tokens and VRAM |
| Prompt | Task-specific instruction + user question | Separate caption/VQA/grounding output contracts |
| Decode | Greedy (`do_sample=False`) | Reproducible demo behavior |
| Output | Pydantic specialist schema | Reject malformed model-service responses |

The generated RGB image is an inference preview. The original upload, SHA-256, CRS, transform, and
metadata remain the provenance source.

## Grounding and overlay pipeline

```mermaid
flowchart TD
    Q[Grounding question] --> G[Constrained 0..1000 box prompt]
    G --> T[Generated text]
    T --> P{Valid JSON or box tag?}
    P -->|Yes| N[Normalize to 0..1]
    N --> C{Non-empty and in bounds?}
    C -->|Yes| E[Asset-linked EvidenceItem]
    E --> UI[CSS overlay + downloadable burned JPEG]
    P -->|No| W[Text answer + parsing warning]
    C -->|No| W
```

An absent or invalid box is never replaced with a decorative placeholder. The released adapter is
primarily trained for remote-sensing VQA; grounding quality must be judged independently.

## Confidence semantics

| Model output | API label | UI behavior |
|---|---|---|
| Qwen open-ended answer | `uncalibrated` / evidence quality | No correctness percentage; display warning |
| CPU pair analytical tools | `evidence_quality` | Candidate fractions/boxes plus explicit proxy warning |
| Future temperature-calibrated classifier | `calibrated_probability` | Percentage allowed with calibration version |
| Demo simulator | `simulated` | Visibly marked; forbidden in production mode |

Token likelihood is not treated as the probability that an answer is factually correct.

## Change pipeline: runnable baseline and learned upgrade

```mermaid
flowchart LR
    A[Time A] --> Align[Bounded common geospatial grid]
    B[Time B] --> Align
    Align --> Scale[Joint robust channel scaling]
    Scale --> Diff[Mean absolute spectral difference]
    Diff --> Threshold[Bounded/user-permitted threshold]
    Threshold --> Out[Changed fraction + direction + candidate box]
```

The runnable CPU tool is deterministic and makes no semantic-class claim. Its evidence-quality
score is capped below calibrated-confidence display. The learned upgrade path is:

```mermaid
flowchart LR
    A[Time A image] --> EA[Shared visual encoder]
    B[Time B image] --> EB[Shared visual encoder]
    EA --> X[Difference/product fusion]
    EB --> X
    Q[Question tokens] --> QE[Question encoder]
    X --> H[Multimodal fusion]
    QE --> H
    H --> Ans[Answer head]
    H --> Mask[Change-mask head]
    Ans --> Gate[Accuracy gate]
    Mask --> Gate2[IoU/Dice gate]
```

The learned artifact must not replace the baseline until CDVQA/SECOND imagery and labels are
attached, leakage-safe evaluation is run, and both answer and localization gates pass.

## Optical/SAR pipeline: runnable baseline and learned upgrade

The runnable baseline aligns SAR onto the optical grid, combines relative backscatter with optical
luminance/colour/edge cues, and returns explicitly labelled water and built-up **candidate proxies**.
It supports one- or two-band SAR and never passes raw SAR through Qwen. The learned upgrade path is:

```mermaid
flowchart LR
    S2[Sentinel-2 bands] --> O[Optical encoder]
    S1[Sentinel-1 VV/VH] --> R[SAR encoder]
    O --> F[TerraMind multimodal fusion]
    R --> F
    F --> Class[Multilabel class head]
    F --> Dense[Evidence head]
    Class --> Eval[Macro-F1 / AP]
    Dense --> Eval
    Eval --> Ablate[S2-only and S1-only ablations]
```

The TerraMind artifact may replace the baseline only after prescribed paired evaluation plus
fused/S2-only/S1-only ablations. Raw SAR is never passed to Qwen as if it were optical colour.

## Reproducibility and leakage controls

| Control | Enforcement |
|---|---|
| Immutable sources | Model and dataset revisions recorded before training/inference |
| Split identity | Patch/image IDs cannot cross train/validation/test |
| Test isolation | Test used once after selection; no checkpoint selection on test |
| Artifact integrity | SafeTensors and SHA-256 manifest |
| Fresh reload | Release recreated without the in-memory Trainer object |
| Runtime parity | Matching processor, chat template, base SHA, adapter SHA |
| Deterministic smoke | Fixed seeds where applicable and greedy generation |
