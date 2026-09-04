# System design

Studio additions are documented in [Architecture](ARCHITECTURE.md) and [Studio guide](STUDIO_GUIDE.md).
Cases remain local SQLite records. External context queries have fixed hosts, bounded response
sizes, timeouts, concurrency and cache limits; they neither follow arbitrary URLs nor download
imagery. Input profiles/registration provenance are stored and cannot be replaced by camera GPS.
Model-derived mask counts are recomputed against the source grid. No paid infrastructure is used.

## Design goals

| Goal | Mechanism | Failure behavior |
|---|---|---|
| Evidence-backed answers | Specialist response carries asset-linked geometry | No geometry is shown when parsing/validation fails |
| Auditable execution | Closed task plan, model revisions, durations, warnings, provenance | Observable trace remains even when a job fails |
| Local-first control plane | Loopback frontend/controller, SQLite, local artifacts | GPU tunnel outage is explicit |
| Model isolation | Controller calls a typed specialist HTTP boundary | A model crash does not expose arbitrary execution to the browser |
| Reproducibility | Base/adapter commits, SafeTensors, manifests, pinned runtime | Startup rejects mutable/malformed revisions |
| Zero billed infrastructure | Laptop GPU or temporary free notebook GPU | Quota/OOM is explicit; no paid fallback is provisioned |

## Component view

```mermaid
C4Context
    title SatQuery local system context
    Person(user, "Analyst or SIH judge", "Uploads imagery and inspects evidence")
    System_Boundary(local, "Local computer") {
        System(web, "Evidence workspace", "Vinext/React")
        System(api, "SatQuery controller", "FastAPI")
        System(model, "Specialist service", "PyTorch/PEFT")
        SystemDb(db, "Job store", "SQLite")
        SystemDb(files, "Asset/artifact store", "Local filesystem")
    }
    System_Ext(hub, "Hugging Face Hub", "Public immutable base + LoRA downloads")
    System_Ext(notebook, "Free GPU notebook", "Temporary protected FastAPI model service")
    System_Ext(tunnel, "ngrok", "Ephemeral HTTPS reverse proxy")
    Rel(user, web, "Uses", "HTTP localhost")
    Rel(web, api, "Uploads and polls", "Same-origin proxy")
    Rel(api, model, "Typed inference contract", "HTTP localhost")
    Rel(api, db, "Persists status/result")
    Rel(api, files, "Stores immutable uploads and artifacts")
    Rel(model, hub, "Downloads once", "HTTPS")
    Rel(api, tunnel, "Bearer-authenticated multipart call", "HTTPS")
    Rel(tunnel, notebook, "Forwards to port 8080")
```

## Trust boundaries

```mermaid
flowchart TB
    subgraph Untrusted[Untrusted input zone]
        Browser
        Upload[Uploaded TIFF]
        Prompt[User question]
        Context[User lat/lon/altitude metadata]
    end
    subgraph Controller[Controller trust boundary]
        Limit[Size and schema limits]
        Raster[Raster validation]
        Route[Closed-set router]
        Integrate[Response validation and integration]
    end
    subgraph Model[Model boundary]
        Preview[Bounded RGB preview]
        VLM[Qwen3-VL + LoRA]
    end
    Browser --> Limit
    Upload --> Limit --> Raster --> Route
    Prompt --> Limit --> Route
    Context --> Limit --> Route
    Route --> Preview --> VLM --> Integrate --> Browser
```

User text never becomes a URL, file path, model identifier, shell command, or arbitrary tool name.
The router can select only `single_vqa`, `caption`, `grounding`, `change_vqa`, or
`optical_sar_fusion`, and compatibility checks reject tasks without the required assets.

## Runtime processes

| Process | Port | Startup readiness | Concurrency | Persistent state |
|---|---:|---|---:|---|
| Frontend | 3000 | Route compiles and responds | Framework-managed | None |
| Controller | 8000 | SQLite + model gateway healthy | Two jobs max by default | SQLite, uploads, reports |
| VLM service | 8080 | Base and LoRA loaded, CUDA ready | One request lock | HF cache only |
| ngrok agent | ephemeral HTTPS | Tunnel URL responds | Free-plan limits | None |

The controller uses asynchronous jobs so upload/API polling remains responsive while generation is
blocking. The model service serializes GPU work to prevent VRAM overcommit.

## State model

```mermaid
stateDiagram-v2
    [*] --> queued
    queued --> validating
    validating --> planning
    planning --> running
    running --> integrating
    integrating --> succeeded
    queued --> cancelled
    validating --> failed
    planning --> failed
    running --> failed
    integrating --> failed
    running --> cancelled
    succeeded --> [*]
    failed --> [*]
    cancelled --> [*]
```

Each transition is persisted. On controller restart, unfinished in-process jobs are marked failed
instead of being presented as still running.

## Data model

| Entity | Key fields | Invariant |
|---|---|---|
| Asset | ID, original name, SHA-256, modality, role, raster metadata | Stored name is random; source bytes are immutable |
| Analysis | ID, request, location context, state, progress, plan, result/error | Terminal states do not resume silently |
| Plan step | Task enum, asset IDs, permitted parameters, policy reason | No arbitrary model/tool execution |
| Specialist output | Text, facts, evidence, score semantics, revision, warnings | Pydantic schema validation before integration |
| Evidence | Type, label, score, coordinate space, geometry, asset ID | Coordinates must remain within the declared space |
| Trace event | Step, tool, revision, duration, status, reason | Records observable operations, not hidden reasoning |

## Model loading design

```mermaid
sequenceDiagram
    participant Runner
    participant ModelService
    participant Hub
    participant GPU
    Runner->>ModelService: start with immutable revisions
    ModelService->>Hub: load processor from adapter revision
    ModelService->>Hub: stream base weights at pinned SHA
    ModelService->>GPU: construct NF4 model (FP16 compute)
    ModelService->>Hub: load LoRA at pinned SHA
    ModelService->>GPU: attach adapter and eval()
    ModelService-->>Runner: /ready = model version
```

Weights are not merged. Keeping the adapter separate preserves provenance and avoids writing a
large merged checkpoint. No token is required because both repositories are public.

## Failure policy

| Failure | Surface | System response |
|---|---|---|
| Invalid/corrupt raster | Upload | `422 validation_failed`; no model call |
| Unsupported asset/task combination | Planning | `422 routing_failed` |
| Model offline/OOM | Running | Analysis becomes `failed`; API stays alive |
| Model timeout | Running | Bounded timeout; no fabricated answer |
| Unparseable box | Integration | Text retained, warning added, evidence omitted |
| Report generation error | Artifact stage | Valid answer retained with a warning |
| Controller restart | Job store | Interrupted jobs marked failed |

## Scale and future deployment

The current local profile is intended for one analyst and one GPU. Scaling later should replace
only infrastructure adapters: SQLite → PostgreSQL, local files → object storage, in-process jobs →
durable queue, and local model URL → authenticated GPU service. Browser and specialist schemas do
not need to change. No such deployment is authorized or performed in the current phase.
