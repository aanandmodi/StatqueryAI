# Product requirements document

## Product

SatQuery AI lets a user upload validated satellite imagery and ask a natural-language question.
It selects a permitted specialist, shows evidence and uncertainty, exposes the execution trace,
and produces a report suitable for a hackathon demonstration.

## Users

- Disaster-response analyst comparing affected regions.
- Agriculture/water/urban planning analyst exploring a scene.
- SIH judge verifying that routing, evidence and model provenance are real.

## Core jobs

1. Ask a question about one optical scene.
2. Request a factual caption.
3. Locate a visible feature and receive grounded geometry.
4. Compare two dates and identify what/where changed.
5. Fuse co-registered optical and SAR observations.
6. Inspect the task, model, parameters, timings, warnings and report.

## Functional requirements

- Accept GeoTIFF/TIFF; allow PNG/JPEG only when explicitly labelled benchmark imagery.
- Reject empty, oversized, corrupt, excessively large, or incompatible inputs.
- Route only among the five named task types.
- Never use the single-image model for change or raw SAR fusion.
- Preserve source asset identity and evidence coordinate space.
- Support asynchronous analysis status, cancellation, events, result retrieval and PDF report.
- Cache identical VLM requests to conserve free GPU quota.
- Pin model/data revisions and expose model versions in the trace.
- Keep all tokens and API keys out of the browser bundle and repository.

## UX requirements

- One clear question composer and upload workflow, not a generic admin dashboard.
- Calm geospatial visual language using frost, glass and topographic/cartographic details.
- Honest empty/loading/error/quota states.
- Never display a fake confidence percentage; distinguish calibrated, evidence-quality and
  uncalibrated scores in plain language.
- Evidence remains linked to the image; the trace is readable without exposing hidden reasoning.
- Keyboard operation, visible focus, reduced-motion support, descriptive labels and strong text
  contrast are required.

## Non-functional requirements

- Zero billed infrastructure in the demo profile.
- API tests cover upload → analysis → trace → report and the external Space queue contract.
- Transient remote calls have timeouts, bounded retries and explicit errors.
- Source uploads are immutable; generated artifacts are kept separate.
- The free public deployment is documented as quota-limited and ephemeral.

## Release acceptance

- Qwen adapter release gate passes fresh reload and checksum validation.
- Evaluation-only notebook writes task-specific metrics on a pinned split.
- Change artifact reports answer accuracy plus mask IoU/Dice using real SECOND labels.
- Fusion artifact reports fused, S2-only and S1-only macro-F1/AP ablations.
- Backend and Space helper tests pass; frontend build passes.
- End-to-end public smoke test succeeds without any paid hardware selection.
- Security scan finds no token in source, notebook output or frontend bundle.

## Out of scope for the zero-cost hackathon release

- Unlimited or SLA-backed traffic.
- Paid persistent Hugging Face storage.
- Autonomous use for emergency or legal decisions.
- Claims of accuracy on Indian Cartosat/RISAT imagery before an Indian-domain evaluation set exists.

