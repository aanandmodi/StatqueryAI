# Local API contract

## Report/mask extension

`AnalysisResult.sections` contains source-labelled report sections (`title`, `source`, `paragraphs`).
Binary mask evidence is stored locally with `type=mask`, `geometry.encoding=binary-png-artifact` and
an analysis-scoped `artifact_url`. `GET /analyses/{id}/masks/ev_mask_1` returns binary PNG;
`?colored=true` returns a transparent cyan PNG for the UI. Only mask IDs belonging to the selected
analysis are served. The source width/height, mask width/height, valid/selected pixel counts,
coverage, projected area and method are included in geometry. See [quality semantics](ANALYSIS_QUALITY.md).

Base URL: `http://127.0.0.1:8000/v1`. The frontend normally calls the same endpoints through
`http://localhost:3000/api/satquery/*` so the browser remains same-origin.

## Endpoints

Studio additions retain normal local API authentication:

| Method/path | Contract |
|---|---|
| GET `/analyses?limit=20&offset=0` | Bounded recent-first case summaries |
| POST `/evidence/history/search` | `bbox`, ISO acquisition/start/end dates, `max_cloud_cover`, `limit`, `footprint_confirmed:true` |
| POST `/evidence/weather/search` | `latitude`, `longitude`, ISO `start_date`, `end_date`; max31 days |

History field names are `acquisition_date`, `start_date`, `end_date`; results explicitly have
`comparison_ready:false`. Upload adds `input_profile=strict|exploration` and
`registration_basis=geospatial|pixel_grid`, defaults strict/geospatial. Pixel-grid declarations
are temporal-only and require compatibility checks. Sensor modalities still require TIFF.
Rejected stored assets cannot be previewed. Context providers receive coordinates/dates, not images.
See [Studio guide](STUDIO_GUIDE.md) for bounds and source semantics.

| Method | Path | Purpose | Success |
|---|---|---|---|
| GET | `/health/live` | Process liveness | `200` |
| GET | `/health/ready` | SQLite and model-gateway readiness | `200`, possibly `status=degraded` |
| GET | `/capabilities` | Available registered tasks, limits, formats, model/tool versions | `200` |
| POST | `/assets` | Stream and validate one raster | `201` |
| GET | `/assets/{asset_id}` | Read asset metadata | `200` |
| GET | `/assets/{asset_id}/preview` | Bounded RGB JPEG | `200 image/jpeg` |
| POST | `/analyses` | Create an asynchronous analysis | `202` |
| GET | `/analyses/{analysis_id}` | Poll status/result | `200` |
| GET | `/analyses/{analysis_id}/events` | Server-sent progress events | `200 text/event-stream` |
| POST | `/analyses/{analysis_id}/cancel` | Cancel a non-terminal job | `200` |
| GET | `/analyses/{analysis_id}/report` | Download audit PDF | `200 application/pdf` |
| GET | `/analyses/{analysis_id}/overlay?asset_id=ast_...` | Download marked JPEG for an input asset | `200 image/jpeg` |

Readiness consumers must inspect the JSON `status` and `checks`, not just HTTP 200. A model outage
returns `status=degraded`. Overlay `asset_id` is optional: by default the first evidence-associated
asset is used, or the first input when there is no geometry. A selector must belong to this
analysis; unrelated assets return 422. Only evidence linked to the selected asset is drawn.
The exported canvas may be enlarged/letterboxed and extended with a footer to keep labels readable;
all scene pixels remain present. Floating raster NaN/Infinity nodata sentinels are represented as
JSON `null` with an explicit metadata warning; inference masks still come from original TIFF bytes.

## Upload example (PowerShell 7+)

```powershell
$asset = Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/v1/assets `
  -Form @{
    file = Get-Item "C:\path\scene.tif"
    modality = "optical"
    role = "primary"
  }
$asset | ConvertTo-Json -Depth 8
```

Windows PowerShell 5.1 does not support `-Form`. Use the browser upload or the Python acceptance
runner instead; the remaining `Invoke-RestMethod` JSON examples work in both versions.

The returned asset includes `id`, `sha256`, byte size, declared modality/role, timestamps, raster
metadata, and validation errors. An invalid raster is retained as a failed asset record for audit
but returns a validation error and is not accepted into an analysis.

## Create and poll example

```powershell
$request = @{
  query = "Locate the largest visible built-up region."
  asset_ids = @($asset.id)
  requested_tasks = @("grounding")
  context = @{
    latitude = 28.6139
    longitude = 77.2090
    altitude_m = 216
    sensor = "Sentinel-2"
    source = "user"
    metadata = @{ mission = "SIH demo" }
  }
  parameters = @{}
} | ConvertTo-Json

$analysis = Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/v1/analyses `
  -ContentType "application/json" `
  -Headers @{ "Idempotency-Key" = [guid]::NewGuid().ToString() } `
  -Body $request

do {
  Start-Sleep -Seconds 2
  $analysis = Invoke-RestMethod "http://127.0.0.1:8000/v1/analyses/$($analysis.id)"
} while ($analysis.status -notin @("succeeded", "failed", "cancelled"))

$analysis | ConvertTo-Json -Depth 12
```

## Analysis result

| Field | Meaning |
|---|---|
| `answer` | Integrated human-readable model answer |
| `facts` | Small named structured facts returned by specialists |
| `evidence` | Asset-linked normalized boxes/masks/polygons |
| `confidence` | Score, level, calibration version, and exact semantics |
| `trace` | Observable validation/planning/model/integration events |
| `warnings` | Domain, calibration, parsing, or artifact caveats |
| `provenance` | Planner, backend, asset hashes, and model revisions |
| `report_url` | Controller route for the generated PDF |

## Model-service contract

The controller calls `POST {SATQUERY_MODEL_SERVICE_URL}/v1/infer/{task}` as multipart form data.
That URL may be loopback or the temporary ngrok HTTPS URL:

| Part | Type | Description |
|---|---|---|
| `payload` | JSON string | Plan step, query, asset metadata, and optional geospatial context |
| `assets` | One or more files | Validated source rasters selected by the plan |

The model response must contain task, non-empty text, facts, evidence, raw score, score kind, model
version, and warnings. A route task mismatch is `409`; an invalid contract is `422`.

## Authentication

Local development leaves `SATQUERY_API_KEY` empty and binds services to loopback. A future network
deployment must set the key; protected controller routes then require `X-API-Key`. The model service
supports a separate bearer `SATQUERY_MODEL_SERVICE_TOKEN`. Neither value may enter React client
code.

## Error envelope

```json
{
  "error": {
    "code": "validation_failed",
    "message": "Human-readable safe message",
    "details": {}
  }
}
```

| Status | Typical code | Meaning |
|---:|---|---|
| 401 | framework auth detail | Missing/invalid API key |
| 404 | `not_found` | Asset, analysis, or artifact absent |
| 409 | `conflict` | Idempotency key reused with a different request |
| 413 | upload limit | Body exceeds the streaming limit |
| 422 | `validation_failed` / `routing_failed` | Input or task compatibility rejected |
| 503 | `model_unavailable` | Model process/temporary GPU unavailable |

Every controller response receives an `X-Request-ID`; use it with the analysis ID when reporting a
problem.
