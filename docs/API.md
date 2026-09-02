# Local API contract

Base URL: `http://127.0.0.1:8000/v1`. The frontend normally calls the same endpoints through
`http://localhost:3000/api/satquery/*` so the browser remains same-origin.

## Endpoints

| Method | Path | Purpose | Success |
|---|---|---|---|
| GET | `/health/live` | Process liveness | `200` |
| GET | `/health/ready` | SQLite and model-gateway readiness | `200`, possibly `status=degraded` |
| GET | `/capabilities` | Released tasks, limits, formats, model versions | `200` |
| POST | `/assets` | Stream and validate one raster | `201` |
| GET | `/assets/{asset_id}` | Read asset metadata | `200` |
| GET | `/assets/{asset_id}/preview` | Bounded RGB JPEG | `200 image/jpeg` |
| POST | `/analyses` | Create an asynchronous analysis | `202` |
| GET | `/analyses/{analysis_id}` | Poll status/result | `200` |
| GET | `/analyses/{analysis_id}/events` | Server-sent progress events | `200 text/event-stream` |
| POST | `/analyses/{analysis_id}/cancel` | Cancel a non-terminal job | `200` |
| GET | `/analyses/{analysis_id}/report` | Download audit PDF | `200 application/pdf` |
| GET | `/analyses/{analysis_id}/overlay` | Download marked JPEG | `200 image/jpeg` |

## Upload example

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
