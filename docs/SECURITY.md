# Security and data handling

## Secrets

- `HF_TOKEN` is used only for deployment/upload and optional authenticated Space quota.
- `SATQUERY_API_KEY` is shared only between the server-side frontend proxy and API Space.
- Neither value may enter React client code, notebook source/output, Git history or screenshots.
- Production startup fails when the API key is absent or the demo gateway is selected.

## Input controls

- Streamed upload size cap and randomized immutable storage name.
- Filename sanitization and path containment checks.
- Raster driver/band/pixel/CRS/transform validation.
- Pair co-registration checks before change/fusion routing.
- Closed task enum, bounded plan, bounded prompt and generation lengths.
- Base64 image byte and megapixel caps at the model boundary.

## Output controls

- Pydantic rejects unexpected response fields and invalid evidence geometry.
- Uncalibrated scores are capped by integration policy and accompanied by warnings.
- Trace exposes observable actions and versions, not hidden chain-of-thought.
- Reports are generated into a separate artifact directory.

## Free public deployment limitations

The CPU Space uses ephemeral local storage. Do not upload imagery that is confidential, regulated,
or unsafe to process on public third-party infrastructure. A production deployment would require
private networking, authenticated object storage, retention/deletion controls, malware scanning,
rate limiting, audit export and an organizational privacy assessment.

## Reporting a problem

Do not include access tokens or private imagery in an issue. Record the request ID, analysis ID,
error code, model revision and sanitized trace.

