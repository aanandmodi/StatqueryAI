from __future__ import annotations

import asyncio
import secrets
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, Header, Request, UploadFile, status
from fastapi.responses import FileResponse, Response, StreamingResponse

from app.config import Settings
from app.errors import NotFoundError, ValidationFailure
from app.models.gateway import render_rgb_preview
from app.schemas import (
    AnalysisCreate,
    AnalysisRecord,
    AssetRecord,
    AssetRole,
    HealthResponse,
    Modality,
    utc_now,
)

router = APIRouter()


def container(request: Request) -> Any:
    return request.app.state.container


async def require_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    settings: Settings = request.app.state.container.settings
    if settings.api_key is None:
        return
    expected = settings.api_key.get_secret_value()
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        from fastapi import HTTPException

        raise HTTPException(status_code=401, detail="Missing or invalid API key")


protected = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/health/live", response_model=HealthResponse, tags=["health"])
async def liveness(request: Request) -> HealthResponse:
    state = container(request)
    return HealthResponse(
        status="ok",
        version=state.version,
        model_backend=state.settings.model_backend,
        checks={"process": True},
    )


@router.get("/health/ready", response_model=HealthResponse, tags=["health"])
async def readiness(request: Request) -> HealthResponse:
    state = container(request)
    db_ok = True
    try:
        await state.repository.initialize()
    except Exception:
        db_ok = False
    model_ok = await state.gateway.health()
    ready = db_ok and model_ok
    return HealthResponse(
        status="ok" if ready else "degraded",
        version=state.version,
        model_backend=state.settings.model_backend,
        checks={"database": db_ok, "model_gateway": model_ok},
    )


@protected.get("/capabilities", tags=["system"])
async def capabilities(request: Request) -> dict[str, Any]:
    state = container(request)
    return {
        "tasks": state.gateway.supported_tasks(),
        "formats": {
            "primary": ["GeoTIFF", "TIFF"],
            "benchmark_only": ["PNG", "JPEG"],
        },
        "limits": {
            "upload_bytes": state.settings.max_upload_bytes,
            "raster_pixels": state.settings.max_raster_pixels,
            "raster_bands": state.settings.max_raster_bands,
            "plan_steps": state.settings.max_plan_steps,
        },
        "model_backend": state.settings.model_backend,
        "model_versions": state.gateway.versions(),
        "trace_policy": "observable tool decisions only; no hidden chain-of-thought",
    }


@protected.post(
    "/assets", response_model=AssetRecord, status_code=status.HTTP_201_CREATED, tags=["assets"]
)
async def upload_asset(
    request: Request,
    file: Annotated[UploadFile, File(description="GeoTIFF/TIFF remote-sensing image")],
    modality: Annotated[Modality, Form()],
    role: Annotated[AssetRole, Form()] = AssetRole.PRIMARY,
    source_dataset: Annotated[str | None, Form(max_length=80)] = None,
) -> AssetRecord:
    state = container(request)
    if source_dataset and state.settings.environment == "production":
        raise ValidationFailure(
            "Public benchmark-image import is disabled in production; use a server-side manifest"
        )
    stored = await state.asset_store.save_upload(file)
    metadata = None
    errors: list[str] = []
    try:
        metadata = await asyncio.to_thread(
            state.validator.inspect,
            stored.path,
            modality=modality,
            source_dataset=source_dataset,
        )
    except ValidationFailure as exc:
        errors.append(exc.message)

    asset = AssetRecord(
        id=stored.asset_id,
        original_name=stored.original_name,
        content_type=stored.content_type,
        size_bytes=stored.size_bytes,
        sha256=stored.sha256,
        role=role,
        modality=modality,
        source_dataset=source_dataset,
        created_at=utc_now(),
        metadata=metadata,
        validation_errors=errors,
    )
    await state.repository.create_asset(asset)
    if errors:
        raise ValidationFailure(
            "Asset was stored but failed raster validation",
            details={"asset_id": asset.id, "errors": errors},
        )
    return asset


@protected.get("/assets/{asset_id}", response_model=AssetRecord, tags=["assets"])
async def get_asset(asset_id: str, request: Request) -> AssetRecord:
    state = container(request)
    return await state.repository.get_asset(asset_id)


@protected.get("/assets/{asset_id}/preview", tags=["assets"])
async def preview_asset(asset_id: str, request: Request) -> Response:
    state = container(request)
    await state.repository.get_asset(asset_id)
    preview = await asyncio.to_thread(
        render_rgb_preview,
        state.asset_store.resolve(asset_id),
        max_edge=state.settings.space_preview_max_edge,
        jpeg_quality=state.settings.space_preview_jpeg_quality,
    )
    return Response(
        content=preview,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@protected.post(
    "/analyses",
    response_model=AnalysisRecord,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["analyses"],
)
async def create_analysis(
    payload: AnalysisCreate,
    request: Request,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> AnalysisRecord:
    state = container(request)
    return await state.analysis_service.create(payload, idempotency_key=idempotency_key)


@protected.get("/analyses/{analysis_id}", response_model=AnalysisRecord, tags=["analyses"])
async def get_analysis(analysis_id: str, request: Request) -> AnalysisRecord:
    state = container(request)
    return await state.repository.get_analysis(analysis_id)


@protected.get("/analyses/{analysis_id}/events", tags=["analyses"])
async def analysis_events(analysis_id: str, request: Request) -> StreamingResponse:
    state = container(request)
    await state.repository.get_analysis(analysis_id)
    return StreamingResponse(
        state.analysis_service.events(analysis_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store",
            "X-Accel-Buffering": "no",
        },
    )


@protected.post("/analyses/{analysis_id}/cancel", response_model=AnalysisRecord, tags=["analyses"])
async def cancel_analysis(analysis_id: str, request: Request) -> AnalysisRecord:
    state = container(request)
    return await state.analysis_service.cancel(analysis_id)


@protected.get("/analyses/{analysis_id}/report", tags=["analyses"])
async def download_report(analysis_id: str, request: Request) -> FileResponse:
    state = container(request)
    record = await state.repository.get_analysis(analysis_id)
    if record.result is None or record.result.report_url is None:
        raise NotFoundError("Report is not available for this analysis")
    path = state.artifact_store.allocate(analysis_id, "satquery-report.pdf")
    if not path.exists():
        raise NotFoundError("Report artifact is missing")
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=f"satquery-{analysis_id}.pdf",
    )


router.include_router(protected)
