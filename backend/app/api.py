from __future__ import annotations

import asyncio
import io
import secrets
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, Header, Request, UploadFile, status
from fastapi.responses import FileResponse, Response, StreamingResponse
from PIL import Image, ImageDraw, ImageFont

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
        "pair_backend": state.settings.pair_backend,
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


def _draw_overlay(
    preview: bytes,
    evidence: list[Any],
    *,
    context: Any | None = None,
    answer: str = "",
) -> bytes:
    """Burn validated boxes plus labelled metadata/answer into a JPEG artifact."""

    with Image.open(io.BytesIO(preview)) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(2, round(min(image.size) / 180))
    for item in evidence:
        if item.type != "box" or item.coordinate_space != "normalized":
            continue
        geometry = item.geometry
        try:
            x = max(0.0, min(1.0, float(geometry["x"])))
            y = max(0.0, min(1.0, float(geometry["y"])))
            width = max(0.0, min(1.0 - x, float(geometry["width"])))
            height = max(0.0, min(1.0 - y, float(geometry["height"])))
        except (KeyError, TypeError, ValueError):
            continue
        if width <= 0 or height <= 0:
            continue
        box = (
            round(x * image.width),
            round(y * image.height),
            round((x + width) * image.width),
            round((y + height) * image.height),
        )
        color = (109, 255, 197)
        draw.rectangle(box, outline=color, width=line_width)
        label = str(item.label).strip()[:80] or "model evidence"
        text_box = draw.textbbox((box[0], box[1]), label)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_top = max(0, box[1] - text_height - 8)
        draw.rectangle(
            (box[0], label_top, min(image.width, box[0] + text_width + 10), box[1]),
            fill=(9, 22, 30),
        )
        draw.text((box[0] + 5, label_top + 3), label, fill=color)

    font_size = max(12, round(min(image.size) / 55))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    lines: list[str] = []
    if context is not None:
        altitude = (
            f"{context.altitude_m:g} m"
            if context.altitude_m is not None
            else "not supplied"
        )
        lines.append(
            "USER METADATA · "
            f"lat {context.latitude:.6f} · lon {context.longitude:.6f} · alt {altitude}"
        )
    clean_answer = " ".join(answer.split())
    if clean_answer:
        max_chars = max(36, round(image.width / max(font_size * 0.56, 1)))
        while clean_answer and len(lines) < 3:
            prefix = "MODEL · " if not any(line.startswith("MODEL · ") for line in lines) else ""
            lines.append(prefix + clean_answer[:max_chars])
            clean_answer = clean_answer[max_chars:]
    if lines:
        line_height = font_size + 6
        panel_height = line_height * len(lines) + 14
        top = max(0, image.height - panel_height)
        draw.rectangle((0, top, image.width, image.height), fill=(9, 22, 30))
        for index, line in enumerate(lines):
            draw.text(
                (12, top + 7 + index * line_height),
                line,
                fill=(225, 247, 242),
                font=font,
            )

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=92, optimize=True)
    return output.getvalue()


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


@protected.get("/analyses/{analysis_id}/overlay", tags=["analyses"])
async def download_overlay(analysis_id: str, request: Request) -> Response:
    state = container(request)
    record = await state.repository.get_analysis(analysis_id)
    if record.result is None:
        raise NotFoundError("Marked image is not available for this analysis")
    assets = await state.repository.get_assets(record.request.asset_ids)
    if not assets:
        raise NotFoundError("Source asset is missing")
    preview = await asyncio.to_thread(
        render_rgb_preview,
        state.asset_store.resolve(assets[0].id),
        max_edge=state.settings.space_preview_max_edge,
        jpeg_quality=state.settings.space_preview_jpeg_quality,
    )
    overlay = await asyncio.to_thread(
        _draw_overlay,
        preview,
        record.result.evidence,
        context=record.request.context,
        answer=record.result.answer,
    )
    return Response(
        content=overlay,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "private, max-age=3600",
            "Content-Disposition": f'attachment; filename="satquery-{analysis_id}-overlay.jpg"',
        },
    )


router.include_router(protected)
