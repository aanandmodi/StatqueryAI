from __future__ import annotations

import asyncio
import io
import secrets
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, Header, Query, Request, UploadFile, status
from fastapi.responses import FileResponse, Response, StreamingResponse
from PIL import Image, ImageDraw, ImageFont

from app.config import Settings
from app.core.external_evidence import HistorySearch, WeatherSearch, search_history, search_weather
from app.errors import NotFoundError, ValidationFailure
from app.models.gateway import render_rgb_preview
from app.models.masks import evidence_color
from app.schemas import (
    AnalysisCreate,
    AnalysisRecord,
    AssetRecord,
    AssetRole,
    HealthResponse,
    InputProfile,
    Modality,
    RegistrationBasis,
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
            "strict_geospatial": ["GeoTIFF", "TIFF"],
            "exploration_optical": ["TIFF", "PNG", "JPEG", "WebP"],
            "exploration_sar_display": ["TIFF", "PNG", "JPEG", "WebP"],
        },
        "format_notes": {
            "exploration_sar_display": (
                "Requires an explicitly attested pixel-aligned optical/SAR pair; display "
                "intensities are not calibrated backscatter."
            )
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
    file: Annotated[
        UploadFile,
        File(description="TIFF/GeoTIFF or a bounded exploration PNG/JPEG/WebP image"),
    ],
    modality: Annotated[Modality, Form()],
    role: Annotated[AssetRole, Form()] = AssetRole.PRIMARY,
    registration_basis: Annotated[RegistrationBasis, Form()] = RegistrationBasis.GEOSPATIAL,
    input_profile: Annotated[InputProfile, Form()] = InputProfile.STRICT,
    source_dataset: Annotated[str | None, Form(max_length=80)] = None,
) -> AssetRecord:
    state = container(request)
    if source_dataset and state.settings.environment == "production":
        raise ValidationFailure(
            "Public benchmark-image import is disabled in production; use a server-side manifest"
        )
    if registration_basis == RegistrationBasis.PIXEL_GRID and role not in {
        AssetRole.TIME_A,
        AssetRole.TIME_B,
        AssetRole.OPTICAL,
        AssetRole.SAR,
    }:
        raise ValidationFailure(
            "Image-grid registration may only be declared for a temporal or optical/SAR pair"
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
            allow_image_grid=registration_basis == RegistrationBasis.PIXEL_GRID,
            exploration=input_profile == InputProfile.EXPLORATION,
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
        registration_basis=registration_basis,
        input_profile=input_profile,
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
    asset = await state.repository.get_asset(asset_id)
    if not asset.valid:
        raise ValidationFailure("Cannot preview an asset that failed validation")
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


def _fit_overlay_line(
    text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, max_width: int
) -> str:
    """Fit one display line by measured glyph width, not a character estimate."""
    if font.getlength(text) <= max_width:
        return text
    suffix = "..."
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if font.getlength(text[:middle] + suffix) <= max_width:
            low = middle
        else:
            high = middle - 1
    return text[:low].rstrip() + suffix


def _wrap_overlay_text(
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
    max_lines: int,
) -> tuple[list[str], bool]:
    """Wrap even unbroken tokens, with explicit truncation for bounded artifacts."""
    lines: list[str] = []
    current = ""
    for character in " ".join(text.split()):
        candidate = current + character
        if current and font.getlength(candidate) > max_width:
            # Prefer a word boundary, while still supporting long IDs/URLs.
            break_at = current.rfind(" ")
            if break_at > 0:
                lines.append(current[:break_at])
                current = current[break_at + 1 :] + character
            else:
                lines.append(current)
                current = character.lstrip()
            if len(lines) >= max_lines:
                lines[-1] = _fit_overlay_line(lines[-1] + "...", font, max_width)
                return lines, True
        else:
            current = candidate
    if current.strip():
        lines.append(current.strip())
    return lines, False


def _draw_overlay(
    preview: bytes,
    evidence: list[Any],
    *,
    context: Any | None = None,
    answer: str = "",
    masks: dict[str, bytes] | None = None,
) -> bytes:
    """Preserve the scene; add boxes and a separate readable annotation footer.

    Coordinates are applied to the proportional scene before it is letterboxed.
    Only the artifact canvas expands: no image pixels are cropped for labels.
    """

    with Image.open(io.BytesIO(preview)) as source:
        scene = source.convert("RGB")
    original_width, original_height = scene.size
    target_edge = min(1600, max(768, max(scene.size)))
    scale = target_edge / max(scene.size)
    scene_size = (max(1, round(original_width * scale)), max(1, round(original_height * scale)))
    if scene.size != scene_size:
        scene = scene.resize(scene_size, Image.Resampling.LANCZOS)
    canvas_width = max(640, scene.width)
    scene_left = (canvas_width - scene.width) // 2
    font_size = max(16, min(24, round(canvas_width / 44)))
    font = None
    for font_name in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
        try:
            font = ImageFont.truetype(font_name, font_size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default(size=font_size)
    margin = 20
    text_width = canvas_width - 2 * margin
    background = (9, 22, 30)
    foreground = (225, 247, 242)
    muted = (159, 190, 193)
    colors = [(109, 255, 197), (255, 211, 109), (133, 198, 255), (239, 158, 255)]
    footer: list[tuple[str, tuple[int, int, int]]] = []
    region_count = 0
    draw = ImageDraw.Draw(scene)
    line_width = max(2, round(min(scene.size) / 180))
    for item in evidence:
        if item.type == "mask" and masks and item.id in masks:
            with Image.open(io.BytesIO(masks[item.id])) as binary:
                alpha = binary.convert("L").resize(scene.size, Image.Resampling.NEAREST)
            # Zero-valued pixels, including holes, remain completely untouched.
            alpha = alpha.point(lambda value: 105 if value else 0)
            mask_hex = evidence_color(item)
            mask_color = tuple(bytes.fromhex(mask_hex.removeprefix("#")))
            tint = Image.new("RGB", scene.size, mask_color)
            scene.paste(tint, (0, 0), alpha)
            draw = ImageDraw.Draw(scene)
            region_count += 1
            footer.append(
                (
                    _fit_overlay_line(
                        f"MASK {region_count} · {item.label} (candidate)", font, text_width
                    ),
                    mask_color,
                )
            )
            continue
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
            min(scene.width - 1, round(x * scene.width)),
            min(scene.height - 1, round(y * scene.height)),
            min(scene.width - 1, round((x + width) * scene.width)),
            min(scene.height - 1, round((y + height) * scene.height)),
        )
        color = colors[region_count % len(colors)]
        draw.rectangle(box, outline=color, width=line_width)
        region_count += 1
        if region_count <= 8:
            label = " ".join(str(item.label).split())[:500] or "model evidence"
            footer.append(
                (_fit_overlay_line(f"REGION {region_count} · {label}", font, text_width), color)
            )
    if region_count > 8:
        footer.append(
            (f"{region_count - 8} more region labels are available in the audit report.", muted)
        )
    if region_count:
        footer.append(
            ("Region colors match the outlines; labels are not drawn over scene pixels.", muted)
        )
    if context is not None:
        altitude = f"{context.altitude_m:g} m" if context.altitude_m is not None else "not supplied"
        metadata = (
            "USER METADATA · "
            f"lat {context.latitude:.6f} · lon {context.longitude:.6f} · alt {altitude}"
        )
        metadata_lines, _ = _wrap_overlay_text(metadata, font, text_width, max_lines=3)
        footer.extend((line, muted) for line in metadata_lines)
    clean_answer = " ".join(answer.split())
    if clean_answer:
        answer_lines, shortened = _wrap_overlay_text(
            "MODEL OUTPUT · " + clean_answer, font, text_width, max_lines=12
        )
        footer.extend((line, foreground) for line in answer_lines)
        if shortened:
            footer.append(
                ("Text shortened here; download the audit report for the full output.", muted)
            )
    # Fixed legend/help lines are measured too; nothing may spill past the canvas.
    footer = [(_fit_overlay_line(line, font, text_width), color) for line, color in footer]
    line_height = font_size + 9
    footer_height = 2 * margin + line_height * len(footer) if footer else 0
    image = Image.new("RGB", (canvas_width, scene.height + footer_height), background)
    image.paste(scene, (scene_left, 0))
    draw = ImageDraw.Draw(image)
    for index, (line, color) in enumerate(footer):
        draw.text(
            (margin, scene.height + margin + index * line_height),
            line,
            fill=color,
            font=font,
            anchor="lt",
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


@protected.get("/analyses", tags=["analyses"])
async def list_analyses(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 30,
    offset: Annotated[int, Query(ge=0, le=10000)] = 0,
) -> dict:
    records = await container(request).repository.list_analyses(limit, offset)
    return {
        "items": [
            {
                "id": r.id,
                "status": r.status.value,
                "query": r.request.query,
                "created_at": r.created_at.isoformat(),
                "asset_count": len(r.request.asset_ids),
                "answer_excerpt": r.result.answer[:260] if r.result else None,
            }
            for r in records
        ]
    }


@protected.post("/evidence/history/search", tags=["context"])
async def history_search(payload: HistorySearch) -> dict:
    return await search_history(payload)


@protected.post("/evidence/weather/search", tags=["context"])
async def weather_search(payload: WeatherSearch) -> dict:
    return await search_weather(payload)


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
async def download_overlay(
    analysis_id: str,
    request: Request,
    asset_id: Annotated[str | None, Query(pattern=r"^ast_[a-f0-9]+$")] = None,
) -> Response:
    state = container(request)
    record = await state.repository.get_analysis(analysis_id)
    if record.result is None:
        raise NotFoundError("Marked image is not available for this analysis")
    if asset_id is not None and asset_id not in record.request.asset_ids:
        raise ValidationFailure("Overlay source asset does not belong to this analysis")
    assets = await state.repository.get_assets(record.request.asset_ids)
    if not assets:
        raise NotFoundError("Source asset is missing")
    if asset_id is None:
        asset_id = next(
            (
                item.asset_id
                for item in record.result.evidence
                if item.asset_id in record.request.asset_ids
            ),
            assets[0].id,
        )
    selected_evidence = [item for item in record.result.evidence if item.asset_id == asset_id]
    preview = await asyncio.to_thread(
        render_rgb_preview,
        state.asset_store.resolve(asset_id),
        max_edge=state.settings.space_preview_max_edge,
        jpeg_quality=state.settings.space_preview_jpeg_quality,
    )
    overlay = await asyncio.to_thread(
        _draw_overlay,
        preview,
        selected_evidence,
        context=record.request.context,
        answer=record.result.answer,
        masks={
            item.id: state.artifact_store.allocate(analysis_id, f"{item.id}.png").read_bytes()
            for item in selected_evidence
            if item.type == "mask" and item.geometry.get("encoding") == "binary-png-artifact"
        },
    )
    return Response(
        content=overlay,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "private, max-age=3600",
            "Content-Disposition": (
                f'attachment; filename="satquery-{analysis_id}-{asset_id}-overlay.jpg"'
            ),
        },
    )


@protected.get("/analyses/{analysis_id}/masks/{mask_id}", tags=["analyses"])
async def download_mask(
    analysis_id: str,
    mask_id: str,
    request: Request,
    colored: bool = False,
) -> Response:
    state = container(request)
    record = await state.repository.get_analysis(analysis_id)
    item = (
        next(
            (
                item
                for item in record.result.evidence
                if item.id == mask_id
                and item.type == "mask"
                and item.geometry.get("encoding") == "binary-png-artifact"
            ),
            None,
        )
        if record.result
        else None
    )
    if item is None:
        raise NotFoundError("Mask does not belong to this analysis")
    path = state.artifact_store.allocate(analysis_id, f"{item.id}.png")
    if not path.exists():
        raise NotFoundError("Mask artifact is missing")
    raw = path.read_bytes()
    if colored:
        with Image.open(io.BytesIO(raw)) as mask:
            color = tuple(bytes.fromhex(evidence_color(item).removeprefix("#")))
            image = Image.new("RGBA", mask.size, (*color, 0))
            image.putalpha(mask.convert("L").point(lambda value: 110 if value else 0))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        raw = stream.getvalue()
    return Response(
        content=raw, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"}
    )


router.include_router(protected)
