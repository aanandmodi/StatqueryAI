from __future__ import annotations

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api import router
from app.config import Settings, get_settings
from app.core.router import PolicyRouter
from app.core.validation import RasterValidator
from app.errors import SatQueryError
from app.models.gateway import SpecialistGateway, build_gateway
from app.repository import SQLiteRepository
from app.schemas import ErrorDetail, ErrorResponse
from app.services import AnalysisService
from app.storage import LocalArtifactStore, LocalAssetStore


@dataclass
class AppContainer:
    settings: Settings
    repository: SQLiteRepository
    asset_store: LocalAssetStore
    artifact_store: LocalArtifactStore
    validator: RasterValidator
    router: PolicyRouter
    gateway: SpecialistGateway
    analysis_service: AnalysisService
    version: str = __version__


def build_container(settings: Settings) -> AppContainer:
    settings.prepare_directories()
    settings.assert_safe_production_configuration()
    repository = SQLiteRepository(settings.database_path)
    asset_store = LocalAssetStore(settings.upload_dir, settings.max_upload_bytes)
    artifact_store = LocalArtifactStore(settings.artifact_dir)
    validator = RasterValidator(settings)
    policy_router = PolicyRouter(settings)
    gateway = build_gateway(settings, asset_store)
    service = AnalysisService(
        settings=settings,
        repository=repository,
        validator=validator,
        router=policy_router,
        gateway=gateway,
        artifacts=artifact_store,
        asset_store=asset_store,
    )
    return AppContainer(
        settings=settings,
        repository=repository,
        asset_store=asset_store,
        artifact_store=artifact_store,
        validator=validator,
        router=policy_router,
        gateway=gateway,
        analysis_service=service,
    )


def build_lifespan(settings: Settings):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.container = build_container(settings)
        await app.state.container.repository.initialize()
        await app.state.container.repository.mark_interrupted_jobs_failed()
        yield
        await app.state.container.analysis_service.shutdown()
        close_gateway = getattr(app.state.container.gateway, "close", None)
        if close_gateway is not None:
            await close_gateway()

    return lifespan


def configure_app(app: FastAPI, settings: Settings) -> FastAPI:
    """Attach SatQuery middleware, errors and routes to a FastAPI-compatible app."""

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key", "Idempotency-Key"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", f"req_{uuid4().hex}")[:128]
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Server-Timing"] = f"app;dur={(time.perf_counter() - started) * 1000:.1f}"
        return response

    @app.exception_handler(SatQueryError)
    async def satquery_error_handler(_: Request, exc: SatQueryError) -> JSONResponse:
        payload = ErrorResponse(
            error=ErrorDetail(code=exc.code, message=exc.message, details=exc.details)
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump(mode="json"))

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's wrapper does not accept Pydantic's include_* keyword arguments.
        # Omit raw input/context: they may contain secrets or non-JSON exceptions.
        issues = [
            {key: issue[key] for key in ("type", "loc", "msg") if key in issue}
            for issue in exc.errors()
        ]
        payload = ErrorResponse(
            error=ErrorDetail(
                code="request_validation_failed",
                message="Request does not match the API contract",
                details={"issues": issues},
            )
        )
        return JSONResponse(status_code=422, content=payload.model_dump(mode="json"))

    app.include_router(router, prefix=settings.api_prefix)
    return app


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Auditable remote-sensing analysis gateway with fail-closed raster validation, "
            "deterministic routing and specialist model isolation."
        ),
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
        lifespan=build_lifespan(settings),
    )
    return configure_app(app, settings)


app = create_app()
