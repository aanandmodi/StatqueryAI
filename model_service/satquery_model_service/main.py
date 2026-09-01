from __future__ import annotations

import asyncio
import json
import secrets
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile

from satquery_model_service import __version__
from satquery_model_service.adapters import Adapter, build_adapter
from satquery_model_service.config import ModelSettings
from satquery_model_service.contracts import InferencePayload, SpecialistResponse


def create_app(settings: ModelSettings | None = None) -> FastAPI:
    settings = settings or ModelSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.validate_startup()
        app.state.adapter = await asyncio.to_thread(build_adapter, settings)
        app.state.lock = asyncio.Lock()
        yield

    app = FastAPI(
        title="SatQuery Model Service",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    async def authorize(authorization: str | None) -> None:
        if settings.service_token is None:
            return
        expected = f"Bearer {settings.service_token.get_secret_value()}"
        if authorization is None or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="Invalid service credential")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/ready")
    async def ready(request: Request) -> dict[str, str]:
        adapter: Adapter = request.app.state.adapter
        return {
            "status": "ready",
            "capability": adapter.capability,
            "model_version": adapter.version,
        }

    @app.post("/v1/infer/{task}", response_model=SpecialistResponse)
    async def infer(
        task: str,
        request: Request,
        payload: Annotated[str, Form()],
        assets: Annotated[list[UploadFile], File()],
        authorization: Annotated[str | None, Header()] = None,
    ) -> SpecialistResponse:
        await authorize(authorization)
        try:
            contract = InferencePayload.model_validate(json.loads(payload))
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(
                status_code=422, detail="Invalid inference contract"
            ) from exc
        if task != contract.step.task:
            raise HTTPException(
                status_code=409, detail="Route task does not match payload task"
            )
        if len(assets) != len(contract.assets):
            raise HTTPException(
                status_code=422, detail="Asset count does not match payload"
            )

        with tempfile.TemporaryDirectory(prefix="satquery-model-") as temporary:
            root = Path(temporary).resolve()
            paths = []
            for index, upload in enumerate(assets):
                destination = root / f"asset-{index}.tif"
                total = 0
                with destination.open("xb") as handle:
                    while chunk := await upload.read(1024 * 1024):
                        total += len(chunk)
                        if total > settings.max_upload_bytes:
                            raise HTTPException(
                                status_code=413, detail="Inference asset is too large"
                            )
                        handle.write(chunk)
                await upload.close()
                paths.append(destination)
            adapter: Adapter = request.app.state.adapter
            async with request.app.state.lock:
                try:
                    return await asyncio.to_thread(adapter.infer, contract, paths)
                except ValueError as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()
