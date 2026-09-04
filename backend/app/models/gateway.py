from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import httpx
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import ColorInterp, Resampling

from app.config import Settings
from app.errors import ModelUnavailableError
from app.models.pair_tools import PAIR_TASKS, LocalPairSpecialistGateway
from app.schemas import (
    AssetRecord,
    EvidenceItem,
    GeospatialContext,
    PlannedStep,
    SpecialistOutput,
    TaskType,
)
from app.storage import LocalAssetStore


class SpecialistGateway(Protocol):
    async def infer(
        self,
        step: PlannedStep,
        assets: list[AssetRecord],
        query: str,
        context: GeospatialContext | None = None,
    ) -> SpecialistOutput: ...

    async def health(self) -> bool: ...

    def versions(self) -> dict[str, str]: ...

    def supported_tasks(self) -> list[str]: ...


class DemoSpecialistGateway:
    """Deterministic simulator for API/UI integration without GPU weights.

    This output is intentionally low-confidence and visibly marked as simulated.
    It must never be enabled when ``environment=production``.
    """

    VERSION = "demo-simulator-v1"

    async def infer(
        self,
        step: PlannedStep,
        assets: list[AssetRecord],
        query: str,
        context: GeospatialContext | None = None,
    ) -> SpecialistOutput:
        digest = hashlib.sha256(
            f"{step.task.value}:{query}:{':'.join(a.sha256 for a in assets)}".encode()
        ).digest()
        score = 0.48 + (digest[0] / 255) * 0.12
        targets = step.permitted_params.get("targets", ["requested region"])
        target = str(targets[0])

        answers = {
            TaskType.SINGLE_VQA: (
                "Demo pipeline completed the single-image VQA route. Connect the exported "
                "Qwen3-VL adapter to produce an evidence-backed scene answer."
            ),
            TaskType.CAPTION: (
                "Demo pipeline identified the captioning route. The production VLM will return "
                "a remote-sensing scene description after its adapter is mounted."
            ),
            TaskType.GROUNDING: (
                f"Demo grounding output marks a placeholder region for {target}; this geometry "
                "is not a model prediction."
            ),
            TaskType.CHANGE_VQA: (
                "Demo change route completed. The production change expert will quantify the "
                "direction, area and location of change from the co-registered pair."
            ),
            TaskType.OPTICAL_SAR_FUSION: (
                "Demo fusion route completed. The production TerraMind expert will combine "
                "optical reflectance with SAR backscatter and return dense evidence."
            ),
        }
        x = 0.08 + (digest[1] / 255) * 0.35
        y = 0.08 + (digest[2] / 255) * 0.35
        width = 0.22 + (digest[3] / 255) * 0.18
        height = 0.18 + (digest[4] / 255) * 0.22
        evidence = EvidenceItem(
            id=f"ev_{uuid4().hex}",
            type="box",
            label=f"demo:{target}",
            score=score,
            coordinate_space="normalized",
            geometry={"x": x, "y": y, "width": width, "height": height},
            asset_id=assets[0].id,
        )
        return SpecialistOutput(
            task=step.task,
            text=answers[step.task],
            facts=[
                {"name": "execution_mode", "value": "simulated"},
                *(
                    [
                        {
                            "name": "user_location",
                            "value": {
                                "latitude": context.latitude,
                                "longitude": context.longitude,
                                "altitude_m": context.altitude_m,
                            },
                        }
                    ]
                    if context
                    else []
                ),
            ],
            evidence=[evidence],
            raw_score=score,
            score_kind="uncalibrated",
            model_version=self.VERSION,
            warnings=[
                "SIMULATED OUTPUT: no neural model ran; use "
                "SATQUERY_MODEL_BACKEND=http for real inference."
            ],
        )

    async def health(self) -> bool:
        return True

    def versions(self) -> dict[str, str]:
        return {task.value: self.VERSION for task in TaskType}

    def supported_tasks(self) -> list[str]:
        return [task.value for task in TaskType]


SINGLE_IMAGE_TASKS = {
    TaskType.SINGLE_VQA,
    TaskType.CAPTION,
    TaskType.GROUNDING,
}


class HttpSpecialistGateway:
    """Bounded HTTP bridge to the long-lived GPU inference service."""

    def __init__(
        self,
        settings: Settings,
        asset_store: LocalAssetStore,
        *,
        allowed_tasks: set[TaskType] | None = None,
    ) -> None:
        self.settings = settings
        self.asset_store = asset_store
        self.allowed_tasks = allowed_tasks or SINGLE_IMAGE_TASKS
        token = (
            settings.model_service_token.get_secret_value()
            if settings.model_service_token
            else None
        )
        headers = {"ngrok-skip-browser-warning": "1"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.model_timeout_seconds, connect=10.0),
            headers=headers,
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
        )
        self._versions: dict[str, str] = {}

    async def infer(
        self,
        step: PlannedStep,
        assets: list[AssetRecord],
        query: str,
        context: GeospatialContext | None = None,
    ) -> SpecialistOutput:
        if step.task not in self.allowed_tasks:
            raise ModelUnavailableError(
                "The remote specialist gateway cannot execute this task",
                details={"task": step.task.value},
            )
        handles = []
        try:
            files = []
            for asset in assets:
                path = self.asset_store.resolve(asset.id)
                if asset.metadata and asset.metadata.driver == "WEBP":
                    # Lossless transport derivative; keep the original hash/grid in provenance.
                    # Kaggle need not have GDAL's optional WebP driver installed.
                    with Image.open(path) as image:
                        handle = io.BytesIO()
                        image.save(handle, format="PNG")
                    handle.seek(0)
                    handles.append(handle)
                    files.append(("assets", ("transport.png", handle, "image/png")))
                    continue
                handle = path.open("rb")
                handles.append(handle)
                files.append(
                    (
                        "assets",
                        (asset.original_name, handle, asset.content_type),
                    )
                )
            payload = {
                "step": step.model_dump(mode="json"),
                "query": query,
                # Keep the existing running Kaggle contract compatible. These controller-only
                # fields remain in local provenance, not in the v1 remote Asset schema.
                "assets": [
                    asset.model_dump(mode="json", exclude={"input_profile", "registration_basis"})
                    for asset in assets
                ],
                "context": context.model_dump(mode="json") if context else None,
            }
            response = await self.client.post(
                f"{self.settings.service_url_for_task(step.task.value).rstrip('/')}/v1/infer/{step.task.value}",
                data={"payload": json.dumps(payload)},
                files=files,
            )
            response.raise_for_status()
            output = SpecialistOutput.model_validate(response.json())
            self._versions[step.task.value] = output.model_version
            return output
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelUnavailableError(
                "Specialist model service failed",
                details={"task": step.task.value, "reason": str(exc)},
            ) from exc
        finally:
            for handle in handles:
                handle.close()

    async def health(self) -> bool:
        tasks_by_url: dict[str, set[TaskType]] = {}
        for task in self.allowed_tasks:
            url = self.settings.service_url_for_task(task.value).rstrip("/")
            tasks_by_url.setdefault(url, set()).add(task)
        capability_tasks = {
            "vlm": SINGLE_IMAGE_TASKS,
            "change": {TaskType.CHANGE_VQA},
            "fusion": {TaskType.OPTICAL_SAR_FUSION},
        }
        for url, required_tasks in tasks_by_url.items():
            try:
                response = await self.client.get(f"{url}/ready", timeout=5.0)
                if response.status_code != 200:
                    return False
                payload = response.json()
                if not isinstance(payload, dict) or payload.get("status") not in ("ready", "ok"):
                    return False
                if "checks" in payload:
                    checks = payload["checks"]
                    if not isinstance(checks, dict) or not all(
                        value is True for value in checks.values()
                    ):
                        return False
                # A healthy VLM URL cannot stand in for a missing change/fusion
                # service. Both supported /ready implementations declare their
                # capability; older generic readiness contracts may omit it.
                if "capability" in payload:
                    capability = payload["capability"]
                    if not isinstance(capability, str) or not required_tasks.issubset(
                        capability_tasks.get(capability, set())
                    ):
                        return False
            except (httpx.HTTPError, ValueError):
                return False
        return True

    def versions(self) -> dict[str, str]:
        return dict(self._versions)

    def supported_tasks(self) -> list[str]:
        return sorted(task.value for task in self.allowed_tasks)

    async def close(self) -> None:
        await self.client.aclose()


def _scale_band(band: np.ndarray) -> np.ndarray:
    finite = np.isfinite(band)
    if not finite.any():
        return np.zeros(band.shape, dtype=np.uint8)
    values = band[finite]
    low, high = np.percentile(values, [2.0, 98.0])
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros(band.shape, dtype=np.uint8)
    scaled = np.clip((band - low) / (high - low), 0.0, 1.0)
    scaled[~finite] = 0.0
    return (scaled * 255.0).round().astype(np.uint8)


def _rgb_indexes(dataset: rasterio.io.DatasetReader) -> list[int]:
    interpretations = list(dataset.colorinterp)
    colors = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
    if all(color in interpretations for color in colors):
        return [interpretations.index(color) + 1 for color in colors]
    descriptions = [str(item or "").lower().strip() for item in dataset.descriptions]
    aliases = (
        ("red", "b04", "b4"),
        ("green", "b03", "b3"),
        ("blue", "b02", "b2"),
    )
    indexes: list[int] = []
    for names in aliases:
        match = next(
            (
                index
                for index, description in enumerate(descriptions, start=1)
                if description in names
            ),
            None,
        )
        if match is None:
            indexes = []
            break
        indexes.append(match)
    if indexes:
        return indexes
    if dataset.count >= 3:
        return [1, 2, 3]
    return [1, 1, 1]


def render_rgb_preview(path: Path, *, max_edge: int, jpeg_quality: int) -> bytes:
    """Convert an accepted raster into a bounded RGB JPEG for the VLM.

    Match the Kaggle decoder's band and radiometric policy: declared RGB first,
    named RGB bands second, otherwise first-three bands (or grayscale). Preserve
    uint8 imagery; independently stretch other types over valid 2nd-98th
    percentiles. This preview never replaces the immutable original raster.
    """

    try:
        with rasterio.open(path) as dataset:
            scale = min(1.0, max_edge / max(dataset.width, dataset.height))
            width = max(1, round(dataset.width * scale))
            height = max(1, round(dataset.height * scale))
            indexes = _rgb_indexes(dataset)
            data = dataset.read(
                indexes,
                out_shape=(3, height, width),
                resampling=Resampling.bilinear,
                masked=True,
            )
            raster = np.ma.filled(data.astype(np.float32), np.nan)
            if all(dataset.dtypes[index - 1] == "uint8" for index in indexes):
                rgb = np.moveaxis(np.nan_to_num(raster, nan=0.0), 0, -1)
                rgb = np.clip(rgb, 0, 255).round().astype(np.uint8)
            else:
                rgb = np.stack([_scale_band(raster[index]) for index in range(3)], axis=-1)
            image = Image.fromarray(rgb, mode="RGB")
    except rasterio.errors.RasterioIOError:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=jpeg_quality, optimize=True)
    return output.getvalue()


def _parse_gradio_sse(body: str) -> object:
    event = ""
    for line in body.splitlines():
        if line.startswith("event:"):
            event = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            raw = line.removeprefix("data:").strip()
            if event == "error":
                raise ValueError(f"ZeroGPU Space returned an error: {raw[:500]}")
            if event in {"complete", "completed"}:
                return json.loads(raw)
    raise ValueError("ZeroGPU Space did not return a complete queue event")


def _unwrap_gradio_result(value: object) -> dict[str, object]:
    payload = value
    if isinstance(payload, list) and len(payload) == 1:
        payload = payload[0]
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("ZeroGPU Space returned an unexpected response shape")
    return payload


class SpaceSpecialistGateway:
    """Queue-aware bridge to a free public Hugging Face ZeroGPU Gradio Space."""

    def __init__(self, settings: Settings, asset_store: LocalAssetStore) -> None:
        if not settings.space_url:
            raise RuntimeError("SATQUERY_SPACE_URL is required for the Space gateway")
        self.settings = settings
        self.asset_store = asset_store
        self.base_url = settings.space_url.rstrip("/")
        token = settings.space_token.get_secret_value() if settings.space_token else None
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.model_timeout_seconds, connect=15.0),
            headers={"Authorization": f"Bearer {token}"} if token else {},
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )
        self._versions: dict[str, str] = {}
        self._cache: OrderedDict[str, tuple[float, SpecialistOutput]] = OrderedDict()
        self._cache_lock = asyncio.Lock()

    def _cache_key(self, step: PlannedStep, assets: list[AssetRecord], query: str) -> str:
        canonical = json.dumps(
            {
                "task": step.task.value,
                "query": query,
                "assets": [asset.sha256 for asset in assets],
                "params": step.permitted_params,
                "max_new_tokens": self.settings.space_max_new_tokens,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    async def _cache_get(self, key: str) -> SpecialistOutput | None:
        async with self._cache_lock:
            row = self._cache.get(key)
            if row is None:
                return None
            expires_at, output = row
            if expires_at <= time.monotonic():
                self._cache.pop(key, None)
                return None
            self._cache.move_to_end(key)
            return output.model_copy(deep=True)

    async def _cache_put(self, key: str, output: SpecialistOutput) -> None:
        async with self._cache_lock:
            self._cache[key] = (
                time.monotonic() + self.settings.space_cache_ttl_seconds,
                output.model_copy(deep=True),
            )
            self._cache.move_to_end(key)
            while len(self._cache) > self.settings.space_cache_max_entries:
                self._cache.popitem(last=False)

    async def _request(self, method: str, url: str, **kwargs: object) -> httpx.Response:
        attempts = max(1, self.settings.space_retry_attempts)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = await self.client.request(method, url, **kwargs)
                if response.status_code not in {429, 502, 503, 504}:
                    response.raise_for_status()
                    return response
                last_error = httpx.HTTPStatusError(
                    f"transient Space response {response.status_code}",
                    request=response.request,
                    response=response,
                )
            except httpx.HTTPError as exc:
                last_error = exc
            if attempt + 1 < attempts:
                await asyncio.sleep(min(2**attempt, 8))
        assert last_error is not None
        raise last_error

    async def infer(
        self,
        step: PlannedStep,
        assets: list[AssetRecord],
        query: str,
        context: GeospatialContext | None = None,
    ) -> SpecialistOutput:
        if step.task not in SINGLE_IMAGE_TASKS:
            raise ModelUnavailableError(
                "This ZeroGPU deployment does not contain the requested specialist",
                details={
                    "task": step.task.value,
                    "available": sorted(task.value for task in SINGLE_IMAGE_TASKS),
                    "next_step": "train and deploy the separate change/fusion specialist",
                },
            )
        if len(assets) != 1:
            raise ModelUnavailableError(
                "The Qwen3-VL specialist accepts exactly one validated image",
                details={"task": step.task.value, "assets": len(assets)},
            )

        context_suffix = context.model_dump_json() if context else ""
        cache_key = self._cache_key(step, assets, f"{query}\n{context_suffix}")
        cached = await self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            preview = await asyncio.to_thread(
                render_rgb_preview,
                self.asset_store.resolve(assets[0].id),
                max_edge=self.settings.space_preview_max_edge,
                jpeg_quality=self.settings.space_preview_jpeg_quality,
            )
            submit_url = (
                f"{self.base_url}/gradio_api/call/{self.settings.space_api_name.lstrip('/')}"
            )
            submit = await self._request(
                "POST",
                submit_url,
                json={
                    "data": [
                        base64.b64encode(preview).decode("ascii"),
                        step.task.value,
                        (
                            f"{query}\nUser-supplied context (not pixel-derived): "
                            f"{context.model_dump_json()}"
                            if context
                            else query
                        ),
                        self.settings.space_max_new_tokens,
                    ]
                },
            )
            event_id = str(submit.json().get("event_id", "")).strip()
            if not event_id:
                raise ValueError("ZeroGPU Space did not return an event_id")
            completed = await self._request("GET", f"{submit_url}/{event_id}")
            payload = _unwrap_gradio_result(_parse_gradio_sse(completed.text))

            evidence = []
            for index, item in enumerate(payload.get("evidence", [])):
                if not isinstance(item, dict):
                    continue
                evidence.append(
                    EvidenceItem(
                        id=f"ev_{uuid4().hex}",
                        type=str(item.get("type", "box")),
                        label=str(item.get("label", f"grounded region {index + 1}")),
                        score=float(item.get("score", 0.5)),
                        coordinate_space=str(item.get("coordinate_space", "normalized")),
                        geometry=dict(item.get("geometry", {})),
                        asset_id=assets[0].id,
                    )
                )
            output = SpecialistOutput(
                task=step.task,
                text=str(payload["text"]).strip(),
                facts=list(payload.get("facts", [])),
                evidence=evidence,
                raw_score=float(payload.get("raw_score", 0.5)),
                score_kind=str(payload.get("score_kind", "uncalibrated")),
                model_version=str(payload.get("model_version", "satquery-space:unknown")),
                warnings=[str(item) for item in payload.get("warnings", [])],
            )
            self._versions[step.task.value] = output.model_version
            await self._cache_put(cache_key, output)
            return output
        except (httpx.HTTPError, OSError, ValueError, KeyError) as exc:
            raise ModelUnavailableError(
                "Free ZeroGPU inference is unavailable or its quota is exhausted",
                details={
                    "task": step.task.value,
                    "reason": str(exc),
                    "retry": (
                        "ZeroGPU has no uptime SLA; retry after the queue or daily quota resets."
                    ),
                },
            ) from exc

    async def health(self) -> bool:
        try:
            response = await self.client.get(f"{self.base_url}/gradio_api/info", timeout=10.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def versions(self) -> dict[str, str]:
        return dict(self._versions)

    def supported_tasks(self) -> list[str]:
        return sorted(task.value for task in SINGLE_IMAGE_TASKS)

    async def close(self) -> None:
        await self.client.aclose()


class HybridSpecialistGateway:
    """Route released single-image inference and local paired tools together."""

    def __init__(
        self,
        single_gateway: SpecialistGateway,
        pair_gateway: SpecialistGateway,
    ) -> None:
        self.single_gateway = single_gateway
        self.pair_gateway = pair_gateway

    async def infer(
        self,
        step: PlannedStep,
        assets: list[AssetRecord],
        query: str,
        context: GeospatialContext | None = None,
    ) -> SpecialistOutput:
        gateway = self.pair_gateway if step.task in PAIR_TASKS else self.single_gateway
        return await gateway.infer(step, assets, query, context)

    async def health(self) -> bool:
        single_ok, pair_ok = await asyncio.gather(
            self.single_gateway.health(), self.pair_gateway.health(), return_exceptions=True
        )
        # Local CPU tools being available does not make the remote VLM ready.
        # An exception or malformed child response must also fail closed.
        return single_ok is True and pair_ok is True

    def versions(self) -> dict[str, str]:
        return {**self.single_gateway.versions(), **self.pair_gateway.versions()}

    def supported_tasks(self) -> list[str]:
        return sorted(
            set(self.single_gateway.supported_tasks()) | set(self.pair_gateway.supported_tasks())
        )

    async def close(self) -> None:
        for gateway in (self.single_gateway, self.pair_gateway):
            close = getattr(gateway, "close", None)
            if close is not None:
                await close()


def build_gateway(settings: Settings, asset_store: LocalAssetStore) -> SpecialistGateway:
    pair_gateway: SpecialistGateway = (
        HttpSpecialistGateway(settings, asset_store, allowed_tasks=PAIR_TASKS)
        if settings.pair_backend == "http"
        else LocalPairSpecialistGateway(asset_store)
    )
    if settings.model_backend == "http":
        return HybridSpecialistGateway(HttpSpecialistGateway(settings, asset_store), pair_gateway)
    if settings.model_backend == "space":
        return HybridSpecialistGateway(SpaceSpecialistGateway(settings, asset_store), pair_gateway)
    return HybridSpecialistGateway(DemoSpecialistGateway(), pair_gateway)
