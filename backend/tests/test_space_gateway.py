from __future__ import annotations

import io
from pathlib import Path

import httpx
import pytest
from PIL import Image

from app.config import Settings
from app.models.gateway import SpaceSpecialistGateway, _parse_gradio_sse, render_rgb_preview
from app.schemas import PlannedStep, TaskType
from app.storage import LocalAssetStore


def test_parse_gradio_complete_event() -> None:
    payload = _parse_gradio_sse('event: complete\ndata: [{"text":"water"}]\n')
    assert payload == [{"text": "water"}]


def test_render_benchmark_rgb_preview(tmp_path: Path) -> None:
    path = tmp_path / "scene.png"
    Image.new("RGB", (1_600, 800), (21, 70, 130)).save(path)
    preview = render_rgb_preview(path, max_edge=640, jpeg_quality=85)
    with Image.open(io.BytesIO(preview)) as image:
        assert image.mode == "RGB"
        assert image.size == (640, 320)


@pytest.mark.asyncio
async def test_space_gateway_calls_queue_once_then_uses_cache(tmp_path: Path, make_asset) -> None:
    asset = make_asset("ast_a")
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    Image.new("RGB", (24, 24), (10, 80, 140)).save(upload_dir / "ast_a__scene.png")
    store = LocalAssetStore(upload_dir, max_upload_bytes=1_000_000)
    settings = Settings(
        environment="test",
        model_backend="space",
        space_url="https://example.hf.space",
        space_retry_attempts=1,
    )
    gateway = SpaceSpecialistGateway(settings, store)
    await gateway.client.aclose()
    calls = {"post": 0, "get": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            calls["post"] += 1
            return httpx.Response(200, json={"event_id": "evt-1"})
        calls["get"] += 1
        body = (
            'event: complete\ndata: [{"task":"single_vqa","text":"A river is visible.",'
            '"facts":[],"evidence":[],"raw_score":0.5,"score_kind":"uncalibrated",'
            '"model_version":"adapter@abc","warnings":["uncalibrated"]}]\n'
        )
        return httpx.Response(200, text=body)

    gateway.client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        timeout=10,
    )
    step = PlannedStep(
        step_id="step-1",
        task=TaskType.SINGLE_VQA,
        asset_ids=[asset.id],
        policy_reason="test",
    )
    first = await gateway.infer(step, [asset], "Is a river visible?")
    second = await gateway.infer(step, [asset], "Is a river visible?")

    assert first.text == "A river is visible."
    assert second == first
    assert calls == {"post": 1, "get": 1}
    await gateway.close()

