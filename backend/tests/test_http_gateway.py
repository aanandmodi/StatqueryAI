from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.models.gateway import HttpSpecialistGateway
from app.schemas import GeospatialContext, PlannedStep, TaskType
from app.storage import LocalAssetStore


@pytest.mark.asyncio
async def test_http_gateway_sends_context_and_bearer_token(
    tmp_path: Path, make_asset
) -> None:
    asset = make_asset("ast_a")
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    (upload_dir / "ast_a__scene.tif").write_bytes(b"II*\x00" + b"x" * 128)
    store = LocalAssetStore(upload_dir, max_upload_bytes=1_000_000)
    settings = Settings(
        environment="test",
        model_backend="http",
        model_service_url="https://example.ngrok-free.app",
        model_service_token="test-service-token-with-more-than-32-chars",
    )
    gateway = HttpSpecialistGateway(settings, store)
    client_headers = dict(gateway.client.headers)
    await gateway.client.aclose()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"].startswith("Bearer test-service-token")
        assert request.headers["ngrok-skip-browser-warning"] == "1"
        assert request.url.path == "/v1/infer/single_vqa"
        body = request.content.decode("utf-8", errors="ignore")
        assert '"latitude": 28.6139' in body
        assert '"longitude": 77.209' in body
        assert "ast_a.tif" in body
        return httpx.Response(
            200,
            json={
                "task": "single_vqa",
                "text": "Vegetation is visible.",
                "facts": [
                    {
                        "name": "user_location",
                        "value": {"latitude": 28.6139, "longitude": 77.209},
                    }
                ],
                "evidence": [],
                "raw_score": 0.5,
                "score_kind": "uncalibrated",
                "model_version": "adapter@ed12e59",
                "warnings": ["uncalibrated"],
            },
        )

    gateway.client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), headers=client_headers, timeout=10
    )
    step = PlannedStep(
        step_id="step-1",
        task=TaskType.SINGLE_VQA,
        asset_ids=[asset.id],
        policy_reason="test",
    )
    context = GeospatialContext(
        latitude=28.6139,
        longitude=77.209,
        altitude_m=216,
    )
    output = await gateway.infer(step, [asset], "What is visible?", context)

    assert output.text == "Vegetation is visible."
    assert output.model_version == "adapter@ed12e59"
    await gateway.close()
