from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from app.config import Settings
from app.errors import ModelUnavailableError
from app.models.gateway import (
    FallbackPairSpecialistGateway,
    HttpSpecialistGateway,
    HybridSpecialistGateway,
)
from app.models.pair_tools import PAIR_TASKS
from app.schemas import EvidenceItem, PlannedStep, SpecialistOutput, TaskType
from app.storage import LocalAssetStore


def _gateway(tmp_path: Path, *, paired: bool = False) -> HttpSpecialistGateway:
    settings = Settings(
        environment="test",
        model_backend="http",
        model_service_url="https://model.example.test",
        change_service_url="https://change.example.test" if paired else None,
        fusion_service_url="https://fusion.example.test" if paired else None,
    )
    return HttpSpecialistGateway(
        settings,
        LocalAssetStore(tmp_path / "uploads", max_upload_bytes=1_000_000),
        allowed_tasks=PAIR_TASKS if paired else None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "status", "expected"),
    [
        ({"status": "ready", "capability": "vlm", "model_version": "test-v1"}, 200, True),
        ({"status": "ok", "checks": {"weights": True, "gpu": True}}, 200, True),
        ({"status": "ready"}, 200, True),
        ({"status": "degraded"}, 200, False),
        ({"status": "loading"}, 200, False),
        ({"status": "ok", "checks": {"weights": False}}, 200, False),
        ({"status": "ok", "checks": {"weights": "true"}}, 200, False),
        ({"status": "ok", "checks": {"weights": 1}}, 200, False),
        ({"status": "ok", "checks": None}, 200, False),
        ({"status": "ok", "checks": [True]}, 200, False),
        ({"status": "ready", "capability": "change"}, 200, False),
        ({"status": "ready", "capability": ["vlm"]}, 200, False),
        ({"status": "ready"}, 503, False),
        ({"status": ["ready"]}, 200, False),
        ({}, 200, False),
        (["ready"], 200, False),
        (None, 200, False),
        ("<html>ngrok browser interstitial</html>", 200, False),
    ],
)
async def test_http_health_requires_json_readiness_contract(tmp_path, body, status, expected):
    gateway = _gateway(tmp_path)
    await gateway.client.aclose()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/ready"
        if isinstance(body, str):
            return httpx.Response(status, text=body, headers={"content-type": "text/html"})
        return httpx.Response(status, json=body)

    gateway.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        assert await gateway.health() is expected
    finally:
        await gateway.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("broken_service", [None, "change.example.test", "fusion.example.test"])
async def test_pair_gateway_checks_every_specialist_contract(tmp_path, broken_service):
    gateway = _gateway(tmp_path, paired=True)
    await gateway.client.aclose()
    observed: set[str] = set()

    def handler(request: httpx.Request) -> httpx.Response:
        observed.add(request.url.host)
        capability = request.url.host.split(".")[0]
        body = {"status": "ready", "capability": capability}
        if request.url.host == broken_service:
            body["status"] = "degraded"
        return httpx.Response(200, json=body)

    gateway.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        assert await gateway.health() is (broken_service is None)
        if broken_service is None:
            assert observed == {"change.example.test", "fusion.example.test"}
    finally:
        await gateway.close()


@pytest.mark.asyncio
async def test_http_health_connection_failure_is_not_ready(tmp_path):
    gateway = _gateway(tmp_path)
    await gateway.client.aclose()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("tunnel unavailable", request=request)

    gateway.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        assert await gateway.health() is False
    finally:
        await gateway.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("single_ready", "pair_ready", "expected"),
    [(True, True, True), (False, True, False), (True, False, False), (False, False, False)],
)
async def test_hybrid_ready_requires_both_branches(single_ready, pair_ready, expected):
    single = AsyncMock()
    single.health.return_value = single_ready
    pair = AsyncMock()
    pair.health.return_value = pair_ready

    assert await HybridSpecialistGateway(single, pair).health() is expected
    single.health.assert_awaited_once()
    pair.health.assert_awaited_once()


@pytest.mark.asyncio
async def test_hybrid_child_failure_does_not_claim_ready():
    single = AsyncMock()
    single.health.side_effect = RuntimeError("model startup failed")
    pair = AsyncMock()
    pair.health.return_value = True

    assert await HybridSpecialistGateway(single, pair).health() is False


def _pair_output(task: TaskType, version: str) -> SpecialistOutput:
    return SpecialistOutput(
        task=task,
        text="Pair result",
        facts=[],
        evidence=[
            EvidenceItem(
                id="ev_a",
                type="box",
                label="candidate",
                score=0.5,
                coordinate_space="normalized",
                geometry={"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                asset_id="ast_a",
            )
        ],
        raw_score=0.5,
        score_kind="evidence_quality",
        model_version=version,
    )


@pytest.mark.asyncio
async def test_pair_gateway_labels_reachable_learned_change_service():
    learned = AsyncMock()
    learned.infer.return_value = _pair_output(TaskType.CHANGE_VQA, "checkpoint@immutable-sha")
    analytical = AsyncMock()
    gateway = FallbackPairSpecialistGateway(learned, analytical)
    step = PlannedStep(
        step_id="step-1",
        task=TaskType.CHANGE_VQA,
        asset_ids=["ast_a", "ast_b"],
        policy_reason="test",
    )

    output = await gateway.infer(step, [], "What changed?")

    assert {item["name"]: item["value"] for item in output.facts}[
        "pair_execution_method"
    ] == "learned-cdvqa-v1"
    assert output.evidence[0].geometry["method"] == "learned-cdvqa-v1"
    analytical.infer.assert_not_awaited()


@pytest.mark.asyncio
async def test_pair_gateway_falls_back_and_labels_analytical_method():
    learned = AsyncMock()
    learned.infer.side_effect = ModelUnavailableError("checkpoint offline")
    analytical = AsyncMock()
    analytical.infer.return_value = _pair_output(
        TaskType.OPTICAL_SAR_FUSION, "satquery-optical-sar-proxy-tool-v2"
    )
    gateway = FallbackPairSpecialistGateway(learned, analytical)
    step = PlannedStep(
        step_id="step-1",
        task=TaskType.OPTICAL_SAR_FUSION,
        asset_ids=["ast_a", "ast_b"],
        policy_reason="test",
    )

    output = await gateway.infer(step, [], "Fuse these scenes")

    assert {item["name"]: item["value"] for item in output.facts}[
        "pair_execution_method"
    ] == "analytical-fusion-baseline"
    assert output.evidence[0].geometry["method"] == "analytical-fusion-baseline"
    assert any("checkpoint offline" in warning for warning in output.warnings)
    analytical.infer.assert_awaited_once()
