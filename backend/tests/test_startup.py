from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[2] / "scripts" / "run-local.py"
spec = importlib.util.spec_from_file_location("satquery_local_runner", RUNNER)
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize(
    ("body", "status", "expected"),
    [
        ({"status": "ready"}, 200, True),
        ({"status": "ok", "checks": {"database": True, "model_gateway": True}}, 200, True),
        ({"status": "degraded", "checks": {"model_gateway": False}}, 200, False),
        ({"status": "ok", "checks": {"model_gateway": False}}, 200, False),
        ({"status": "ready"}, 503, False),
        ("ngrok interstitial", 200, False),
        (["ready"], 200, False),
    ],
)
def test_readiness_requires_healthy_contract(body, status, expected):
    assert runner.readiness_ok(body, status) is expected

