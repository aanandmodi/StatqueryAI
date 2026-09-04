from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import httpx
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "sih-acceptance.py"
spec = importlib.util.spec_from_file_location("satquery_acceptance", SCRIPT)
assert spec and spec.loader
acceptance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acceptance)


@pytest.mark.parametrize("allow_simulated", [False, True])
def test_acceptance_cannot_silently_certify_simulator(allow_simulated):
    def respond(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["requested_tasks"] is None
        return httpx.Response(202, json={
            "id": "anl_test", "status": "succeeded",
            "result": {
                "answer": "Simulated answer.", "trace": [{"task": "single_vqa"}],
                "provenance": {"model_versions": {"single_vqa": "demo-simulator-v1"}},
            },
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        if allow_simulated:
            result = acceptance.analyze(
                client, "http://test/v1", "single_vqa", "Is water visible?", ["ast_a"], 10,
                auto_route=True, allow_simulated=True,
            )
            assert result["status"] == "succeeded"
        else:
            with pytest.raises(RuntimeError, match="simulator response"):
                acceptance.analyze(
                    client, "http://test/v1", "single_vqa", "Is water visible?", ["ast_a"], 10,
                    auto_route=True,
                )
