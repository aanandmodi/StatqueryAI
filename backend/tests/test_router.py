from __future__ import annotations

import pytest

from app.config import Settings
from app.core.router import PolicyRouter
from app.errors import RoutingFailure
from app.schemas import AssetRole, Modality, TaskType


def test_routes_optical_sar_fusion(make_asset):
    router = PolicyRouter(Settings(environment="test"))
    optical = make_asset("ast_1", Modality.OPTICAL, AssetRole.OPTICAL)
    sar = make_asset("ast_2", Modality.SAR, AssetRole.SAR)

    plan = router.plan(
        "Use optical and SAR together to find water and built-up areas",
        [optical, sar],
        None,
        {"threshold": 0.7, "model": "malicious-override"},
    )

    assert [step.task for step in plan.steps] == [TaskType.OPTICAL_SAR_FUSION]
    assert plan.steps[0].permitted_params == {
        "targets": ["built-up", "water"],
        "threshold": 0.7,
    }
    assert "model" not in plan.steps[0].permitted_params


def test_routes_change_before_grounding(make_asset):
    router = PolicyRouter(Settings(environment="test"))
    before = make_asset("ast_a", role=AssetRole.TIME_A)
    after = make_asset("ast_b", role=AssetRole.TIME_B)

    plan = router.plan(
        "Where did built-up land change between the two dates?",
        [before, after],
        None,
        {},
    )

    assert [step.task for step in plan.steps] == [TaskType.CHANGE_VQA]


def test_incompatible_fusion_fails(make_asset):
    router = PolicyRouter(Settings(environment="test"))
    only_optical = make_asset("ast_1")

    with pytest.raises(RoutingFailure):
        router.plan("Fuse optical and SAR", [only_optical], None, {})


def test_explicit_task_cannot_override_contract(make_asset):
    router = PolicyRouter(Settings(environment="test"))
    only_optical = make_asset("ast_1")

    with pytest.raises(RoutingFailure):
        router.plan("anything", [only_optical], [TaskType.CHANGE_VQA], {})
