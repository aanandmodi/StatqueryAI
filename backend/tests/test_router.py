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


@pytest.mark.parametrize(
    "query",
    [
        "Show me all the water bodies in this region",
        '"Show me all the water bodies in this region',
        "Please show all waterbodies on this image",
        "Find the lakes in this scene",
        "Display rivers and ponds in this region",
        "Map the reservoirs and describe the scene",
        "Outline visible water bodies. Describe their spatial pattern, "
        "possible confounders and unresolved questions.",
        "Outline visible vegetation. Describe canopy pattern. Do not invent species or causes.",
    ],
)
def test_display_water_requests_run_grounding_not_caption(make_asset, query):
    plan = PolicyRouter(Settings(environment="test")).plan(query, [make_asset("ast_a")], None, {})
    assert [step.task for step in plan.steps] == [TaskType.GROUNDING]
    if "vegetation" not in query:
        assert plan.steps[0].permitted_params["targets"] == ["water"]


@pytest.mark.parametrize("query", ["Show me the report", "Display scene metadata"])
def test_show_without_a_spatial_feature_does_not_request_masks(make_asset, query):
    plan = PolicyRouter(Settings(environment="test")).plan(query, [make_asset("ast_a")], None, {})
    assert [step.task for step in plan.steps] == [TaskType.CAPTION]


def test_explicit_caption_choice_is_preserved_for_display_query(make_asset):
    plan = PolicyRouter(Settings(environment="test")).plan(
        "Show me all the water bodies in this region",
        [make_asset("ast_a")],
        [TaskType.CAPTION],
        {},
    )
    assert [step.task for step in plan.steps] == [TaskType.CAPTION]


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


def test_auto_routes_ambiguous_temporal_pair_from_input_roles(make_asset):
    router = PolicyRouter(Settings(environment="test"))
    before = make_asset("ast_a", role=AssetRole.TIME_A)
    after = make_asset("ast_b", role=AssetRole.TIME_B)

    plan = router.plan("What can you tell me?", [before, after], None, {})

    assert [step.task for step in plan.steps] == [TaskType.CHANGE_VQA]


def test_auto_routes_ambiguous_optical_sar_pair_from_modalities(make_asset):
    router = PolicyRouter(Settings(environment="test"))
    optical = make_asset("ast_a", Modality.MULTISPECTRAL, AssetRole.OPTICAL)
    sar = make_asset("ast_b", Modality.SAR, AssetRole.SAR)

    plan = router.plan("Assess this paired observation", [optical, sar], None, {})

    assert [step.task for step in plan.steps] == [TaskType.OPTICAL_SAR_FUSION]


@pytest.mark.parametrize(
    "query",
    ["Describe the changes", "Locate changed regions", "Caption the scene and then mark water"],
)
def test_temporal_pair_never_silently_drops_second_asset_in_auto_mode(make_asset, query):
    router = PolicyRouter(Settings(environment="test"))
    before = make_asset("ast_a", role=AssetRole.TIME_A)
    after = make_asset("ast_b", role=AssetRole.TIME_B)

    plan = router.plan(query, [before, after], None, {})

    assert [step.task for step in plan.steps] == [TaskType.CHANGE_VQA]
    assert plan.steps[0].asset_ids == [before.id, after.id]


@pytest.mark.parametrize("query", ["Locate water", "Describe this scene", "What changed here?"])
def test_optical_sar_pair_contract_overrides_ambiguous_query(make_asset, query):
    router = PolicyRouter(Settings(environment="test"))
    optical = make_asset("ast_a", Modality.OPTICAL, AssetRole.OPTICAL)
    sar = make_asset("ast_b", Modality.SAR, AssetRole.SAR)

    plan = router.plan(query, [optical, sar], None, {})

    assert [step.task for step in plan.steps] == [TaskType.OPTICAL_SAR_FUSION]
    assert plan.steps[0].asset_ids == [optical.id, sar.id]


def test_explicit_caption_can_select_one_asset_from_pair(make_asset):
    router = PolicyRouter(Settings(environment="test"))
    before = make_asset("ast_a", Modality.SAR, AssetRole.TIME_A)
    after = make_asset("ast_b", Modality.SAR, AssetRole.TIME_B)

    plan = router.plan("Describe this scene", [before, after], [TaskType.CAPTION], {})

    assert [step.task for step in plan.steps] == [TaskType.CAPTION]
    assert plan.steps[0].asset_ids == [before.id]


def test_temporal_roles_cannot_be_reinterpreted_as_cross_modal_pair(make_asset):
    router = PolicyRouter(Settings(environment="test"))
    before = make_asset("ast_a", Modality.OPTICAL, AssetRole.TIME_A)
    after = make_asset("ast_b", Modality.SAR, AssetRole.TIME_B)

    plan = router.plan("Compare the observations", [before, after], None, {})

    # The downstream temporal validator will reject the modality mismatch.
    assert [step.task for step in plan.steps] == [TaskType.CHANGE_VQA]


@pytest.mark.parametrize("word", ["changes", "increases", "decreases", "differences", "dates"])
def test_change_classifier_recognizes_plural_terms(word):
    assert [candidate.task for candidate in PolicyRouter._classify(f"Describe the {word}")] == [
        TaskType.CHANGE_VQA
    ]
