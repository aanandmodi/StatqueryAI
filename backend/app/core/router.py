from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.errors import RoutingFailure
from app.schemas import AssetRecord, AssetRole, ExecutionPlan, Modality, PlannedStep, TaskType

TARGET_CLASSES = {
    "water",
    "built-up",
    "urban",
    "vegetation",
    "forest",
    "cropland",
    "agriculture",
    "road",
    "building",
    "flood",
    "bare soil",
}

# Display/find requests are spatial when they name a supported feature. "Show my report"
# should remain a caption request; do not turn every occurrence of "show" into grounding.
DISPLAY_FEATURE_PATTERN = re.compile(
    r"\b(?:show|display|draw|find|detect|identify|map)\b[^.!?\n]{0,160}"
    r"\b(?:water(?:\s*bodies?)?|lakes?|rivers?|ponds?|reservoirs?|built-up|urban|"
    r"vegetation|forests?|cropland|agriculture|roads?|buildings?|flood|bare\s+soil)\b",
    re.I,
)

TASK_PATTERNS: list[tuple[TaskType, re.Pattern[str]]] = [
    (
        TaskType.CHANGE_VQA,
        re.compile(
            r"\b(changes?|changed|changing|increase[sd]?|decrease[sd]?|before|after|between|"
<<<<<<< HEAD
            r"temporal|dates?|differences?|compare[sd]?|dropped|declined|lost|loss|gained)\b",
=======
            r"temporal|dates?|differences?)\b",
>>>>>>> 2f620623f8897788bd2df2ce4f5700cb183d84f8
            re.I,
        ),
    ),
    (
        TaskType.OPTICAL_SAR_FUSION,
        re.compile(r"\b(sar|radar|fusion|fuse|multimodal|multi-modal|optical.{0,20}sar)\b", re.I),
    ),
    (
        TaskType.GROUNDING,
        re.compile(
            r"\b(highlight|locate|where|point|ground|bounding|mark|segment|mask|overlay|outline)\b",
            re.I,
        ),
    ),
    (
        TaskType.CAPTION,
        re.compile(
            r"\b(describe|caption|summari[sz]e|what is visible|scene|report|region)\b", re.I
        ),
    ),
]


@dataclass(frozen=True)
class IntentCandidate:
    task: TaskType
    clause: str


class PolicyRouter:
    """Closed-set intent router and compatibility gate.

    Learned intent proposals are compiled here. No query or proposal may choose
    an arbitrary tool, URL, path or model option.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def plan(
        self,
        query: str,
        assets: list[AssetRecord],
        requested_tasks: list[TaskType] | None,
        parameters: dict[str, Any],
        proposal=None,
        proposal_source: str = "deterministic-policy",
    ) -> ExecutionPlan:
        params = self._permit_params(query, parameters)
        objectives = set(proposal.objectives) if proposal else set()
        if proposal and proposal.target != "none":
            params["targets"] = [proposal.target]
        target = (params.get("targets") or [None])[0]
        temporal = len(assets) == 2 and {a.role for a in assets} == {
            AssetRole.TIME_A,
            AssetRole.TIME_B,
        }
        extent_request = bool(
            re.search(
                r"\b(compare|compared|lost|loss|gained|gain|dropped|declined|shrank|expanded|extent|area|change[sd]?)\b",
                query,
                re.I,
            )
        ) or bool(objectives & {"compare", "measure"})
        if (
            not requested_tasks
            and temporal
            and target in {"water", "forest", "vegetation", "building", "road", "cropland"}
            and extent_request
            and all(a.modality in {Modality.OPTICAL, Modality.MULTISPECTRAL} for a in assets)
        ):
            if self.settings.max_plan_steps < 4:
                raise RoutingFailure("Target extent comparison requires a four-step budget")
            before = next(a for a in assets if a.role == AssetRole.TIME_A)
            after = next(a for a in assets if a.role == AssetRole.TIME_B)
            omitted = [name for name in params.get("targets", []) if name != target]
            params["targets"] = [target]
            return ExecutionPlan(
                version="bounded-dag-v2",
                proposal_source=proposal_source,
                steps=[
                    PlannedStep(
                        step_id="ground-before",
                        task=TaskType.GROUNDING,
                        asset_ids=[before.id],
                        permitted_params=params,
                        query=(
                            f"Outline all visible {target} in this image. "
                            "Return source-aligned candidate masks and explain limitations."
                        ),
                        policy_reason="Segment the same target independently at time A",
                    ),
                    PlannedStep(
                        step_id="ground-after",
                        task=TaskType.GROUNDING,
                        asset_ids=[after.id],
                        permitted_params=params,
                        query=(
                            f"Outline all visible {target} in this image. "
                            "Return source-aligned candidate masks and explain limitations."
                        ),
                        policy_reason="Segment the same target independently at time B",
                    ),
                    PlannedStep(
                        step_id="compare-scenes",
                        task=TaskType.CHANGE_VQA,
                        asset_ids=[before.id, after.id],
                        permitted_params=params,
                        query=(
                            "Describe observable changes between time A and time B. "
                            "Do not infer water level, event cause or acquisition dates."
                        ),
                        policy_reason="Pair specialist provides a separately attributed comparison",
                    ),
                    PlannedStep(
                        step_id="measure-extent",
                        task=TaskType.CHANGE_VQA,
                        operation="measure_mask_change",
                        depends_on=["ground-before", "ground-after"],
                        asset_ids=[before.id, after.id],
                        permitted_params=params,
                        policy_reason="Measure compatible target masks on a verified common grid",
                    ),
                ],
                rejected_intents=(
                    [f"Four-step budget analyses {target}; not measured: {', '.join(omitted)}"]
                    if omitted
                    else []
                )
                + [
                    "Before/after roles do not verify 'last month': dates are unverified.",
                    "Surface extent does not establish water level/depth, species or cause.",
                    "All targets are analysed; selecting one reservoir needs a reviewed ROI.",
                ],
            )
        candidates = (
            [IntentCandidate(task, query) for task in requested_tasks]
            if requested_tasks
            else self._classify(query)
        )
<<<<<<< HEAD
        if proposal and not requested_tasks:
            mapping = {
                "describe": TaskType.CAPTION,
                "ground": TaskType.GROUNDING,
                "compare": TaskType.CHANGE_VQA,
                "measure": TaskType.GROUNDING,
                "fuse": TaskType.OPTICAL_SAR_FUSION,
            }
            candidates = [IntentCandidate(mapping[item], query) for item in proposal.objectives]
=======
>>>>>>> 2f620623f8897788bd2df2ce4f5700cb183d84f8
        if not requested_tasks and len(assets) == 2:
            # In automatic mode the pair contract is authoritative. Words such
            # as "describe" or "locate" must not silently discard one upload.
            # A caller can explicitly request a single-image task when intended.
            candidates = [IntentCandidate(self._task_from_pair(assets), query)]
        if not candidates:
            candidates = [IntentCandidate(TaskType.SINGLE_VQA, query)]
        if not requested_tasks and len(assets) == 1:
            # Every specialist receives the complete question. A grounding response
            # already contains the report; sentence boundaries must not double GPU work.
            tasks = {candidate.task for candidate in candidates}
            if tasks <= {TaskType.SINGLE_VQA, TaskType.CAPTION, TaskType.GROUNDING}:
                preferred = (
                    TaskType.GROUNDING
                    if TaskType.GROUNDING in tasks
                    else TaskType.CAPTION
                    if TaskType.CAPTION in tasks
                    else TaskType.SINGLE_VQA
                )
                candidates = [IntentCandidate(preferred, query)]

        unique: list[IntentCandidate] = []
        seen: set[TaskType] = set()
        for candidate in candidates:
            if candidate.task not in seen:
                unique.append(candidate)
                seen.add(candidate.task)
        if len(unique) > self.settings.max_plan_steps:
            raise RoutingFailure(
                "Query expands beyond the permitted plan length",
                details={"steps": len(unique), "limit": self.settings.max_plan_steps},
            )

        steps: list[PlannedStep] = []
        rejected: list[str] = []
        for index, candidate in enumerate(unique, start=1):
            selected = self._select_assets(candidate.task, assets)
            if not selected:
                rejected.append(f"{candidate.task.value}: incompatible input configuration")
                continue
            steps.append(
                PlannedStep(
                    step_id=f"step-{index}",
                    task=candidate.task,
                    asset_ids=[asset.id for asset in selected],
                    permitted_params=params,
                    policy_reason=self._reason(candidate.task, selected),
                    query=candidate.clause,
                )
            )

        if not steps:
            if proposal and not requested_tasks:
                return self.plan(
                    query,
                    assets,
                    None,
                    parameters,
                    proposal_source="deterministic-fallback (asset-incompatible learned proposal)",
                )
            if not requested_tasks and len(assets) == 1 and target and extent_request:
                return ExecutionPlan(
                    steps=[
                        PlannedStep(
                            step_id="ground-current",
                            task=TaskType.GROUNDING,
                            asset_ids=[assets[0].id],
                            permitted_params=params,
                            query=f"Outline visible {target}. Do not invent a past scene.",
                            policy_reason="Only the present image is available",
                        )
                    ],
                    rejected_intents=[
                        "Comparison withheld: a second aligned, dated image is needed."
                    ],
                    proposal_source=proposal_source,
                )
            raise RoutingFailure(
                "No requested task is compatible with the supplied imagery",
                details={
                    "modalities": [asset.modality.value for asset in assets],
                    "asset_count": len(assets),
                    "rejected": rejected,
                },
            )
        if len(assets) == 1 and extent_request:
            rejected.append(
                "Historical comparison needs a second aligned, dated image; no past scene invented."
            )
        return ExecutionPlan(
            steps=steps, rejected_intents=rejected, proposal_source=proposal_source
        )

    @staticmethod
    def _task_from_pair(assets: list[AssetRecord]) -> TaskType:
        if {asset.role for asset in assets} == {AssetRole.TIME_A, AssetRole.TIME_B}:
            return TaskType.CHANGE_VQA
        modalities = {asset.modality for asset in assets}
        has_optical = bool(modalities & {Modality.OPTICAL, Modality.MULTISPECTRAL})
        if has_optical and Modality.SAR in modalities:
            return TaskType.OPTICAL_SAR_FUSION
        return TaskType.CHANGE_VQA

    @staticmethod
    def _classify(query: str) -> list[IntentCandidate]:
        clauses = [
            part.strip()
            for part in re.split(r"[.;]|\band\s+then\b", query, flags=re.I)
            if part.strip()
        ]
        candidates: list[IntentCandidate] = []
        for clause in clauses:
            matches = [task for task, pattern in TASK_PATTERNS if pattern.search(clause)]
            if DISPLAY_FEATURE_PATTERN.search(clause) and TaskType.GROUNDING not in matches:
                matches.append(TaskType.GROUNDING)
            if TaskType.CHANGE_VQA in matches:
                matches = [TaskType.CHANGE_VQA]
            elif TaskType.OPTICAL_SAR_FUSION in matches:
                matches = [TaskType.OPTICAL_SAR_FUSION]
            elif TaskType.GROUNDING in matches:
                # A grounding response already includes a narrative; avoid two GPU reports
                # for a single clause such as "outline water and describe this scene".
                matches = [TaskType.GROUNDING]
            for task in matches or [TaskType.SINGLE_VQA]:
                candidates.append(IntentCandidate(task, clause))
        return candidates

    @staticmethod
    def _select_assets(task: TaskType, assets: list[AssetRecord]) -> list[AssetRecord]:
        if task in {TaskType.SINGLE_VQA, TaskType.CAPTION, TaskType.GROUNDING}:
            if len(assets) == 1:
                return assets
            optical = [
                a for a in assets if a.modality in {Modality.OPTICAL, Modality.MULTISPECTRAL}
            ]
            return optical[:1] or assets[:1]
        if task == TaskType.OPTICAL_SAR_FUSION:
            optical = [
                a for a in assets if a.modality in {Modality.OPTICAL, Modality.MULTISPECTRAL}
            ]
            sar = [a for a in assets if a.modality == Modality.SAR]
            return [optical[0], sar[0]] if optical and sar else []
        if task == TaskType.CHANGE_VQA and len(assets) == 2:
            return assets
        return []

    @staticmethod
    def _permit_params(query: str, provided: dict[str, Any]) -> dict[str, Any]:
        permitted: dict[str, Any] = {}
        lowered = query.lower()
        targets = [name for name in TARGET_CLASSES if name in lowered]
        if re.search(r"\b(water|waterbodies|lakes?|rivers?|ponds?|reservoirs?)\b", lowered):
            targets = list(set(targets) | {"water"})
        if targets:
            permitted["targets"] = sorted(targets)[:8]
        # Spectral threshold is not a confidence percentage.
        water_threshold = provided.get("water_index_threshold")
        if isinstance(water_threshold, int | float) and -1 <= water_threshold <= 1:
            permitted["water_index_threshold"] = float(water_threshold)

        threshold = provided.get("threshold")
        if isinstance(threshold, int | float):
            permitted["threshold"] = max(0.05, min(0.95, float(threshold)))
        elif match := re.search(r"(?:above|over|threshold)\s*(\d{1,2}(?:\.\d+)?)\s*%", query, re.I):
            permitted["threshold"] = max(0.05, min(0.95, float(match.group(1)) / 100))

        bbox = provided.get("bbox")
        if (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(value, int | float) for value in bbox)
        ):
            permitted["bbox"] = [float(value) for value in bbox]
        return permitted

    @staticmethod
    def _reason(task: TaskType, assets: Iterable[AssetRecord]) -> str:
        modalities = ", ".join(asset.modality.value for asset in assets)
        reasons = {
            TaskType.SINGLE_VQA: "one validated scene and a free-text question were supplied",
            TaskType.CAPTION: "the query asks for a scene description from one validated image",
            TaskType.GROUNDING: "the query asks for a target region from one validated image",
            TaskType.CHANGE_VQA: (
                "the input contract supplies two temporal scenes for paired change analysis"
            ),
            TaskType.OPTICAL_SAR_FUSION: (
                "the input contract supplies an optical/SAR pair for cross-modal analysis"
            ),
        }
        return f"{reasons[task]} (modalities: {modalities})"
