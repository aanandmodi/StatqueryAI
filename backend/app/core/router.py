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
            r"temporal|dates?|differences?)\b",
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

    An LLM may propose the same schema in a future planner, but this policy remains
    authoritative: no query may choose an arbitrary tool, URL, path or model option.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def plan(
        self,
        query: str,
        assets: list[AssetRecord],
        requested_tasks: list[TaskType] | None,
        parameters: dict[str, Any],
    ) -> ExecutionPlan:
        candidates = (
            [IntentCandidate(task, query) for task in requested_tasks]
            if requested_tasks
            else self._classify(query)
        )
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
                    permitted_params=self._permit_params(query, parameters),
                    policy_reason=self._reason(candidate.task, selected),
                )
            )

        if not steps:
            raise RoutingFailure(
                "No requested task is compatible with the supplied imagery",
                details={
                    "modalities": [asset.modality.value for asset in assets],
                    "asset_count": len(assets),
                    "rejected": rejected,
                },
            )
        return ExecutionPlan(steps=steps, rejected_intents=rejected)

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
