from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Modality(StrEnum):
    OPTICAL = "optical"
    SAR = "sar"
    MULTISPECTRAL = "multispectral"
    UNKNOWN = "unknown"


class AssetRole(StrEnum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    OPTICAL = "optical"
    SAR = "sar"
    TIME_A = "time_a"
    TIME_B = "time_b"


class TaskType(StrEnum):
    SINGLE_VQA = "single_vqa"
    CAPTION = "caption"
    GROUNDING = "grounding"
    CHANGE_VQA = "change_vqa"
    OPTICAL_SAR_FUSION = "optical_sar_fusion"


class AnalysisStatus(StrEnum):
    QUEUED = "queued"
    VALIDATING = "validating"
    PLANNING = "planning"
    RUNNING = "running"
    INTEGRATING = "integrating"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = {
    AnalysisStatus.SUCCEEDED,
    AnalysisStatus.FAILED,
    AnalysisStatus.CANCELLED,
}


class RasterMetadata(StrictModel):
    driver: str
    width: int
    height: int
    count: int
    dtypes: list[str]
    crs: str | None
    transform: list[float]
    bounds: list[float]
    resolution: list[float]
    nodata: float | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    quality_score: Annotated[float, Field(ge=0, le=1)] = 1.0


class AssetRecord(StrictModel):
    id: str
    original_name: str
    content_type: str
    size_bytes: int
    sha256: str
    role: AssetRole
    modality: Modality
    source_dataset: str | None = None
    created_at: datetime
    metadata: RasterMetadata | None = None
    validation_errors: list[str] = Field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.validation_errors and self.metadata is not None


class GeospatialContext(StrictModel):
    """User-supplied location context kept separate from pixel-derived evidence."""

    latitude: Annotated[float, Field(ge=-90, le=90)]
    longitude: Annotated[float, Field(ge=-180, le=180)]
    altitude_m: Annotated[float | None, Field(ge=-500, le=100_000)] = None
    captured_at: datetime | None = None
    sensor: Annotated[str | None, Field(max_length=120)] = None
    source: Literal["user", "gps", "exif", "raster"] = "user"
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def bound_metadata(
        cls, value: dict[str, str | int | float | bool]
    ) -> dict[str, str | int | float | bool]:
        if len(value) > 24:
            raise ValueError("metadata may contain at most 24 entries")
        cleaned: dict[str, str | int | float | bool] = {}
        for key, item in value.items():
            clean_key = str(key).strip()
            if not clean_key or len(clean_key) > 80:
                raise ValueError("metadata keys must contain 1 to 80 characters")
            if isinstance(item, str) and len(item) > 500:
                raise ValueError("metadata string values may contain at most 500 characters")
            cleaned[clean_key] = item
        return cleaned


class AnalysisCreate(StrictModel):
    query: Annotated[str, Field(min_length=2, max_length=2_000)]
    asset_ids: Annotated[list[str], Field(min_length=1, max_length=4)]
    session_id: Annotated[str | None, Field(max_length=80)] = None
    requested_tasks: Annotated[list[TaskType] | None, Field(max_length=4)] = None
    context: GeospatialContext | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("asset_ids")
    @classmethod
    def unique_assets(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("asset_ids must be unique")
        return value


class PlannedStep(StrictModel):
    step_id: str
    task: TaskType
    asset_ids: list[str]
    permitted_params: dict[str, Any] = Field(default_factory=dict)
    policy_reason: str


class ExecutionPlan(StrictModel):
    version: str = "planner-policy-v1"
    steps: list[PlannedStep]
    rejected_intents: list[str] = Field(default_factory=list)


class TraceEvent(StrictModel):
    step_id: str
    task: TaskType | Literal["validation", "planning", "integration"]
    tool: str
    model_version: str | None = None
    status: Literal["started", "succeeded", "failed", "skipped"]
    policy_reason: str
    permitted_params: dict[str, Any] = Field(default_factory=dict)
    duration_ms: int | None = None
    timestamp: datetime = Field(default_factory=utc_now)


class EvidenceItem(StrictModel):
    id: str
    type: Literal["box", "polygon", "mask", "heatmap", "text_region"]
    label: str
    score: Annotated[float, Field(ge=0, le=1)]
    coordinate_space: Literal["normalized", "pixel", "geographic"]
    geometry: dict[str, Any]
    asset_id: str
    artifact_url: str | None = None


class Confidence(StrictModel):
    score: Annotated[float, Field(ge=0, le=1)]
    level: Literal["low", "medium", "high"]
    calibration_version: str
    factors: dict[str, Annotated[float, Field(ge=0, le=1)]]
    meaning: str


class SpecialistOutput(StrictModel):
    task: TaskType
    text: str
    facts: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    raw_score: Annotated[float, Field(ge=0, le=1)]
    score_kind: Literal[
        "calibrated_probability", "evidence_quality", "uncalibrated"
    ] = "uncalibrated"
    model_version: str
    warnings: list[str] = Field(default_factory=list)


class AnalysisResult(StrictModel):
    answer: str
    facts: list[dict[str, Any]]
    evidence: list[EvidenceItem]
    confidence: Confidence
    trace: list[TraceEvent]
    warnings: list[str] = Field(default_factory=list)
    report_url: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class AnalysisRecord(StrictModel):
    id: str
    status: AnalysisStatus
    progress: Annotated[float, Field(ge=0, le=1)] = 0
    request: AnalysisCreate
    created_at: datetime
    updated_at: datetime
    plan: ExecutionPlan | None = None
    result: AnalysisResult | None = None
    error_code: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def result_matches_status(self) -> AnalysisRecord:
        if self.status == AnalysisStatus.SUCCEEDED and self.result is None:
            raise ValueError("a succeeded analysis must include a result")
        return self


class HealthResponse(StrictModel):
    status: Literal["ok", "degraded"]
    version: str
    model_backend: str
    checks: dict[str, bool]


class ErrorDetail(StrictModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(StrictModel):
    error: ErrorDetail
