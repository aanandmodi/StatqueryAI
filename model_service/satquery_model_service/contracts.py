from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Step(StrictModel):
    step_id: str
    task: Literal[
        "single_vqa", "caption", "grounding", "change_vqa", "optical_sar_fusion"
    ]
    asset_ids: list[str]
    permitted_params: dict[str, Any]
    policy_reason: str


class Asset(StrictModel):
    id: str
    original_name: str
    content_type: str
    size_bytes: int
    sha256: str
    role: str
    modality: str
    source_dataset: str | None = None
    created_at: str
    metadata: dict[str, Any] | None = None
    validation_errors: list[str] = Field(default_factory=list)


class InferencePayload(StrictModel):
    step: Step
    query: str = Field(min_length=2, max_length=2_000)
    assets: list[Asset] = Field(min_length=1, max_length=2)


class Evidence(StrictModel):
    id: str
    type: Literal["box", "polygon", "mask", "heatmap", "text_region"]
    label: str
    score: float = Field(ge=0, le=1)
    coordinate_space: Literal["normalized", "pixel", "geographic"]
    geometry: dict[str, Any]
    asset_id: str
    artifact_url: str | None = None


class SpecialistResponse(StrictModel):
    task: str
    text: str
    facts: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    raw_score: float = Field(ge=0, le=1)
    score_kind: Literal["calibrated_probability", "evidence_quality", "uncalibrated"]
    model_version: str
    warnings: list[str] = Field(default_factory=list)
