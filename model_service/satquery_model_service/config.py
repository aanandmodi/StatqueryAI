from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SATQUERY_MODEL_",
        case_sensitive=False,
        extra="ignore",
    )

    capability: Literal["vlm", "change", "fusion"] = "vlm"
    artifact_dir: Path | None = None
    base_model: str = "Qwen/Qwen3-VL-4B-Instruct"
    base_revision: str | None = None
    adapter_dir: Path | None = None
    four_bit: bool = True
    max_upload_bytes: int = 1_073_741_824
    max_new_tokens: int = 256
    device: str = "cuda"
    service_token: SecretStr | None = None

    def validate_startup(self) -> None:
        if self.capability == "vlm" and not self.base_revision:
            raise RuntimeError(
                "SATQUERY_MODEL_BASE_REVISION must pin an immutable commit"
            )
        if self.capability in {"change", "fusion"} and not self.artifact_dir:
            raise RuntimeError(
                "SATQUERY_MODEL_ARTIFACT_DIR is required for this capability"
            )
