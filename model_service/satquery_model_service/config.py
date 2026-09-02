from __future__ import annotations

from pathlib import Path
import re
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
    base_model: str = "Qwen/Qwen3-VL-2B-Instruct"
    base_revision: str = "89644892e4d85e24eaac8bacfd4f463576704203"
    adapter_model: str = "aanandmodi/satquery-qwen3vl-bigearthnet-txt-lora"
    adapter_revision: str = "ed12e59e0def9468bdf4a226789fc1b77c7900e7"
    adapter_dir: Path | None = None
    four_bit: bool = True
    max_upload_bytes: int = 1_073_741_824
    max_new_tokens: int = 128
    max_image_edge: int = 448
    max_pixels: int = 448 * 448
    device: str = "cuda"
    service_token: SecretStr | None = None

    def validate_startup(self) -> None:
        if self.capability == "vlm":
            for name, revision in (
                ("SATQUERY_MODEL_BASE_REVISION", self.base_revision),
                ("SATQUERY_MODEL_ADAPTER_REVISION", self.adapter_revision),
            ):
                if not re.fullmatch(r"[0-9a-f]{40}", revision):
                    raise RuntimeError(f"{name} must pin an immutable 40-character commit SHA")
            if self.four_bit and self.device != "cuda":
                raise RuntimeError("The bounded local VLM profile requires a CUDA device")
        if self.capability in {"change", "fusion"} and not self.artifact_dir:
            raise RuntimeError(
                "SATQUERY_MODEL_ARTIFACT_DIR is required for this capability"
            )
