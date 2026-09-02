from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables.

    Defaults are deliberately safe for a single-machine demo. Production must set
    ``environment=production`` and use an HTTP model service rather than demo models.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SATQUERY_",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "SatQuery API"
    environment: Literal["development", "test", "production"] = "development"
    api_prefix: str = "/v1"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    data_dir: Path = Path("./data/runtime")
    database_path: Path = Path("./data/runtime/satquery.sqlite3")
    upload_dir: Path = Path("./data/runtime/uploads")
    artifact_dir: Path = Path("./data/runtime/artifacts")
    report_dir: Path = Path("./data/runtime/reports")

    model_backend: Literal["demo", "http", "space"] = "demo"
    model_service_url: str = "http://model-service:8080"
    vlm_service_url: str | None = None
    change_service_url: str | None = None
    fusion_service_url: str | None = None
    model_service_token: SecretStr | None = None
    model_timeout_seconds: float = 180.0

    # Public Gradio/ZeroGPU Space integration. A token is optional for a public
    # Space; keeping it server-side gives the deployment account its normal quota.
    space_url: str | None = None
    space_token: SecretStr | None = None
    space_api_name: str = "analyze"
    space_retry_attempts: int = 3
    space_cache_ttl_seconds: int = 86_400
    space_cache_max_entries: int = 128
    space_preview_max_edge: int = 1_280
    space_preview_jpeg_quality: int = 90
    space_max_new_tokens: int = 128

    max_upload_bytes: int = 1_073_741_824
    max_raster_pixels: int = 500_000_000
    max_raster_bands: int = 32
    max_query_chars: int = 2_000
    max_plan_steps: int = 4
    max_parallel_jobs: int = 2
    event_poll_seconds: float = 0.5
    job_timeout_seconds: int = 900

    allowed_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    api_key: SecretStr | None = None
    enable_docs: bool = True
    allow_benchmark_images: bool = True

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def split_origins(cls, value: object) -> object:
        if isinstance(value, str) and not value.lstrip().startswith("["):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    def prepare_directories(self) -> None:
        for path in (
            self.data_dir,
            self.database_path.parent,
            self.upload_dir,
            self.artifact_dir,
            self.report_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def assert_safe_production_configuration(self) -> None:
        if self.environment == "production" and self.model_backend == "demo":
            raise RuntimeError("SATQUERY_MODEL_BACKEND=demo is forbidden in production")
        if self.environment == "production" and not self.api_key:
            raise RuntimeError("SATQUERY_API_KEY is required in production")
        if self.model_backend == "space" and not self.space_url:
            raise RuntimeError("SATQUERY_SPACE_URL is required when model_backend=space")
        if (
            self.model_backend == "http"
            and self.model_service_url.lower().startswith("https://")
            and not self.model_service_token
        ):
            raise RuntimeError(
                "SATQUERY_MODEL_SERVICE_TOKEN is required for a remote HTTPS model service"
            )

    def service_url_for_task(self, task: str) -> str:
        if task in {"single_vqa", "caption", "grounding"}:
            return self.vlm_service_url or self.model_service_url
        if task == "change_vqa":
            return self.change_service_url or self.model_service_url
        if task == "optical_sar_fusion":
            return self.fusion_service_url or self.model_service_url
        raise ValueError(f"Unknown task: {task}")


@lru_cache
def get_settings() -> Settings:
    return Settings()
