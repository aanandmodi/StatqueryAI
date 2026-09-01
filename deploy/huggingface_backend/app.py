from __future__ import annotations

import os

# Import ZeroGPU's runtime shim before Gradio or any dependency can touch torch.
import spaces

import gradio as gr

from app.config import get_settings
from app.main import build_lifespan, configure_app


settings = get_settings()
server = gr.Server(
    title=settings.app_name,
    version="1.0.0",
    description="Auditable SatQuery orchestration API",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=build_lifespan(settings),
)
configure_app(server, settings)


@spaces.GPU(duration=1)
def _zerogpu_runtime_marker() -> str:
    """ZeroGPU eligibility marker; normal SatQuery API routes never call this."""

    return "SatQuery API routes are CPU orchestration only."


@server.api(name="runtime-marker", api_visibility="private", time_limit=5)
def runtime_marker() -> str:
    """Internal deployment probe for the free ZeroGPU runtime."""

    return _zerogpu_runtime_marker()


@server.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "SatQuery API",
        "health": "/v1/health/ready",
        "cost_policy": "ZeroGPU only; no paid hardware fallback",
    }


if __name__ == "__main__":
    server.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        ssr_mode=False,
    )
