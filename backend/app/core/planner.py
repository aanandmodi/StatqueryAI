"""Learned intent proposals; the controller, never the LLM, owns the executable DAG."""

from __future__ import annotations

from typing import Literal

import httpx
from pydantic import Field

from app.schemas import StrictModel


class IntentProposal(StrictModel):
    objectives: list[Literal["describe", "ground", "compare", "measure", "fuse"]] = Field(
        min_length=1, max_length=5
    )
    target: Literal["water", "forest", "vegetation", "building", "road", "cropland", "none"]


async def propose_intents(settings, query, assets):
    if settings.planner_backend == "policy" or settings.model_backend != "http":
        return None, "deterministic-policy (learned planner not configured)"
    headers = {"ngrok-skip-browser-warning": "1"}
    if settings.model_service_token:
        headers["Authorization"] = f"Bearer {settings.model_service_token.get_secret_value()}"
    try:
        async with httpx.AsyncClient(
            timeout=settings.planner_timeout_seconds, headers=headers
        ) as client:
            response = await client.post(
                f"{settings.service_url_for_task('single_vqa').rstrip('/')}/v1/plan",
                json={
                    "query": query,
                    "assets": [
                        {"role": asset.role.value, "modality": asset.modality.value}
                        for asset in assets
                    ],
                },
            )
            response.raise_for_status()
            if len(response.content) > 16_384:
                raise ValueError("Oversized planner response")
            body = response.json()
            proposal = IntentProposal.model_validate(body["proposal"])
            version = str(body.get("model_version", "unversioned"))[:240]
            return proposal, f"learned-intent:{version}"
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return None, "deterministic-fallback (learned planner unavailable or invalid)"
