from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from uuid import uuid4

from app.config import Settings
from app.core.integration import integrate_outputs
from app.core.router import PolicyRouter
from app.core.validation import RasterValidator
from app.errors import ConflictError, SatQueryError
from app.models.gateway import SpecialistGateway
from app.reporting import build_pdf_report
from app.repository import SQLiteRepository
from app.schemas import (
    TERMINAL_STATUSES,
    AnalysisCreate,
    AnalysisRecord,
    AnalysisResult,
    AnalysisStatus,
    ExecutionPlan,
    TraceEvent,
    utc_now,
)
from app.storage import LocalArtifactStore


@dataclass
class ActiveJob:
    task: asyncio.Task[None]


class AnalysisService:
    """Bounded in-process workflow executor for local and free-demo profiles.

    Jobs are intentionally ephemeral on the no-cost CPU Space. A funded production
    profile should call the same orchestration logic from a durable external queue.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        repository: SQLiteRepository,
        validator: RasterValidator,
        router: PolicyRouter,
        gateway: SpecialistGateway,
        artifacts: LocalArtifactStore,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.validator = validator
        self.router = router
        self.gateway = gateway
        self.artifacts = artifacts
        self._semaphore = asyncio.Semaphore(settings.max_parallel_jobs)
        self._active: dict[str, ActiveJob] = {}

    async def create(
        self,
        request: AnalysisCreate,
        *,
        idempotency_key: str | None,
    ) -> AnalysisRecord:
        await self.repository.get_assets(request.asset_ids)
        safe_key = self._normalize_idempotency_key(idempotency_key) if idempotency_key else None
        if safe_key:
            existing = await self.repository.get_by_idempotency_key(safe_key)
            if existing:
                if existing.request != request:
                    raise ConflictError(
                        "Idempotency key was already used for a different request",
                        details={"analysis_id": existing.id},
                    )
                return existing

        now = utc_now()
        record = AnalysisRecord(
            id=f"anl_{uuid4().hex}",
            status=AnalysisStatus.QUEUED,
            progress=0,
            request=request,
            created_at=now,
            updated_at=now,
        )
        record = await self.repository.create_analysis(record, safe_key)
        self.enqueue(record.id)
        return record

    def enqueue(self, analysis_id: str) -> None:
        if analysis_id in self._active and not self._active[analysis_id].task.done():
            return
        task = asyncio.create_task(self.run(analysis_id), name=f"analysis:{analysis_id}")
        self._active[analysis_id] = ActiveJob(task=task)
        task.add_done_callback(lambda _: self._active.pop(analysis_id, None))

    async def run(self, analysis_id: str) -> None:
        trace: list[TraceEvent] = []
        async with self._semaphore:
            try:
                record = await self.repository.get_analysis(analysis_id)
                if record.status in TERMINAL_STATUSES:
                    return
                assets = await self.repository.get_assets(record.request.asset_ids)

                record = await self._transition(record, AnalysisStatus.VALIDATING, 0.08)
                started = time.perf_counter()
                invalid = [asset.id for asset in assets if not asset.valid]
                if invalid:
                    from app.errors import ValidationFailure

                    raise ValidationFailure(
                        "Analysis contains invalid assets", details={"asset_ids": invalid}
                    )
                trace.append(
                    TraceEvent(
                        step_id="validation",
                        task="validation",
                        tool="raster-policy-v1",
                        status="succeeded",
                        policy_reason="all asset records passed format and metadata validation",
                        duration_ms=self._duration_ms(started),
                    )
                )

                record = await self._transition(record, AnalysisStatus.PLANNING, 0.18)
                started = time.perf_counter()
                plan = self.router.plan(
                    record.request.query,
                    assets,
                    record.request.requested_tasks,
                    record.request.parameters,
                )
                task_assets: dict[str, list] = {}
                compatibility_warnings: list[str] = []
                for step in plan.steps:
                    selected = [asset for asset in assets if asset.id in step.asset_ids]
                    compatibility_warnings.extend(
                        self.validator.validate_for_task(step.task, selected)
                    )
                    task_assets[step.step_id] = selected
                trace.append(
                    TraceEvent(
                        step_id="planning",
                        task="planning",
                        tool="closed-set-policy-router",
                        model_version=plan.version,
                        status="succeeded",
                        policy_reason=f"selected {len(plan.steps)} compatible registered task(s)",
                        duration_ms=self._duration_ms(started),
                    )
                )
                record = await self._set_plan(record, plan)

                outputs = []
                for index, step in enumerate(plan.steps):
                    latest = await self.repository.get_analysis(analysis_id)
                    if latest.status == AnalysisStatus.CANCELLED:
                        return
                    progress = 0.28 + (index / max(1, len(plan.steps))) * 0.48
                    record = await self._transition(latest, AnalysisStatus.RUNNING, progress)
                    started = time.perf_counter()
                    trace.append(
                        TraceEvent(
                            step_id=step.step_id,
                            task=step.task,
                            tool=f"{step.task.value}-specialist",
                            status="started",
                            policy_reason=step.policy_reason,
                            permitted_params=step.permitted_params,
                        )
                    )
                    output = await asyncio.wait_for(
                        self.gateway.infer(
                            step,
                            task_assets[step.step_id],
                            record.request.query,
                            record.request.context,
                        ),
                        timeout=self.settings.model_timeout_seconds,
                    )
                    outputs.append(output)
                    trace.append(
                        TraceEvent(
                            step_id=step.step_id,
                            task=step.task,
                            tool=f"{step.task.value}-specialist",
                            model_version=output.model_version,
                            status="succeeded",
                            policy_reason=step.policy_reason,
                            permitted_params=step.permitted_params,
                            duration_ms=self._duration_ms(started),
                        )
                    )

                record = await self._transition(record, AnalysisStatus.INTEGRATING, 0.86)
                started = time.perf_counter()
                answer, facts, evidence, confidence, warnings = integrate_outputs(
                    outputs,
                    assets,
                    simulated=any(
                        output.model_version.startswith("demo-simulator") for output in outputs
                    ),
                )
                warnings = list(
                    dict.fromkeys(compatibility_warnings + plan.rejected_intents + warnings)
                )
                trace.append(
                    TraceEvent(
                        step_id="integration",
                        task="integration",
                        tool="structured-result-integrator",
                        status="succeeded",
                        policy_reason="merged only schema-validated specialist outputs",
                        duration_ms=self._duration_ms(started),
                    )
                )
                result = AnalysisResult(
                    answer=answer,
                    facts=facts,
                    evidence=evidence,
                    confidence=confidence,
                    trace=trace,
                    warnings=warnings,
                    provenance={
                        "planner_version": plan.version,
                        "model_backend": self.settings.model_backend,
                        "asset_hashes": {asset.id: asset.sha256 for asset in assets},
                        "model_versions": {
                            output.task.value: output.model_version for output in outputs
                        },
                        "user_context": (
                            record.request.context.model_dump(mode="json")
                            if record.request.context
                            else None
                        ),
                    },
                )
                record = record.model_copy(update={"result": result, "updated_at": utc_now()})

                report_path = self.artifacts.allocate(record.id, "satquery-report.pdf")
                try:
                    await asyncio.to_thread(build_pdf_report, record, report_path)
                    result = result.model_copy(
                        update={
                            "report_url": f"{self.settings.api_prefix}/analyses/{record.id}/report"
                        }
                    )
                except Exception as exc:  # report is non-critical after valid inference
                    result = result.model_copy(
                        update={"warnings": [*result.warnings, f"Report generation failed: {exc}"]}
                    )
                record = record.model_copy(
                    update={
                        "status": AnalysisStatus.SUCCEEDED,
                        "progress": 1.0,
                        "result": result,
                        "updated_at": utc_now(),
                    }
                )
                await self.repository.update_analysis(record)
            except asyncio.CancelledError:
                await self._mark_cancelled(analysis_id)
                raise
            except Exception as exc:
                await self._mark_failed(analysis_id, exc)

    async def cancel(self, analysis_id: str) -> AnalysisRecord:
        record = await self.repository.get_analysis(analysis_id)
        if record.status in TERMINAL_STATUSES:
            return record
        cancelled = record.model_copy(
            update={
                "status": AnalysisStatus.CANCELLED,
                "updated_at": utc_now(),
                "error_code": "cancelled_by_user",
                "error_message": "Analysis was cancelled by the user.",
            }
        )
        await self.repository.update_analysis(cancelled)
        active = self._active.get(analysis_id)
        if active and not active.task.done():
            active.task.cancel()
        return cancelled

    async def events(self, analysis_id: str) -> AsyncIterator[str]:
        last_fingerprint = ""
        sequence = 0
        while True:
            record = await self.repository.get_analysis(analysis_id)
            payload = record.model_dump_json()
            fingerprint = hashlib.sha256(payload.encode()).hexdigest()
            if fingerprint != last_fingerprint:
                sequence += 1
                event = "complete" if record.status in TERMINAL_STATUSES else "progress"
                yield f"id: {sequence}\nevent: {event}\ndata: {payload}\n\n"
                last_fingerprint = fingerprint
            if record.status in TERMINAL_STATUSES:
                break
            await asyncio.sleep(self.settings.event_poll_seconds)

    async def shutdown(self) -> None:
        active = [job.task for job in self._active.values() if not job.task.done()]
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)

    async def _transition(
        self, record: AnalysisRecord, status: AnalysisStatus, progress: float
    ) -> AnalysisRecord:
        updated = record.model_copy(
            update={"status": status, "progress": progress, "updated_at": utc_now()}
        )
        return await self.repository.update_analysis(updated)

    async def _set_plan(self, record: AnalysisRecord, plan: ExecutionPlan) -> AnalysisRecord:
        updated = record.model_copy(update={"plan": plan, "updated_at": utc_now()})
        return await self.repository.update_analysis(updated)

    async def _mark_cancelled(self, analysis_id: str) -> None:
        try:
            record = await self.repository.get_analysis(analysis_id)
            if record.status != AnalysisStatus.CANCELLED:
                await self.repository.update_analysis(
                    record.model_copy(
                        update={
                            "status": AnalysisStatus.CANCELLED,
                            "updated_at": utc_now(),
                            "error_code": "cancelled",
                            "error_message": "Analysis execution was cancelled.",
                        }
                    )
                )
        except Exception:
            return

    async def _mark_failed(self, analysis_id: str, exc: Exception) -> None:
        try:
            record = await self.repository.get_analysis(analysis_id)
            if record.status == AnalysisStatus.CANCELLED:
                return
            code = exc.code if isinstance(exc, SatQueryError) else "internal_pipeline_error"
            message = exc.message if isinstance(exc, SatQueryError) else "Analysis failed safely."
            await self.repository.update_analysis(
                record.model_copy(
                    update={
                        "status": AnalysisStatus.FAILED,
                        "updated_at": utc_now(),
                        "error_code": code,
                        "error_message": message,
                    }
                )
            )
        except Exception:
            return

    @staticmethod
    def _duration_ms(started: float) -> int:
        return max(0, int((time.perf_counter() - started) * 1000))

    @staticmethod
    def _normalize_idempotency_key(value: str) -> str:
        normalized = value.strip()
        if not 8 <= len(normalized) <= 128 or not all(32 < ord(char) < 127 for char in normalized):
            raise ConflictError("Idempotency-Key must be 8-128 printable ASCII characters")
        return normalized
