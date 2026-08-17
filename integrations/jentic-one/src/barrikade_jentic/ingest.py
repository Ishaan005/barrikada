"""Registered OpenAPI/tool-metadata assessment stage."""

from __future__ import annotations

import httpx
from jentic_one.registry.ingest.exc import IngestStageError
from jentic_one.registry.ingest.pipeline.ctx import PipelineContext
from jentic_one.registry.ingest.stages.base import BasePipelineStage

from barrikade_jentic.config import get_barrikade_config
from barrikade_jentic.events import SecurityEvent
from barrikade_jentic.extraction import ExtractionLimitError, extract_specification
from barrikade_jentic.generated.client import BarrikadeApiError
from barrikade_jentic.generated.models import AssessmentContext, AssessmentRequest
from barrikade_jentic.runtime import get_runtime


class BarrikadeSpecStage(BasePipelineStage):
    name = "BarrikadeSpecificationAssessment"

    @staticmethod
    def _emit(runtime, event_type: str, summary: str, data: dict) -> None:
        runtime.events.emit(
            SecurityEvent(
                type=event_type,
                severity="warning",
                summary=summary,
                data=data,
            )
        )

    async def _run(self, ctx: PipelineContext) -> None:
        config = get_barrikade_config(ctx.config)
        if not config.enabled:
            return
        runtime = get_runtime()
        if runtime is None:
            if config.enforcement_policy == "shadow":
                return
            raise IngestStageError("Barrikade specification scanner is unavailable")
        specification = ctx.specification.content
        if not isinstance(specification, dict):
            if config.enforcement_policy == "shadow":
                self._emit(
                    runtime,
                    "barrikade.scan_failed",
                    "Barrikade shadow policy could not inspect a specification",
                    {"reason": "invalid_specification", "enforced": False},
                )
                return
            raise IngestStageError("Barrikade could not inspect the specification")
        try:
            digest, segments = extract_specification(
                specification,
                max_text_bytes=config.max_text_bytes,
                max_segment_bytes=config.max_segment_bytes,
                max_segments=config.max_segments,
            )
        except ExtractionLimitError:
            if config.enforcement_policy == "shadow":
                self._emit(
                    runtime,
                    "barrikade.scan_failed",
                    "Barrikade shadow policy could not completely inspect a specification",
                    {"reason": "extraction_limit", "enforced": False},
                )
                return
            raise IngestStageError(
                "Barrikade rejected an incompletely inspected specification"
            ) from None
        if not segments:
            return

        request = AssessmentRequest(
            request_id=f"spec:{digest}",
            profile=config.specification_profile,
            deadline_ms=max(1, int(config.request_timeout_seconds * 1000)),
            content_sha256=digest,
            segments=segments,
            context=AssessmentContext(
                surface="specification",
                api_id=f"spec-{digest[:24]}",
            ),
        )
        try:
            assessment = await runtime.client.assess(request)
        except (BarrikadeApiError, httpx.HTTPError, OSError, TimeoutError):
            if config.enforcement_policy == "shadow":
                self._emit(
                    runtime,
                    "barrikade.scan_failed",
                    "Barrikade shadow specification assessment failed",
                    {
                        "specification_digest": digest,
                        "enforced": False,
                        "enforcement_policy": "shadow",
                    },
                )
                return
            raise IngestStageError("Barrikade specification scanner is unavailable") from None

        metadata = {
            "assessment_id": assessment.assessment_id,
            "categories": assessment.categories,
            "model_bundle_version": assessment.model_bundle_version,
            "specification_digest": digest,
            "enforced": config.enforcement_policy != "shadow",
            "enforcement_policy": config.enforcement_policy,
        }
        if assessment.override is not None:
            runtime.events.emit(
                SecurityEvent(
                    type="barrikade.override_applied",
                    severity="warning",
                    summary="Barrikade specification override applied",
                    data={**metadata, "override_id": assessment.override.id},
                )
            )
            return
        if assessment.status == "complete" and assessment.verdict == "allow":
            return

        if config.enforcement_policy == "shadow":
            event_type = (
                "barrikade.content_flagged"
                if assessment.status == "complete" and assessment.verdict == "flag"
                else "barrikade.content_blocked"
            )
            self._emit(
                runtime,
                event_type,
                "Barrikade shadow policy observed a rejected specification",
                metadata,
            )
            return

        locators = {
            segment.id: segment.locator for segment in segments if segment.locator is not None
        }
        finding_locators = sorted(
            {
                locators[segment_id]
                for finding in assessment.findings
                for segment_id in finding.segment_ids
                if segment_id in locators
            }
        )[:10]
        suffix = f"; fields={','.join(finding_locators)}" if finding_locators else ""
        raise IngestStageError(
            f"Barrikade rejected specification assessment={assessment.assessment_id}{suffix}"
        )
