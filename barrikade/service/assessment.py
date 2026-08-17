"""Assessment orchestration without persisting or emitting assessed text."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from barrikade.service.auth import Principal
from barrikade.service.runtime import BoundedInferenceRuntime
from barrikade.service.schemas import (
    AssessmentProfile,
    AssessmentRequest,
    AssessmentResponse,
    AssessmentStatus,
    AssessmentVerdict,
    Finding,
)
from barrikade.service.storage import IdempotencyConflictError, MetadataStore


@dataclass(frozen=True)
class SegmentOutcome:
    verdict: AssessmentVerdict
    risk_score: float
    categories: tuple[str, ...]
    deciding_layer: str


class SegmentEvaluator(Protocol):
    def evaluate(
        self, text: str, profile: AssessmentProfile, source: str | None = None
    ) -> SegmentOutcome: ...


class PipelineEvaluator:
    """Compatibility adapter around the existing A-D detector cascade."""

    def __init__(self, pipeline: object) -> None:
        self._pipeline = pipeline

    def evaluate(
        self, text: str, profile: AssessmentProfile, source: str | None = None
    ) -> SegmentOutcome:
        result = self._pipeline.detect(text, profile=profile.value, source=source)
        verdict = AssessmentVerdict(result.final_verdict.value)
        layer = result.decision_layer.value
        categories: list[str] = []
        if verdict in {AssessmentVerdict.FLAG, AssessmentVerdict.BLOCK}:
            categories.append("prompt_injection")
        layer_a = getattr(result, "layer_a_result", None)
        flags = getattr(layer_a, "flags", []) if layer_a is not None else []
        if "direction_override" in flags or "confusable_chars" in flags:
            categories.append("unicode_obfuscation")
        if "embedded_instruction" in flags:
            categories.append("encoded_instruction")

        confidence = float(result.confidence_score)
        risk_score = confidence if verdict != AssessmentVerdict.ALLOW else max(0.0, 1 - confidence)
        return SegmentOutcome(verdict, min(1.0, risk_score), tuple(categories), layer)


class AssessmentService:
    def __init__(
        self,
        store: MetadataStore,
        runtime: BoundedInferenceRuntime,
        evaluator: SegmentEvaluator,
        model_bundle_version: str,
    ) -> None:
        self.store = store
        self.runtime = runtime
        self.evaluator = evaluator
        self.model_bundle_version = model_bundle_version

    async def assess(self, request: AssessmentRequest, principal: Principal) -> AssessmentResponse:
        fingerprint = request.idempotency_fingerprint()
        existing = self.store.get_assessment_by_request(principal.storage_hash, request.request_id)
        if existing is not None:
            digest, stored_fingerprint, response = existing
            if digest != request.content_sha256 or stored_fingerprint != fingerprint:
                raise IdempotencyConflictError("request ID was already used for different content")
            return response

        started = time.perf_counter()
        outcomes = await self.runtime.run(
            request.deadline_ms,
            lambda: [
                self.evaluator.evaluate(segment.text, request.profile, segment.source.value)
                for segment in request.segments
            ],
        )

        severity = {
            AssessmentVerdict.UNKNOWN: 0,
            AssessmentVerdict.ALLOW: 1,
            AssessmentVerdict.FLAG: 2,
            AssessmentVerdict.BLOCK: 3,
        }
        verdict = max(outcomes, key=lambda outcome: severity[outcome.verdict]).verdict
        deciding = max(outcomes, key=lambda outcome: severity[outcome.verdict])
        categories = sorted({category for outcome in outcomes for category in outcome.categories})
        findings = [
            Finding(category=category, segment_ids=[segment.id])
            for segment, outcome in zip(request.segments, outcomes, strict=True)
            for category in outcome.categories
        ]
        response = AssessmentResponse(
            assessment_id=f"asm_{uuid4().hex}",
            status=AssessmentStatus.COMPLETE,
            verdict=verdict,
            risk_score=max(outcome.risk_score for outcome in outcomes),
            categories=categories,
            findings=findings,
            deciding_layer=deciding.deciding_layer,
            profile=request.profile,
            model_bundle_version=self.model_bundle_version,
            processing_time_ms=(time.perf_counter() - started) * 1000,
        )
        return self.store.save_assessment(principal.storage_hash, request, response)
