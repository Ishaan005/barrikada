"""Jentic Broker wrapper that assesses buffered upstream responses."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Awaitable, Callable
from dataclasses import replace

import httpx
from fastapi import Response
from jentic_one.broker.core.exceptions import (
    ActionDeniedError,
    AgentDirective,
    RunnerUnavailableError,
)
from jentic_one.shared.broker.broker import Broker
from jentic_one.shared.broker.execution import (
    ErrorOrigin,
    ExecutionContext,
    ExecutionOutcome,
    RunnerRequest,
    RunnerResult,
    StreamingOutcome,
    StreamingUpstreamRunner,
)
from jentic_one.shared.broker.schemas import ExecuteRequestContext

from barrikade_jentic.config import BarrikadeConfig
from barrikade_jentic.events import SecurityEvent
from barrikade_jentic.extraction import extract_response
from barrikade_jentic.generated.client import BarrikadeApiError, BarrikadeClient
from barrikade_jentic.generated.models import AssessmentContext, AssessmentRequest

_STABLE_ID = re.compile(r"^[\w.:-]{1,128}$")


def _safe_correlation_id(value: str | None) -> str | None:
    if value is None:
        return None
    if _STABLE_ID.fullmatch(value):
        return value
    return f"sha256:{hashlib.sha256(value.encode()).hexdigest()}"


def _identity_encoding(headers: dict[str, str]) -> dict[str, str]:
    result = {key: value for key, value in headers.items() if key.lower() != "accept-encoding"}
    result["Accept-Encoding"] = "identity"
    return result


class BarrikadeBroker:
    def __init__(
        self,
        delegate: Broker,
        client: BarrikadeClient,
        config: BarrikadeConfig,
        event_queue,
    ) -> None:
        self._delegate = delegate
        self._client = client
        self._config = config
        self._events = event_queue

    async def execute(self, request: RunnerRequest, context: ExecutionContext) -> ExecutionOutcome:
        protected_request = replace(request, headers=_identity_encoding(request.headers))
        outcome = await self._delegate.execute(protected_request, context)
        extraction = extract_response(
            body=outcome.result.body,
            headers=outcome.result.headers,
            content_type=outcome.result.content_type,
            max_text_bytes=self._config.max_text_bytes,
            max_segment_bytes=self._config.max_segment_bytes,
            max_segments=self._config.max_segments,
            decompression_ratio_limit=self._config.decompression_ratio_limit,
        )
        if extraction.applicability == "not_applicable":
            return outcome
        if extraction.applicability == "incomplete":
            self._emit(
                "barrikade.content_blocked",
                "error",
                "Barrikade blocked an incompletely inspected response",
                context,
                {"upstream_status": outcome.result.status_code, "reason": "incomplete"},
            )
            raise self._blocked_error(
                upstream_status=outcome.result.status_code,
                assessment_id=None,
                categories=[],
                bundle_version=None,
                incomplete=True,
            )

        assessment_request = AssessmentRequest(
            request_id=(
                "runtime:"
                f"{hashlib.sha256(context.execution_id.encode()).hexdigest()[:32]}:"
                f"{extraction.content_sha256}"
            ),
            profile=self._config.runtime_profile,
            deadline_ms=max(1, int(self._config.request_timeout_seconds * 1000)),
            content_sha256=extraction.content_sha256,
            segments=extraction.segments,
            context=AssessmentContext(
                surface="runtime_response",
                upstream_status=outcome.result.status_code,
                content_type=(outcome.result.content_type or "")[:255] or None,
                execution_id=_safe_correlation_id(context.execution_id),
                api_id=_safe_correlation_id(context.toolkit_id),
                operation_id=_safe_correlation_id(context.operation_id),
            ),
        )
        try:
            assessment = await self._client.assess(assessment_request)
        except (BarrikadeApiError, httpx.HTTPError, TimeoutError):
            self._emit(
                "barrikade.scan_failed",
                "error",
                "Barrikade response assessment failed",
                context,
                {"upstream_status": outcome.result.status_code},
            )
            raise RunnerUnavailableError(
                "The response security scanner is temporarily unavailable.",
                type="barrikade_scanner_unavailable",
                origin=ErrorOrigin.BROKER,
                directive=AgentDirective(
                    strategy="retry",
                    parameters={},
                    human_readable_instruction="Retry after the security scanner recovers.",
                ),
            ) from None

        metadata = {
            "assessment_id": assessment.assessment_id,
            "categories": assessment.categories,
            "model_bundle_version": assessment.model_bundle_version,
            "upstream_status": outcome.result.status_code,
        }
        if assessment.status != "complete" or assessment.verdict == "unknown":
            self._emit(
                "barrikade.content_blocked",
                "error",
                "Barrikade blocked an incompletely inspected response",
                context,
                metadata,
            )
            raise self._blocked_error(
                upstream_status=outcome.result.status_code,
                assessment_id=assessment.assessment_id,
                categories=assessment.categories,
                bundle_version=assessment.model_bundle_version,
                incomplete=True,
            )
        if assessment.verdict == "block":
            self._emit(
                "barrikade.content_blocked",
                "error",
                "Barrikade blocked a tool response",
                context,
                metadata,
            )
            raise self._blocked_error(
                upstream_status=outcome.result.status_code,
                assessment_id=assessment.assessment_id,
                categories=assessment.categories,
                bundle_version=assessment.model_bundle_version,
            )
        if assessment.verdict == "flag":
            self._emit(
                "barrikade.content_flagged",
                "warning",
                "Barrikade flagged a tool response",
                context,
                metadata,
            )

        normalized_result = RunnerResult(
            status_code=outcome.result.status_code,
            body=extraction.body,
            headers=extraction.headers,
            content_type=outcome.result.content_type,
            duration_ms=outcome.result.duration_ms,
        )
        return replace(outcome, result=normalized_result)

    async def execute_streaming(
        self,
        runner: StreamingUpstreamRunner,
        request: RunnerRequest,
        ctx_req: ExecuteRequestContext,
        execution_id: str,
        *,
        transfer_deadline_s: float,
        background_callback: Callable[[StreamingOutcome], Awaitable[None]] | None = None,
    ) -> Response:
        del runner, request, ctx_req, execution_id, transfer_deadline_s, background_callback
        raise ActionDeniedError(
            "Streaming responses are unavailable while response security scanning is enabled.",
            type="barrikade_streaming_unsupported",
            origin=ErrorOrigin.BROKER,
            directive=AgentDirective(
                strategy="fatal",
                parameters={},
                human_readable_instruction="Use the buffered tool execution path.",
            ),
        )

    def _emit(
        self,
        event_type: str,
        severity: str,
        summary: str,
        context: ExecutionContext,
        data: dict,
    ) -> None:
        self._events.emit(
            SecurityEvent(
                type=event_type,
                severity=severity,
                summary=summary,
                data=data,
                execution_id=context.execution_id,
                trace_id=context.trace_id,
            )
        )

    @staticmethod
    def _blocked_error(
        *,
        upstream_status: int,
        assessment_id: str | None,
        categories: list[str],
        bundle_version: str | None,
        incomplete: bool = False,
    ) -> ActionDeniedError:
        extra = {
            "assessment_id": assessment_id,
            "categories": categories,
            "model_bundle_version": bundle_version,
            "upstream_status": upstream_status,
        }
        return ActionDeniedError(
            (
                "The upstream response could not be completely inspected."
                if incomplete
                else "The upstream response was blocked by the response security policy."
            ),
            type="barrikade_prompt_injection",
            extra={key: value for key, value in extra.items() if value is not None},
            origin=ErrorOrigin.BROKER,
            directive=AgentDirective(
                strategy="fatal",
                parameters={"assessment_id": assessment_id} if assessment_id else {},
                human_readable_instruction="Do not use content from this tool response.",
            ),
        )


def broker_factory(runtime):
    from jentic_one.broker.services.execution.service import default_broker  # noqa: PLC0415

    def factory(runner):
        return BarrikadeBroker(
            default_broker(runner),
            runtime.client,
            runtime.config,
            runtime.events,
        )

    return factory


def install_broker_factory(app, ctx) -> None:
    del ctx
    from barrikade_jentic.runtime import get_runtime  # noqa: PLC0415

    runtime = get_runtime()
    if runtime is not None:
        app.state.broker_factory = broker_factory(runtime)
