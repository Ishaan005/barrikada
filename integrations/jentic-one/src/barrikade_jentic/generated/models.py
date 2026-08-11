"""Generated v2 API models. Do not add Jentic-specific behavior here."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Category = Literal["prompt_injection", "unicode_obfuscation", "encoded_instruction"]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AssessmentContext(ContractModel):
    surface: Literal["runtime_response", "specification"]
    upstream_status: int | None = None
    content_type: str | None = None
    execution_id: str | None = None
    api_id: str | None = None
    operation_id: str | None = None


class TextSegment(ContractModel):
    id: str
    text: str
    source: Literal[
        "json_key",
        "json_value",
        "text",
        "html",
        "xml",
        "yaml",
        "problem",
        "specification",
    ]
    locator: str | None = Field(
        default=None,
        max_length=512,
        pattern=r"^(?:\$[.\[\]A-Za-z0-9_:-]*|locator_sha256:[0-9a-f]{64})$",
    )
    sha256: str


class AssessmentRequest(ContractModel):
    request_id: str
    profile: Literal["jentic_gateway_fast", "jentic_spec"]
    deadline_ms: int
    content_sha256: str
    segments: list[TextSegment]
    context: AssessmentContext | None = None


class Finding(ContractModel):
    category: Category
    segment_ids: list[str]


class AppliedOverride(ContractModel):
    id: str
    actor_id: str
    reason: str
    created_at: str


class AssessmentResponse(ContractModel):
    assessment_id: str
    status: Literal["complete", "partial"]
    verdict: Literal["allow", "flag", "block", "unknown"]
    risk_score: float
    categories: list[Category]
    findings: list[Finding]
    deciding_layer: Literal["A", "B", "C", "D", "E"]
    profile: Literal["jentic_gateway_fast", "jentic_spec"]
    model_bundle_version: str
    processing_time_ms: float
    override: AppliedOverride | None = None


class OverrideCreateRequest(ContractModel):
    assessment_id: str
    content_sha256: str
    profile: Literal["jentic_gateway_fast", "jentic_spec"]
    model_bundle_version: str
    actor_id: str
    reason: str


class OverrideResponse(OverrideCreateRequest):
    id: str
    state: Literal["active", "resolved", "revoked"]
    created_at: str
    resolved_at: str | None = None
    revoked_at: str | None = None


class ProblemResponse(ContractModel):
    type: str = "about:blank"
    title: str = "Barrikade API error"
    status: int
    detail: str
    request_id: str | None = None
    assessment_id: str | None = None
    extensions: dict[str, Any] = Field(default_factory=dict)
