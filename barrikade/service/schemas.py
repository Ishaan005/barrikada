"""Closed, content-safe Pydantic contract for Assessment API v2."""

from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


MAX_SEGMENTS = 256
MAX_SEGMENT_BYTES = 64 * 1024
MAX_AGGREGATE_BYTES = 2 * 1024 * 1024
MAX_LOCATOR_CHARS = 512
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

StableId = Annotated[str, StringConstraints(min_length=1, max_length=128, pattern=r"^[\w.:-]+$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
SafeLocator = Annotated[
    str,
    StringConstraints(
        max_length=MAX_LOCATOR_CHARS,
        pattern=r"^(?:\$[.\[\]A-Za-z0-9_:-]*|locator_sha256:[0-9a-f]{64})$",
    ),
]


class ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AssessmentProfile(str, Enum):
    JENTIC_GATEWAY_FAST = "jentic_gateway_fast"
    JENTIC_SPEC = "jentic_spec"


class SegmentSource(str, Enum):
    JSON_KEY = "json_key"
    JSON_VALUE = "json_value"
    TEXT = "text"
    HTML = "html"
    XML = "xml"
    YAML = "yaml"
    PROBLEM = "problem"
    SPECIFICATION = "specification"
    SPECIFICATION_TITLE = "specification_title"
    SPECIFICATION_SUMMARY = "specification_summary"
    SPECIFICATION_DESCRIPTION = "specification_description"
    SPECIFICATION_DEFAULT = "specification_default"
    SPECIFICATION_EXAMPLE = "specification_example"
    SPECIFICATION_EXTERNAL_DOCS = "specification_external_docs"
    SPECIFICATION_TAGS = "specification_tags"


class AssessmentSurface(str, Enum):
    RUNTIME_RESPONSE = "runtime_response"
    SPECIFICATION = "specification"


class AssessmentStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"


class AssessmentVerdict(str, Enum):
    ALLOW = "allow"
    FLAG = "flag"
    BLOCK = "block"
    UNKNOWN = "unknown"


class FindingCategory(str, Enum):
    PROMPT_INJECTION = "prompt_injection"
    UNICODE_OBFUSCATION = "unicode_obfuscation"
    ENCODED_INSTRUCTION = "encoded_instruction"


class AssessmentContext(ClosedModel):
    surface: AssessmentSurface
    upstream_status: int | None = Field(default=None, ge=100, le=599)
    content_type: str | None = Field(default=None, max_length=255)
    execution_id: StableId | None = None
    api_id: StableId | None = None
    operation_id: StableId | None = None


class TextSegment(ClosedModel):
    id: StableId
    text: str = Field(min_length=1)
    source: SegmentSource
    locator: SafeLocator | None = None
    sha256: Sha256

    @model_validator(mode="after")
    def validate_bytes_and_digest(self) -> "TextSegment":
        encoded = self.text.encode("utf-8")
        if len(encoded) > MAX_SEGMENT_BYTES:
            raise ValueError(f"segment text exceeds {MAX_SEGMENT_BYTES} UTF-8 bytes")
        actual = hashlib.sha256(encoded).hexdigest()
        if actual != self.sha256:
            raise ValueError("segment sha256 does not match text")
        return self


class AssessmentRequest(ClosedModel):
    request_id: StableId
    profile: AssessmentProfile
    deadline_ms: int = Field(ge=1, le=30_000)
    content_sha256: Sha256
    segments: list[TextSegment] = Field(min_length=1, max_length=MAX_SEGMENTS)
    context: AssessmentContext | None = None

    @model_validator(mode="after")
    def validate_segments(self) -> "AssessmentRequest":
        segment_ids = [segment.id for segment in self.segments]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("segment IDs must be unique")
        aggregate = sum(len(segment.text.encode("utf-8")) for segment in self.segments)
        if aggregate > MAX_AGGREGATE_BYTES:
            raise ValueError(f"aggregate text exceeds {MAX_AGGREGATE_BYTES} UTF-8 bytes")
        return self

    def idempotency_fingerprint(self) -> str:
        canonical = {
            "profile": self.profile.value,
            "content_sha256": self.content_sha256,
            "segments": [
                {
                    "id": segment.id,
                    "source": segment.source.value,
                    "locator": segment.locator,
                    "sha256": segment.sha256,
                }
                for segment in self.segments
            ],
            "context": self.context.model_dump(mode="json") if self.context else None,
        }
        payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


class Finding(ClosedModel):
    category: FindingCategory
    segment_ids: list[StableId] = Field(min_length=1, max_length=MAX_SEGMENTS)


class AppliedOverride(ClosedModel):
    id: StableId
    actor_id: StableId
    reason: str = Field(min_length=1, max_length=512)
    created_at: str


class AssessmentResponse(ClosedModel):
    assessment_id: StableId
    status: AssessmentStatus
    verdict: AssessmentVerdict
    risk_score: float = Field(ge=0.0, le=1.0)
    categories: list[FindingCategory]
    findings: list[Finding]
    deciding_layer: Literal["A", "B", "C", "D", "E"]
    profile: AssessmentProfile
    model_bundle_version: str = Field(min_length=1, max_length=128)
    processing_time_ms: float = Field(ge=0.0)
    override: AppliedOverride | None = None


class OverrideCreateRequest(ClosedModel):
    assessment_id: StableId
    content_sha256: Sha256
    profile: AssessmentProfile
    model_bundle_version: str = Field(min_length=1, max_length=128)
    actor_id: StableId
    reason: str = Field(min_length=1, max_length=512)


class OverrideResponse(ClosedModel):
    id: StableId
    assessment_id: StableId
    content_sha256: Sha256
    profile: AssessmentProfile
    model_bundle_version: str
    actor_id: StableId
    reason: str
    state: Literal["active", "resolved", "revoked"]
    created_at: str
    resolved_at: str | None = None
    revoked_at: str | None = None


class ProblemResponse(ClosedModel):
    type: str
    title: str
    status: int
    detail: str
    request_id: str | None = None
    assessment_id: str | None = None
