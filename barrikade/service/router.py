"""FastAPI routes and safe problem mapping for Assessment API v2."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from barrikade.service.assessment import AssessmentService
from barrikade.service.auth import (
    AuthenticationError,
    AuthorizationError,
    Principal,
    TokenVerifier,
)
from barrikade.service.runtime import (
    InferenceDeadlineError,
    InferenceQueueFullError,
    InferenceUnavailableError,
)
from barrikade.service.schemas import (
    AssessmentRequest,
    AssessmentResponse,
    OverrideCreateRequest,
    OverrideResponse,
    ProblemResponse,
)
from barrikade.service.storage import (
    IdempotencyConflictError,
    MetadataStore,
    StorageUnavailableError,
)


@dataclass
class ServiceComponents:
    assessments: AssessmentService
    store: MetadataStore
    tokens: TokenVerifier


router = APIRouter(prefix="/v2", tags=["assessments-v2"])


def _components(request: Request) -> ServiceComponents:
    components = getattr(request.app.state, "barrikade_v2", None)
    if components is None:
        raise HTTPException(status_code=503, detail="Assessment service unavailable")
    return components


def _problem(
    status_code: int,
    title: str,
    detail: str,
    *,
    request_id: str | None = None,
    assessment_id: str | None = None,
) -> JSONResponse:
    body = ProblemResponse(
        type=f"urn:barrikade:problem:{title.lower().replace(' ', '_')}",
        title=title,
        status=status_code,
        detail=detail,
        request_id=request_id,
        assessment_id=assessment_id,
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(exclude_none=True),
        media_type="application/problem+json",
    )


def _authenticate(
    components: ServiceComponents, authorization: str | None, scope: str
) -> Principal | JSONResponse:
    try:
        return components.tokens.authenticate(authorization, scope)
    except AuthenticationError:
        return _problem(401, "Authentication required", "A valid service token is required.")
    except AuthorizationError:
        return _problem(403, "Forbidden", "The service token lacks the required permission.")


@router.post(
    "/assessments",
    response_model=AssessmentResponse,
    responses={
        401: {"model": ProblemResponse},
        403: {"model": ProblemResponse},
        409: {"model": ProblemResponse},
        413: {"model": ProblemResponse},
        422: {"model": ProblemResponse},
        429: {"model": ProblemResponse},
        503: {"model": ProblemResponse},
        504: {"model": ProblemResponse},
    },
)
async def create_assessment(
    payload: AssessmentRequest,
    request: Request,
    authorization: str | None = Header(default=None),
):
    components = _components(request)
    principal = _authenticate(components, authorization, "assessments:write")
    if isinstance(principal, JSONResponse):
        return principal
    try:
        return await components.assessments.assess(payload, principal)
    except IdempotencyConflictError:
        return _problem(
            status.HTTP_409_CONFLICT,
            "Idempotency conflict",
            "The request ID was already used for different content.",
            request_id=payload.request_id,
        )
    except InferenceQueueFullError:
        return _problem(429, "Inference queue full", "The inference queue is at capacity.")
    except InferenceDeadlineError:
        return _problem(504, "Deadline exceeded", "The assessment deadline was exceeded.")
    except (InferenceUnavailableError, StorageUnavailableError):
        return _problem(503, "Service unavailable", "The assessment service is unavailable.")


@router.get("/assessments/{assessment_id}", response_model=AssessmentResponse)
def get_assessment(
    assessment_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
):
    components = _components(request)
    principal = _authenticate(components, authorization, "assessments:read")
    if isinstance(principal, JSONResponse):
        return principal
    assessment = components.store.get_assessment(assessment_id)
    if assessment is None:
        return _problem(404, "Not found", "The assessment does not exist.")
    return assessment


@router.post("/overrides", response_model=OverrideResponse, status_code=201)
def create_override(
    payload: OverrideCreateRequest,
    request: Request,
    authorization: str | None = Header(default=None),
):
    components = _components(request)
    principal = _authenticate(components, authorization, "overrides:write")
    if isinstance(principal, JSONResponse):
        return principal
    try:
        return components.store.create_override(payload)
    except KeyError:
        return _problem(404, "Not found", "The assessment does not exist.")
    except IdempotencyConflictError:
        return _problem(
            409,
            "Override binding conflict",
            "The override must exactly match the assessment content, profile, and bundle.",
            assessment_id=payload.assessment_id,
        )


@router.post("/overrides/{override_id}/resolve", response_model=OverrideResponse)
def resolve_override(
    override_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
):
    return _transition_override(override_id, "resolved", request, authorization)


@router.delete("/overrides/{override_id}", response_model=OverrideResponse)
def revoke_override(
    override_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
):
    return _transition_override(override_id, "revoked", request, authorization)


def _transition_override(override_id: str, state: str, request: Request, authorization: str | None):
    components = _components(request)
    principal = _authenticate(components, authorization, "overrides:write")
    if isinstance(principal, JSONResponse):
        return principal
    try:
        return components.store.transition_override(override_id, state)
    except KeyError:
        return _problem(404, "Not found", "The active override does not exist.")
