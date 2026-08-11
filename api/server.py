import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from barrikade.service.artifact_verifier import verify_bundle
from barrikade.service.assessment import AssessmentService, PipelineEvaluator
from barrikade.service.auth import TokenVerifier
from barrikade.service.config import ServiceSettings
from barrikade.service.router import ServiceComponents
from barrikade.service.router import router as assessment_v2_router
from barrikade.service.runtime import BoundedInferenceRuntime
from barrikade.service.storage import MetadataStore
from core.__version__ import __version__
from core.orchestrator import PIPipeline
from core.session import SessionNotActiveError
from core.session_orchestrator import SessionOrchestrator, create_session_orchestrator
from core.session_settings import SessionSettings
from core.settings import Settings, env_value
from models.verdicts import InputProvenance


# Configure logging at startup so that internal core.artifacts and other
# module loggers have a default stream handler configured and print progress to stdout.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


@dataclass
class AppState:
    pipeline: PIPipeline | None = None
    session_orchestrator: SessionOrchestrator | None = None
    startup_error: str | None = None
    profile: str | None = None
    bundle_version: str | None = None
    v2_components: ServiceComponents | None = None


# Stateless detect request/response


class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)
    include_diagnostics: bool = False


class DetectResponse(BaseModel):
    final_verdict: str
    decision_layer: str
    confidence_score: float
    total_processing_time_ms: float
    result: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    pipeline_initialized: bool
    session_orchestrator_initialized: bool = False
    details: str | None = None
    profile: str | None = None
    model_bundle_version: str | None = None


# Session request/response models


class CreateSessionRequest(BaseModel):
    declared_intent: str = Field(..., min_length=1, max_length=10000)
    permissions: list[str] = Field(default_factory=list)
    provenance: str = "unknown"
    delegation_chain: list[str] = Field(default_factory=list)
    risk_budget: int | None = None


class CreateSessionResponse(BaseModel):
    session_id: str


class SessionDetectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)
    provenance: str = "unknown"
    tool_name: str | None = None
    target_domain: str | None = None


class SessionDetectResponse(BaseModel):
    pipeline_result: dict[str, Any]
    session_id: str
    drift: dict[str, Any] | None = None
    risk_assessment: dict[str, Any] | None = None
    intervention: str = "none"


class SessionSummaryResponse(BaseModel):
    session_id: str
    declared_intent: str
    status: str
    created_at: str
    closed_at: str | None = None
    event_count: int
    permissions_granted: list[str]
    permissions_requested: list[str]
    external_domains_contacted: list[str]
    delegation_chain: list[str]
    risk_budget_initial: int
    risk_budget_remaining: int


class IncidentReportResponse(BaseModel):
    report: dict[str, Any]


# App lifecycle


state = AppState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime: BoundedInferenceRuntime | None = None
    try:
        service_settings = ServiceSettings.from_env()
        bundle_version = service_settings.model_bundle_version
        prepare_artifacts = True
        if service_settings.require_artifact_verification:
            if not service_settings.artifact_manifest or not service_settings.artifact_public_key:
                raise RuntimeError("artifact_verification_configuration_missing")
            verified = verify_bundle(
                service_settings.artifact_manifest,
                service_settings.artifact_public_key,
            )
            bundle_version = verified.version
            prepare_artifacts = False

        state.pipeline = PIPipeline(
            profile=service_settings.active_profile,
            prepare_artifacts=prepare_artifacts,
        )
        state.startup_error = None
        state.profile = service_settings.active_profile
        state.bundle_version = bundle_version
        log.info("Barrikade pipeline initialized")

        store = MetadataStore(service_settings.database_url)
        if service_settings.auto_migrate:
            store.migrate()
        if service_settings.token_file:
            tokens = TokenVerifier.from_file(service_settings.token_file)
        elif service_settings.local_token:
            tokens = TokenVerifier.local(service_settings.local_token)
        else:
            tokens = TokenVerifier()
        runtime = BoundedInferenceRuntime(
            workers=service_settings.inference_workers,
            queue_size=service_settings.inference_queue_size,
        )
        assessment_service = AssessmentService(
            store,
            runtime,
            PipelineEvaluator(state.pipeline),
            bundle_version,
        )
        state.v2_components = ServiceComponents(assessment_service, store, tokens)
        app.state.barrikade_v2 = state.v2_components

        # Initialise the session orchestrator, reusing the pipeline
        try:
            session_settings = SessionSettings()
            state.session_orchestrator = create_session_orchestrator(
                settings=session_settings,
                pipeline=state.pipeline,
            )
            log.info("Barrikade session orchestrator initialized")
        except Exception as exc:
            log.warning(
                "Session orchestrator initialization failed (stateless "
                "endpoint still available): %s",
                exc,
            )
            state.session_orchestrator = None

    except Exception:  # pragma: no cover
        state.pipeline = None
        state.session_orchestrator = None
        state.v2_components = None
        state.startup_error = "startup_failed"
        log.exception("Failed to initialize Barrikade pipeline")
    yield
    if runtime is not None:
        await runtime.close(ServiceSettings.from_env().shutdown_grace_seconds)


app = FastAPI(
    title="Barrikade Detection API",
    version=__version__,
    description="Production API for the Barrikade detection pipeline.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(assessment_v2_router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    if not request.url.path.startswith("/v2/"):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})
    limit_error = any(
        error.get("type") in {"too_long", "string_too_long", "list_too_long"}
        or "exceeds" in str(error.get("ctx", {}).get("error", ""))
        for error in exc.errors()
    )
    status_code = 413 if limit_error else 422
    title = "Payload too large" if limit_error else "Invalid request"
    return JSONResponse(
        status_code=status_code,
        media_type="application/problem+json",
        content={
            "type": f"urn:barrikade:problem:{title.lower().replace(' ', '_')}",
            "title": title,
            "status": status_code,
            "detail": "The assessment request did not satisfy the v2 contract.",
        },
    )


# Health endpoints


@app.get("/health/live", response_model=HealthResponse)
def live():
    return HealthResponse(status="alive")


@app.get("/health/ready", response_model=ReadinessResponse)
def ready():
    if state.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(state.startup_error or "Pipeline not initialized"),
        )

    details = (
        "Fast assessment profile loaded."
        if state.profile in {"jentic_gateway_fast", "jentic_spec"}
        else f"Full assessment profile loaded ({Settings().layer_e_judge_mode})."
    )

    return ReadinessResponse(
        status="ready",
        pipeline_initialized=True,
        session_orchestrator_initialized=state.session_orchestrator is not None,
        details=details,
        profile=state.profile,
        model_bundle_version=state.bundle_version,
    )


# Stateless detect endpoint


@app.post("/v1/detect", response_model=DetectResponse)
def detect(payload: DetectRequest):
    if state.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=state.startup_error or "Pipeline unavailable",
        )
    diagnostics_allowed = (env_value("BARRIKADE_ALLOW_DIAGNOSTICS", "true") or "true").lower()
    if payload.include_diagnostics and diagnostics_allowed not in {"1", "true", "yes", "on"}:
        raise HTTPException(status_code=403, detail="Diagnostics are disabled in this service.")

    try:
        result = state.pipeline.detect(payload.text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Pipeline artifacts are unavailable.") from exc
    except Exception as exc:  # pragma: no cover
        log.exception("Detection request failed")
        raise HTTPException(status_code=500, detail="Detection failed.") from exc

    details = result.to_dict() if payload.include_diagnostics else None
    return DetectResponse(
        final_verdict=result.final_verdict.value,
        decision_layer=result.decision_layer.value,
        confidence_score=result.confidence_score,
        total_processing_time_ms=result.total_processing_time_ms,
        result=details,
    )


# Session-aware endpoints


def _require_session_orchestrator() -> SessionOrchestrator:
    if state.session_orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="Session orchestrator not initialized",
        )
    return state.session_orchestrator


def _parse_provenance(value: str) -> InputProvenance:
    try:
        return InputProvenance(value)
    except ValueError:
        return InputProvenance.UNKNOWN


@app.post("/v1/sessions", response_model=CreateSessionResponse)
def create_session(payload: CreateSessionRequest):
    """Create a new workload session."""
    orch = _require_session_orchestrator()
    try:
        session_id = orch.start_session(
            declared_intent=payload.declared_intent,
            permissions=payload.permissions,
            provenance=_parse_provenance(payload.provenance),
            delegation_chain=payload.delegation_chain,
            risk_budget=payload.risk_budget,
        )
    except Exception as exc:
        log.exception("Session creation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return CreateSessionResponse(session_id=session_id)


@app.post(
    "/v1/sessions/{session_id}/detect",
    response_model=SessionDetectResponse,
)
def session_detect(session_id: str, payload: SessionDetectRequest):
    """Run a session-aware detection."""
    orch = _require_session_orchestrator()
    try:
        result = orch.detect_with_session(
            session_id=session_id,
            input_text=payload.text,
            provenance=_parse_provenance(payload.provenance),
            tool_name=payload.tool_name,
            target_domain=payload.target_domain,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    except SessionNotActiveError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Session detection failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SessionDetectResponse(**result.to_dict())


@app.get(
    "/v1/sessions/{session_id}",
    response_model=SessionSummaryResponse,
)
def get_session(session_id: str):
    """Get session status and summary."""
    orch = _require_session_orchestrator()
    try:
        summary = orch.get_session_summary(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return SessionSummaryResponse(**summary)


@app.post(
    "/v1/sessions/{session_id}/end",
    response_model=IncidentReportResponse,
)
def end_session(session_id: str):
    """End a session and get the incident report."""
    orch = _require_session_orchestrator()
    try:
        report = orch.end_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    except Exception as exc:
        log.exception("Session end failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IncidentReportResponse(report=report.model_dump(mode="json"))


@app.get(
    "/v1/sessions/{session_id}/report",
    response_model=IncidentReportResponse,
)
def get_session_report(session_id: str):
    """Get the incident report for a (completed) session."""
    orch = _require_session_orchestrator()
    try:
        report = orch._reporter.generate_report(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    except Exception as exc:
        log.exception("Report generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IncidentReportResponse(report=report.model_dump(mode="json"))
