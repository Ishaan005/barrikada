"""Exceptional admin facade; normal enablement requires no API calls."""

from __future__ import annotations

import hashlib
from typing import Literal

import httpx
from fastapi import APIRouter, HTTPException
from jentic_one.shared.auth.identity import Identity
from jentic_one.shared.web.deps import get_current_identity
from pydantic import BaseModel, ConfigDict, Field

from barrikade_jentic import __version__ as plugin_version
from barrikade_jentic.generated.client import BarrikadeApiError
from barrikade_jentic.generated.contract import CONTRACT_SHA256, CONTRACT_VERSION
from barrikade_jentic.generated.models import OverrideCreateRequest
from barrikade_jentic.runtime import get_runtime

router = APIRouter()
_admin = get_current_identity(required_permissions=["org:admin"])


class AdminModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SpecOverrideRequest(AdminModel):
    assessment_id: str = Field(min_length=1, max_length=128)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile: Literal["jentic_spec"] = "jentic_spec"
    model_bundle_version: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=1, max_length=512)


def _runtime():
    runtime = get_runtime()
    if runtime is None:
        raise HTTPException(status_code=503, detail="Barrikade is not enabled in this process.")
    return runtime


def _actor_id(identity: Identity) -> str:
    """Persist a stable pseudonymous actor ID, never an email-shaped subject."""
    return f"jentic:{hashlib.sha256(identity.sub.encode()).hexdigest()}"


def _core_error(exc: Exception, operation: str) -> HTTPException:
    if isinstance(exc, BarrikadeApiError) and exc.status_code in {404, 409, 422}:
        return HTTPException(
            status_code=exc.status_code,
            detail=f"The Barrikade {operation} request was rejected.",
        )
    return HTTPException(
        status_code=503,
        detail=f"The Barrikade {operation} service is unavailable.",
    )


@router.get("/status")
async def status(identity: Identity = _admin):
    del identity
    runtime = _runtime()
    plugin_metadata = {
        "enabled": True,
        "plugin_version": plugin_version,
        "assessment_api_version": CONTRACT_VERSION,
        "assessment_contract_sha256": CONTRACT_SHA256,
        "enforcement_policy": runtime.config.enforcement_policy,
    }
    try:
        readiness = await runtime.client.status()
    except (BarrikadeApiError, httpx.HTTPError):
        return {
            **plugin_metadata,
            "healthy": False,
            "profile": runtime.config.runtime_profile,
        }
    return {
        **plugin_metadata,
        "healthy": readiness.get("status") == "ready",
        "profile": readiness.get("profile", runtime.config.runtime_profile),
        "model_bundle_version": readiness.get("model_bundle_version"),
    }


@router.get("/assessments/{assessment_id}")
async def assessment(assessment_id: str, identity: Identity = _admin):
    del identity
    try:
        result = await _runtime().client.get_assessment(assessment_id)
    except (BarrikadeApiError, httpx.HTTPError) as exc:
        raise _core_error(exc, "assessment") from exc
    return result.model_dump(mode="json")


@router.post("/spec-overrides", status_code=201)
async def create_spec_override(payload: SpecOverrideRequest, identity: Identity = _admin):
    request = OverrideCreateRequest(
        assessment_id=payload.assessment_id,
        content_sha256=payload.content_sha256,
        profile=payload.profile,
        model_bundle_version=payload.model_bundle_version,
        actor_id=_actor_id(identity),
        reason=payload.reason,
    )
    try:
        result = await _runtime().client.create_override(request)
    except (BarrikadeApiError, httpx.HTTPError) as exc:
        raise _core_error(exc, "override") from exc
    return result.model_dump(mode="json")


@router.delete("/spec-overrides/{override_id}")
async def revoke_spec_override(override_id: str, identity: Identity = _admin):
    del identity
    try:
        result = await _runtime().client.revoke_override(override_id)
    except (BarrikadeApiError, httpx.HTTPError) as exc:
        raise _core_error(exc, "override") from exc
    return result.model_dump(mode="json")
