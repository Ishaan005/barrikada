"""Async HTTP client generated for Barrikade Assessment API v2."""

from __future__ import annotations

import httpx

from barrikade_jentic.generated.contract import CONTRACT_SHA256, CONTRACT_VERSION
from barrikade_jentic.generated.models import (
    AssessmentRequest,
    AssessmentResponse,
    OverrideCreateRequest,
    OverrideResponse,
)


class BarrikadeApiError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class BarrikadeClient:
    def __init__(
        self,
        endpoint: str,
        token: str,
        timeout_seconds: float,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=endpoint.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
            headers={
                "Authorization": f"Bearer {token}",
                "X-Barrikade-Contract-Version": CONTRACT_VERSION,
                "X-Barrikade-Contract-SHA256": CONTRACT_SHA256,
            },
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def status(self) -> dict:
        response = await self._client.get("/health/ready")
        if response.is_success:
            return response.json()
        raise BarrikadeApiError(response.status_code, "Barrikade service is not ready")

    async def assess(self, request: AssessmentRequest) -> AssessmentResponse:
        response = await self._client.post(
            "/v2/assessments",
            content=request.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        return self._parse(response, AssessmentResponse)

    async def get_assessment(self, assessment_id: str) -> AssessmentResponse:
        response = await self._client.get(f"/v2/assessments/{assessment_id}")
        return self._parse(response, AssessmentResponse)

    async def create_override(self, request: OverrideCreateRequest) -> OverrideResponse:
        response = await self._client.post(
            "/v2/overrides",
            content=request.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        return self._parse(response, OverrideResponse)

    async def revoke_override(self, override_id: str) -> OverrideResponse:
        response = await self._client.delete(f"/v2/overrides/{override_id}")
        return self._parse(response, OverrideResponse)

    @staticmethod
    def _parse(response: httpx.Response, model_type):
        if response.is_success:
            return model_type.model_validate(response.json())
        try:
            detail = response.json().get("detail", "Barrikade service request failed")
        except ValueError:
            detail = "Barrikade service request failed"
        raise BarrikadeApiError(response.status_code, detail)
