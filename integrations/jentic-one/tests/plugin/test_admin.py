from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from jentic_one.shared.auth.identity import Identity
from jentic_one.shared.web.deps import resolve_identity

from barrikade_jentic.admin import _actor_id, router
from barrikade_jentic.config import BarrikadeConfig
from barrikade_jentic.runtime import set_runtime


class _Client:
    async def status(self):
        return {
            "status": "ready",
            "profile": "jentic_gateway_fast",
            "model_bundle_version": "bundle-1",
        }


def _app(identity: Identity) -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/plugins/barrikade")
    app.dependency_overrides[resolve_identity] = lambda: identity
    return app


def test_admin_routes_reject_non_admin_identity():
    set_runtime(SimpleNamespace(config=BarrikadeConfig(enabled=True), client=_Client()))
    client = TestClient(_app(Identity(sub="user-1", permissions=["events:read"])))

    response = client.get("/plugins/barrikade/status")

    assert response.status_code == 403
    set_runtime(None)


def test_admin_status_is_available_to_org_admin():
    set_runtime(SimpleNamespace(config=BarrikadeConfig(enabled=True), client=_Client()))
    client = TestClient(_app(Identity(sub="admin-1", permissions=["org:admin"])))

    response = client.get("/plugins/barrikade/status")

    assert response.status_code == 200
    assert response.json()["model_bundle_version"] == "bundle-1"
    assert response.json()["plugin_version"] == "0.2.0"
    assert response.json()["assessment_api_version"] == "2"
    assert response.json()["enforcement_policy"] == "balanced"
    set_runtime(None)


def test_override_actor_is_pseudonymous_not_email_shaped():
    actor = _actor_id(Identity(sub="admin@example.test", email="admin@example.test"))
    assert actor.startswith("jentic:")
    assert "admin@example.test" not in actor
