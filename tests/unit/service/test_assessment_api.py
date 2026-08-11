import hashlib
import sqlite3

from fastapi.testclient import TestClient

from api.server import app
from barrikade.service.assessment import AssessmentService, SegmentOutcome
from barrikade.service.auth import TokenVerifier
from barrikade.service.router import ServiceComponents
from barrikade.service.runtime import BoundedInferenceRuntime
from barrikade.service.schemas import AssessmentVerdict
from barrikade.service.storage import MetadataStore


TOKEN = "test-service-token"


class _Evaluator:
    def evaluate(self, text, profile):
        if "attack" in text:
            return SegmentOutcome(
                AssessmentVerdict.BLOCK,
                0.99,
                ("prompt_injection",),
                "D",
            )
        if "uncertain" in text:
            return SegmentOutcome(
                AssessmentVerdict.FLAG,
                0.65,
                ("prompt_injection",),
                "C",
            )
        return SegmentOutcome(AssessmentVerdict.ALLOW, 0.01, (), "B")


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _payload(text="hello", request_id="req-1"):
    return {
        "request_id": request_id,
        "profile": "jentic_gateway_fast",
        "deadline_ms": 1000,
        "content_sha256": _digest(text),
        "segments": [
            {
                "id": "segment-1",
                "text": text,
                "source": "text",
                "locator": "$body[0]",
                "sha256": _digest(text),
            }
        ],
        "context": {
            "surface": "runtime_response",
            "upstream_status": 200,
            "content_type": "text/plain",
        },
    }


def _client(tmp_path):
    database_path = tmp_path / "metadata.db"
    store = MetadataStore(f"sqlite:///{database_path}")
    store.migrate()
    runtime = BoundedInferenceRuntime(workers=1, queue_size=2)
    service = AssessmentService(store, runtime, _Evaluator(), "bundle-test")
    app.state.barrikade_v2 = ServiceComponents(service, store, TokenVerifier.local(TOKEN))
    return TestClient(app), database_path


def _headers():
    return {"Authorization": f"Bearer {TOKEN}"}


def test_assessment_requires_authentication(tmp_path):
    client, _ = _client(tmp_path)
    response = client.post("/v2/assessments", json=_payload())
    assert response.status_code == 401
    assert response.headers["content-type"].startswith("application/problem+json")
    assert "hello" not in response.text


def test_assessment_is_idempotent_and_metadata_only(tmp_path):
    client, database_path = _client(tmp_path)
    text = "unique raw canary content"
    payload = _payload(text)

    first = client.post("/v2/assessments", json=payload, headers=_headers())
    second = client.post("/v2/assessments", json=payload, headers=_headers())

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["assessment_id"] == second.json()["assessment_id"]
    assert first.json()["verdict"] == "allow"
    assert text.encode() not in database_path.read_bytes()

    with sqlite3.connect(database_path) as connection:
        metadata = connection.execute(
            "SELECT segment_metadata_json FROM barrikade_assessments"
        ).fetchone()[0]
    assert _digest(text) in metadata
    assert text not in metadata


def test_request_id_reuse_with_different_content_returns_409(tmp_path):
    client, _ = _client(tmp_path)
    assert client.post("/v2/assessments", json=_payload(), headers=_headers()).status_code == 200
    changed_text = "different raw canary"
    response = client.post(
        "/v2/assessments",
        json=_payload(changed_text, request_id="req-1"),
        headers=_headers(),
    )
    assert response.status_code == 409
    assert changed_text not in response.text


def test_block_response_references_segment_without_matching_text(tmp_path):
    client, _ = _client(tmp_path)
    text = "attack instructions"
    response = client.post("/v2/assessments", json=_payload(text), headers=_headers())
    body = response.json()
    assert body["verdict"] == "block"
    assert body["categories"] == ["prompt_injection"]
    assert body["findings"] == [{"category": "prompt_injection", "segment_ids": ["segment-1"]}]
    assert text not in response.text


def test_segment_and_locator_limits_return_413_without_echo(tmp_path):
    client, _ = _client(tmp_path)
    text = "x" * (64 * 1024 + 1)
    response = client.post("/v2/assessments", json=_payload(text), headers=_headers())
    assert response.status_code == 413
    assert text not in response.text

    payload = _payload()
    payload["segments"][0]["locator"] = "l" * 513
    response = client.post("/v2/assessments", json=payload, headers=_headers())
    assert response.status_code == 413
    assert "l" * 513 not in response.text


def test_override_requires_exact_binding_and_is_returned_on_read(tmp_path):
    client, _ = _client(tmp_path)
    created = client.post("/v2/assessments", json=_payload("attack"), headers=_headers()).json()
    override = {
        "assessment_id": created["assessment_id"],
        "content_sha256": _digest("attack"),
        "profile": "jentic_gateway_fast",
        "model_bundle_version": "bundle-test",
        "actor_id": "admin-1",
        "reason": "Reviewed the source out of band",
    }
    created_override = client.post("/v2/overrides", json=override, headers=_headers())
    assert created_override.status_code == 201

    read = client.get(f"/v2/assessments/{created['assessment_id']}", headers=_headers()).json()
    assert read["override"]["id"] == created_override.json()["id"]

    wrong = {**override, "content_sha256": _digest("modified")}
    conflict = client.post("/v2/overrides", json=wrong, headers=_headers())
    assert conflict.status_code == 409

    revoked = client.delete(f"/v2/overrides/{created_override.json()['id']}", headers=_headers())
    assert revoked.status_code == 200
    assert revoked.json()["state"] == "revoked"
