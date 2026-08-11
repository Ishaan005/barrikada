import base64
import hashlib
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from barrikade.service.artifact_verifier import (
    ArtifactVerificationError,
    verify_bundle,
)


def _write_bundle(tmp_path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"verified model")
    document = {
        "bundle_version": "2026.08.1",
        "artifacts": [
            {
                "path": "model.bin",
                "size": artifact.stat().st_size,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        ],
    }
    private_key = Ed25519PrivateKey.generate()
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    document["signature"] = base64.b64encode(private_key.sign(canonical)).decode()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(document))
    public_key = tmp_path / "public.pem"
    public_key.write_bytes(
        private_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return manifest, public_key, artifact


def test_signed_bundle_is_verified(tmp_path):
    manifest, public_key, _ = _write_bundle(tmp_path)
    verified = verify_bundle(manifest, public_key)
    assert verified.version == "2026.08.1"
    assert verified.root == tmp_path


def test_artifact_tampering_is_rejected(tmp_path):
    manifest, public_key, artifact = _write_bundle(tmp_path)
    artifact.write_bytes(b"tampered model")
    with pytest.raises(ArtifactVerificationError, match="artifact_digest_mismatch"):
        verify_bundle(manifest, public_key)
