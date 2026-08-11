"""Offline verification for signed, digest-pinned runtime bundles."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ArtifactVerificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class VerifiedBundle:
    version: str
    root: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_unsigned_manifest(document: dict[str, Any]) -> bytes:
    unsigned = {key: value for key, value in document.items() if key != "signature"}
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()


def verify_bundle(manifest_path: Path, public_key_path: Path) -> VerifiedBundle:
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactVerificationError("artifact_manifest_invalid") from exc

    version = document.get("bundle_version")
    artifacts = document.get("artifacts")
    signature = document.get("signature")
    if not isinstance(version, str) or not version or not isinstance(artifacts, list):
        raise ArtifactVerificationError("artifact_manifest_invalid")
    if not isinstance(signature, str):
        raise ArtifactVerificationError("artifact_signature_missing")

    try:
        from cryptography.hazmat.primitives import serialization  # noqa: PLC0415
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # noqa: PLC0415
            Ed25519PublicKey,
        )

        key = serialization.load_pem_public_key(public_key_path.read_bytes())
        if not isinstance(key, Ed25519PublicKey):
            raise TypeError("not an Ed25519 key")
        key.verify(
            base64.b64decode(signature, validate=True), _canonical_unsigned_manifest(document)
        )
    except Exception as exc:
        raise ArtifactVerificationError("artifact_signature_invalid") from exc

    root = manifest_path.resolve().parent
    for item in artifacts:
        if not isinstance(item, dict):
            raise ArtifactVerificationError("artifact_manifest_invalid")
        relative = item.get("path")
        expected_digest = item.get("sha256")
        expected_size = item.get("size")
        if not isinstance(relative, str) or not isinstance(expected_digest, str):
            raise ArtifactVerificationError("artifact_manifest_invalid")
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ArtifactVerificationError("artifact_path_invalid") from exc
        if candidate.is_symlink() or not candidate.is_file():
            raise ArtifactVerificationError("artifact_missing")
        if not isinstance(expected_size, int) or candidate.stat().st_size != expected_size:
            raise ArtifactVerificationError("artifact_size_mismatch")
        digest = _sha256_file(candidate)
        if digest != expected_digest:
            raise ArtifactVerificationError("artifact_digest_mismatch")
    return VerifiedBundle(version=version, root=root)
