"""Create the signed, digest-pinned manifest consumed by the Core release image."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


REQUIRED_FAST_ARTIFACTS = (
    "layer_b/embeddings/prompt_encoder_onnx/onnx/model.onnx",
    "layer_b/embeddings/faiss_index.bin",
    "layer_b/embeddings/centroids.npy",
    "layer_c/classifier.onnx",
    "layer_c/encoder_onnx/onnx/model.onnx",
    "layer_d/onnx/model.onnx",
    "layer_d/onnx/tokenizer.json",
    "layer_d/onnx/jentic-reviewed-calibration.json",
    "parity/layer_b.json",
    "parity/layer_c.json",
    "parity/layer_d.json",
)


def _canonical(document: dict) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_parity_reports(root: Path) -> None:
    for layer in ("layer_b", "layer_c", "layer_d"):
        path = root / "parity" / f"{layer}.json"
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            passed = (
                report["passed"] is True
                and float(report["verdict_disagreement_rate"]) <= 0.005
                and float(report["candidate_false_block_rate"])
                <= float(report["reference_false_block_rate"])
                and float(report["candidate_recall"]) >= float(report["reference_recall"])
                and (
                    float(report["candidate_p95_ms"]) < float(report["reference_p95_ms"])
                    or float(report["candidate_recall"]) > float(report["reference_recall"])
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid ONNX parity report: {path}") from exc
        if not passed:
            raise SystemExit(f"ONNX parity gate failed: {path}")


def build_manifest(bundle: Path, bundle_version: str, private_key_path: Path) -> dict:
    root = bundle.resolve()
    for relative in REQUIRED_FAST_ARTIFACTS:
        if not (root / relative).is_file():
            raise SystemExit(f"fast-profile artifact is missing: {relative}")
    _validate_parity_reports(root)

    artifacts = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        artifacts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    unsigned = {
        "schema_version": 1,
        "bundle_version": bundle_version,
        "profiles": {
            "jentic_gateway_fast": bundle_version,
            "jentic_spec": bundle_version,
        },
        "signing": {"algorithm": "Ed25519"},
        "artifacts": artifacts,
    }
    key = serialization.load_pem_private_key(private_key_path.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise SystemExit("bundle signing key must be an Ed25519 private key")
    return {
        **unsigned,
        "signature": base64.b64encode(key.sign(_canonical(unsigned))).decode("ascii"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-version", required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.bundle, args.bundle_version, args.private_key)
    destination = args.bundle / "manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
