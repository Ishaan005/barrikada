"""Download and verify the pinned inputs for the Jentic corpus review pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import BinaryIO

INTEGRATION_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MANIFEST = INTEGRATION_ROOT / "evaluation" / "sources" / "jentic-corpus-sources.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "build" / "jentic-corpus-sources" / "raw"
SHA256 = re.compile(r"[0-9a-f]{64}")
REVISION = re.compile(r"[0-9a-f]{40}")


class SourceManifestError(ValueError):
    """Raised when a source manifest is not safe or reproducible."""


def safe_destination(root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise SourceManifestError(f"unsafe source path: {relative_path}")
    destination = (root / candidate).resolve()
    try:
        destination.relative_to(root.resolve())
    except ValueError as exc:
        raise SourceManifestError(f"source path escapes output directory: {relative_path}") from exc
    return destination


def _digest_stream(handle: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _existing_digest(path: Path) -> tuple[str, int] | None:
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        return _digest_stream(handle)


def load_source_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("sources"), list):
        raise SourceManifestError("unsupported source manifest schema")
    source_ids: set[str] = set()
    file_paths: set[str] = set()
    for source in manifest["sources"]:
        required = {
            "id",
            "homepage",
            "revision",
            "license",
            "roles",
            "review_status",
            "files",
        }
        if set(source) != required:
            raise SourceManifestError(f"source {source.get('id', '<unknown>')} has invalid fields")
        if not isinstance(source["id"], str) or source["id"] in source_ids:
            raise SourceManifestError("source IDs must be unique strings")
        source_ids.add(source["id"])
        if not str(source["homepage"]).startswith("https://"):
            raise SourceManifestError(f"source {source['id']} homepage must use HTTPS")
        if not REVISION.fullmatch(str(source["revision"])):
            raise SourceManifestError(f"source {source['id']} must pin a full commit revision")
        if not source["license"] or source["review_status"] != "approved_for_draft_pool":
            raise SourceManifestError(f"source {source['id']} is not approved for the draft pool")
        if not source["roles"] or not source["files"]:
            raise SourceManifestError(f"source {source['id']} has no roles or files")
        for asset in source["files"]:
            if set(asset) != {"path", "url", "sha256", "bytes"}:
                raise SourceManifestError(f"source {source['id']} has invalid file fields")
            if asset["path"] in file_paths:
                raise SourceManifestError(f"duplicate asset path: {asset['path']}")
            file_paths.add(asset["path"])
            if not str(asset["url"]).startswith("https://"):
                raise SourceManifestError(f"asset {asset['path']} must use HTTPS")
            if not SHA256.fullmatch(str(asset["sha256"])):
                raise SourceManifestError(f"asset {asset['path']} has an invalid SHA-256")
            if not isinstance(asset["bytes"], int) or asset["bytes"] <= 0:
                raise SourceManifestError(f"asset {asset['path']} has an invalid byte count")
    return manifest


def download_verified(
    url: str, destination: Path, expected_digest: str, expected_size: int
) -> tuple[int, bool]:
    existing = _existing_digest(destination)
    if existing and existing == (expected_digest, expected_size):
        return existing[1], True

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Barrikade-corpus-builder/1"})
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        digest = hashlib.sha256()
        size = 0
        with (
            os.fdopen(descriptor, "wb") as output,
            urllib.request.urlopen(  # noqa: S310
                request, timeout=60
            ) as response,
        ):
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
                if size > expected_size:
                    raise SourceManifestError(f"asset exceeds pinned size: {destination.name}")
        actual_digest = digest.hexdigest()
        if actual_digest != expected_digest or size != expected_size:
            raise SourceManifestError(
                f"asset mismatch for {destination.name}: expected {expected_digest}/{expected_size}, "
                f"received {actual_digest}/{size}"
            )
        temporary.replace(destination)
        return size, False
    finally:
        temporary.unlink(missing_ok=True)


def acquire_sources(manifest_path: Path, output_dir: Path) -> dict:
    manifest = load_source_manifest(manifest_path)
    assets: list[dict] = []
    for source in manifest["sources"]:
        for asset in source["files"]:
            destination = safe_destination(output_dir, asset["path"])
            size, reused = download_verified(
                asset["url"], destination, asset["sha256"], asset["bytes"]
            )
            assets.append(
                {
                    "source": source["id"],
                    "revision": source["revision"],
                    "license": source["license"],
                    "path": asset["path"],
                    "sha256": asset["sha256"],
                    "bytes": size,
                    "reused": reused,
                }
            )
    return {"schema_version": 1, "asset_count": len(assets), "assets": assets}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = acquire_sources(args.manifest, args.output_dir)
    report_path = args.report or args.output_dir.parent / "acquisition-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
