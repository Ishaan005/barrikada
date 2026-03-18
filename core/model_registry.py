from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def current_git_commit(project_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_sha256(path: Path) -> str:
    h = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file())
    for f in files:
        rel = f.relative_to(path).as_posix().encode("utf-8")
        h.update(rel)
        with f.open("rb") as fp:
            for chunk in iter(lambda: fp.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def update_latest_pointer(release_root: Path, version: str) -> None:
    release_root.mkdir(parents=True, exist_ok=True)
    (release_root / "LATEST").write_text(version.strip() + "\n")


def read_latest_pointer(release_root: Path) -> str | None:
    pointer = release_root / "LATEST"
    if not pointer.exists():
        return None
    value = pointer.read_text().strip()
    return value or None
