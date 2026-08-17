"""Build redacted AgentDojo attack training rows from its pinned benchmark archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath

INTEGRATION_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[4]
for path in (ROOT, INTEGRATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.evaluation.build_jentic_review_pool import (  # noqa: E402
    normalize_text,
    redact_text,
)
from scripts.evaluation.jentic_corpus import load_corpus  # noqa: E402

DEFAULT_ARCHIVE = ROOT / "build" / "jentic-training-sources" / "ethz-spylab-agentdojo.tar.gz"
DEFAULT_REVIEWED = INTEGRATION_ROOT / "evaluation" / "corpora" / "jentic-v1" / "reviewed.jsonl"
CONTEXTS = (
    ("runtime_response", "Tool response field:\n"),
    ("runtime_response", "Upstream error detail:\n"),
    ("specification", "Operation example:\n"),
    ("specification", "Parameter description:\n"),
)


def _digest(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _archive_injections(path: Path) -> set[str]:
    injections = set()
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            member_path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or "runs" not in member_path.parts
                or member_path.suffix != ".json"
            ):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            raw = json.load(handle)
            values = raw.get("injections") or {}
            if isinstance(values, dict):
                injections.update(value for value in values.values() if isinstance(value, str))
    return injections


def build_rows(archive: Path, reviewed: Path) -> tuple[list[dict], dict]:
    held_out = {_digest(case.text) for case in load_corpus(reviewed)}
    redactions: Counter = Counter()
    rows = []
    seen = set()
    raw_injections = _archive_injections(archive)
    for raw_text in sorted(raw_injections):
        injection = redact_text(raw_text, redactions)
        family = hashlib.sha256(normalize_text(injection).encode("utf-8")).hexdigest()[:20]
        for surface, prefix in CONTEXTS:
            text = f"{prefix}{injection}"
            digest = _digest(text)
            if digest in held_out or digest in seen:
                continue
            seen.add(digest)
            rows.append(
                {
                    "id": f"agentdojo-{digest[:20]}",
                    "text": text,
                    "label": 1,
                    "surface": surface,
                    "family": f"agentdojo_{family}",
                    "split": "train",
                    "provenance": "public_redacted",
                }
            )
    rows.sort(key=lambda row: row["id"])
    report = {
        "schema_version": 1,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "reviewed_corpus_sha256": hashlib.sha256(reviewed.read_bytes()).hexdigest(),
        "unique_injections": len(raw_injections),
        "rows": len(rows),
        "held_out_overlap_count": 0,
        "redactions": dict(sorted(redactions.items())),
    }
    return rows, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--reviewed", type=Path, default=DEFAULT_REVIEWED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rows, report = build_rows(args.archive, args.reviewed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
