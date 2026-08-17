"""Add non-held-out Jentic-shaped benign examples to fast-student training data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

INTEGRATION_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[4]
for path in (ROOT, INTEGRATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.evaluation.build_jentic_review_pool import (  # noqa: E402
    normalize_text,
    read_json,
    redact_text,
    walk_json_strings,
    walk_spec_strings,
)
from scripts.evaluation.jentic_corpus import load_corpus  # noqa: E402

DEFAULT_SOURCE_DIR = ROOT / "build" / "jentic-corpus-sources" / "raw"
DEFAULT_REVIEWED = INTEGRATION_ROOT / "evaluation" / "corpora" / "jentic-v1" / "reviewed.jsonl"


def _digest(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _select(
    values: Iterable[tuple[str, str]],
    *,
    limit: int,
    excluded: set[str],
    seen: set[str],
    redactions: Counter,
) -> list[tuple[str, str, str]]:
    candidates = {}
    for source, value in values:
        text = redact_text(value, redactions)
        digest = _digest(text)
        if (
            len(normalize_text(text)) < 3
            or len(text.encode("utf-8")) > 64 * 1024
            or digest in excluded
            or digest in seen
        ):
            continue
        candidates.setdefault(digest, (source, text))
    selected = []
    for digest, (source, text) in sorted(candidates.items())[:limit]:
        seen.add(digest)
        selected.append((source, digest, text))
    return selected


def _runtime_values(source_dir: Path) -> Iterable[tuple[str, str]]:
    stripe = read_json(source_dir / "openapi" / "stripe-fixtures.json")
    yield from (("stripe", text) for text in walk_json_strings(stripe))
    responses = read_json(source_dir / "injecagent" / "attacker_simulated_responses.json")
    for response in responses.values():
        try:
            value = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            value = response
        yield from (("injecagent", text) for text in walk_json_strings(value))


def _specification_values(source_dir: Path) -> Iterable[tuple[str, str]]:
    for source in ("stripe", "github", "openai", "jentic"):
        specification = read_json(source_dir / "openapi" / f"{source}.json")
        yield from ((source, text) for text in walk_spec_strings(specification))


def build_complement(
    base_path: Path,
    reviewed_path: Path,
    source_dir: Path,
    *,
    runtime_limit: int,
    specification_limit: int,
    additional_paths: list[Path] | None = None,
) -> tuple[list[dict], dict]:
    base_rows = [
        json.loads(line) for line in base_path.read_text(encoding="utf-8").splitlines() if line
    ]
    held_out = {_digest(case.text) for case in load_corpus(reviewed_path)}
    seen = {_digest(row["text"]) for row in base_rows}
    if held_out & seen:
        raise ValueError("base training data overlaps the reviewed corpus")
    additional_rows = []
    for path in additional_paths or []:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            digest = _digest(row["text"])
            if digest in held_out:
                raise ValueError(f"additional data {path.name} overlaps the reviewed corpus")
            if digest in seen:
                continue
            seen.add(digest)
            additional_rows.append(row)
    redactions: Counter = Counter()
    runtime = _select(
        _runtime_values(source_dir),
        limit=runtime_limit,
        excluded=held_out,
        seen=seen,
        redactions=redactions,
    )
    specification = _select(
        _specification_values(source_dir),
        limit=specification_limit,
        excluded=held_out,
        seen=seen,
        redactions=redactions,
    )
    additions = []
    for surface, selected in (("runtime_response", runtime), ("specification", specification)):
        for source, digest, text in selected:
            additions.append(
                {
                    "id": f"jentic-complement-{digest[:20]}",
                    "text": text,
                    "label": 0,
                    "surface": surface,
                    "family": f"complement_{source}_{digest[:20]}",
                    "split": "train",
                    "provenance": "public_redacted",
                }
            )
    rows = sorted([*base_rows, *additional_rows, *additions], key=lambda row: row["id"])
    report = {
        "schema_version": 1,
        "base_file_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "reviewed_corpus_sha256": hashlib.sha256(reviewed_path.read_bytes()).hexdigest(),
        "held_out_overlap_count": 0,
        "base_rows": len(base_rows),
        "added_rows": len(additions),
        "additional_attack_rows": len(additional_rows),
        "total_rows": len(rows),
        "added_by_surface": {
            "runtime_response": len(runtime),
            "specification": len(specification),
        },
        "redactions": dict(sorted(redactions.items())),
    }
    return rows, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--reviewed", type=Path, default=DEFAULT_REVIEWED)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--runtime-limit", type=int, default=2000)
    parser.add_argument("--specification-limit", type=int, default=2000)
    parser.add_argument("--additional", type=Path, action="append", default=[])
    args = parser.parse_args()

    rows, report = build_complement(
        args.base,
        args.reviewed,
        args.source_dir,
        runtime_limit=args.runtime_limit,
        specification_limit=args.specification_limit,
        additional_paths=args.additional,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
