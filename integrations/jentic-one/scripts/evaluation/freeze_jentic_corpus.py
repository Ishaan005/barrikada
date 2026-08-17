"""Freeze an approved draft pool as the independent Jentic release-test corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

try:
    from scripts.evaluation.jentic_corpus import corpus_summary, load_corpus
except ModuleNotFoundError:  # Direct execution places this script's directory on sys.path.
    from jentic_corpus import corpus_summary, load_corpus


INTEGRATION_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = REPOSITORY_ROOT / "build" / "jentic-corpus-sources" / "jentic-review-pool.jsonl"
DEFAULT_OUTPUT = INTEGRATION_ROOT / "evaluation" / "corpora" / "jentic-v1" / "reviewed.jsonl"
DEFAULT_MANIFEST = (
    INTEGRATION_ROOT / "evaluation" / "corpora" / "jentic-v1" / "reviewed.manifest.json"
)
DEFAULT_SOURCES = INTEGRATION_ROOT / "evaluation" / "sources" / "jentic-corpus-sources.json"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_digest(text: str) -> str:
    normalized = " ".join(text.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _training_digests(paths: list[Path]) -> tuple[set[str], list[dict]]:
    digests: set[str] = set()
    files = []
    for path in paths:
        row_count = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if not isinstance(raw, dict) or not isinstance(raw.get("text"), str):
                    raise ValueError(f"{path.name}:{line_number}: invalid training row")
                digests.add(_normalized_digest(raw["text"]))
                row_count += 1
        files.append({"name": path.name, "sha256": _file_sha256(path), "row_count": row_count})
    return digests, files


def _assign_splits(cases: list, minimum_test_cases: int) -> dict[str, str]:
    family_strata: dict[str, set[tuple[str, str]]] = defaultdict(set)
    stratum_families: dict[tuple[str, str], set[str]] = defaultdict(set)
    for case in cases:
        stratum = (case.surface, case.label)
        family_strata[case.family].add(stratum)
        stratum_families[stratum].add(case.family)

    attack_runtime = stratum_families[("runtime_response", "prompt_injection")]
    attack_specification = stratum_families[("specification", "prompt_injection")]
    if attack_runtime != attack_specification:
        raise ValueError("attack families must pair runtime and specification variants")

    assignments: dict[str, str] = {}
    strata = (
        attack_runtime,
        stratum_families[("runtime_response", "benign")],
        stratum_families[("specification", "benign")],
    )
    for families in strata:
        if len(families) < minimum_test_cases:
            raise ValueError(
                f"not enough independent families for test: {len(families)} < {minimum_test_cases}"
            )
        ordered = sorted(
            families,
            key=lambda family: hashlib.sha256(family.encode("utf-8")).hexdigest(),
        )
        for family in ordered[:minimum_test_cases]:
            assignments[family] = "test"
        for family in ordered[minimum_test_cases:]:
            assignments[family] = "validation"
    if set(assignments) != set(family_strata):
        raise ValueError("a corpus family was not assigned to a release split")
    return assignments


def freeze_corpus(
    input_path: Path,
    output_path: Path,
    source_manifest_path: Path,
    training_paths: list[Path],
    *,
    approval_reference: str,
    approval_date: str,
    minimum_test_cases: int = 1000,
) -> dict:
    draft_cases = load_corpus(input_path)
    if any(case.review_status != "draft" for case in draft_cases):
        raise ValueError("freeze input must contain only draft cases")

    training_digests, training_files = _training_digests(training_paths)
    overlaps = sorted(
        case.id for case in draft_cases if _normalized_digest(case.text) in training_digests
    )
    if overlaps:
        raise ValueError(f"release corpus overlaps training data: {len(overlaps)} rows")

    assignments = _assign_splits(draft_cases, minimum_test_cases)
    rows = []
    for case in draft_cases:
        row = dict(case.__dict__)
        row["review_status"] = "reviewed"
        row["split"] = assignments[case.family]
        rows.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )

    reviewed_cases = load_corpus(output_path)
    sources = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    attribution = [
        {
            "id": source["id"],
            "homepage": source["homepage"],
            "license": source["license"],
            "revision": source["revision"],
        }
        for source in sources["sources"]
    ]
    return {
        "schema_version": 1,
        "approval": {
            "scope": "all_cases",
            "status": "approved",
            "reviewer_role": "project_owner",
            "reference": approval_reference,
            "date": approval_date,
        },
        "source_pool": {
            "name": input_path.name,
            "sha256": _file_sha256(input_path),
        },
        "source_manifest_sha256": _file_sha256(source_manifest_path),
        "reviewed_corpus_file": output_path.name,
        "reviewed_corpus_file_sha256": _file_sha256(output_path),
        "corpus": corpus_summary(reviewed_cases),
        "leakage_check": {
            "normalized_text_overlap_count": 0,
            "training_files": training_files,
        },
        "split_policy": {
            "method": "sha256_family_stratified",
            "minimum_test_cases_per_surface_label": minimum_test_cases,
            "training_rows": 0,
        },
        "attribution": attribution,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--training-data", type=Path, action="append", default=[])
    parser.add_argument("--approval-reference", required=True)
    parser.add_argument("--approval-date", required=True)
    parser.add_argument("--minimum-test-cases", type=int, default=1000)
    args = parser.parse_args()

    manifest = freeze_corpus(
        args.input,
        args.output,
        args.source_manifest,
        args.training_data,
        approval_reference=args.approval_reference,
        approval_date=args.approval_date,
        minimum_test_cases=args.minimum_test_cases,
    )
    args.manifest_out.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
