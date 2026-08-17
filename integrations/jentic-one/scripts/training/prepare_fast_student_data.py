"""Prepare pinned public and local synthetic data for fast student training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

INTEGRATION_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = INTEGRATION_ROOT / "evaluation" / "sources" / "jentic-training-sources.json"


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _read_synthetic(path: Path | None) -> list[dict]:
    if path is None:
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def prepare(args) -> tuple[list[dict], dict]:
    from datasets import load_dataset  # noqa: PLC0415

    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    source = manifest["sources"][0]
    dataset = {
        split: load_dataset(
            source["dataset_id"],
            revision=source["revision"],
            split=split,
            trust_remote_code=False,
        )
        for split in source["allowed_splits"]
    }
    rows = []
    for split in source["allowed_splits"]:
        if split not in dataset:
            raise ValueError(f"pinned dataset does not contain required split {split}")
        for item in dataset[split]:
            text = item.get("text")
            label = item.get("label")
            if not isinstance(text, str) or not text.strip() or label not in {0, 1}:
                raise ValueError(f"invalid row in pinned dataset split {split}")
            digest = hashlib.sha256(_normalize(text).encode()).hexdigest()
            rows.append(
                {
                    "id": f"slabs-{digest[:20]}",
                    "text": text,
                    "label": int(label),
                    "surface": "runtime_response" if int(digest[0], 16) % 2 else "specification",
                    "family": f"slabs_{digest[:24]}",
                    "split": split,
                    "provenance": "public_mit",
                }
            )

    rows.extend(_read_synthetic(args.synthetic))
    deduplicated = {}
    duplicate_count = 0
    # Validation wins on cross-source duplicates so it cannot leak into training.
    for row in sorted(rows, key=lambda value: value["split"] == "train"):
        digest = hashlib.sha256(_normalize(row["text"]).encode()).hexdigest()
        if digest in deduplicated:
            duplicate_count += 1
            continue
        deduplicated[digest] = row
    prepared = sorted(deduplicated.values(), key=lambda row: row["id"])
    report = {
        "rows": len(prepared),
        "duplicates_removed": duplicate_count,
        "source_manifest_sha256": hashlib.sha256(SOURCE_MANIFEST.read_bytes()).hexdigest(),
        "source_revision": source["revision"],
        "source_license": source["license"],
        "included_splits": source["allowed_splits"],
        "test_split_included": False,
        "counts": {
            f"{split}:{label}": sum(
                row["split"] == split and row["label"] == label for row in prepared
            )
            for split in ("train", "validation")
            for label in (0, 1)
        },
    }
    return prepared, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path)
    args = parser.parse_args()
    rows, report = prepare(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
