"""Acquire digest-pinned archive sources used for Jentic fast-model training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

INTEGRATION_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[4]
for path in (ROOT, INTEGRATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.evaluation.acquire_jentic_corpus_sources import (  # noqa: E402
    download_verified,
)

DEFAULT_MANIFEST = INTEGRATION_ROOT / "evaluation" / "sources" / "jentic-training-sources.json"
DEFAULT_OUTPUT = ROOT / "build" / "jentic-training-sources"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    reports = []
    for source in manifest["sources"]:
        archive = source.get("archive")
        if archive is None:
            continue
        destination = args.output_dir / f"{source['dataset_id'].replace('/', '-')}.tar.gz"
        size, reused = download_verified(
            archive["url"], destination, archive["sha256"], archive["bytes"]
        )
        reports.append(
            {
                "source": source["dataset_id"],
                "revision": source["revision"],
                "license": source["license"],
                "path": destination.name,
                "bytes": size,
                "sha256": archive["sha256"],
                "reused": reused,
            }
        )
    report = {"schema_version": 1, "archives": reports}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "acquisition-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
