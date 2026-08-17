"""Validate a Jentic corpus and emit a content-free manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jentic_corpus import corpus_summary, load_corpus

INTEGRATION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = INTEGRATION_ROOT / "evaluation" / "corpora" / "jentic-v1" / "reviewed.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument(
        "--require-reviewed",
        action="store_true",
        help="Fail unless every row has completed human review.",
    )
    args = parser.parse_args()

    summary = corpus_summary(load_corpus(args.corpus))
    if args.manifest_out:
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if args.require_reviewed and not summary["release_review_ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
