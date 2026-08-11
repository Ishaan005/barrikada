"""Fail CI when Core's v2 OpenAPI changes without updating the thin client."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parents[1]
EXPECTED_HASH = ROOT / "openapi" / "barrikade-v2.sha256"
GENERATED_CONTRACT = ROOT / "src" / "barrikade_jentic" / "generated" / "contract.py"

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from api.server import app  # noqa: E402


def _references(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        reference = value.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/components/schemas/"):
            found.add(reference.rsplit("/", 1)[-1])
        for child in value.values():
            found.update(_references(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_references(child))
    return found


def canonical_v2_contract() -> bytes:
    document = app.openapi()
    paths = {key: value for key, value in document["paths"].items() if key.startswith("/v2/")}
    all_schemas = document.get("components", {}).get("schemas", {})
    selected: dict[str, Any] = {}
    pending = _references(paths)
    while pending:
        name = pending.pop()
        if name in selected or name not in all_schemas:
            continue
        selected[name] = all_schemas[name]
        pending.update(_references(all_schemas[name]))
    contract = {
        "openapi": document["openapi"],
        "paths": paths,
        "components": {"schemas": selected},
    }
    return json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-hash", action="store_true")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Update generated contract linkage after the client change has been reviewed.",
    )
    args = parser.parse_args(argv)
    actual = hashlib.sha256(canonical_v2_contract()).hexdigest()
    if args.print_hash:
        print(actual)
        return 0
    expected = EXPECTED_HASH.read_text(encoding="utf-8").strip()
    generated = (
        '"""Generated from Barrikade Core OpenAPI. Do not edit manually."""\n\n'
        f'CONTRACT_SHA256 = "{actual}"\n'
        'CONTRACT_VERSION = "2"\n'
    )
    if args.write:
        EXPECTED_HASH.write_text(actual + "\n", encoding="utf-8")
        GENERATED_CONTRACT.write_text(generated, encoding="utf-8")
        return 0
    generated_matches = GENERATED_CONTRACT.read_text(encoding="utf-8") == generated
    if actual != expected or not generated_matches:
        print(
            "Barrikade v2 OpenAPI drifted. Regenerate/review the checked-in Jentic client, "
            f"then update {EXPECTED_HASH} and {GENERATED_CONTRACT} to {actual}."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
