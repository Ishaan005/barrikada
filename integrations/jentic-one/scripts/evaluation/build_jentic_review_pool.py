"""Build a redacted, deterministic draft pool for independent security review."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

try:
    from scripts.evaluation.jentic_corpus import corpus_summary, load_corpus
except ModuleNotFoundError:  # Direct execution places this script's directory on sys.path.
    from jentic_corpus import corpus_summary, load_corpus


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE_DIR = REPOSITORY_ROOT / "build" / "jentic-corpus-sources" / "raw"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "build" / "jentic-corpus-sources" / "jentic-review-pool.jsonl"
SPEC_CONTENT_KINDS = {
    "title": "specification_title",
    "summary": "specification_summary",
    "description": "specification_description",
    "default": "specification_default",
    "example": "specification_example",
    "examples": "specification_example",
    "externalDocs": "specification_external_docs",
    "tags": "specification_tags",
}
EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])")
SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
PHONE = re.compile(r"(?<![\w])(?:\+?\d[ ()-]*){9,}\d(?![\w])")
SUBJECT_ID = re.compile(r"\bP-[A-Z0-9]{4,}\b", re.IGNORECASE)
PRIVATE_KEY = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----.*?"
    r"-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    re.DOTALL,
)
AWS_KEY = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
API_KEY = re.compile(r"\b(?:gh[opusr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{24,})\b")
BEARER = re.compile(r"(?i)authorization\s*:\s*bearer\s+[A-Za-z0-9._~-]{16,}")


def redact_text(text: str, counters: Counter) -> str:
    rules = (
        (PRIVATE_KEY, "<PRIVATE_KEY>", "private_key"),
        (BEARER, "Authorization: Bearer <TOKEN>", "bearer_token"),
        (AWS_KEY, "<API_KEY>", "api_key"),
        (API_KEY, "<API_KEY>", "api_key"),
        (EMAIL, "<EMAIL_ADDRESS>", "email"),
        (SSN, "<SSN>", "ssn"),
        (PHONE, "<PHONE>", "phone"),
        (SUBJECT_ID, "<SUBJECT_ID>", "subject_id"),
    )
    result = text
    for pattern, replacement, name in rules:
        result, count = pattern.subn(replacement, result)
        counters[name] += count
    return result.strip()


def walk_json_items(value: Any, *, include_keys: bool = True) -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            if include_keys:
                yield "json_key", str(key)
            yield from walk_json_items(child, include_keys=include_keys)
    elif isinstance(value, list):
        for child in value:
            yield from walk_json_items(child, include_keys=include_keys)
    elif isinstance(value, str):
        yield "json_value", value


def walk_json_strings(value: Any, *, include_keys: bool = True) -> Iterable[str]:
    for _, text in walk_json_items(value, include_keys=include_keys):
        yield text


def walk_spec_items(value: Any, selected: str | None = None) -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_selected = SPEC_CONTENT_KINDS.get(str(key), selected)
            yield from walk_spec_items(child, child_selected)
    elif isinstance(value, list):
        for child in value:
            yield from walk_spec_items(child, selected)
    elif selected is not None and isinstance(value, str):
        yield selected, value


def walk_spec_strings(value: Any, selected: str | None = None) -> Iterable[str]:
    for _, text in walk_spec_items(value, selected):
        yield text


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _select_unique(
    values: Iterable[str], *, limit: int, redactions: Counter, globally_seen: set[str]
) -> list[str]:
    candidates: dict[str, str] = {}
    for value in values:
        text = redact_text(value, redactions)
        if not text or len(text.encode("utf-8")) > 64 * 1024:
            continue
        normalized = normalize_text(text)
        if len(normalized) < 3:
            continue
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if digest not in globally_seen:
            candidates.setdefault(digest, text)
    selected = sorted(candidates.items())[:limit]
    globally_seen.update(digest for digest, _ in selected)
    return [text for _, text in selected]


def _select_unique_items(
    values: Iterable[tuple[str, str]],
    *,
    limit: int,
    redactions: Counter,
    globally_seen: set[str],
) -> list[tuple[str, str]]:
    candidates: dict[str, tuple[str, str]] = {}
    for content_kind, value in values:
        text = redact_text(value, redactions)
        normalized = normalize_text(text)
        if not text or len(text.encode("utf-8")) > 64 * 1024 or len(normalized) < 3:
            continue
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if digest not in globally_seen:
            candidates.setdefault(digest, (content_kind, text))
    selected = sorted(candidates.items())[:limit]
    globally_seen.update(digest for digest, _ in selected)
    return [item for _, item in selected]


def _case_id(surface: str, label: str, source: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    surface_name = "runtime" if surface == "runtime_response" else "spec"
    return f"jtc-v1-{surface_name}-{label.replace('_', '-')}-{source}-{digest}"


def _benign_case(
    surface: str,
    source: str,
    text: str,
    family_index: int,
    *,
    content_kind: str,
    provenance: str = "curated",
) -> dict:
    surface_name = "runtime" if surface == "runtime_response" else "spec"
    return {
        "id": _case_id(surface, "benign", source, text),
        "surface": surface,
        "content_kind": content_kind,
        "label": "benign",
        "expected_verdict": "allow",
        "attack_category": None,
        "severity": "none",
        "family": f"{source}_{surface_name}_benign_{family_index}",
        "split": "validation",
        "review_status": "draft",
        "provenance": provenance,
        "text": text,
    }


def _attack_case(
    surface: str,
    *,
    source: str,
    source_id: str,
    raw_category: str,
    content_kind: str,
    text: str,
) -> dict:
    category = re.sub(r"[^a-z0-9_]+", "_", raw_category.casefold()).strip("_")
    family_digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:16]
    return {
        "id": _case_id(surface, "prompt-injection", source, text),
        "surface": surface,
        "content_kind": content_kind,
        "label": "prompt_injection",
        "expected_verdict": "block",
        "attack_category": category,
        "severity": "critical" if category == "exfiltration" else "high",
        "family": f"{source}_{family_digest}",
        "split": "validation",
        "review_status": "draft",
        "provenance": "synthetic",
        "text": text,
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_review_pool(source_dir: Path) -> tuple[list[dict], dict]:
    redactions: Counter = Counter()
    seen: set[str] = set()
    cases: list[dict] = []
    source_counts: Counter = Counter()

    with (source_dir / "nvidia-agentic-ipi-v1" / "train.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            injection = redact_text(str(row["injection"]["injection_text"]), redactions)
            for surface, prefix in (
                ("runtime_response", "Tool response note:\n"),
                ("specification", "Operation description:\n"),
            ):
                text = f"{prefix}{injection}"
                digest = hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                cases.append(
                    _attack_case(
                        surface,
                        source="nvidia",
                        source_id=str(row["id"]),
                        raw_category=str(
                            row.get("attack_category")
                            or row["injection"].get("category")
                            or "indirect_instruction"
                        ),
                        content_kind=(
                            "json_value"
                            if surface == "runtime_response"
                            else "specification_description"
                        ),
                        text=text,
                    )
                )
                source_counts[f"nvidia-agentic-ipi-v1:{surface}:prompt_injection"] += 1

    injecagent_attack_rows = []
    for filename in ("test_cases_ds_enhanced.json", "test_cases_dh_enhanced.json"):
        injecagent_attack_rows.extend(read_json(source_dir / "injecagent" / filename))
    injecagent_candidates: dict[str, tuple[str, str]] = {}
    for row in injecagent_attack_rows:
        raw_response = redact_text(str(row["Tool Response"]), redactions)
        normalized = normalize_text(raw_response)
        if not normalized or len(raw_response.encode("utf-8")) > 64 * 1024:
            continue
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        injecagent_candidates.setdefault(
            digest, (raw_response, str(row.get("Attack Type", "other")))
        )
    for digest, (response, category) in sorted(injecagent_candidates.items())[:500]:
        for surface, prefix in (
            ("runtime_response", "Tool response:\n"),
            ("specification", "Response example:\n"),
        ):
            text = f"{prefix}{response}"
            text_digest = hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
            if text_digest in seen:
                continue
            seen.add(text_digest)
            cases.append(
                _attack_case(
                    surface,
                    source="injecagent",
                    source_id=digest,
                    raw_category=category,
                    content_kind=(
                        "json_value" if surface == "runtime_response" else "specification_example"
                    ),
                    text=text,
                )
            )
            source_counts[f"injecagent:{surface}:prompt_injection"] += 1

    specification_sources = (
        ("stripe", source_dir / "openapi" / "stripe.json"),
        ("github", source_dir / "openapi" / "github.json"),
        ("openai", source_dir / "openapi" / "openai.json"),
        ("jentic", source_dir / "openapi" / "jentic.json"),
    )
    for source, path in specification_sources:
        strings = _select_unique_items(
            walk_spec_items(read_json(path)),
            limit=500,
            redactions=redactions,
            globally_seen=seen,
        )
        for index, (content_kind, text) in enumerate(strings):
            cases.append(
                _benign_case(
                    "specification",
                    source,
                    text,
                    index,
                    content_kind=content_kind,
                )
            )
            source_counts[f"{source}:specification:benign"] += 1

    stripe_runtime = _select_unique_items(
        walk_json_items(read_json(source_dir / "openapi" / "stripe-fixtures.json")),
        limit=600,
        redactions=redactions,
        globally_seen=seen,
    )
    for index, (content_kind, text) in enumerate(stripe_runtime):
        cases.append(
            _benign_case(
                "runtime_response",
                "stripe",
                text,
                index,
                content_kind=content_kind,
            )
        )
        source_counts["stripe:runtime_response:benign"] += 1

    injecagent = read_json(source_dir / "injecagent" / "attacker_simulated_responses.json")

    def injecagent_strings() -> Iterable[tuple[str, str]]:
        for response in injecagent.values():
            try:
                value = json.loads(response)
            except (json.JSONDecodeError, TypeError):
                value = response
            yield from walk_json_items(value)

    injecagent_runtime = _select_unique_items(
        injecagent_strings(), limit=600, redactions=redactions, globally_seen=seen
    )
    for index, (content_kind, text) in enumerate(injecagent_runtime):
        cases.append(
            _benign_case(
                "runtime_response",
                "injecagent",
                text,
                index,
                content_kind=content_kind,
                provenance="synthetic",
            )
        )
        source_counts["injecagent:runtime_response:benign"] += 1

    cases.sort(key=lambda case: case["id"])
    report = {
        "schema_version": 1,
        "purpose": "draft_review_pool_only",
        "source_counts": dict(sorted(source_counts.items())),
        "redactions": dict(sorted(redactions.items())),
    }
    return cases, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    cases, report = build_review_pool(args.source_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(case, sort_keys=True) + "\n" for case in cases), encoding="utf-8"
    )
    report["corpus"] = corpus_summary(load_corpus(args.output))
    report_path = args.report or args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
