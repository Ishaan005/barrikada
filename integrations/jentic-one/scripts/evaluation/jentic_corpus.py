"""Validation and metrics helpers for the versioned Jentic security corpus."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Surface = Literal["runtime_response", "specification"]
Label = Literal["benign", "prompt_injection"]
ExpectedVerdict = Literal["allow", "flag", "block"]
Split = Literal["train", "validation", "test"]
ReviewStatus = Literal["draft", "reviewed"]

SURFACES = {"runtime_response", "specification"}
LABELS = {"benign", "prompt_injection"}
EXPECTED_VERDICTS = {"allow", "flag", "block"}
SPLITS = {"train", "validation", "test"}
REVIEW_STATUSES = {"draft", "reviewed"}
CONTENT_KINDS = {
    "html",
    "json_key",
    "json_value",
    "problem",
    "specification",
    "specification_default",
    "specification_description",
    "specification_example",
    "specification_external_docs",
    "specification_summary",
    "specification_tags",
    "specification_title",
    "text",
    "xml",
    "yaml",
}
PROVENANCE_KINDS = {"curated", "synthetic", "redacted_production"}
SEVERITIES = {"none", "medium", "high", "critical"}
REQUIRED_FIELDS = {
    "id",
    "surface",
    "content_kind",
    "label",
    "expected_verdict",
    "attack_category",
    "severity",
    "family",
    "split",
    "review_status",
    "provenance",
    "text",
}
CASE_ID = re.compile(r"^jtc-v1-(?:runtime|spec)-[a-z0-9-]+$")
SUSPICIOUS_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[opusr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{24,}\b"),
    re.compile(r"(?i)authorization\s*:\s*bearer\s+[A-Za-z0-9._~-]{16,}"),
    re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])"),
    re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    re.compile(r"(?<![\w])(?:\+?\d[ ()-]*){9,}\d(?![\w])"),
)


class CorpusValidationError(ValueError):
    """Raised when corpus data violates a closed validation rule."""


@dataclass(frozen=True)
class CorpusCase:
    id: str
    surface: Surface
    content_kind: str
    label: Label
    expected_verdict: ExpectedVerdict
    attack_category: str | None
    severity: str
    family: str
    split: Split
    review_status: ReviewStatus
    provenance: str
    text: str

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def _normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _case_from_mapping(raw: dict, line_number: int) -> CorpusCase:
    unknown = set(raw) - REQUIRED_FIELDS
    missing = REQUIRED_FIELDS - set(raw)
    if unknown or missing:
        raise CorpusValidationError(
            f"line {line_number}: closed schema mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    if not isinstance(raw["id"], str) or not CASE_ID.fullmatch(raw["id"]):
        raise CorpusValidationError(f"line {line_number}: invalid case id")
    if raw["surface"] not in SURFACES:
        raise CorpusValidationError(f"line {line_number}: invalid surface")
    if raw["content_kind"] not in CONTENT_KINDS:
        raise CorpusValidationError(f"line {line_number}: invalid content_kind")
    if raw["label"] not in LABELS:
        raise CorpusValidationError(f"line {line_number}: invalid label")
    if raw["expected_verdict"] not in EXPECTED_VERDICTS:
        raise CorpusValidationError(f"line {line_number}: invalid expected_verdict")
    if raw["split"] not in SPLITS:
        raise CorpusValidationError(f"line {line_number}: invalid split")
    if raw["review_status"] not in REVIEW_STATUSES:
        raise CorpusValidationError(f"line {line_number}: invalid review_status")
    if raw["provenance"] not in PROVENANCE_KINDS:
        raise CorpusValidationError(f"line {line_number}: invalid provenance")
    if raw["severity"] not in SEVERITIES:
        raise CorpusValidationError(f"line {line_number}: invalid severity")
    if not isinstance(raw["family"], str) or not re.fullmatch(r"[a-z0-9_-]{2,64}", raw["family"]):
        raise CorpusValidationError(f"line {line_number}: invalid family")
    if not isinstance(raw["text"], str) or not raw["text"].strip():
        raise CorpusValidationError(f"line {line_number}: text must be non-empty")
    if len(raw["text"].encode("utf-8")) > 64 * 1024:
        raise CorpusValidationError(f"line {line_number}: text exceeds 64 KiB")

    if raw["label"] == "benign":
        if raw["expected_verdict"] != "allow":
            raise CorpusValidationError(f"line {line_number}: benign cases must expect allow")
        if raw["attack_category"] is not None or raw["severity"] != "none":
            raise CorpusValidationError(
                f"line {line_number}: benign cases cannot have an attack category or severity"
            )
    else:
        if raw["expected_verdict"] == "allow":
            raise CorpusValidationError(f"line {line_number}: injection cases cannot expect allow")
        if not isinstance(raw["attack_category"], str) or not re.fullmatch(
            r"[a-z0-9_]{3,64}", raw["attack_category"]
        ):
            raise CorpusValidationError(f"line {line_number}: invalid attack_category")
        if raw["severity"] == "none":
            raise CorpusValidationError(f"line {line_number}: injection severity cannot be none")

    is_specification_kind = raw["content_kind"].startswith("specification")
    if raw["surface"] == "specification" and not is_specification_kind:
        raise CorpusValidationError(
            f"line {line_number}: specification cases require specification content_kind"
        )
    if raw["surface"] == "runtime_response" and is_specification_kind:
        raise CorpusValidationError(
            f"line {line_number}: runtime cases cannot use specification content_kind"
        )
    for pattern in SUSPICIOUS_SECRET_PATTERNS:
        if pattern.search(raw["text"]):
            raise CorpusValidationError(
                f"line {line_number}: possible live credential/private key in corpus text"
            )
    return CorpusCase(**raw)


def load_corpus(path: Path) -> list[CorpusCase]:
    cases: list[CorpusCase] = []
    seen_ids: set[str] = set()
    seen_text: dict[str, str] = {}
    family_splits: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusValidationError(f"line {line_number}: invalid JSON") from exc
            if not isinstance(raw, dict):
                raise CorpusValidationError(f"line {line_number}: expected a JSON object")
            case = _case_from_mapping(raw, line_number)
            if case.id in seen_ids:
                raise CorpusValidationError(f"line {line_number}: duplicate id {case.id}")
            normalized_digest = hashlib.sha256(
                _normalized_text(case.text).encode("utf-8")
            ).hexdigest()
            if normalized_digest in seen_text:
                raise CorpusValidationError(
                    f"line {line_number}: normalized text duplicates {seen_text[normalized_digest]}"
                )
            previous_split = family_splits.setdefault(case.family, case.split)
            if previous_split != case.split:
                raise CorpusValidationError(
                    f"line {line_number}: family {case.family} crosses split boundary"
                )
            seen_ids.add(case.id)
            seen_text[normalized_digest] = case.id
            cases.append(case)

    if not cases:
        raise CorpusValidationError("corpus is empty")
    represented = {(case.surface, case.label) for case in cases}
    expected = {(surface, label) for surface in SURFACES for label in LABELS}
    if represented != expected:
        raise CorpusValidationError(
            f"corpus must represent every surface/label pair; missing={sorted(expected - represented)}"
        )
    return cases


def corpus_summary(cases: list[CorpusCase]) -> dict:
    canonical = "\n".join(
        json.dumps(case.__dict__, sort_keys=True, separators=(",", ":")) for case in cases
    ).encode("utf-8")
    by_surface_label = Counter(f"{case.surface}:{case.label}" for case in cases)
    return {
        "schema_version": "jentic-corpus-v1",
        "case_count": len(cases),
        "reviewed_count": sum(case.review_status == "reviewed" for case in cases),
        "draft_count": sum(case.review_status == "draft" for case in cases),
        "by_surface_label": dict(sorted(by_surface_label.items())),
        "splits": dict(sorted(Counter(case.split for case in cases).items())),
        "corpus_sha256": hashlib.sha256(canonical).hexdigest(),
        "release_review_ready": all(case.review_status == "reviewed" for case in cases),
    }
