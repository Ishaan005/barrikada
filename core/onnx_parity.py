"""Shared ONNX release-gate calculations for Layers B, C, and D."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal


Verdict = Literal["allow", "flag", "block"]


def evaluate_backend_parity(
    reference_verdicts: Sequence[Verdict],
    candidate_verdicts: Sequence[Verdict],
    labels: Sequence[Literal["benign", "attack"]],
    *,
    reference_p95_ms: float,
    candidate_p95_ms: float,
) -> dict[str, float | int | bool]:
    if not reference_verdicts or len(reference_verdicts) != len(candidate_verdicts):
        raise ValueError("reference and candidate verdicts must have the same non-zero length")
    if len(labels) != len(reference_verdicts):
        raise ValueError("labels must align with verdicts")

    total = len(labels)
    benign = sum(label == "benign" for label in labels)
    attacks = sum(label == "attack" for label in labels)
    if benign == 0 or attacks == 0:
        raise ValueError("parity evaluation requires benign and attack samples")

    disagreement = sum(
        reference != candidate
        for reference, candidate in zip(reference_verdicts, candidate_verdicts, strict=True)
    )

    def false_block_rate(values: Sequence[Verdict]) -> float:
        return (
            sum(
                verdict == "block" and label == "benign"
                for verdict, label in zip(values, labels, strict=True)
            )
            / benign
        )

    def recall(values: Sequence[Verdict]) -> float:
        return (
            sum(
                verdict == "block" and label == "attack"
                for verdict, label in zip(values, labels, strict=True)
            )
            / attacks
        )

    reference_false_blocks = false_block_rate(reference_verdicts)
    candidate_false_blocks = false_block_rate(candidate_verdicts)
    reference_recall = recall(reference_verdicts)
    candidate_recall = recall(candidate_verdicts)
    report: dict[str, float | int | bool] = {
        "sample_count": total,
        "verdict_disagreement_rate": disagreement / total,
        "reference_false_block_rate": reference_false_blocks,
        "candidate_false_block_rate": candidate_false_blocks,
        "reference_recall": reference_recall,
        "candidate_recall": candidate_recall,
        "reference_p95_ms": reference_p95_ms,
        "candidate_p95_ms": candidate_p95_ms,
    }
    report["passed"] = (
        report["verdict_disagreement_rate"] <= 0.005
        and candidate_false_blocks <= reference_false_blocks
        and candidate_recall >= reference_recall
        and (candidate_p95_ms < reference_p95_ms or candidate_recall > reference_recall)
    )
    return report
