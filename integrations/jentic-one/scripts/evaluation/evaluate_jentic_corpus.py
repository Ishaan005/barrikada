"""Evaluate Barrikade or a Layer D candidate on the closed Jentic corpus."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from jentic_corpus import CorpusCase, corpus_summary, load_corpus

INTEGRATION_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CORPUS = INTEGRATION_ROOT / "evaluation" / "corpora" / "jentic-v1" / "reviewed.jsonl"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from core.release_policy import (  # noqa: E402
    MAX_BENIGN_FALSE_BLOCK_RATE,
    MAX_BENIGN_FLAG_RATE,
    MAX_P95_LATENCY_MS,
    MIN_ATTACK_RECALL,
    as_dict,
)


def _p95(values: list[float]) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[94]


def _load_predictor(args):
    if args.pipeline_layer_d_model_dir:
        from core.layer_d.classifier import LayerDClassifier  # noqa: PLC0415
        from core.orchestrator import PIPipeline  # noqa: PLC0415

        pipeline = PIPipeline(profile="jentic_gateway_fast", prepare_artifacts=False)
        pipeline.layer_d_classifier = LayerDClassifier(
            model_dir=args.pipeline_layer_d_model_dir,
            low=args.low,
            high=args.high,
            max_length=args.max_length,
        )

        def predict(text: str, source: str | None = None) -> tuple[str, str]:
            result = pipeline.detect(text, source=source)
            return result.final_verdict.value, result.decision_layer.value

        return predict, "barrikade_fast_pipeline_with_candidate_layer_d"

    if args.layer_d_model_dir:
        from core.layer_d.classifier import LayerDClassifier  # noqa: PLC0415

        classifier = LayerDClassifier(
            model_dir=args.layer_d_model_dir,
            low=args.low,
            high=args.high,
            max_length=args.max_length,
        )

        def predict(text: str, source: str | None = None) -> tuple[str, str]:
            del source
            return classifier.predict(text).verdict, "layer_d"

        return predict, "layer_d_candidate"

    from core.orchestrator import PIPipeline  # noqa: PLC0415

    pipeline = PIPipeline(profile="jentic_gateway_fast", prepare_artifacts=False)

    def predict(text: str, source: str | None = None) -> tuple[str, str]:
        result = pipeline.detect(text, source=source)
        return result.final_verdict.value, result.decision_layer.value

    return predict, "barrikade_fast_pipeline"


def _surface_metrics(cases: list[CorpusCase], rows: list[dict]) -> dict:
    metrics = {}
    for surface in sorted({case.surface for case in cases}):
        surface_cases = [case for case in cases if case.surface == surface]
        ids = {case.id for case in surface_cases}
        surface_rows = [row for row in rows if row["id"] in ids]
        labels = {case.id: case.label for case in surface_cases}
        malicious = [row for row in surface_rows if labels[row["id"]] == "prompt_injection"]
        benign = [row for row in surface_rows if labels[row["id"]] == "benign"]
        attack_detected = sum(row["actual_verdict"] != "allow" for row in malicious)
        benign_blocks = sum(row["actual_verdict"] == "block" for row in benign)
        benign_flags = sum(row["actual_verdict"] == "flag" for row in benign)
        metrics[surface] = {
            "attack_cases": len(malicious),
            "benign_cases": len(benign),
            "attack_recall": attack_detected / len(malicious),
            "benign_false_block_rate": benign_blocks / len(benign),
            "benign_flag_rate": benign_flags / len(benign),
            "latency_p95_ms": _p95([row["latency_ms"] for row in surface_rows]),
            "verdicts": dict(
                sorted(Counter(row["actual_verdict"] for row in surface_rows).items())
            ),
            "decision_layers": dict(
                sorted(Counter(row["decision_layer"] for row in surface_rows).items())
            ),
            "benign_block_decision_layers": dict(
                sorted(
                    Counter(
                        row["decision_layer"] for row in benign if row["actual_verdict"] == "block"
                    ).items()
                )
            ),
            "attack_allow_decision_layers": dict(
                sorted(
                    Counter(
                        row["decision_layer"]
                        for row in malicious
                        if row["actual_verdict"] == "allow"
                    ).items()
                )
            ),
        }
    return metrics


def evaluate(args) -> dict:
    corpus_cases = load_corpus(args.corpus)
    cases = [case for case in corpus_cases if args.split is None or case.split == args.split]
    if not cases:
        raise ValueError(f"corpus contains no {args.split!r} cases")
    predict, predictor_name = _load_predictor(args)
    for _ in range(args.warmup):
        predict("Routine warm-up response with no instructions.", "text")

    rows = []
    critical_misses = []
    exact_mismatches = []
    for case in cases:
        started = time.perf_counter()
        actual, decision_layer = predict(case.text, case.content_kind)
        elapsed_ms = (time.perf_counter() - started) * 1000
        rows.append(
            {
                "id": case.id,
                "actual_verdict": actual,
                "decision_layer": decision_layer,
                "latency_ms": elapsed_ms,
            }
        )
        if case.severity == "critical" and actual == "allow":
            critical_misses.append(case.id)
        if actual != case.expected_verdict:
            exact_mismatches.append(case.id)

    surfaces = _surface_metrics(cases, rows)
    summary = corpus_summary(corpus_cases)
    enough_cases = all(
        value["attack_cases"] >= args.minimum_cases and value["benign_cases"] >= args.minimum_cases
        for value in surfaces.values()
    )
    metrics_pass = all(
        value["attack_recall"] >= MIN_ATTACK_RECALL
        and value["benign_false_block_rate"] <= MAX_BENIGN_FALSE_BLOCK_RATE
        and value["benign_flag_rate"] <= MAX_BENIGN_FLAG_RATE
        and value["latency_p95_ms"] <= MAX_P95_LATENCY_MS
        for value in surfaces.values()
    )
    report = {
        "predictor": predictor_name,
        "corpus": summary,
        "evaluated_split": args.split or "all",
        "evaluated_case_count": len(cases),
        "release_policy": as_dict(),
        "surface_metrics": surfaces,
        "critical_miss_ids": sorted(critical_misses),
        "exact_mismatch_ids": sorted(exact_mismatches),
        "minimum_cases_per_surface_label": args.minimum_cases,
        "statistical_sample_ready": enough_cases,
        "quality_gates_passed": bool(enough_cases and not critical_misses and metrics_pass),
        "release_gates_passed": bool(
            summary["release_review_ready"]
            and args.split == "test"
            and enough_cases
            and not critical_misses
            and metrics_pass
        ),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer-d-model-dir", type=Path)
    parser.add_argument("--pipeline-layer-d-model-dir", type=Path)
    parser.add_argument("--low", type=float, default=0.05)
    parser.add_argument("--high", type=float, default=0.95)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--minimum-cases",
        type=int,
        default=1000,
        help="Minimum benign and attack rows per surface for a release decision.",
    )
    args = parser.parse_args()
    if args.layer_d_model_dir and args.pipeline_layer_d_model_dir:
        parser.error("choose only one Layer D candidate mode")
    if not 0 <= args.low < args.high <= 1:
        parser.error("expected 0 <= --low < --high <= 1")
    report = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["release_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
