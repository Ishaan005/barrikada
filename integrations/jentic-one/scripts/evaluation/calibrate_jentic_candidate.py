"""Calibrate a Layer D candidate on the approved Jentic validation partition."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

INTEGRATION_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for path in (REPOSITORY_ROOT, INTEGRATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.profile_routing import is_low_risk_short_metadata  # noqa: E402
from core.release_policy import (  # noqa: E402
    MAX_BENIGN_FALSE_BLOCK_RATE,
    MAX_BENIGN_FLAG_RATE,
    MIN_ATTACK_RECALL,
    as_dict,
)

from scripts.evaluation.jentic_corpus import load_corpus  # noqa: E402


def calibrate(
    labels: np.ndarray, scores: np.ndarray, surfaces: np.ndarray
) -> tuple[float, float, dict]:
    candidates = np.linspace(0.001, 0.999, num=999)
    cached = {}
    for surface in sorted(set(surfaces)):
        selected = surfaces == surface
        surface_labels = labels[selected]
        surface_scores = scores[selected]
        attacks = surface_labels == 1
        benign = surface_labels == 0
        cached[str(surface)] = {
            float(threshold): (
                float(np.mean(surface_scores[attacks] >= threshold)),
                float(np.mean(surface_scores[benign] >= threshold)),
            )
            for threshold in candidates
        }
    best = None
    best_valid = None
    for low in candidates:
        for high in candidates:
            if low >= high:
                continue
            rates = {}
            for surface, values in cached.items():
                attack_recall, benign_nonallow = values[float(low)]
                attack_block, benign_block = values[float(high)]
                rates[surface] = {
                    "attack_recall": attack_recall,
                    "attack_block_rate": attack_block,
                    "benign_false_block_rate": benign_block,
                    "benign_flag_rate": max(0.0, benign_nonallow - benign_block),
                }
            minimum_recall = min(item["attack_recall"] for item in rates.values())
            maximum_block = max(item["benign_false_block_rate"] for item in rates.values())
            maximum_flag = max(item["benign_flag_rate"] for item in rates.values())
            mean_attack_block = float(
                np.mean([item["attack_block_rate"] for item in rates.values()])
            )
            quality = minimum_recall + mean_attack_block - (4 * maximum_block) - (2 * maximum_flag)
            item = (quality, float(low), float(high), rates)
            if best is None or (item[0], item[1], -item[2]) > (best[0], best[1], -best[2]):
                best = item
            valid = (
                minimum_recall >= MIN_ATTACK_RECALL
                and maximum_block <= MAX_BENIGN_FALSE_BLOCK_RATE
                and maximum_flag <= MAX_BENIGN_FLAG_RATE
            )
            if valid and (
                best_valid is None
                or (item[0], item[1], -item[2]) > (best_valid[0], best_valid[1], -best_valid[2])
            ):
                best_valid = item
    selected = best_valid or best
    if selected is None:
        raise ValueError("calibration grid is empty")
    _, low, high, rates = selected
    return low, high, {"surfaces": rates, "validation_gates_passed": best_valid is not None}


def main() -> int:
    from core.layer_d.classifier import LayerDClassifier  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    cases = [case for case in load_corpus(args.corpus) if case.split == "validation"]
    if not cases:
        parser.error("corpus has no validation partition")
    classifier = LayerDClassifier(args.model, low=0.05, high=0.95, max_length=args.max_length)
    scores = []
    for start in range(0, len(cases), args.batch_size):
        scores.extend(
            float(value)
            for value in classifier.predict_batch(
                [case.text for case in cases[start : start + args.batch_size]]
            )
        )
    labels = np.asarray([case.label == "prompt_injection" for case in cases], dtype=int)
    surfaces = np.asarray([case.surface for case in cases])
    score_array = np.asarray(scores)
    for index, case in enumerate(cases):
        profile = "jentic_gateway_fast" if case.surface == "runtime_response" else "jentic_spec"
        if is_low_risk_short_metadata(case.text, profile, case.content_kind):
            score_array[index] = 0.0
    low, high, metrics = calibrate(labels, score_array, surfaces)
    report = {
        "model_dir": args.model.name,
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "calibration_split": "validation",
        "calibration_rows": len(cases),
        "release_policy": as_dict(),
        "routing_policy": "jentic_fast_v1",
        "thresholds": {"low": low, "high": high},
        "validation_metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if metrics["validation_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
